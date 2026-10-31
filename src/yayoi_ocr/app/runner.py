"""認識待ちのページを順に認識して DB に書き込む(バックグラウンドのスレッド 1 本)。

- ジョブは作成時に全ページを `processing` で作り、ここが上から順に認識して `review` にする。利用者は最初のページが
  できた時点で確認を始められる(以降のページは確認している間に先読みで認識される)。
- 前のページの最後の年は、その認識が終わったページの `last_year` から引き継ぐ(年を書かない先頭行の補完)。
- 認識に失敗したページは `failed`(例外の型名だけを記録)にして次へ進む。再試行は `store.retry_page` + `submit`。
- アプリの再起動で中断した認識は、`start()` が `processing` のページを持つジョブを拾って続きから走らせる。
- ログには実データの値を出さない(ここではログ出力をしない)。
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from pathlib import Path

from . import gens, learned, settings, store
from .worker import Backend, PagePayload, upload_path


class JobRunner:
    def __init__(
        self,
        con: sqlite3.Connection,
        lock: threading.RLock,
        images_dir: Path,
        uploads_dir: Path,
        backend: Backend,
        models_root: Path | None = None,
        pagedata_root: Path | None = None,
    ) -> None:
        self.con = con
        self.lock = lock
        self.images_dir = images_dir
        self.uploads_dir = uploads_dir
        self.backend = backend
        self.models_root = models_root  # 利用者のデータから作った世代の置き場(<データ領域>/models)
        self.pagedata_root = (
            pagedata_root  # 認識の入力の保存先(<データ領域>/pagedata/<ページ ID>.npz)
        )
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._queued: set[int] = set()

    # ---- 制御

    def start(self) -> None:
        with self.lock:
            rows = self.con.execute(
                "SELECT DISTINCT job_id FROM pages WHERE status = 'processing' ORDER BY job_id"
            ).fetchall()
        for r in rows:
            self.submit(int(r["job_id"]))
        self._thread = threading.Thread(target=self._loop, name="yayoi-runner", daemon=True)
        self._thread.start()

    def submit(self, job_id: int) -> None:
        with self.lock:
            if job_id in self._queued:
                return
            self._queued.add(job_id)
            self._idle.clear()
        self._queue.put(job_id)

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.backend.close()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """キューが空になるまで待つ(テスト用)。"""
        return self._idle.wait(timeout)

    # ---- 実行

    def _loop(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if job_id is None:
                break
            try:
                self.run_job(job_id)
            except Exception as exc:  # 想定外でもスレッドは生かす(次のジョブを処理する)。握りつぶさず、型名を記録する
                try:
                    with self.lock:
                        store.fail_job(self.con, job_id, type(exc).__name__)
                except Exception:  # 記録そのものが失敗しても、スレッドは止めない(型名だけログへ)
                    store.log.error(
                        "job %s: could not record failure (%s)", job_id, type(exc).__name__
                    )
            finally:
                with self.lock:
                    self._queued.discard(job_id)
                    if not self._queued:
                        self._idle.set()

    def run_job(self, job_id: int) -> None:
        """このジョブの認識待ちのページを、上から順に全部認識する。"""
        while not self._stopping.is_set():
            with self.lock:
                rows = store.processing_pages(self.con, job_id)
                if not rows:
                    return
                row = rows[0]
                prev_year = store.previous_year(self.con, job_id, int(row["idx"]))
                profile = int(
                    self.con.execute(
                        "SELECT profile_id FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()[0]
                )
                ctx = learned.build_state(self.con, profile, settings.load(self.con))
                w = (
                    gens.weights_for(self.con, self.models_root, profile)
                    if self.models_root is not None
                    else None
                )
            page_id = int(row["id"])
            try:
                payload = PagePayload(
                    path=str(upload_path(self.uploads_dir, row["upload_path"])),
                    source_index=int(row["source_index"]),
                    page_key=f"{row['upload_id']}:{row['source_index']}",
                    prev_year=prev_year,
                    accounts=tuple(ctx.accounts),
                    titles=tuple(ctx.titles),
                    dictionary=ctx.dictionary,
                    amount_history=ctx.amount_history,
                    corners=(
                        None
                        if row["layout_override"] is None
                        else tuple(tuple(p) for p in json.loads(row["layout_override"]))
                    ),
                    weights_dir=None if w is None or w.dir is None else str(w.dir),
                    confidence=None if w is None or w.confidence is None else str(w.confidence),
                    pagedata_path=(
                        None
                        if self.pagedata_root is None
                        else str(self.pagedata_root / f"{page_id}.npz")
                    ),
                )
                outcome = self.backend.process(payload)
                with self.lock:
                    store.fill_page(
                        self.con,
                        self.images_dir,
                        page_id,
                        outcome.spec,
                        outcome.last_year,
                        outcome.seconds,
                        None if w is None else w.generation_id,
                    )
            except Exception as exc:  # 1 ページの失敗でジョブ全体を止めない
                with self.lock:
                    store.fail_page(self.con, page_id, type(exc).__name__)
