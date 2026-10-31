"""バックアップの作成・復元の準備を管理する(サーバ側)。一度に 1 つの操作だけ。重い処理は `backup_worker`(別プロセス)で行い、ここは進捗を読んで状態を持つ。

流れ:
  作成: DB のスナップショット(SQLite の backup API。DB のロックの下で)→ 別プロセスで zip を作る → 一時置き場 `backups/` に置く
        → 画面が「ダウンロード」を押すと、**1 回だけ使える合言葉**つきの URL を渡す(トークンの要る `/api/` はリンクのダウンロードに使えないため)。
        ダウンロードは WebView2 の「名前を付けて保存」で保存先を選ぶ。置いた zip は、削除・次の作成・次の起動で消える。
  復元: 画面が zip を送る(`restore/incoming.zip`)→ 別プロセスで検査(要約を画面に出す)→ 利用者が確認 → 別プロセスで `restore/staging/` に取り出して検査
        → 「復元の準備ができた」印 → **次の起動の最初に** `backup.apply_pending` が入れ替える。それまで、いまのデータはそのまま使える。
ログ・状態には実データの値を出さない(コード・件数・大きさだけ)。
"""

from __future__ import annotations

import contextlib
import json
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from . import backup
from .paths import AppPaths
from .store import StoreError

TIMEOUT_S = 2 * 3600  # 1 つの操作の上限
DOWNLOAD_TTL_S = 600
POLL_S = 0.4
MAX_INCOMING_BYTES = backup.DEFAULT_LIMITS.max_total_bytes + (1 << 30)
FREE_CHECK_EVERY = (
    64 << 20
)  # 受け取りの途中で、これだけ書くごとに空き容量を確認する(chunked で Content-Length が無い場合にも備える)
FREE_MARGIN = 256 << 20


class BackupManager:
    def __init__(
        self,
        con: sqlite3.Connection,
        lock: threading.RLock,
        paths: AppPaths,
        busy: Callable[[], bool],
        app_version: str,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self.con = con
        self.lock = lock
        self.paths = paths
        self.root = paths.root
        self.busy = busy  # 認識・学習が動いているか(復元の準備は、動いていないときだけ)
        self.app_version = app_version
        self.timeout_s = timeout_s
        self._mu = threading.Lock()
        self._state: dict = self._idle()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._downloads: dict[str, tuple[Path, float]] = {}
        self.cleanup_stale()

    # ------------------------------------------------------------------ 状態

    @staticmethod
    def _idle() -> dict:
        return {
            "op": None,
            "running": False,
            "stage": None,
            "done": 0,
            "total": 0,
            "error": None,
            "created": None,  # 作成したバックアップ(ダウンロード待ち) {name, bytes, summary}
            "incoming": None,  # 検査済みの、復元しようとしている zip の要約
        }

    @property
    def backups_dir(self) -> Path:
        return self.root / backup.BACKUPS_DIR

    @property
    def restore_dir(self) -> Path:
        return self.root / backup.PENDING_DIR

    @property
    def incoming(self) -> Path:
        return self.restore_dir / backup.INCOMING

    def cleanup_stale(self) -> None:
        """起動時: 前回のダウンロード待ちの zip と、準備の途中で止まった復元の作業ファイルを消す(復元の準備ができているものは残す)。"""
        shutil.rmtree(self.backups_dir, ignore_errors=True)
        if backup.read_pending(self.root) is None:
            shutil.rmtree(self.restore_dir, ignore_errors=True)

    def status(self) -> dict:
        with self._mu:
            s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._state.items()}
        s["pending"] = backup.read_pending(self.root)
        s["result"] = backup.read_result(self.root)
        s["undo"] = backup.undo_info(self.root)
        return s

    def _set(self, **kw) -> None:
        with self._mu:
            self._state.update(kw)

    # ------------------------------------------------------------------ 実行

    def _start(
        self,
        op: str,
        cmd: list[str],
        progress: Path,
        on_success: Callable[[dict], None],
        on_failure: Callable[[], None],
    ) -> None:
        with self._mu:
            if self._state["running"]:
                raise StoreError(409, "backup_busy")
            self._state.update(op=op, running=True, stage=None, done=0, total=0, error=None)
        t = threading.Thread(
            target=self._run,
            args=(cmd, progress, on_success, on_failure),
            name=f"yayoi-backup-{op}",
            daemon=True,
        )
        self._thread = t
        t.start()

    def _run(
        self,
        cmd: list[str],
        progress: Path,
        on_success: Callable[[dict], None],
        on_failure: Callable[[], None],
    ) -> None:
        code: str | None = None
        result: dict | None = None
        try:
            progress.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                progress.unlink()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=str(self.root),
                text=True,
            )
            self._proc = proc
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    out, _ = proc.communicate(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    self._read_progress(progress)
                    if time.monotonic() > deadline:
                        proc.kill()
                        proc.communicate()
                        out = ""
                        code = "timeout"
                        break
            if code is None:
                last = (out or "").strip().splitlines()[-1:] or [""]
                try:
                    msg = json.loads(last[0])
                except ValueError:
                    msg = {}
                if (
                    proc.returncode == 0
                    and msg.get("ok") is True
                    and isinstance(msg.get("result"), dict)
                ):
                    result = msg["result"]
                else:
                    c = msg.get("code")
                    code = c if isinstance(c, str) and len(c) <= 64 else "worker_crashed"
        except OSError:
            code = "worker_start_failed"
        finally:
            self._proc = None
        if result is not None:
            try:
                on_success(result)
            except Exception as exc:
                code, result = f"error_{type(exc).__name__}", None
        if result is None:
            with contextlib.suppress(Exception):
                on_failure()
            self._set(running=False, error=code or "unknown")
        else:
            self._set(running=False, error=None)

    def _read_progress(self, path: Path) -> None:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            self._set(stage=str(d["stage"])[:24], done=int(d["done"]), total=int(d["total"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _worker(self, *args: str) -> list[str]:
        return [sys.executable, "-I", "-m", "yayoi_ocr.app.backup_worker", *args]

    def wait(self, timeout: float = 120.0) -> None:
        """テスト用: 実行中の操作が終わるまで待つ。"""
        t = self._thread
        if t is not None:
            t.join(timeout)

    # ------------------------------------------------------------------ 作成

    def start_create(self, include_originals: bool) -> None:
        with self._mu:
            if self._state["running"]:
                raise StoreError(409, "backup_busy")
        self.discard_created()
        d = self.backups_dir
        d.mkdir(parents=True, exist_ok=True)
        snap = d / f"snapshot-{secrets.token_hex(6)}.sqlite3"
        with self.lock:  # DB の整合の取れたスナップショット(動いている DB のファイルをコピーしない)
            dst = sqlite3.connect(snap)
            try:
                self.con.backup(dst)
                dst.execute("PRAGMA journal_mode = DELETE")
            finally:
                dst.close()
        out = d / f"yayoi-backup-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        prog = d / "progress.json"

        def ok(result: dict) -> None:
            with contextlib.suppress(OSError):
                snap.unlink()
            self._set(
                created={"name": out.name, "bytes": int(result["zip_bytes"]), "summary": result}
            )

        def fail() -> None:
            for p in (snap, out, out.with_name(out.name + ".part")):
                with contextlib.suppress(OSError):
                    p.unlink()

        args = [
            "create",
            "--root",
            str(self.root),
            "--snapshot",
            str(snap),
            "--out",
            str(out),
            "--version",
            self.app_version,
            "--progress",
            str(prog),
        ]
        if include_originals:
            args.append("--originals")
        try:
            self._start("create", self._worker(*args), prog, ok, fail)
        except StoreError:
            fail()
            raise

    def discard_created(self) -> bool:
        with self._mu:
            had = self._state["created"] is not None
            self._state["created"] = None
            self._downloads.clear()
        shutil.rmtree(self.backups_dir, ignore_errors=True)
        return had

    def mint_download(self) -> str:
        with self._mu:
            c = self._state["created"]
            if c is None:
                raise StoreError(404, "no_backup")
            path = self.backups_dir / c["name"]
            if not path.is_file():
                raise StoreError(404, "no_backup")
            now = time.time()
            self._downloads = {k: v for k, v in self._downloads.items() if v[1] > now}
            nonce = secrets.token_urlsafe(32)
            self._downloads[nonce] = (path, now + DOWNLOAD_TTL_S)
            return nonce

    def take_download(self, nonce: str) -> Path | None:
        """合言葉に対応するファイル(1 回で失効)。"""
        with self._mu:
            item = self._downloads.pop(nonce, None)
        if item is None or item[1] < time.time() or not item[0].is_file():
            return None
        return item[0]

    # ------------------------------------------------------------------ 復元

    def prepare_incoming(self) -> Path:
        """zip を受け取る前: 以前の準備を片づけ、受け取り先のパスを返す。"""
        with self._mu:
            if self._state["running"]:
                raise StoreError(409, "backup_busy")
        if self.busy():
            raise StoreError(409, "busy")
        self.cancel()
        self.restore_dir.mkdir(parents=True, exist_ok=True)
        return self.incoming

    async def receive_incoming(self, body: AsyncIterator[bytes], declared: int | None) -> None:
        """送られてきた zip を `restore/incoming.zip` に書く(上限つき)。そのあと検査を始める。"""
        dest = self.prepare_incoming()
        if declared is not None and declared > MAX_INCOMING_BYTES:
            raise StoreError(413, "too_large")
        if declared is not None and shutil.disk_usage(self.root).free < declared + (256 << 20):
            raise StoreError(507, "disk_full")
        size = 0
        checked = 0
        try:
            with open(dest, "wb") as fh:
                async for chunk in body:
                    size += len(chunk)
                    if size > MAX_INCOMING_BYTES:
                        raise StoreError(413, "too_large")
                    if size - checked >= FREE_CHECK_EVERY:
                        checked = size
                        if shutil.disk_usage(self.root).free < FREE_MARGIN:
                            raise StoreError(507, "disk_full")
                    await run_in_threadpool(fh.write, chunk)
            if size == 0:
                raise StoreError(422, "empty_file")
        except OSError as exc:
            shutil.rmtree(self.restore_dir, ignore_errors=True)
            raise StoreError(
                507, "disk_full" if getattr(exc, "errno", None) == 28 else "io_error"
            ) from exc
        except BaseException:
            shutil.rmtree(self.restore_dir, ignore_errors=True)
            raise
        self.start_inspect()

    def start_inspect(self) -> None:
        prog = self.restore_dir / "progress.json"

        def ok(summary: dict) -> None:
            self._set(incoming=summary)

        def fail() -> None:
            shutil.rmtree(self.restore_dir, ignore_errors=True)

        self._set(incoming=None)
        self._start(
            "inspect",
            self._worker("inspect", "--zip", str(self.incoming), "--progress", str(prog)),
            prog,
            ok,
            fail,
        )

    def start_prepare(self) -> None:
        """検査済みの zip を取り出して検査し、復元の準備ができた印をつける(入れ替えは次の起動)。"""
        with self._mu:
            summary = self._state["incoming"]
            if self._state["running"]:
                raise StoreError(409, "backup_busy")
        if summary is None or not self.incoming.is_file():
            raise StoreError(409, "no_incoming")
        if self.busy():
            raise StoreError(409, "busy")
        staging = self.restore_dir / backup.STAGING
        shutil.rmtree(staging, ignore_errors=True)
        prog = self.restore_dir / "progress.json"

        def ok(result: dict) -> None:
            backup.write_pending(self.root, result)
            with contextlib.suppress(OSError):
                self.incoming.unlink()  # 取り出し済みなので、zip は要らない(容量)
            self._set(incoming=None)

        def fail() -> None:
            shutil.rmtree(staging, ignore_errors=True)

        self._start(
            "extract",
            self._worker(
                "extract",
                "--zip",
                str(self.incoming),
                "--staging",
                str(staging),
                "--progress",
                str(prog),
            ),
            prog,
            ok,
            fail,
        )

    def cancel(self) -> None:
        """実行中の操作を止め、復元の準備(受け取った zip・取り出したデータ・印)を消す。"""
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.kill()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(10)
        backup.cancel_pending(self.root)
        self._set(incoming=None, error=None)

    def acknowledge_result(self) -> None:
        backup.clear_result(self.root)

    def delete_undo(self) -> bool:
        return backup.delete_undo(self.root)

    def close(self) -> None:
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.kill()
        shutil.rmtree(self.backups_dir, ignore_errors=True)
        if backup.read_pending(self.root) is None:
            shutil.rmtree(self.restore_dir, ignore_errors=True)
