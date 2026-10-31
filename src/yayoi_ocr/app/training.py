"""バックグラウンドの追加学習の管理(`TrainManager`)。設計は `docs/phase4_design.md`。

- **開始**: 自動(確定した新しい非空欄セルが `train_threshold_cells` 件たまり、確定ページが足りて、認識中のジョブがなく、別の学習が動いていないとき)か、手動(学習画面の
  「今すぐ学習する」。しきい値は見ない)。実行は 1 つずつ。学習は別プロセス(`trainers.SubprocessTrainer`)で、確認画面は学習中も使用中の世代のまま使える。
- **分割**: 学習に使える確定ページを確定した順に並べ、新しい側から 25%(最低 2・最大 10 ページ)を評価用にする(`gate.split_pages`)。
- **採否**: 学習器が返した親・新世代の評価の件数から `gate.decide` で決める。採用なら世代として登録して使用中にし、不採用なら理由つきで記録(重みは捨てる)、評価が足りなければ見送る。
- 記録(`training_runs`・世代のメタ)とログには、ページ ID・件数・割合・理由コードだけを入れる(値は入れない)。作業ディレクトリ(確定値を含む計画)は終わったら消す。
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from . import gate, gens, learned
from . import settings as app_settings
from .db import DEFAULT_PROFILE_ID, transaction
from .paths import AppPaths
from .store import StoreError
from .trainers import Trainer, TrainerCancelledError, TrainerError
from .worker import worker_threads

log = logging.getLogger("yayoi.training")

COUNTED_STATUSES = (
    "adopted",
    "rejected",
    "deferred",
    "failed",
)  # 「新しい確定セル」の起点になる実行
PROGRESS_EVERY = 1.0  # 進み具合の書き込みの間隔(秒)
HISTORY = 10
LOG_KEEP_BYTES = 64 * 1024


def _run_dict(r: sqlite3.Row) -> dict:
    return {k: v for k, v in dict(r).items() if k != "profile_id"}


class TrainManager:
    def __init__(
        self,
        con: sqlite3.Connection,
        lock: threading.RLock,
        paths: AppPaths,
        trainer: Trainer | None,
    ) -> None:
        self.con = con
        self.lock = lock
        self.paths = paths
        self.trainer = trainer
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._run_id: int | None = None
        self._last_progress = 0.0
        self.auto_allowed = True  # 開発・E2E 専用の全体のスイッチ(設定 auto_train とは別。共有サーバの試験が勝手に学習を始めないように)

    # ------------------------------------------------------------------ 起動時

    def mark_interrupted(self) -> int:
        """前回の実行中にアプリが閉じた学習を「中断」にし、作業ディレクトリを消す。"""
        with self.lock, transaction(self.con):
            n = self.con.execute(
                "UPDATE training_runs SET status = 'interrupted', reason = 'app_closed', finished_at = ?"
                " WHERE status = 'running'",
                (time.time(),),
            ).rowcount
        shutil.rmtree(self.paths.training, ignore_errors=True)
        return n

    # ------------------------------------------------------------------ 状態

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _cutoff(self, profile_id: int) -> float:
        r = self.con.execute(
            "SELECT MAX(cutoff) FROM training_runs WHERE profile_id = ? AND status IN "
            f"({','.join('?' * len(COUNTED_STATUSES))})",
            (profile_id, *COUNTED_STATUSES),
        ).fetchone()
        return max(float(r[0] or 0.0), gens.halt_cutoff(self.con, profile_id))

    def eligible_pages(self, profile_id: int = DEFAULT_PROFILE_ID) -> list[sqlite3.Row]:
        """学習に使える確定ページ(確定した順)。学習器が認識の入力を要するなら、それが残っているものだけ。"""
        pages = learned.confirmed_pages(self.con, profile_id)
        if self.trainer is not None and self.trainer.needs_pagedata:
            pages = [p for p in pages if self.pagedata_file(p["id"]).is_file()]
        return pages

    def pagedata_file(self, page_id: int) -> Path:
        return self.paths.pagedata / f"{page_id}.npz"

    def new_cells(self, profile_id: int, pages: list[sqlite3.Row] | None = None) -> int:
        """前回までの実行が見たページより後に確定したページの、確定した非空欄セル数。"""
        cutoff = self._cutoff(profile_id)
        ids = [
            p["id"]
            for p in (pages if pages is not None else self.eligible_pages(profile_id))
            if (p["confirmed_at"] or 0) > cutoff
        ]
        if not ids:
            return 0
        q = ",".join("?" * len(ids))
        return int(
            self.con.execute(
                f"SELECT COUNT(*) FROM cells WHERE kind != 'blank' AND page_id IN ({q})", ids
            ).fetchone()[0]
        )

    def status(self, profile_id: int = DEFAULT_PROFILE_ID) -> dict:
        with self.lock:
            cfg = app_settings.load(self.con)
            pages = self.eligible_pages(profile_id)
            runs = [
                _run_dict(r)
                for r in self.con.execute(
                    "SELECT * FROM training_runs WHERE profile_id = ? ORDER BY id DESC LIMIT ?",
                    (profile_id, HISTORY),
                )
            ]
            act = gens.active(self.con, profile_id)
            return {
                "available": self.trainer is not None,
                "auto_train": cfg.auto_train,
                "threshold": cfg.train_threshold_cells,
                "profile_id": profile_id,
                "active_generation": {"id": act["id"], "number": act["number"]},
                "eligible_pages": len(pages),
                "min_pages": gate.MIN_PAGES,
                "new_cells": self.new_cells(profile_id, pages),
                "running": next((r for r in runs if r["status"] == "running"), None),
                "last": next((r for r in runs if r["status"] != "running"), None),
                "runs": runs,
            }

    # ------------------------------------------------------------------ 開始

    def notify_confirmed(self, profile_id: int = DEFAULT_PROFILE_ID) -> int | None:
        """ページが確定したときに呼ぶ。自動開始の条件を満たしていれば学習を始める(始めたら実行 ID)。"""
        if self.trainer is None or self.running or not self.auto_allowed:
            return None
        try:
            with self.lock:
                cfg = app_settings.load(self.con)
                if not cfg.auto_train:
                    return None
                if self.con.execute(
                    "SELECT 1 FROM pages WHERE status = 'processing' LIMIT 1"
                ).fetchone():
                    return None  # 認識の最中は、CPU を取り合わないよう次の機会に回す
                if self.new_cells(profile_id) < cfg.train_threshold_cells:
                    return None
            return self.start(profile_id, "auto")
        except StoreError:
            return None
        except Exception as exc:  # 自動開始の失敗で確定の操作を止めない
            log.warning("auto_start_failed: %s", type(exc).__name__)
            return None

    def start(self, profile_id: int = DEFAULT_PROFILE_ID, trigger: str = "manual") -> int:
        """学習を始める(別スレッドで実行)。始められなければ StoreError(理由コード)。"""
        if self.trainer is None:
            raise StoreError(503, "training_disabled")
        with self.lock:
            if self.running:
                raise StoreError(409, "training_running")
            pages = self.eligible_pages(profile_id)
            split = gate.split_pages([p["id"] for p in pages])
            if split is None:
                raise StoreError(
                    422, "not_enough_pages", {"have": len(pages), "need": gate.MIN_PAGES}
                )
            train_ids, eval_ids = split
            plan = self._plan(profile_id, train_ids, eval_ids)
            now = time.time()
            n_new = self.new_cells(profile_id, pages)
            cutoff = max((p["confirmed_at"] or 0) for p in pages)
            with transaction(self.con):
                run_id = int(
                    self.con.execute(
                        "INSERT INTO training_runs (profile_id, trigger, status, started_at, n_new_cells,"
                        " n_train_pages, n_eval_pages, cutoff, stage, progress)"
                        " VALUES (?, ?, 'running', ?, ?, ?, ?, ?, 'prepare', 0)",
                        (profile_id, trigger, now, n_new, len(train_ids), len(eval_ids), cutoff),
                    ).lastrowid
                )
            plan["run_id"] = run_id
            self._run_id = run_id
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._work,
                args=(run_id, profile_id, plan, trigger),
                name="yayoi-train",
                daemon=True,
            )
            self._thread.start()
        return run_id

    def _plan(self, profile_id: int, train_ids: list[int], eval_ids: list[int]) -> dict:
        """学習器に渡す計画。確定値を含むので、作業ディレクトリの中にだけ置き、終わったら消す。"""
        cfg = app_settings.load(self.con)
        state = learned.build_state(self.con, profile_id, cfg, only_pages=set(train_ids))
        w = gens.weights_for(self.con, self.paths.models, profile_id)

        def entry(page_id: int) -> dict:
            return {
                "page_id": page_id,
                "file": str(self.pagedata_file(page_id)),
                "cells": [
                    [c.row, c.col, c.column, c.kind, c.resolved]
                    for c in learned.confirmed_cells(self.con, page_id)
                ],
            }

        return {
            "profile_id": profile_id,
            "parent": {
                "generation_id": w.generation_id,
                "number": w.number,
                "weights_dir": None if w.dir is None else str(w.dir),
                "confidence": None if w.confidence is None else str(w.confidence),
            },
            "pages": {
                "train": [entry(i) for i in train_ids],
                "eval": [entry(i) for i in eval_ids],
            },
            "context": {
                "accounts": state.accounts,
                "titles": state.titles,
                "dictionary": state.dictionary,
                "amount_history": {
                    k: {str(a): n for a, n in h.items()} for k, h in state.amount_history.items()
                },
            },
            "device": os.environ.get(
                "YAYOI_TRAIN_DEVICE", "auto"
            ),  # auto / cpu / cuda(GPU が動かないときの退避)
            "threads": worker_threads(),
        }

    def cancel(self) -> bool:
        """実行中の学習を止める(結果は使わない)。"""
        if not self.running:
            return False
        self._cancel.set()
        return True

    def join(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def close(self) -> None:
        if self.running:
            self._cancel.set()
            self.join(15)

    # ------------------------------------------------------------------ 実行

    def _progress(self, run_id: int):
        def cb(stage: str, fraction: float) -> None:
            now = time.time()
            if fraction < 1.0 and now - self._last_progress < PROGRESS_EVERY:
                return
            self._last_progress = now
            with self.lock:
                self.con.execute(
                    "UPDATE training_runs SET stage = ?, progress = ? WHERE id = ? AND status = 'running'",
                    (stage, max(0.0, min(1.0, fraction)), run_id),
                )

        return cb

    def _finish(
        self,
        run_id: int,
        status: str,
        reason: str | None,
        generation_id: int | None = None,
    ) -> None:
        with self.lock:
            self.con.execute(
                "UPDATE training_runs SET status = ?, reason = ?, generation_id = ?, finished_at = ?,"
                " stage = 'done', progress = 1 WHERE id = ?",
                (status, reason, generation_id, time.time(), run_id),
            )

    def _work(self, run_id: int, profile_id: int, plan: dict, trigger: str) -> None:
        run_dir = self.paths.training / str(run_id)
        try:
            assert self.trainer is not None
            result = self.trainer.run(plan, run_dir, self._progress(run_id), self._cancel)
            self._conclude(run_id, profile_id, plan, result, trigger)
        except TrainerCancelledError:
            self._finish(run_id, "interrupted", "cancelled")
        except TrainerError as exc:
            log.warning("training_failed run=%d code=%s", run_id, exc.code)
            self._finish(run_id, "failed", exc.code)
        except Exception as exc:  # 型名だけ記録する(例外の文面に値が入りうる)
            log.warning("training_failed run=%d type=%s", run_id, type(exc).__name__)
            self._finish(run_id, "failed", f"error_{type(exc).__name__}")
        finally:
            self._keep_log(run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)

    def _keep_log(self, run_dir: Path) -> None:
        """作業ディレクトリのログ(ステージと件数だけ)を、直近の 1 回分だけ残す。"""
        src = run_dir / "train.log"
        try:
            if src.is_file():
                data = src.read_bytes()[-LOG_KEEP_BYTES:]
                self.paths.training.mkdir(parents=True, exist_ok=True)
                (self.paths.training / "last.log").write_bytes(data)
        except OSError:
            pass

    def _conclude(
        self, run_id: int, profile_id: int, plan: dict, result: dict, trigger: str
    ) -> None:
        parent = gate.EvalSide.from_dict(result["eval"]["parent"])
        cand = gate.EvalSide.from_dict(result["eval"]["candidate"])
        decision = gate.decide(parent, cand)
        with self.lock:
            now_active = gens.active(self.con, profile_id)["id"]
        if (
            now_active != plan["parent"]["generation_id"]
        ):  # 学習中に使用中の世代が変わった(ロールバック)
            self._finish(run_id, "deferred", "active_generation_changed")
            return
        if decision.outcome == "defer":
            self._finish(run_id, "deferred", decision.reason)
            return
        number = None
        with self.lock:
            number = gens.next_number(self.con, profile_id)
        train_ids = [p["page_id"] for p in plan["pages"]["train"]]
        eval_ids = [p["page_id"] for p in plan["pages"]["eval"]]
        meta = {
            "number": number,
            "parent_generation_id": plan["parent"]["generation_id"],
            "parent_number": plan["parent"]["number"],
            "init": "shipped",
            "trigger": trigger,
            "train_page_ids": train_ids,
            "eval_page_ids": eval_ids,
            "n_train_pages": len(train_ids),
            "n_eval_pages": len(eval_ids),
            "train": result.get("train", {}),
            "eval": {
                "parent": parent.to_dict(),
                "candidate": cand.to_dict(),  # クロスフィッティングした判定での成績(ゲートが使った)
                **{
                    k: result["eval"][k]
                    for k in ("candidate_global", "candidate_final")
                    if k in result["eval"]
                },
            },
            "recalibration": result.get("recalibration", {}),
            "decision": {"outcome": decision.outcome, "reason": decision.reason},
            "created_at": time.time(),
        }
        if not decision.adopted:
            with self.lock:
                gid = gens.register(
                    self.con,
                    self.paths.models,
                    profile_id=profile_id,
                    number=number,
                    parent_id=plan["parent"]["generation_id"],
                    meta=meta,
                    adopt=False,
                    reject_reason=decision.reason,
                    keep_dir=False,
                )
            self._finish(run_id, "rejected", decision.reason, gid)
            return
        with self.lock:
            gid = gens.install(
                self.con,
                self.paths.models,
                profile_id=profile_id,
                number=number,
                parent_id=plan["parent"]["generation_id"],
                meta=meta,
                weights_dir=Path(result["weights_dir"]),
                confidence=Path(result["confidence"]),
                run_dir=self.paths.training / str(run_id),
            )
        self._finish(run_id, "adopted", decision.reason, gid)
