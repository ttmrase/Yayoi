"""追加学習の学習器(`Trainer`)。torch を読み込まない(実際の学習は別プロセス `train_worker` が行う)。

- `SubprocessTrainer`: `plan.json` を作業ディレクトリに書き、`python -I -m yayoi_ocr.app.train_worker <作業ディレクトリ>` を起動して、
  `progress.json` を見ながら待ち、`result.json` を読む。CPU のときは優先度を下げる(Windows は BELOW_NORMAL、POSIX は nice。後者はワーカー自身が行う)。
- `MockTrainer`: CI・E2E 用。計画(`plan`)の件数から合成の成績を返す(`mode`: improve / worsen / fail。開発用の `POST /api/dev/trainer` で切り替える)。

計画と結果の形(JSON)は `docs/phase4_design.md`。ログと例外には**コードと件数だけ**を入れる(摘要・金額などの値は入れない)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .gate import Calib, EvalCounts, EvalSide

Progress = Callable[[str, float], None]

WEIGHTS_SUBDIR = "weights"
MAX_SECONDS = int(os.environ.get("YAYOI_TRAIN_TIMEOUT", str(4 * 3600)))  # 1 回の追加学習の上限
POLL_SECONDS = 0.5


class TrainerError(Exception):
    """学習の失敗。`code` は画面とログに出す理由コード(値は含めない)。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TrainerCancelledError(Exception):
    pass


class Trainer(Protocol):
    needs_pagedata: bool  # True: 確定ページの認識の入力(pagedata/<ページ ID>.npz)が必要

    def run(
        self, plan: dict, run_dir: Path, progress: Progress, cancel: threading.Event
    ) -> dict: ...


def _eval_text_cells(plan: dict) -> int:
    return sum(
        1 for pg in plan["pages"]["eval"] for cell in pg["cells"] if cell[3] != "blank"
    )  # cell = [row, col, column, kind, resolved]


def _mock_side(
    n_cells: int,
    n_text: int,
    correct: float,
    review: float,
    miss: int,
    nll: float,
    calibration: float = 1.0,
    amount_miss: int = 0,
) -> EvalSide:
    """計画の件数から作る合成の成績(全体 + 4 列)。見逃し `miss` は日付・科目・摘要に均等に割り、金額は `amount_miss`。
    予測した見逃しの件数は、観測した件数の `calibration` 倍(1.0 = 較正が合っている、小さいほど自信過剰)。"""
    cols = ("date", "account", "description", "amount")

    def share(x: int, i: int, n: int = 4) -> int:
        return x // n + (1 if i < x % n else 0)

    total_miss = miss + amount_miss
    total = EvalCounts(n_cells, n_text, round(correct * n_text), round(review * n_text), total_miss)
    by, cal = {}, {}
    for i, col in enumerate(cols):
        m = amount_miss if col == "amount" else share(miss, i, 3)
        by[col] = EvalCounts(
            share(n_cells, i),
            share(n_text, i),
            share(total.n_correct, i),
            share(total.n_review, i),
            m,
        )
        cal[col] = Calib(share(n_text, i), nll, nll * 0.8, calibration * m, m)
    return EvalSide(
        total, Calib(n_text, nll, nll * 0.8, calibration * total_miss, total_miss), by, cal
    )


class MockTrainer:
    """合成の成績を返す学習器(重みは空のファイル)。評価用ページの非空欄セル数は、計画の実際の件数を使う(評価が足りない場合の見送りも試験できる)。
    mode: improve(採用される)/ worsen(見逃しが増える)/ flat(変わらない)/ fail(失敗)/ amount_miss(金額に要確認でない誤り)/
    overconfident(見逃しが予測より有意に多い)/ calib_worse(較正が悪化)/ suggest(金額に要確認でない誤りがあるが、正解率は有意に高い = 提案モード)/
    promote_prev(今回の世代は不採用だが、すでにある提案モードの世代が採用基準をすべて満たす = 昇格)。"""

    needs_pagedata = False

    def __init__(self) -> None:
        self.mode = "improve"
        self.delay = 0.0
        self.calls = 0

    def run(self, plan: dict, run_dir: Path, progress: Progress, cancel: threading.Event) -> dict:
        from .gens import WEIGHT_FILES

        self.calls += 1
        steps = 5
        for i in range(steps):
            if cancel.is_set():
                raise TrainerCancelledError
            progress("train", i / steps)
            if self.delay:
                time.sleep(self.delay / steps)
        if self.mode == "fail":
            raise TrainerError("mock_failure")
        n_text = _eval_text_cells(plan)
        n_cells = len(plan["pages"]["eval"]) * 116
        miss0 = max(1, round(0.01 * n_cells))
        parent = _mock_side(n_cells, n_text, 0.88, 0.30, miss0, 0.40)
        cand = {
            "improve": _mock_side(n_cells, n_text, 0.94, 0.26, max(0, miss0 - 1), 0.30),
            "worsen": _mock_side(n_cells, n_text, 0.84, 0.30, miss0 + 3, 0.45),
            "flat": _mock_side(n_cells, n_text, 0.88, 0.30, miss0, 0.40),
            "amount_miss": _mock_side(n_cells, n_text, 0.94, 0.26, 0, 0.30, amount_miss=1),
            "overconfident": _mock_side(n_cells, n_text, 0.94, 0.26, miss0, 0.30, calibration=0.02),
            "calib_worse": _mock_side(n_cells, n_text, 0.94, 0.26, max(0, miss0 - 1), 0.55),
            "suggest": _mock_side(n_cells, n_text, 0.94, 0.26, 0, 0.30, amount_miss=1),
            "promote_prev": _mock_side(n_cells, n_text, 0.94, 0.26, 0, 0.30, amount_miss=1),
        }[self.mode]
        # 値が違うセルのうち、どちらが正しいか(提案モードの判定に使う)
        paired = {
            "improve": (round(0.06 * n_text), 0),
            "worsen": (0, round(0.04 * n_text)),
            "suggest": (round(0.10 * n_text), max(1, round(0.01 * n_text))),
            "promote_prev": (round(0.02 * n_text), round(0.02 * n_text)),  # 有意でない
        }.get(self.mode, (0, 0))
        extra = {}
        if self.mode == "promote_prev" and plan.get("suggest_prev"):
            extra["suggest_prev"] = _mock_side(
                n_cells, n_text, 0.94, 0.26, max(0, miss0 - 1), 0.30
            ).to_dict()
        wdir = run_dir / WEIGHTS_SUBDIR
        wdir.mkdir(parents=True, exist_ok=True)
        for f in WEIGHT_FILES:
            (wdir / f).write_bytes(b"mock")
        (wdir / "confidence.json").write_text("{}", encoding="utf-8")
        progress("done", 1.0)
        return {
            "weights_dir": str(wdir),
            "confidence": str(wdir / "confidence.json"),
            "eval": {
                "parent": parent.to_dict(),
                "candidate": cand.to_dict(),
                "paired": {
                    "cand_only_correct": paired[0],
                    "parent_only_correct": paired[1],
                    "both_correct": 0,
                    "both_wrong": 0,
                },
                **extra,
            },
            "recalibration": {},
            "train": {
                "samples": {"256": 0, "384": 0},
                "rehearsal": {"256": 0, "384": 0},
                "epochs": {"256": 0, "384": 0},
                "seconds_total": 0.0,
                "device": "mock",
                "threads": 1,
            },
        }


class SubprocessTrainer:
    """別プロセスで学習・評価する。`command` は起動のしかた(作業ディレクトリが末尾に付く)。既定は `train_worker`。"""

    needs_pagedata = True

    def __init__(self, command: list[str] | None = None, max_seconds: int = MAX_SECONDS) -> None:
        self.command = command or [sys.executable, "-I", "-m", "yayoi_ocr.app.train_worker"]
        self.max_seconds = max_seconds
        self._proc: subprocess.Popen | None = None

    def run(self, plan: dict, run_dir: Path, progress: Progress, cancel: threading.Event) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        flags = 0
        if os.name == "nt":  # 優先度を下げ、コンソールを出さない
            flags = 0x00004000 | 0x08000000  # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
        with open(run_dir / "train.log", "ab") as log:
            self._proc = subprocess.Popen(
                [*self.command, str(run_dir)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(run_dir),
                creationflags=flags,
            )
            deadline = time.time() + self.max_seconds
            last = ("", -1.0)
            try:
                while self._proc.poll() is None:
                    if cancel.is_set():
                        self._kill()
                        raise TrainerCancelledError
                    if time.time() > deadline:
                        self._kill()
                        raise TrainerError("timeout")
                    last = self._relay(run_dir, progress, last)
                    time.sleep(POLL_SECONDS)
            finally:
                if self._proc.poll() is None:
                    self._kill()
        code = self._proc.returncode
        self._proc = None
        res = run_dir / "result.json"
        if res.is_file():
            out = json.loads(res.read_text(encoding="utf-8"))
            if out.get("status") == "failed":
                raise TrainerError(str(out.get("reason", "worker_failed")))
            return out
        raise TrainerError("worker_crashed" if code else "no_result")

    def _relay(
        self, run_dir: Path, progress: Progress, last: tuple[str, float]
    ) -> tuple[str, float]:
        try:
            d = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            cur = (str(d["stage"]), float(d["progress"]))
        except (OSError, ValueError, KeyError):
            return last
        if cur != last:
            progress(*cur)
        return cur

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


def make_trainer(name: str) -> Trainer:
    if name == "fake":
        return MockTrainer()
    if name == "real":
        return SubprocessTrainer()
    raise ValueError(f"未知の学習器: {name}")
