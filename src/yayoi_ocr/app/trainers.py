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

from .gate import EvalCounts

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


class MockTrainer:
    """合成の成績を返す学習器(重みは空のファイル)。評価用ページの非空欄セル数は、計画の実際の件数を使う(評価が足りない場合の見送りも試験できる)。"""

    needs_pagedata = False

    def __init__(self) -> None:
        self.mode = "improve"  # improve / worsen / fail / flat
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
        parent = EvalCounts(
            n_cells,
            n_text,
            round(0.88 * n_text),
            round(0.30 * n_text),
            max(1, round(0.01 * n_cells)),
        )
        cand = {
            "improve": EvalCounts(
                n_cells,
                n_text,
                round(0.94 * n_text),
                round(0.26 * n_text),
                max(0, parent.n_miss - 1),
            ),
            "worsen": EvalCounts(
                n_cells, n_text, round(0.84 * n_text), round(0.30 * n_text), parent.n_miss + 3
            ),
            "flat": EvalCounts(n_cells, n_text, parent.n_correct, parent.n_review, parent.n_miss),
        }[self.mode]
        wdir = run_dir / WEIGHTS_SUBDIR
        wdir.mkdir(parents=True, exist_ok=True)
        for f in WEIGHT_FILES:
            (wdir / f).write_bytes(b"mock")
        (wdir / "confidence.json").write_text("{}", encoding="utf-8")
        progress("done", 1.0)
        return {
            "weights_dir": str(wdir),
            "confidence": str(wdir / "confidence.json"),
            "eval": {"parent": parent.to_dict(), "candidate": cand.to_dict()},
            "recalibration": {},
            "train": {
                "samples": {"256": 0, "384": 0},
                "rehearsal": 0,
                "epochs": 0,
                "seconds": 0.0,
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
