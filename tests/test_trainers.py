"""学習器の差し込み口: SubprocessTrainer(別プロセスの起動・進み具合の中継・失敗/キャンセル/タイムアウトの扱い)と MockTrainer。torch なし(ワーカーは試験用の短いスクリプト)。"""

from __future__ import annotations

import json
import sys
import threading

import pytest

from yayoi_ocr.app.trainers import (
    MockTrainer,
    SubprocessTrainer,
    TrainerCancelledError,
    TrainerError,
)

OK = """
import json, sys, time, pathlib
d = pathlib.Path(sys.argv[1])
plan = json.loads((d / 'plan.json').read_text(encoding='utf-8'))
(d / 'progress.json').write_text(json.dumps({'stage': 'train', 'progress': 0.5}))
time.sleep(1.2)
(d / 'result.json').write_text(json.dumps({'status': 'ok', 'n_eval_pages': len(plan['pages']['eval'])}))
"""
FAIL_RESULT = """
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
(d / 'result.json').write_text(json.dumps({'status': 'failed', 'reason': 'rehearsal_missing'}))
sys.exit(1)
"""
CRASH = "import sys; sys.exit(3)"
HANG = "import time; time.sleep(60)"
NO_RESULT = "pass"


def trainer(code, **kw):
    return SubprocessTrainer([sys.executable, "-c", code], **kw)


PLAN = {"pages": {"train": [], "eval": [{"page_id": 1, "file": "x", "cells": []}]}}


def run(t, tmp_path, cancel=None):
    seen = []
    out = t.run(
        PLAN, tmp_path / "run", lambda s, f: seen.append((s, f)), cancel or threading.Event()
    )
    return out, seen


def test_subprocess_trainer_runs_the_worker_relays_progress_and_returns_the_result(tmp_path):
    out, seen = run(trainer(OK), tmp_path)
    assert out == {"status": "ok", "n_eval_pages": 1}
    assert ("train", 0.5) in seen
    # 計画は作業ディレクトリに書かれる(確定値を含みうるので、消すのは呼び出し側)
    assert (
        json.loads((tmp_path / "run" / "plan.json").read_text())["pages"]["eval"][0]["page_id"] == 1
    )


def test_a_failed_result_becomes_a_trainer_error_with_its_code(tmp_path):
    with pytest.raises(TrainerError) as e:
        run(trainer(FAIL_RESULT), tmp_path)
    assert e.value.code == "rehearsal_missing"


def test_a_crash_without_a_result_is_reported_as_worker_crashed(tmp_path):
    with pytest.raises(TrainerError) as e:
        run(trainer(CRASH), tmp_path)
    assert e.value.code == "worker_crashed"
    with pytest.raises(TrainerError) as e2:
        run(trainer(NO_RESULT), tmp_path / "b")
    assert e2.value.code == "no_result"


def test_cancel_terminates_the_process(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.6, cancel.set).start()
    with pytest.raises(TrainerCancelledError):
        run(trainer(HANG), tmp_path, cancel)


def test_timeout_kills_the_process(tmp_path):
    with pytest.raises(TrainerError) as e:
        run(trainer(HANG, max_seconds=1), tmp_path)
    assert e.value.code == "timeout"


def test_mock_trainer_modes_and_cancel(tmp_path):
    plan = {
        "pages": {
            "train": [],
            "eval": [
                {"page_id": i, "file": "", "cells": [[0, 0, "date", "text", "x"]] * 100}
                for i in (1, 2)
            ],
        }
    }
    m = MockTrainer()
    out = m.run(plan, tmp_path / "a", lambda *_: None, threading.Event())
    assert out["eval"]["candidate"]["n_correct"] > out["eval"]["parent"]["n_correct"]
    assert (tmp_path / "a" / "weights" / "confidence.json").is_file()
    m.mode = "fail"
    with pytest.raises(TrainerError):
        m.run(plan, tmp_path / "b", lambda *_: None, threading.Event())
    ev = threading.Event()
    ev.set()
    with pytest.raises(TrainerCancelledError):
        MockTrainer().run(plan, tmp_path / "c", lambda *_: None, ev)
