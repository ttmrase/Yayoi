"""採用ゲートの感度の回帰試験: 報告 4 の追加学習の曲線(少数ページ N = 1・3 で学習すると自信のある誤りが増えた)を、新しいゲートで再生する。

合成の書き手 12 人 × 評価 12 ページの記録(`YAYOI_PRIVATE_DATA/evals/wa_main`。`scripts/writer_adapt_curve.py` の出力)が必要。無ければ飛ばす。
- 前提: N = 1・3 では、ほぼ全員で見逃しが増える(全体の重みの確信度のままなので)。
- 循環(同じページで較正したしきい値でそのページの見逃しを数える)では、見逃しが増えた世代が採用されうる(クロスフィッティング導入前のゲート)。
- 新しいゲート(クロスフィッティング + 感度の高い指標)は、見逃しが増えた世代を一つも採用しない。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from yayoi_ocr.app import evalstats, gate, recalibrate
from yayoi_ocr.config import models_dir, private_data_dir
from yayoi_ocr.recognize.confidence import ConfidenceModel

pytestmark = [pytest.mark.slow]

EVAL_DIR = Path(os.environ.get("YAYOI_WA_EVAL") or private_data_dir() / "evals" / "wa_main")
CONF = models_dir() / "adapted" / "confidence.json"


def _replay():
    spec = importlib.util.spec_from_file_location(
        "gate_replay", Path(__file__).resolve().parent.parent / "scripts" / "gate_replay.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def replay():
    if not EVAL_DIR.is_dir() or not CONF.is_file():
        pytest.skip("報告 4 の記録(evals/wa_main)がない(開発機でだけ実行)")
    mod = _replay()
    return mod, mod.load(EVAL_DIR), ConfidenceModel.load(CONF)


def _judge(replay, n):
    mod, data, cal = replay
    out = []
    for _w, by_n in sorted(data.items()):
        if 0 not in by_n or n not in by_n:
            continue
        ppages, pflags = mod.to_pages(by_n[0], cal)
        cpages, cflags = mod.to_pages(by_n[n], cal)
        parent = evalstats.side_from(ppages, pflags)
        glob = evalstats.side_from(cpages, cflags)
        final, _calibs, cf = recalibrate.cross_fit(cal, cpages)
        circular = evalstats.side_from(cpages, [evalstats.review_flags(pg, final) for pg in cpages])
        out.append(
            {
                "harmful": glob.total.n_miss > parent.total.n_miss,
                "circular": gate.decide(parent.total, circular.total),
                "new": gate.decide(parent, evalstats.side_from(cpages, cf)),
            }
        )
    return out


@pytest.mark.parametrize("n", [1, 3])
def test_few_page_generations_that_add_unflagged_errors_are_never_adopted(replay, n):
    rows = _judge(replay, n)
    assert len(rows) >= 10
    harmful = [r for r in rows if r["harmful"]]
    assert len(harmful) >= 8  # 報告 4 の観察: 少数ページでは見逃しが増える
    # 循環では、見逃しが増えた世代が採用されうる(これがクロスフィッティングを入れた理由)
    assert any(r["circular"].adopted for r in harmful)
    # 新しいゲートは、見逃しが増えた世代を採用しない
    assert not any(r["new"].adopted for r in harmful)
    print(  # 件数だけ
        f"\n[gate-replay] N={n} writers={len(rows)} harmful={len(harmful)} "
        f"circular_adopted={sum(r['circular'].adopted for r in harmful)} new_adopted={sum(r['new'].adopted for r in rows)}"
    )


def test_the_new_gate_adopts_only_generations_that_do_not_add_errors_at_every_level(replay):
    for n in (5, 10, 20):
        for r in _judge(replay, n):
            if r["new"].adopted:
                assert not r["harmful"]
