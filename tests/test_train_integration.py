"""Phase 4 の実学習の結合試験(合成の書き手): 確定 → 追加学習(別プロセス)→ 評価・再較正 → ゲート → 採用/不採用 → 認識への反映。

合成データ・出荷時の重み・リハーサルの束(`YAYOI_PRIVATE_DATA` の下)が必要なので、無ければ飛ばす。CPU で数分かかる(学習量は `YAYOI_TRAIN_MAX_PASSES` で下げる)。
実データは使わない。結果(採用か不採用か)は合成データでの学習の効きに依存するので、どちらでも成り立つ不変条件だけを確かめる。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("cv2")

from yayoi_ocr.app import gens, learned, store
from yayoi_ocr.app import settings as app_settings
from yayoi_ocr.app.db import DEFAULT_PROFILE_ID
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.trainers import SubprocessTrainer
from yayoi_ocr.app.worker import FakeBackend
from yayoi_ocr.config import models_dir, private_data_dir
from yayoi_ocr.recognize import variants
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.recognize.pagedata import save_page_input
from yayoi_ocr.recognize.rehearsal import FILE as REHEARSAL_FILE
from yayoi_ocr.synth.content import ACCOUNTS, TITLES
from yayoi_ocr.synth.dataset import list_pages, load_page

pytestmark = [pytest.mark.ml, pytest.mark.slow]

DATA = os.environ.get("YAYOI_SYNTH_WRITER_DATA") or str(private_data_dir() / "datasets" / "wa_seq")
N_PAGES = 16  # 学習 12 + 評価 4(評価の非空欄セルが 150 件に届く)


def _need() -> None:
    needed = [
        models_dir() / "celltype.pt",
        models_dir() / REHEARSAL_FILE,
        variants.confidence_path(variants.default_variant()),
    ]
    if not all(p.is_file() for p in needed) or not os.path.isdir(DATA):
        pytest.skip("合成データ・出荷時の重み・リハーサルの束が無い(開発機でだけ実行)")


def _writer_pages() -> list:
    by: dict[int, list] = {}
    for p in list_pages(DATA):
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        by.setdefault(m["page"]["writer_id"], []).append((m["page"]["seed"], p))
    w = sorted(by)[0]
    return [load_page(p) for _, p in sorted(by[w])[:N_PAGES]]


def _spec(pg) -> store.PageSpec:
    cells = [
        store.CellSpec(
            row=c.row,
            col=c.col,
            column=c.column,
            kind="wave" if c.kind == "elided" else c.kind,
            recognized_text=c.raw,
            resolved_value=c.resolved,
            review=False,
        )
        for c in pg.cells
    ]
    return store.PageSpec(cells)


def test_real_training_adopts_or_rejects_consistently_and_the_generation_is_usable(
    tmp_path, monkeypatch
):
    _need()
    monkeypatch.setenv("YAYOI_TRAIN_MAX_PASSES", "600")
    token = new_token()
    app = create_app(
        tmp_path / "a.sqlite3",
        token,
        allowed_hosts_for(8133),
        dev=True,
        backend=FakeBackend(),
        trainer=SubprocessTrainer(),
    )
    con = app.state.db
    mgr = app.state.trainer
    pages = _writer_pages()
    assert len(pages) == N_PAGES
    app_settings.save(
        con,
        app_settings.Settings(
            payments=[app_settings.PaymentMap(name=a, credit_account=a) for a in ACCOUNTS],
            titles=[app_settings.TitleMap(name=t, debit_account=t) for t in TITLES[:3]],
        ),
    )
    _job_id, page_ids = store.create_job(
        con,
        tmp_path / "images",
        "synthetic-writer",
        DEFAULT_PROFILE_ID,
        [_spec(pg) for pg in pages],
    )
    prev = None
    for pid, pg in zip(page_ids, pages, strict=True):
        save_page_input(mgr.pagedata_file(pid), from_page_data(pg), prev)
        store.confirm_page(con, pid, 30.0)
    assert len(mgr.eligible_pages()) == N_PAGES

    run_id = mgr.start(DEFAULT_PROFILE_ID, "manual")
    mgr.join(1500)
    assert not mgr.running
    run = dict(con.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone())
    assert run["status"] in ("adopted", "rejected", "deferred"), run  # 失敗ではない
    meta_row = con.execute("SELECT meta FROM generations WHERE number = 1").fetchone()
    ev = (json.loads(meta_row[0]) if meta_row and meta_row[0] else {}).get("eval", {})
    print(  # 件数と割合だけ(値は出さない)
        f"\n[train-integration] status={run['status']} reason={run['reason']} "
        f"parent={ev.get('parent')} candidate={ev.get('candidate')}"
    )
    assert (run["n_train_pages"], run["n_eval_pages"]) == (12, 4)
    gl = {g["number"]: g for g in gens.list_generations(con, tmp_path / "models")}
    log = (tmp_path / "training" / "last.log").read_text(encoding="utf-8")
    assert "label:" in log and "train: width=256" in log and "eval: parent" in log
    # ログと記録に確定値(摘要・金額)は出ない
    secrets = {
        c.resolved
        for pg in pages
        for c in pg.cells
        if c.column == "description" and len(c.resolved) >= 3
    }
    dump = log + json.dumps(
        [dict(r) for r in con.execute("SELECT * FROM generations")], ensure_ascii=False
    )
    dump += json.dumps(run, ensure_ascii=False)
    assert not any(s in dump for s in secrets)
    assert not (
        tmp_path / "training" / str(run_id)
    ).exists()  # 作業ディレクトリ(確定値を含む計画)は消える

    if run["status"] == "adopted":
        g = gl[1]
        assert g["status"] == "active"
        d = tmp_path / "models" / "1" / "gen_0001"
        assert all((d / f).is_file() for f in (*gens.WEIGHT_FILES, "confidence.json", "meta.json"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        assert meta["decision"]["outcome"] == "adopt" and meta["init"] == "shipped"
        w = gens.weights_for(con, tmp_path / "models")
        assert w.dir == d and w.number == 1
        # 新しい世代の重みで、実際に認識できる(重み・確信度が読める)
        from yayoi_ocr.recognize.celltype import CellTypeClassifier
        from yayoi_ocr.recognize.confidence import ConfidenceModel
        from yayoi_ocr.recognize.page import JobContext, PageRecognizer
        from yayoi_ocr.recognize.parseq.engine import ParseqEngine

        rec = PageRecognizer(
            ParseqEngine(models_dir=d),
            CellTypeClassifier.load(models_dir() / "celltype.pt"),
            calibrator=ConfidenceModel.load(d / "confidence.json"),
        )
        res = rec.recognize_page(from_page_data(pages[-1]), JobContext(accounts=list(ACCOUNTS)))
        assert len(res.cells) == 116
        # ロールバックで出荷時に戻り、戻したあと世代 1 にまた進める
        gens.activate(con, tmp_path / "models", gens.shipped(con)["id"])
        assert gens.weights_for(con, tmp_path / "models").dir is None
    elif run["status"] == "rejected":
        g = gl[1]
        assert g["status"] == "rejected" and g["reject_reason"] == run["reason"]
        assert not (tmp_path / "models" / "1" / "gen_0001").exists()
        assert gens.active(con)["kind"] == "shipped"
    # 学習に使ったページの文脈(辞書など)は、評価用ページを含まない
    state = learned.build_state(
        con, DEFAULT_PROFILE_ID, app_settings.load(con), only_pages=set(page_ids[:12])
    )
    assert state.dictionary is not None
