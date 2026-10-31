"""Phase 4 基盤 2b/2c: バックグラウンドの追加学習の管理(自動・手動の開始、採用・不採用・見送り・失敗、中断、ロールバックとの競合)。
学習器は `MockTrainer`(torch なし)。値(摘要・金額)は計画にだけ入り、記録・ログ・世代のメタには入らない。"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app import gens
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend

PORT = 8132
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make(tmp_path):
    app = create_app(
        tmp_path / "a.sqlite3",
        TOKEN,
        allowed_hosts_for(PORT),
        dev=True,
        backend=FakeBackend(error_rate=0.0),
    )
    return app, TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH)


@pytest.fixture
def env(tmp_path):
    app, c = make(tmp_path)
    with c:
        # 自動開始は既定で有効。試験ごとに明示して始める
        c.put("/api/settings", json={"auto_train": False, "train_threshold_cells": 50})
        c.app_ = app
        c.dir_ = tmp_path
        yield c
    # TestClient の終了で close_app が呼ばれる


def confirm_pages(c, n: int, *, fill_rows: int = 29, seed: int = 1) -> list[int]:
    ids = []
    for i in range(n):
        job = c.post(
            "/api/dev/jobs",
            json={
                "pages": 1,
                "seed": seed + i,
                "fill_rows": fill_rows,
                "error_rate": 0.0,
                "review_rate": 0.0,
                "ambiguous_rows": [],
            },
        ).json()
        pid = job["pages"][0]["id"]
        confirm_with(c, pid, {})
        ids.append(pid)
    return ids


def wait(c):
    c.app_.state.trainer.join(20)
    assert not c.app_.state.trainer.running


def set_mock(c, mode="improve", delay=0.0):
    assert c.post("/api/dev/trainer", json={"mode": mode, "delay": delay}).status_code == 200


def last(c):
    return c.get("/api/training").json()["last"]


# ---- 状態・開始の条件


def test_status_without_enough_pages_and_manual_start_is_refused(env):
    st = env.get("/api/training").json()
    assert st["available"] and st["eligible_pages"] == 0 and st["new_cells"] == 0
    assert st["running"] is None and st["last"] is None and st["min_pages"] == 6
    r = env.post("/api/training/start")
    assert r.status_code == 422 and r.json() == {"error": "not_enough_pages", "have": 0, "need": 6}
    confirm_pages(env, 5)
    assert env.post("/api/training/start").json()["error"] == "not_enough_pages"


def test_new_cells_count_only_confirmed_non_blank_cells(env):
    confirm_pages(env, 2, fill_rows=10)
    st = env.get("/api/training").json()
    assert st["eligible_pages"] == 2
    con = env.app_.state.db
    expect = con.execute(
        "SELECT COUNT(*) FROM cells WHERE kind != 'blank' AND page_id IN"
        " (SELECT id FROM pages WHERE status = 'confirmed')"
    ).fetchone()[0]
    assert st["new_cells"] == expect and expect > 0


# ---- 手動の開始 → 採用


def test_manual_run_adopts_a_better_generation_and_makes_it_active(env):
    set_mock(env, "improve")
    ids = confirm_pages(env, 8)
    rid = env.post("/api/training/start").json()["run_id"]
    wait(env)
    run = last(env)
    assert run["id"] == rid and run["status"] == "adopted" and run["trigger"] == "manual"
    assert (run["n_train_pages"], run["n_eval_pages"]) == (6, 2)  # 新しい側 25%(ceil)= 2
    assert run["reason"] == "adopted_accuracy_up" and run["generation_id"]
    act = gens.active(env.app_.state.db)
    assert act["id"] == run["generation_id"] and act["number"] == 1 and act["kind"] == "trained"
    d = env.dir_ / "models" / "1" / "gen_0001"
    assert all((d / f).is_file() for f in (*gens.WEIGHT_FILES, "confidence.json", "meta.json"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["train_page_ids"] == ids[:6] and meta["eval_page_ids"] == ids[6:]  # 時系列で分ける
    assert meta["parent_number"] == 0 and meta["init"] == "shipped"
    assert meta["eval"]["candidate"]["accuracy"] > meta["eval"]["parent"]["accuracy"]
    assert meta["decision"] == {"outcome": "adopt", "reason": "adopted_accuracy_up"}
    # 作業ディレクトリ(確定値を含む計画)は残らない
    assert not (env.dir_ / "training" / str(rid)).exists()
    # 新しい確定セルは 0 に戻る(前回の実行が見たページより後に確定したものだけ数える)
    assert env.get("/api/training").json()["new_cells"] == 0
    # 以降に認識したページは新しい世代で認識する
    gl = env.get("/api/generations").json()
    assert gl["active_id"] == run["generation_id"]


def test_rejected_worse_candidate_keeps_the_active_generation_and_records_the_reason(env):
    set_mock(env, "worsen")
    confirm_pages(env, 7)
    env.post("/api/training/start")
    wait(env)
    run = last(env)
    assert run["status"] == "rejected" and run["reason"] == "miss_rate_worse"
    assert gens.active(env.app_.state.db)["kind"] == "shipped"
    g = next(x for x in env.get("/api/generations").json()["generations"] if x["number"] == 1)
    assert g["status"] == "rejected" and g["reject_reason"] == "miss_rate_worse"
    assert (
        not g["restorable"]
        and g["eval"]["candidate"]["miss_rate"] > g["eval"]["parent"]["miss_rate"]
    )
    assert not (env.dir_ / "models" / "1" / "gen_0001").exists()  # 重みは残さない
    assert env.post(f"/api/generations/{g['id']}/activate").status_code == 409


def test_flat_candidate_is_rejected_for_no_gain(env):
    set_mock(env, "flat")
    confirm_pages(env, 6)
    env.post("/api/training/start")
    wait(env)
    assert last(env)["reason"] == "no_gain" and last(env)["status"] == "rejected"


def test_run_is_deferred_when_the_evaluation_pages_are_too_few_cells(env):
    set_mock(env, "improve")
    confirm_pages(env, 6, fill_rows=6)  # 評価 2 ページ × 約 24 セル = 150 件に届かない
    env.post("/api/training/start")
    wait(env)
    run = last(env)
    assert run["status"] == "deferred" and run["reason"] == "eval_too_small"
    assert run["generation_id"] is None
    assert [g["number"] for g in env.get("/api/generations").json()["generations"]] == [0]


def test_trainer_failure_is_recorded_with_a_code_only(env):
    set_mock(env, "fail")
    confirm_pages(env, 6)
    env.post("/api/training/start")
    wait(env)
    run = last(env)
    assert run["status"] == "failed" and run["reason"] == "mock_failure"
    assert gens.active(env.app_.state.db)["kind"] == "shipped"


# ---- 自動の開始


def test_auto_start_fires_once_when_enough_new_cells_have_accumulated(env):
    set_mock(env, "improve")
    env.put("/api/settings", json={"auto_train": True, "train_threshold_cells": 50})
    confirm_pages(env, 5)  # ページ数が足りない間は始まらない
    assert env.get("/api/training").json()["last"] is None
    confirm_pages(env, 1, seed=100)  # 6 ページ目で条件がそろう
    wait(env)
    run = last(env)
    assert run["trigger"] == "auto" and run["status"] == "adopted"
    n_runs = len(env.get("/api/training").json()["runs"])
    confirm_pages(env, 1, seed=200)  # 1 ページ = 約 116 セル ≥ 50 なので、また始まる
    wait(env)
    assert len(env.get("/api/training").json()["runs"]) == n_runs + 1


def test_auto_start_respects_the_setting_the_threshold_and_running_jobs(env):
    from test_app_hardening import png_bytes, upload

    set_mock(env, "improve")
    env.put("/api/settings", json={"auto_train": False, "train_threshold_cells": 50})
    confirm_pages(env, 7)
    assert env.get("/api/training").json()["last"] is None  # 自動を止めている
    # しきい値に届かない
    env.put("/api/settings", json={"auto_train": True, "train_threshold_cells": 5000})
    confirm_pages(env, 1, seed=300)
    assert env.get("/api/training").json()["last"] is None
    # 認識中のページがあれば、次の機会に回す(CPU を取り合わない)
    env.put("/api/settings", json={"auto_train": True, "train_threshold_cells": 50})
    up = upload(env, "a.png", png_bytes()).json()
    job = env.post(
        "/api/jobs", json={"upload_ids": [up["id"]]}
    ).json()  # runner は動かしていない = 認識中のまま
    confirm_pages(env, 1, seed=400)
    assert env.get("/api/training").json()["last"] is None
    env.app_.state.runner.run_job(job["job_id"])  # 認識が終われば、次の確定で始まる
    confirm_pages(env, 1, seed=500)
    wait(env)
    assert last(env)["trigger"] == "auto"


def test_settings_validate_the_threshold_range(env):
    assert env.put("/api/settings", json={"train_threshold_cells": 10}).status_code == 422
    assert env.put("/api/settings", json={"train_threshold_cells": 9999}).status_code == 422
    s = env.get("/api/settings").json()["settings"]
    assert "auto_train" in s and "train_threshold_cells" in s


# ---- 実行中の制約


def test_only_one_run_at_a_time_and_reset_is_refused_while_running(env):
    set_mock(env, "improve", delay=1.0)
    confirm_pages(env, 6)
    env.post("/api/training/start")
    st = env.get("/api/training").json()
    assert st["running"] and st["running"]["status"] == "running"
    assert env.post("/api/training/start").json()["error"] == "training_running"
    assert env.post("/api/generations/reset", json={}).json()["error"] == "training_running"
    assert env.post("/api/learning/reset").json()["error"] == "training_running"
    wait(env)
    assert last(env)["status"] == "adopted"
    assert env.post("/api/generations/reset", json={}).status_code == 200


def test_cancel_stops_the_run_and_discards_the_result(env):
    set_mock(env, "improve", delay=5.0)
    confirm_pages(env, 6)
    env.post("/api/training/start")
    assert env.post("/api/training/cancel").json() == {"cancelled": True}
    wait(env)
    run = last(env)
    assert run["status"] == "interrupted" and run["reason"] == "cancelled"
    assert gens.active(env.app_.state.db)["kind"] == "shipped"
    assert env.post("/api/training/cancel").json() == {"cancelled": False}


def test_rollback_during_a_run_defers_it(env):
    set_mock(env, "improve")
    confirm_pages(env, 6)
    env.post("/api/training/start")
    wait(env)
    g1 = last(env)["generation_id"]
    set_mock(env, "improve", delay=1.0)
    confirm_pages(env, 1, seed=500)
    env.post("/api/training/start")
    shipped = gens.shipped(env.app_.state.db)["id"]
    assert env.post(f"/api/generations/{shipped}/activate").status_code == 200  # 学習中に戻す
    wait(env)
    run = last(env)
    assert run["status"] == "deferred" and run["reason"] == "active_generation_changed"
    assert gens.active(env.app_.state.db)["id"] == shipped and g1 != shipped


def test_a_run_left_running_by_a_closed_app_is_marked_interrupted_on_next_start(tmp_path):
    app, c = make(tmp_path)
    with c:
        con = app.state.db
        con.execute(
            "INSERT INTO training_runs (profile_id, trigger, status, started_at, stage, progress)"
            " VALUES (1, 'auto', 'running', 1.0, 'train', 0.3)"
        )
        (tmp_path / "training" / "7").mkdir(parents=True)
        (tmp_path / "training" / "7" / "plan.json").write_text("{}")
    _app2, c2 = make(tmp_path)  # 同じデータ領域でもう一度起動
    with c2:
        run = c2.get("/api/training").json()["last"]
        assert run["status"] == "interrupted" and run["reason"] == "app_closed"
        assert c2.get("/api/training").json()["running"] is None
        assert not (tmp_path / "training").exists()


# ---- 認識の入力・データの削除


def test_reset_with_data_deletion_removes_pagedata_but_keeping_data_does_not(env):
    from test_app_hardening import png_bytes, upload

    up = upload(env, "a.png", png_bytes()).json()
    job = env.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
    env.app_.state.runner.run_job(job["job_id"])
    pid = job["page_ids"][0]
    f = env.dir_ / "pagedata" / f"{pid}.npz"
    assert f.is_file()  # 認識のとき、追加学習用の入力を保存する
    confirm_with(env, pid, {})
    env.post("/api/generations/reset", json={"delete_learning_data": False})
    assert f.is_file()
    env.post("/api/generations/reset", json={"delete_learning_data": True})
    assert not f.exists()


def test_the_mock_run_record_and_meta_contain_no_cell_values(env):
    set_mock(env, "improve")
    secret = "秘密の摘要テキスト"
    pid = env.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": 9,
            "fill_rows": 29,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
        },
    ).json()["pages"][0]["id"]
    confirm_with(env, pid, {(0, 2): secret})
    confirm_pages(env, 6, seed=50)
    env.post("/api/training/start")
    wait(env)
    con = env.app_.state.db
    dump = json.dumps(
        [dict(r) for r in con.execute("SELECT * FROM training_runs")], ensure_ascii=False
    )
    dump += json.dumps(
        [dict(r) for r in con.execute("SELECT * FROM generations")], ensure_ascii=False
    )
    assert secret not in dump
    meta = (env.dir_ / "models" / "1" / "gen_0001" / "meta.json").read_text(encoding="utf-8")
    assert secret not in meta


def test_accuracy_report_has_generation_and_training_statistics_only(env):
    set_mock(env, "improve")
    secret = "秘密の摘要テキスト"
    pid = env.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": 11,
            "fill_rows": 29,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
        },
    ).json()["pages"][0]["id"]
    confirm_with(env, pid, {(0, 2): secret})
    confirm_pages(env, 6, seed=60)
    env.post("/api/training/start")
    wait(env)
    r = env.get("/api/learning/report")
    rep = r.json()
    assert rep["schema"] == 2
    gens_ = {g["number"]: g for g in rep["generations"]}
    assert set(gens_) == {0, 1} and gens_[1]["status"] == "active" and gens_[0]["kind"] == "shipped"
    assert gens_[1]["eval"]["candidate"]["n_text"] > 0 and gens_[1]["n_train_pages"] == 5
    assert "train_page_ids" not in json.dumps(rep)  # ページの ID は含めない
    assert (
        rep["training_runs"][0]["status"] == "adopted"
        and rep["training_runs"][0]["trigger"] == "manual"
    )
    assert rep["settings"]["auto_train"] is False and rep["settings"]["train_threshold_cells"] == 50
    assert all(p["generation"] in (0, 1) for p in rep["pages"])  # 世代の ID ではなく番号
    assert secret not in r.text


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("amount_miss", "amount_miss"),
        ("overconfident", "overconfident"),
        ("calib_worse", "calibration_worse"),
    ],
)
def test_sensitive_gate_rules_reject_a_generation_that_looks_better_by_accuracy(env, mode, reason):
    set_mock(env, mode)
    confirm_pages(env, 6)
    env.post("/api/training/start")
    wait(env)
    run = last(env)
    assert run["status"] == "rejected" and run["reason"] == reason
    assert gens.active(env.app_.state.db)["kind"] == "shipped"
    g = next(x for x in env.get("/api/generations").json()["generations"] if x["number"] == 1)
    assert (
        g["eval"]["candidate"]["accuracy"] > g["eval"]["parent"]["accuracy"]
        or mode == "calib_worse"
    )
    assert (
        g["eval"]["candidate"]["calib"]["n"] > 0 and "amount" in g["eval"]["candidate"]["by_column"]
    )


def test_the_first_pages_after_adoption_use_the_warm_thresholds(tmp_path):
    from test_app_generations import add_trained
    from test_app_hardening import png_bytes, upload
    from yayoi_ocr.app.recalibrate import WARM_PAGES

    class Rec(FakeBackend):
        def __init__(self):
            super().__init__(error_rate=0.0)
            self.payloads = []

        def process(self, payload):
            self.payloads.append(payload)
            return super().process(payload)

    backend = Rec()
    app = create_app(
        tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True, backend=backend
    )
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        con = app.state.db
        add_trained(con, tmp_path)
        warm = []
        for _ in range(WARM_PAGES + 1):
            up = upload(c, "a.png", png_bytes()).json()
            job = c.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
            app.state.runner.run_job(job["job_id"])
            warm.append(backend.payloads[-1].confidence_warm)
        assert warm == [True] * WARM_PAGES + [False]
        # 出荷時の世代では warm_threshold を使わない
        gens.activate(con, tmp_path / "models", gens.shipped(con)["id"])
        up = upload(c, "a.png", png_bytes()).json()
        job = c.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
        app.state.runner.run_job(job["job_id"])
        assert backend.payloads[-1].confidence_warm is False


# ---- 提案モード


def run_training(env, mode, seed=0):
    set_mock(env, mode)
    env.post("/api/training/start")
    wait(env)
    return last(env)


def test_a_generation_rejected_for_safety_but_significantly_more_accurate_becomes_a_suggestion(env):
    confirm_pages(env, 7)
    run = run_training(env, "suggest")
    assert run["status"] == "suggested" and run["reason"] == "amount_miss" and run["generation_id"]
    con = env.app_.state.db
    assert gens.active(con)["kind"] == "shipped"  # 使用中の世代は変わらない(要確認の判定は親のまま)
    g = next(x for x in env.get("/api/generations").json()["generations"] if x["number"] == 1)
    assert g["status"] == "suggest" and g["suggest_in_effect"] is True and g["restorable"] is False
    assert g["suggest"]["better"] > g["suggest"]["worse"] and g["suggest"]["p"] < 0.05
    assert env.post(f"/api/generations/{g['id']}/activate").status_code == 409
    d = env.dir_ / "models" / "1" / "gen_0001"
    assert all((d / f).is_file() for f in (*gens.WEIGHT_FILES, "confidence.json", "meta.json"))
    # 学習画面の API にも出る
    learning = env.get("/api/learning").json()
    assert any(x["suggest_in_effect"] for x in learning["generations"])


def test_safety_rejections_without_a_significant_gain_are_not_suggestions(env):
    confirm_pages(env, 7)
    for mode in ("amount_miss", "overconfident", "calib_worse", "worsen", "flat"):
        run = run_training(env, mode)
        assert run["status"] == "rejected", mode
    assert not [
        g for g in env.get("/api/generations").json()["generations"] if g["status"] == "suggest"
    ]


def test_pages_recognized_in_suggestion_mode_record_it_and_add_extra_review_cells(env):
    from test_app_hardening import png_bytes, upload

    confirm_pages(env, 7)
    run_training(env, "suggest")
    con = env.app_.state.db
    gid = gens.weights_for(con, env.dir_ / "models").suggest.generation_id
    up = upload(env, "a.png", png_bytes()).json()
    job = env.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
    env.app_.state.runner.run_job(job["job_id"])
    pid = job["page_ids"][0]
    assert (
        con.execute("SELECT suggest_generation FROM pages WHERE id = ?", (pid,)).fetchone()[0]
        == gid
    )
    cells = env.get(f"/api/pages/{pid}").json()["cells"]
    assert any("suggest_disagreement" in c["reasons"] and c["review"] for c in cells)
    # 通常の世代で認識したページは、提案モードの記録を持たない
    assert (
        con.execute(
            "SELECT COUNT(*) FROM pages WHERE suggest_generation IS NOT NULL AND id != ?", (pid,)
        ).fetchone()[0]
        == 0
    )
    confirm_with(env, pid, {})
    eff = env.get("/api/learning").json()["suggest_effect"]
    assert eff["suggest"]["n_pages"] == 1 and eff["suggest"]["n_review"] > 0
    assert eff["suggest"]["review_unchanged_share"] is not None
    assert eff["normal"]["n_pages"] >= 7
    rep = env.get("/api/learning/report").json()
    assert rep["suggest_effect"]["suggest"]["n_pages"] == 1
    assert next(g for g in rep["generations"] if g["number"] == 1)["suggest"]["better"] > 0


def test_a_later_run_promotes_the_suggestion_when_it_meets_every_criterion(env):
    confirm_pages(env, 7)
    sug = run_training(env, "suggest")["generation_id"]
    confirm_pages(env, 1, seed=900)
    run = run_training(env, "promote_prev")
    assert run["status"] == "adopted" and run["reason"] == "promoted_suggestion"
    assert run["generation_id"] == sug
    con = env.app_.state.db
    assert gens.active(con)["id"] == sug
    assert (
        con.execute("SELECT status FROM generations WHERE id = ?", (sug,)).fetchone()[0] == "active"
    )
    # 昇格した世代の重みと確信度の組は、そのまま使える
    assert gens.weights_for(con, env.dir_ / "models").generation_id == sug


def test_a_newly_adopted_generation_replaces_the_suggestion(env):
    confirm_pages(env, 7)
    sug = run_training(env, "suggest")["generation_id"]
    confirm_pages(env, 1, seed=901)
    run = run_training(env, "improve")
    assert run["status"] == "adopted"
    con = env.app_.state.db
    row = con.execute(
        "SELECT status, reject_reason FROM generations WHERE id = ?", (sug,)
    ).fetchone()
    assert (row["status"], row["reject_reason"]) == ("rejected", "superseded")
    assert not (env.dir_ / "models" / "1" / "gen_0001").exists()
    assert gens.weights_for(con, env.dir_ / "models").suggest is None
