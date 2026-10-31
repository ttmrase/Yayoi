"""採用した世代の監視と自動ロールバック: 確定時に「要確認でなかったのに直したセル」(観測した見逃し)を世代ごと・列ごとに数え、基準に当たれば親の世代に戻す。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from test_app_generations import add_trained, models
from yayoi_ocr.app import gens, monitor
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend
from yayoi_ocr.recognize.confidence import cp_lower

PORT = 8134
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def env(tmp_path):
    app = create_app(
        tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True, backend=FakeBackend()
    )
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        c.app_ = app
        c.dir_ = tmp_path
        c.put("/api/settings", json={"auto_train": False})
        yield c


def new_page(c, seed):
    """要確認のないページ(全セルが要確認でない = 直せば、すべて「観測した見逃し」)。現在の使用中の世代で認識したことになる。"""
    job = c.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": seed,
            "fill_rows": 29,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
        },
    ).json()
    return job["pages"][0]["id"]


def cells_of(c, pid, column):
    return [
        x
        for x in c.get(f"/api/pages/{pid}").json()["cells"]
        if x["column"] == column and x["kind"] == "text"
    ]


def edit(c, pid, cell_id, value="999"):
    r = c.patch(f"/api/pages/{pid}/cells/{cell_id}", json={"action": "edit", "value": value})
    assert r.status_code == 200


def observed_pair(c, pid):
    return monitor.observed(
        c.app_.state.db,
        c.app_.state.db.execute("SELECT generation FROM pages WHERE id = ?", (pid,)).fetchone()[0],
    )


def test_one_observed_amount_miss_halts_the_adopted_generation_and_returns_to_the_parent(env):
    con = env.app_.state.db
    g1 = add_trained(con, env.dir_)
    shipped = gens.shipped(con)["id"]
    pid = new_page(env, 1)
    assert con.execute("SELECT generation FROM pages WHERE id = ?", (pid,)).fetchone()[0] == g1
    amount = cells_of(env, pid, "amount")[0]
    edit(env, pid, amount["id"])  # 要確認でなかった金額を利用者が直した
    r = env.post(f"/api/pages/{pid}/confirm", json={})
    assert r.status_code == 200 and r.json()["generation_halted"] == {
        "number": 1,
        "code": "halted_amount_miss",
    }
    row = con.execute("SELECT * FROM generations WHERE id = ?", (g1,)).fetchone()
    assert (row["status"], row["reject_reason"], row["ack"]) == ("halted", "halted_amount_miss", 0)
    assert gens.active(con)["id"] == shipped  # 親の世代(ここでは出荷時)に戻った
    # 通知・学習画面・精度レポートに理由が出る(件数だけ)
    n = env.get("/api/notices").json()["notices"]
    assert [(x["number"], x["status"], x["code"]) for x in n] == [
        (1, "halted", "halted_amount_miss")
    ]
    g = next(x for x in env.get("/api/generations").json()["generations"] if x["number"] == 1)
    assert g["status"] == "halted" and g["restorable"] is False and g["halt"]["k_amount"] == 1
    rep = env.get("/api/learning/report").json()
    assert (
        next(x for x in rep["generations"] if x["number"] == 1)["halt"]["code"]
        == "halted_amount_miss"
    )
    # 戻せない。また、次に認識するページは親の世代を使う
    assert env.post(f"/api/generations/{g1}/activate").status_code == 409
    assert gens.weights_for(con, models(env.dir_)).dir is None


def test_a_miss_rate_above_the_target_by_the_cp_lower_bound_halts_but_a_few_misses_do_not(env):
    con = env.app_.state.db
    g1 = add_trained(con, env.dir_)
    n_cells = 116
    k_needed = next(k for k in range(1, 40) if cp_lower(k, n_cells) > monitor.TARGET_MISS)
    assert k_needed >= 2  # 1 件では下側限界は目標を超えない
    p1 = new_page(env, 2)
    desc = cells_of(env, p1, "description")
    for x in desc[: k_needed - 1]:
        edit(env, p1, x["id"], "別の値")
    r = env.post(f"/api/pages/{p1}/confirm", json={})
    assert "generation_halted" not in r.json()  # まだ目標を超えたと言えない
    assert gens.active(con)["id"] == g1
    # 次のページで直したセルが累積し、(全セル数も増えるが)下側限界が目標を超える
    p2 = new_page(env, 3)
    for x in cells_of(env, p2, "description")[:k_needed]:
        edit(env, p2, x["id"], "別の値")
    r = env.post(f"/api/pages/{p2}/confirm", json={})
    assert r.json()["generation_halted"]["code"] == "halted_miss_rate"
    assert gens.active(con)["kind"] == "shipped"


def test_edits_to_cells_that_were_flagged_for_review_are_not_observed_misses(env):
    con = env.app_.state.db
    add_trained(con, env.dir_)
    job = env.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": 3,
            "fill_rows": 6,
            "error_rate": 0.6,
            "review_rate": 0.5,
            "ambiguous_rows": [],
        },
    ).json()
    pid = job["pages"][0]["id"]
    confirm_with(env, pid, {})  # 要確認のセルを確認して確定(要確認でないセルは触らない)
    obs = observed_pair(env, pid)
    assert obs["k_total"] == 0 and gens.active(con)["kind"] == "trained"


def test_only_the_active_trained_generation_is_watched(env):
    con = env.app_.state.db
    # 世代 0(出荷時)で認識したページでは、観測した見逃しがあっても何も起きない(戻す先がない)
    p0 = new_page(env, 5)
    edit(env, p0, cells_of(env, p0, "amount")[0]["id"])
    assert "generation_halted" not in env.post(f"/api/pages/{p0}/confirm", json={}).json()
    # 過去の世代(retired)で認識したページも、現在の使用中の世代には関係しない
    g1 = add_trained(con, env.dir_, 1)
    p1 = new_page(env, 6)
    g2 = add_trained(con, env.dir_, 2)  # p1 を認識したあとに世代 2 が採用された
    edit(env, p1, cells_of(env, p1, "amount")[0]["id"])
    assert "generation_halted" not in env.post(f"/api/pages/{p1}/confirm", json={}).json()
    assert (
        gens.active(con)["id"] == g2
        and con.execute("SELECT status FROM generations WHERE id = ?", (g1,)).fetchone()[0]
        == "retired"
    )


def test_a_halt_makes_the_next_auto_training_wait_for_new_confirmed_data(env):
    con = env.app_.state.db
    mgr = env.app_.state.trainer
    add_trained(con, env.dir_)
    pid = new_page(env, 7)
    edit(env, pid, cells_of(env, pid, "amount")[0]["id"])
    env.post(f"/api/pages/{pid}/confirm", json={})
    assert (
        mgr.new_cells(1) == 0
    )  # 停止の時点までに確定したデータは、再挑戦の「新しい確定セル」に数えない
    p2 = new_page(env, 8)
    env.post(f"/api/pages/{p2}/confirm", json={})
    assert mgr.new_cells(1) > 0


def test_decide_is_a_pure_function_of_the_observed_counts():
    assert (
        monitor.decide(
            {"by_column": {"amount": 1}, "k_total": 1, "n_cells": 100000, "n_pages": 900}
        )
        == "halted_amount_miss"
    )
    assert (
        monitor.decide({"by_column": {"date": 2}, "k_total": 2, "n_cells": 1160, "n_pages": 10})
        is None
    )
    assert (
        monitor.decide({"by_column": {"date": 12}, "k_total": 12, "n_cells": 1160, "n_pages": 10})
        == "halted_miss_rate"
    )
