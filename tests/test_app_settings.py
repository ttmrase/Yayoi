"""設定・ページのタイトル・確定済みページからの学習状態(辞書・科目の語彙・金額の履歴)のテスト。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app import db, learned, settings
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app

PORT = 8126
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        c.app_ = app
        yield c


def test_settings_default_and_round_trip(client):
    r = client.get("/api/settings").json()
    assert r["settings"]["payments"] == [] and set(r["missing"]) == {"no_payments", "no_titles"}
    body = {
        "business_type": "corporate",
        "tax_method": "exclusive",
        "payments": [{"name": "現金", "credit_account": "現金"}],
        "titles": [{"name": "消耗品費", "debit_account": "消耗品費", "debit_sub": ""}],
        "default_tax": [{"account": "消耗品費", "tax_category": "課対仕入10%"}],
    }
    r = client.put("/api/settings", json=body)
    assert r.status_code == 200 and r.json()["missing"] == []
    got = client.get("/api/settings").json()["settings"]
    assert got["business_type"] == "corporate" and got["payments"][0]["name"] == "現金"
    assert got["default_tax"][0]["tax_category"] == "課対仕入10%"


@pytest.mark.parametrize(
    ("body", "path", "code"),
    [
        ({"payments": [{"name": "現金"}, {"name": "現金"}]}, "payments[1].name", "duplicate"),
        ({"payments": [{"name": ""}]}, "payments[0].name", "empty"),
        ({"payments": [{"name": " 現金"}]}, "payments[0].name", "surrounding_space"),
        (
            {"titles": [{"name": "会議費", "debit_sub": "😀"}]},
            "titles[0].debit_sub",
            "cp932_unencodable",
        ),
    ],
)
def test_invalid_settings_are_rejected_with_the_field(client, body, path, code):
    r = client.put("/api/settings", json=body)
    assert r.status_code == 422 and r.json()["error"] == "invalid_settings"
    assert {"path": path, "code": code} in r.json()["problems"]


def test_settings_type_errors_and_unknown_choices_are_422(client):
    assert client.put("/api/settings", json={"tax_method": "whatever"}).status_code == 422
    assert client.put("/api/settings", json={"payments": [{"name": "x" * 21}]}).status_code == 422


def test_readiness_lists_what_export_still_needs():
    s = settings.Settings(payments=[settings.PaymentMap(name="現金")], titles=[])
    assert settings.readiness(s) == ["payment_without_credit_account", "no_titles"]


def test_a_broken_stored_setting_falls_back_to_defaults(client):
    client.app_.state.db.execute("INSERT INTO settings (key, value) VALUES ('app', '{nope')")
    assert client.get("/api/settings").json()["settings"]["payments"] == []


# ---- ページのタイトル


def new_job(client, **kw):
    r = client.post("/api/dev/jobs", json={"pages": 2, "seed": 3, **kw})
    assert r.status_code == 200
    return r.json()


def test_title_is_saved_trimmed_and_suggested_to_the_next_page(client):
    job = new_job(client)
    p1, p2 = (p["id"] for p in job["pages"])
    page = client.get(f"/api/pages/{p1}").json()
    assert page["title"] is None and page["title_suggestion"] is None
    r = client.put(f"/api/pages/{p1}/title", json={"title": "  旅費交通費 "})
    assert r.status_code == 200 and r.json()["title"] == "旅費交通費"
    assert client.get(f"/api/pages/{p2}").json()["title_suggestion"] == "旅費交通費"
    assert client.put(f"/api/pages/{p1}/title", json={"title": ""}).json()["title"] is None


def test_title_validation(client):
    p = new_job(client)["pages"][0]["id"]
    assert client.put(f"/api/pages/{p}/title", json={"title": "あ" * 41}).status_code == 422
    r = client.put(f"/api/pages/{p}/title", json={"title": "😀"})
    assert r.status_code == 422 and r.json()["error"] == "cp932_unencodable"
    assert client.put("/api/pages/9999/title", json={"title": "x"}).status_code == 404


# ---- 確定済みページからの学習状態


def test_learned_state_is_rebuilt_from_confirmed_pages_only(client):
    job = new_job(client, error_rate=0, review_rate=0, ambiguous_rows=[], fill_rows=4)
    p1 = job["pages"][0]["id"]
    con = client.app_.state.db
    cfg = settings.Settings(
        payments=[settings.PaymentMap(name="現金", credit_account="現金")],
        titles=[settings.TitleMap(name="消耗品費", debit_account="消耗品費")],
    )
    before = learned.build_state(con, db.DEFAULT_PROFILE_ID, cfg)
    assert before.accounts == ["現金"] and before.dictionary == {} and before.amount_history == {}
    confirm_with(client, p1, {(0, 1): "掛売", (0, 2): "テスト摘要", (0, 3): "1200"})
    after = learned.build_state(con, db.DEFAULT_PROFILE_ID, cfg)
    assert (
        after.accounts[0] == "現金" and "掛売" in after.accounts
    )  # 設定の語彙が先、確定した科目が後
    assert after.dictionary["テスト摘要"] == 1
    assert after.amount_history["d:テスト摘要"] == {1200: 1}
    assert after.amount_history["a:掛売"] == {1200: 1}
    assert (
        learned.build_state(con, db.DEFAULT_PROFILE_ID, cfg).dictionary == after.dictionary
    )  # 何度でも同じ
    # 再オープン → 状態から外れる(二重に数えない)
    client.post(f"/api/pages/{p1}/reopen")
    assert learned.build_state(con, db.DEFAULT_PROFILE_ID, cfg).dictionary == {}
    confirm_with(client, p1, {})
    again = learned.build_state(con, db.DEFAULT_PROFILE_ID, cfg)
    assert again.dictionary["テスト摘要"] == 1


def test_learned_state_is_per_profile(client):
    job = new_job(client, error_rate=0, review_rate=0, ambiguous_rows=[], fill_rows=3)
    p1 = job["pages"][0]["id"]
    confirm_with(client, p1, {(0, 2): "共通の摘要"})
    con = client.app_.state.db
    con.execute("INSERT INTO profiles (id, name, created_at) VALUES (2, '別の人', 0)")
    mine = learned.build_state(con, db.DEFAULT_PROFILE_ID, settings.Settings())
    other = learned.build_state(con, 2, settings.Settings())
    assert "共通の摘要" in mine.dictionary and other.dictionary == {}


def test_confirmed_titles_join_the_title_vocabulary(client):
    job = new_job(client, error_rate=0, review_rate=0, ambiguous_rows=[], fill_rows=2)
    p1 = job["pages"][0]["id"]
    client.put(f"/api/pages/{p1}/title", json={"title": "新しい勘定"})
    con = client.app_.state.db
    assert (
        "新しい勘定"
        not in learned.build_state(con, db.DEFAULT_PROFILE_ID, settings.Settings()).titles
    )
    confirm_with(client, p1, {})
    assert (
        "新しい勘定" in learned.build_state(con, db.DEFAULT_PROFILE_ID, settings.Settings()).titles
    )


# ---- 学習画面の指標


def test_learning_metrics_count_only_confirmed_pages_and_no_values(client):
    job = new_job(client, error_rate=0.6, review_rate=0.5, ambiguous_rows=[], fill_rows=6)
    p1 = job["pages"][0]["id"]
    empty = client.get("/api/learning").json()
    assert empty["summary"]["n_pages"] == 0 and empty["generations"][0]["id"] > 0
    assert not empty["can_rollback"] and not empty["can_reset"] and empty["phase"] == 3
    # 要確認でないセルを利用者が直した = 見逃しの観測
    page = client.get(f"/api/pages/{p1}").json()
    quiet = next(
        c
        for c in page["cells"]
        if c["kind"] != "blank" and not c["review"] and c["column"] == "amount"
    )
    client.patch(f"/api/pages/{p1}/cells/{quiet['id']}", json={"action": "edit", "value": "123"})
    confirm_with(client, p1, {})
    m = client.get("/api/learning").json()
    assert m["summary"]["n_pages"] == 1 and m["summary"]["unflagged_corrected"] >= 1
    row = m["pages"][0]
    assert row["n_text"] == 24 and row["n_corrected"] >= 1 and 0 < row["review_rate"] <= 1
    assert all(not isinstance(v, str) for k, v in row.items())  # 値(文字列)を含まない
