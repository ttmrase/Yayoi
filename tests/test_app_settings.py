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
        "taxpayer": "taxable",
        "tax_scheme": "simplified",
        "tax_method": "exclusive",
        "tax_rounding": "round",
        "payments": [{"name": "現金", "credit_account": "現金"}],
        "titles": [{"name": "消耗品費", "debit_account": "消耗品費", "debit_sub": ""}],
        "default_tax": [{"account": "消耗品費", "tax_category": "課対仕入内10%"}],
    }
    r = client.put("/api/settings", json=body)
    assert r.status_code == 200 and r.json()["missing"] == []
    got = client.get("/api/settings").json()["settings"]
    assert (got["taxpayer"], got["tax_scheme"], got["tax_method"], got["tax_rounding"]) == (
        "taxable",
        "simplified",
        "exclusive",
        "round",
    )
    assert got["payments"][0]["name"] == "現金"
    assert got["default_tax"][0]["tax_category"] == "課対仕入内10%"


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
    assert not empty["can_rollback"] and not empty["can_reset"] and empty["phase"] == 4
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


# ---- 精度レポート(統計だけ)


def test_accuracy_report_has_statistics_only_and_never_cell_values(client):
    secret = "秘密の摘要テキスト"
    client.put(
        "/api/settings",
        json={
            "payments": [{"name": "秘密の支払手段", "credit_account": "秘密の科目"}],
            "titles": [{"name": "秘密のタイトル", "debit_account": "秘密の借方"}],
        },
    )
    job = new_job(client, error_rate=0.5, review_rate=0.7, ambiguous_rows=[], fill_rows=6)
    p1 = job["pages"][0]["id"]
    client.put(f"/api/pages/{p1}/title", json={"title": "秘密のタイトル"})
    confirm_with(client, p1, {(0, 2): secret})
    r = client.get("/api/learning/report")
    assert r.status_code == 200 and "accuracy_report.json" in r.headers["content-disposition"]
    rep = r.json()
    assert rep["schema"] == 1 and rep["summary"]["n_pages"] == 1 and rep["pages"][0]["order"] == 1
    assert set(rep["per_column"]) == {"date", "account", "description", "amount"}
    assert sum(rep["edit_kinds"].values()) == 24
    text = r.text
    for word in (secret, "秘密の支払手段", "秘密の科目", "秘密のタイトル", "秘密の借方"):
        assert word not in text  # 値・設定の名前・タイトルは含まない
    assert rep["settings"] == {
        "n_payments": 1, "n_titles": 1, "taxpayer": "taxable", "tax_scheme": "general", "tax_method": "inclusive",
    }  # fmt: skip

    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)

    from yayoi_ocr import __version__

    allowed_values = {__version__, "taxable", "general", "inclusive"}
    reason_codes = set(rep["review_reasons"]) | set(rep["edit_kinds"])
    keys = set()

    def collect_keys(x):
        if isinstance(x, dict):
            keys.update(x)
            for v in x.values():
                collect_keys(v)
        elif isinstance(x, list):
            for v in x:
                collect_keys(v)

    collect_keys(rep)
    leftovers = [s for s in strings(rep) if s not in allowed_values | reason_codes | keys]
    assert leftovers == []  # キー・理由コード・選択肢のコード・バージョン以外の文字列は無い


def _put_tax(client, category, **kw):
    body = {
        "payments": [{"name": "現金", "credit_account": "現金"}],
        "titles": [{"name": "消耗品費", "debit_account": "消耗品費"}],
        "default_tax": [{"account": "消耗品費", "tax_category": category}],
        **kw,
    }
    return client.put("/api/settings", json=body)


@pytest.mark.parametrize(
    ("kw", "category", "ok"),
    [
        ({}, "課対仕入込10%", True),
        ({}, "共対仕入込軽減8%", True),
        ({}, "非課仕入", True),
        ({}, "課対仕入内10%", False),  # 税込経理に内税の区分は使えない
        ({}, "課対仕入", False),  # 税率なし = 弥生で 5% 扱い
        ({}, "課対仕入込10％", False),  # 全角の %
        ({}, "売上", False),  # 仕入系以外
        ({"tax_method": "exclusive"}, "課対仕入内10%", True),
        ({"tax_method": "exclusive"}, "共対仕入別軽減8%", True),
        ({"tax_method": "exclusive"}, "課対仕入込10%", False),
        ({"tax_scheme": "simplified"}, "課対仕入込10%", True),
        ({"tax_scheme": "simplified"}, "共対仕入込10%", False),  # 簡易課税は課対のみ
        ({"tax_scheme": "simplified", "tax_method": "exclusive"}, "課対仕入外5%", True),
        ({"taxpayer": "exempt"}, "対象外", True),
    ],
)
def test_default_tax_must_be_in_the_allow_list_for_the_scheme_and_method(client, kw, category, ok):
    r = _put_tax(client, category, **kw)
    assert (r.status_code == 200) is ok, r.text


def test_the_error_code_tells_why(client):
    def code(category, **kw):
        r = _put_tax(client, category, **kw)
        return r.json()["problems"][0]["code"]

    assert code("課対仕入") == "tax_rate_missing"
    assert code("課対仕入込10％") == "tax_percent_fullwidth"
    assert code("課対仕入内10%") == "tax_category_not_allowed"


def test_settings_response_has_the_allow_lists_for_the_screen(client):
    r = client.get("/api/settings").json()
    assert set(r["tax_options"]) == {
        "general_inclusive",
        "general_exclusive",
        "simplified_inclusive",
        "simplified_exclusive",
    }
    assert r["tax_options"]["general_inclusive"][0] == "課対仕入込10%"
    assert r["tax_options"]["general_exclusive"][0] == "課対仕入内10%"


def test_changing_the_method_leaves_old_categories_flagged_as_missing_settings(client):
    assert _put_tax(client, "課対仕入込10%").status_code == 200
    s = client.get("/api/settings").json()["settings"]
    s["tax_method"] = "exclusive"
    # 保存は拒否される(古い税区分が許可リストにない)ので、画面で直させる
    assert client.put("/api/settings", json=s).status_code == 422
