"""行ごとの税区分の上書きと、摘要ごとの記憶(app/rowtax.py)。合成した認識結果だけを使う。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app.fake import ACCOUNTS
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app

PORT = 8128
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DEFAULT = "課対仕入込10%"
SETTINGS = {
    "wizard_done": True,
    "payments": [{"name": a, "credit_account": a} for a in ACCOUNTS],
    "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
    "default_tax": [{"account": "旅費交通費", "tax_category": DEFAULT}],
}
STAMP, REGISTRY = "切手代", "登記手数料"


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        assert c.put("/api/settings", json=SETTINGS).status_code == 200
        yield c


def open_page(client, descs: dict[int, str], seed=5, title="旅費交通費"):
    """合成ページを作り、行ごとの摘要を指定の値にする(確定はしない)。"""
    job = client.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": seed,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
            "fill_rows": 4,
        },
    ).json()
    pid = job["pages"][0]["id"]
    assert client.put(f"/api/pages/{pid}/title", json={"title": title}).status_code == 200
    cells = {(c["row"], c["col"]): c for c in client.get(f"/api/pages/{pid}").json()["cells"]}
    for row, text in descs.items():
        r = client.patch(
            f"/api/pages/{pid}/cells/{cells[(row, 2)]['id']}",
            json={"action": "edit", "value": text},
        )
        assert r.status_code == 200, r.text
    return pid


def tax_of(client, pid):
    return {r["row"]: r for r in client.get(f"/api/pages/{pid}").json()["tax"]["rows"]}


def set_tax(client, pid, row, category):
    return client.put(f"/api/pages/{pid}/rows/{row}/tax", json={"tax_category": category})


def export_h(client, pid):
    rec = client.post("/api/exports", json={"page_ids": [pid], "accept_duplicates": True}).json()
    text = client.get(f"/api/exports/{rec['id']}/file").content.decode("cp932")
    return [line.split(",")[7] for line in text.split("\r\n") if line]


def test_the_page_lists_a_tax_category_per_row_with_the_default_and_the_options(client):
    pid = open_page(client, {})
    tax = client.get(f"/api/pages/{pid}").json()["tax"]
    assert tax["enabled"] is True and DEFAULT in tax["options"] and "課対仕入" not in tax["options"]
    assert [r["row"] for r in tax["rows"]] == [0, 1, 2, 3]
    assert all(r["category"] == DEFAULT and r["source"] == "default" for r in tax["rows"])


def test_a_user_choice_is_per_row_validated_and_used_in_the_export(client):
    pid = open_page(client, {0: STAMP, 1: REGISTRY})
    assert set_tax(client, pid, 0, "非課仕入").status_code == 200
    assert set_tax(client, pid, 1, "対外仕入").status_code == 200
    rows = tax_of(client, pid)
    assert (rows[0]["category"], rows[0]["source"]) == ("非課仕入", "user")
    assert (rows[2]["category"], rows[2]["source"]) == (DEFAULT, "default")
    # 許可リストにないもの・存在しない行
    assert set_tax(client, pid, 2, "課対仕入").status_code == 422
    assert set_tax(client, pid, 2, "課対仕入内10%").status_code == 422  # 税込経理に内税は使えない
    assert set_tax(client, pid, 20, "非課仕入").status_code == 404
    confirm_with(client, pid, {})
    assert export_h(client, pid) == ["非課仕入", "対外仕入", DEFAULT, DEFAULT]


def test_clearing_a_choice_returns_to_the_default(client):
    pid = open_page(client, {})
    set_tax(client, pid, 1, "対象外")
    assert tax_of(client, pid)[1]["category"] == "対象外"
    assert set_tax(client, pid, 1, None).status_code == 200
    assert tax_of(client, pid)[1]["source"] == "default"


def test_a_confirmed_page_cannot_be_changed_until_reopened(client):
    pid = open_page(client, {})
    confirm_with(client, pid, {})
    assert set_tax(client, pid, 0, "非課仕入").status_code == 409
    client.post(f"/api/pages/{pid}/reopen", json={})
    assert set_tax(client, pid, 0, "非課仕入").status_code == 200


def test_exempt_businesses_have_no_row_choices(client):
    assert client.put("/api/settings", json={**SETTINGS, "taxpayer": "exempt"}).status_code == 200
    pid = open_page(client, {})
    assert client.get(f"/api/pages/{pid}").json()["tax"]["enabled"] is False
    assert set_tax(client, pid, 0, "対象外").status_code == 409


def test_confirming_remembers_the_choice_by_description_and_suggests_it_next_time(client):
    pid = open_page(client, {0: STAMP, 1: REGISTRY})
    set_tax(client, pid, 0, "非課仕入")
    set_tax(client, pid, 1, "対外仕入")
    assert client.get("/api/tax-memory").json()["entries"] == []  # 確定するまで記憶しない
    confirm_with(client, pid, {})
    mem = client.get("/api/tax-memory").json()["entries"]
    assert {(e["description"], e["tax_category"]) for e in mem} == {
        (STAMP, "非課仕入"),
        (REGISTRY, "対外仕入"),
    }
    # 別のページで同じ摘要 → 提案(source = suggested)。摘要が違う行は既定のまま(〃の行は上の行と同じ摘要になる)
    pid2 = open_page(client, {0: STAMP, 2: REGISTRY}, seed=6)
    desc = {
        c["row"]: c["value"]
        for c in client.get(f"/api/pages/{pid2}").json()["cells"]
        if c["col"] == 2
    }
    rows = tax_of(client, pid2)
    assert (rows[0]["category"], rows[0]["source"]) == ("非課仕入", "suggested")
    assert (rows[2]["category"], rows[2]["source"]) == ("対外仕入", "suggested")
    for r, row in rows.items():
        assert (row["source"] == "suggested") == (desc[r] in {STAMP, REGISTRY})


def test_the_match_ignores_spaces_and_width_differences(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    pid2 = open_page(client, {1: " 切手 代 "}, seed=6)
    assert tax_of(client, pid2)[1]["source"] == "suggested"


def test_the_suggestion_follows_the_current_description(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    pid2 = open_page(client, {0: STAMP}, seed=6)
    assert tax_of(client, pid2)[0]["source"] == "suggested"
    cells = {(c["row"], c["col"]): c for c in client.get(f"/api/pages/{pid2}").json()["cells"]}
    client.patch(
        f"/api/pages/{pid2}/cells/{cells[(0, 2)]['id']}", json={"action": "edit", "value": "文具代"}
    )
    assert tax_of(client, pid2)[0]["source"] == "default"


def test_confirming_fixes_the_suggestion_for_the_export_and_reopening_releases_it(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    pid2 = open_page(client, {0: STAMP}, seed=6)
    confirm_with(client, pid2, {})
    mem_id = client.get("/api/tax-memory").json()["entries"][0]["id"]
    # 記憶を変えても、確定済みのページの出力は変わらない(提案の値を固定してある)
    assert (
        client.put(f"/api/tax-memory/{mem_id}", json={"tax_category": "対象外"}).status_code == 200
    )
    assert export_h(client, pid2)[0] == "非課仕入"
    assert tax_of(client, pid2)[0]["source"] == "suggested"
    # 再オープンすると固定が外れ、新しい記憶で提案し直す
    client.post(f"/api/pages/{pid2}/reopen", json={})
    assert tax_of(client, pid2)[0]["category"] == "対象外"


def test_choosing_the_default_over_a_suggestion_forgets_the_memory(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    pid2 = open_page(client, {0: STAMP}, seed=6)
    assert tax_of(client, pid2)[0]["source"] == "suggested"
    set_tax(client, pid2, 0, DEFAULT)  # 提案を断って、既定を選ぶ
    assert tax_of(client, pid2)[0]["source"] == "user"
    confirm_with(client, pid2, {})
    assert client.get("/api/tax-memory").json()["entries"] == []


def test_memory_can_be_edited_deleted_and_cleared(client):
    pid = open_page(client, {0: STAMP, 1: REGISTRY})
    set_tax(client, pid, 0, "非課仕入")
    set_tax(client, pid, 1, "対外仕入")
    confirm_with(client, pid, {})
    entries = client.get("/api/tax-memory").json()["entries"]
    stamp = next(e for e in entries if e["description"] == STAMP)
    r = client.put(f"/api/tax-memory/{stamp['id']}", json={"tax_category": "対象外"})
    assert r.status_code == 200
    assert {e["description"]: e["tax_category"] for e in r.json()["entries"]}[STAMP] == "対象外"
    assert (
        client.put(f"/api/tax-memory/{stamp['id']}", json={"tax_category": "課対仕入"}).status_code
        == 422
    )
    assert client.put("/api/tax-memory/9999", json={"tax_category": "対象外"}).status_code == 404
    r = client.delete(f"/api/tax-memory/{stamp['id']}")
    assert [e["description"] for e in r.json()["entries"]] == [REGISTRY]
    assert client.delete(f"/api/tax-memory/{stamp['id']}").status_code == 404
    assert client.post("/api/tax-memory/clear").json()["entries"] == []


def test_a_changed_tax_row_is_not_a_duplicate_of_the_earlier_export(client):
    pid = open_page(client, {0: STAMP})
    confirm_with(client, pid, {})
    assert client.post("/api/exports", json={"page_ids": [pid]}).status_code == 200
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["duplicates"] != []
    client.post(f"/api/pages/{pid}/reopen", json={})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert {d["n_rows"] for d in v["duplicates"]} == {3}  # 税区分を変えた 1 行は別の行


def test_a_row_choice_that_became_invalid_is_an_export_error_with_a_link(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "共対仕入込5%")
    confirm_with(client, pid, {})
    s = client.get("/api/settings").json()["settings"]
    s["tax_scheme"] = "simplified"  # 簡易課税は課対だけ → 共対は選べない
    assert client.put("/api/settings", json=s).status_code == 200
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    errs = [e for e in v["errors"] if e["code"] == "tax_category_not_allowed"]
    assert len(errs) == 1 and errs[0]["row"] == 0 and errs[0]["cell_id"] is not None
    assert client.post("/api/exports", json={"page_ids": [pid]}).status_code == 409


def test_export_validation_warns_about_rows_exported_with_a_suggested_tax_category(client):
    pid = open_page(client, {0: STAMP})
    set_tax(client, pid, 0, "非課仕入")
    confirm_with(client, pid, {})
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert [
        w for w in v["warnings"] if w["code"] == "tax_suggested"
    ] == []  # 自分で選んだ行は注意しない
    pid2 = open_page(client, {0: STAMP}, seed=6)
    suggested = [r for r, x in tax_of(client, pid2).items() if x["source"] == "suggested"]
    assert 0 in suggested
    confirm_with(client, pid2, {})
    v = client.post("/api/exports/validate", json={"page_ids": [pid2]}).json()
    warns = [w for w in v["warnings"] if w["code"] == "tax_suggested"]
    assert sorted(w["row"] for w in warns) == sorted(suggested)
    assert all(w["cell_id"] is not None and w["page_id"] == pid2 for w in warns)  # 行へのリンク
    assert v["errors"] == []
    assert client.post("/api/exports", json={"page_ids": [pid2]}).status_code == 200  # 止めない
