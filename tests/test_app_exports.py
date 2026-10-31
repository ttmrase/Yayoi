"""出力(検証・重複の警告・履歴・ファイル)のテスト。形式は暫定。合成した認識結果だけを使う。"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app.fake import ACCOUNTS
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app

PORT = 8127
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}

SETTINGS = {
    "payments": [
        {"name": a, "credit_account": a, "credit_sub": "補助" if a == "普通預金" else ""}
        for a in ACCOUNTS
    ],
    "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
    "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入"}],
}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        c.root_ = tmp_path
        yield c


def confirmed_page(client, title="旅費交通費", settings=SETTINGS, **kw):
    assert client.put("/api/settings", json=settings).status_code == 200
    job = client.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": 5,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
            "fill_rows": 4,
            **kw,
        },
    ).json()
    pid = job["pages"][0]["id"]
    if title:
        assert client.put(f"/api/pages/{pid}/title", json={"title": title}).status_code == 200
    confirm_with(client, pid, {})
    return pid


def issues_of(r):
    return {(i["code"], i["page_id"]) for i in r.json()["issues"]}


def test_validate_and_export_a_confirmed_page(client):
    pid = confirmed_page(client)
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["issues"] == [] and v["n_rows"] == 4 and v["duplicates"] == [] and v["provisional"]
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["n_rows"] == 4 and e["n_pages"] == 1 and "仮の形式" in e["format_label"]
    data = client.get(f"/api/exports/{e['id']}/file").content
    assert hashlib.sha256(data).hexdigest() == e["file_sha256"]
    text = data.decode("cp932")
    lines = text.split("\r\n")
    assert lines[0] == "日付,借方科目,借方補助科目,貸方科目,貸方補助科目,摘要,金額,税区分"
    assert len(lines) == 4 + 2 and lines[-1] == ""
    first = lines[1].split(",")
    assert first[0].startswith("2026/") and first[1] == "旅費交通費" and first[7] == "課対仕入"
    assert first[3] in ACCOUNTS and int(first[6]) > 0


def test_export_requires_a_complete_valid_page(client):
    pid = confirmed_page(client, title="")
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 409 and r.json()["error"] == "validation_failed"
    assert ("title_missing", pid) in issues_of(r)
    client.post(f"/api/pages/{pid}/reopen")
    client.put(f"/api/pages/{pid}/title", json={"title": "未設定の科目"})
    confirm_with(client, pid, {})
    assert ("title_unmapped", pid) in issues_of(
        client.post("/api/exports/validate", json={"page_ids": [pid]})
    )


def test_unmapped_account_links_back_to_the_cell(client):
    pid = confirmed_page(
        client, settings={**SETTINGS, "payments": [{"name": "現金", "credit_account": "現金"}]}
    )
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    bad = [i for i in v["issues"] if i["code"] == "account_unmapped"]
    cells = {c["id"]: c for c in client.get(f"/api/pages/{pid}").json()["cells"]}
    # 設定にない支払手段が書かれた行が、セルへのリンクつきで指摘される(値は指摘に含まれない)
    assert bad and all(cells[i["cell_id"]]["col"] == 1 for i in bad)
    assert all(set(i) == {"code", "label", "page_id", "cell_id", "row"} for i in v["issues"])


def test_unconfirmed_pages_and_missing_settings_are_reported(client):
    assert client.put("/api/settings", json={}).status_code == 200
    job = client.post("/api/dev/jobs", json={"pages": 1, "seed": 6}).json()
    pid = job["pages"][0]["id"]
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert ("page_not_confirmed", pid) in {(i["code"], i["page_id"]) for i in v["issues"]}
    assert set(v["missing_settings"]) == {"no_payments", "no_titles"}
    assert client.post("/api/exports", json={"page_ids": [pid]}).status_code == 409
    assert client.post("/api/exports/validate", json={"page_ids": [9999]}).status_code == 404
    assert client.post("/api/exports/validate", json={"page_ids": []}).status_code == 422
    assert (
        client.post("/api/exports", json={"page_ids": [pid], "format": "nope"}).status_code == 422
    )


def test_exporting_the_same_page_twice_needs_explicit_consent(client):
    pid = confirmed_page(client)
    first = client.post("/api/exports", json={"page_ids": [pid]}).json()
    assert [c["exported"] for c in client.get("/api/exports").json()["candidates"]] == [True]
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["duplicates"] == [{"page_id": pid, "export_ids": [first["id"]]}]
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 409 and r.json()["error"] == "duplicate_pages"
    ok = client.post("/api/exports", json={"page_ids": [pid], "accept_duplicates": True})
    assert ok.status_code == 200 and ok.json()["id"] != first["id"]
    assert [e["id"] for e in client.get("/api/exports").json()["exports"]] == [
        ok.json()["id"],
        first["id"],
    ]


def test_a_page_corrected_after_export_is_no_longer_a_duplicate(client):
    pid = confirmed_page(client)
    client.post("/api/exports", json={"page_ids": [pid]})
    client.post(f"/api/pages/{pid}/reopen")
    cell = next(
        c
        for c in client.get(f"/api/pages/{pid}").json()["cells"]
        if c["column"] == "amount" and c["kind"] != "blank"
    )
    client.patch(f"/api/pages/{pid}/cells/{cell['id']}", json={"action": "edit", "value": "99999"})
    confirm_with(client, pid, {})
    assert client.post("/api/exports/validate", json={"page_ids": [pid]}).json()["duplicates"] == []


def test_history_keeps_no_cell_values_and_the_settings_snapshot(client):
    pid = confirmed_page(client)
    e = client.post("/api/exports", json={"page_ids": [pid]}).json()
    assert set(e) == {
        "id",
        "format",
        "format_label",
        "created_at",
        "n_pages",
        "n_rows",
        "file_sha256",
        "file_name",
    }
    con = client.app.state.db
    snap = con.execute("SELECT settings_snapshot FROM exports WHERE id = ?", (e["id"],)).fetchone()[
        0
    ]
    assert "旅費交通費" in snap  # 設定のスナップショット(確定値ではない)
    assert client.get("/api/exports/999/file").status_code == 404


def test_export_file_download_is_confined_to_the_exports_directory(client):
    pid = confirmed_page(client)
    e = client.post("/api/exports", json={"page_ids": [pid]}).json()
    con = client.app.state.db
    con.execute("UPDATE exports SET file_name = '../../a.sqlite3' WHERE id = ?", (e["id"],))
    assert client.get(f"/api/exports/{e['id']}/file").status_code == 404
