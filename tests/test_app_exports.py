"""出力(検証・行単位の重複警告・履歴・ファイル)のテスト。仕様は docs/yayoi_export.md。合成した認識結果だけを使う。"""

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
    "wizard_done": True,
    "payments": [
        {"name": a, "credit_account": a, "credit_sub": "A銀行" if a == "普通預金" else ""}
        for a in ACCOUNTS
    ],
    "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
    "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入込10%"}],
}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
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


def codes(r, key="issues"):
    return {(i["code"], i["page_id"]) for i in r.json()[key]}


def test_validate_and_export_a_confirmed_page_in_the_yayoi_import_format(client):
    pid = confirmed_page(client)
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert (
        v["errors"] == []
        and v["n_rows"] == 4
        and v["duplicates"] == []
        and v["format"] == "yayoi_import"
    )
    assert any("マッチングリスト" in n for n in v["notes"]) and any(
        "Excel" in n for n in v["notes"]
    )
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["n_rows"] == 4 and e["n_pages"] == 1 and len(e["files"]) == 1
    data = client.get(f"/api/exports/{e['id']}/file").content
    assert hashlib.sha256(data).hexdigest() == e["files"][0]["sha256"]
    lines = data.decode("cp932").split("\r\n")
    assert len(lines) == 5 and lines[-1] == ""  # ヘッダーなし、4 仕訳 + 末尾の CRLF
    first = lines[0].split(",")
    assert len(first) == 25 and first[0] == "2000" and first[3].startswith("2026/")
    assert first[4] == "旅費交通費" and first[7] == "課対仕入込10%" and first[10] in ACCOUNTS
    assert (
        first[8] == first[14]
        and int(first[8]) > 0
        and first[13] == "対象外"
        and first[19] == "0"
        and first[24] == "no"
    )
    assert all(len(ln.split(",")) == 25 for ln in lines[:-1])


def test_smart_torihiki_files_are_split_per_payment_and_have_four_columns(client):
    pid = confirmed_page(client)
    e = client.post("/api/exports", json={"page_ids": [pid], "format": "smart_torihiki"}).json()
    assert e["format"] == "smart_torihiki" and sum(f["n_rows"] for f in e["files"]) == 4
    pays = {f["payment"] for f in e["files"]}
    assert pays <= set(ACCOUNTS) and len(pays) >= 1
    for f in e["files"]:
        assert f["payment"] in f["name"]  # ファイル名に支払手段の名前
        data = client.get(f"/api/exports/{e['id']}/files/{f['idx']}").content
        for ln in data.decode("cp932").split("\r\n")[:-1]:
            cols = ln.split(",")
            assert (
                len(cols) == 4
                and cols[0].startswith("2026/")
                and cols[1] == ""
                and int(cols[2]) > 0
            )
    v = client.post(
        "/api/exports/validate", json={"page_ids": [pid], "format": "smart_torihiki"}
    ).json()
    assert any(w["code"] == "title_lost" for w in v["warnings"]) and v["errors"] == []


def test_export_requires_a_complete_valid_page(client):
    pid = confirmed_page(client, title="")
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 409 and r.json()["error"] == "validation_failed"
    assert ("title_missing", pid) in codes(r)
    client.post(f"/api/pages/{pid}/reopen")
    client.put(f"/api/pages/{pid}/title", json={"title": "未設定の科目"})
    confirm_with(client, pid, {})
    assert ("title_unmapped", pid) in codes(
        client.post("/api/exports/validate", json={"page_ids": [pid]})
    )


def test_the_debit_tax_category_is_required_for_taxable_businesses_only(client):
    s = {**SETTINGS, "default_tax": []}
    pid = confirmed_page(client, settings=s)
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert "tax_category_missing" in {i["code"] for i in v["errors"]}
    assert "debit_account_without_tax_category" in v["missing_settings"]
    assert client.post("/api/exports", json={"page_ids": [pid]}).status_code == 409
    # 免税事業者はすべて「対象外」(税区分の設定は要らない)
    client.put("/api/settings", json={**s, "taxpayer": "exempt"})
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["errors"] == [] and v["missing_settings"] == []
    e = client.post("/api/exports", json={"page_ids": [pid]}).json()
    first = (
        client.get(f"/api/exports/{e['id']}/file")
        .content.decode("cp932")
        .split("\r\n")[0]
        .split(",")
    )
    assert first[7] == "対象外"


EXCLUSIVE = {**SETTINGS, "tax_method": "exclusive"}


def _first_rows(client, export_id):
    text = client.get(f"/api/exports/{export_id}/file").content.decode("cp932")
    return [line.split(",") for line in text.split("\r\n") if line]


def test_tax_exclusive_inner_tax_fills_the_debit_tax_amount(client):
    """税抜経理・内税: J 列(借方税金額)= 金額 × 税率 ÷ (100 + 税率)(端数処理は設定。既定は切り捨て)。"""
    settings = {
        **EXCLUSIVE,
        "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入内10%"}],
    }
    pid = confirmed_page(client, settings=settings)
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["errors"] == [] and v["warnings"] == []
    rec = client.post("/api/exports", json={"page_ids": [pid]}).json()
    rows = _first_rows(client, rec["id"])
    assert len(rows) == 4 and all(len(r) == 25 for r in rows)
    for r in rows:
        amount = int(r[8])
        assert r[7] == "課対仕入内10%" and r[9] == str(amount * 10 // 110) and r[14] == r[8]


def test_tax_exclusive_rounding_modes(client):
    base = {
        **EXCLUSIVE,
        "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入内10%"}],
    }
    seen = {}
    for mode in ("floor", "round", "ceil"):
        pid = confirmed_page(client, settings={**base, "tax_rounding": mode})
        rec = client.post(
            "/api/exports", json={"page_ids": [pid], "accept_duplicates": True}
        ).json()
        seen[mode] = [(int(r[8]), int(r[9])) for r in _first_rows(client, rec["id"])]
    for (amt, fl), (_, rd), (_, ce) in zip(seen["floor"], seen["round"], seen["ceil"], strict=True):
        assert fl == amt * 10 // 110
        assert rd == (2 * amt * 10 + 110) // 220
        assert ce == -(-amt * 10 // 110)


def test_tax_exclusive_outer_tax_and_separate_notation_are_manual_with_a_warning(client):
    for cat in ("課対仕入外10%", "課対仕入別軽減8%"):
        settings = {**EXCLUSIVE, "default_tax": [{"account": "旅費交通費", "tax_category": cat}]}
        pid = confirmed_page(client, settings=settings)
        v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
        assert v["errors"] == [] and [w["code"] for w in v["warnings"]] == ["tax_amount_manual"]
        rec = client.post("/api/exports", json={"page_ids": [pid]}).json()  # 注意は出力を止めない
        assert all(r[9] == "" for r in _first_rows(client, rec["id"]))  # 税額は空(手入力の扱い)


def test_tax_exclusive_non_taxable_categories_have_no_tax_amount_and_no_warning(client):
    for cat in ("非課仕入", "対外仕入", "対象外"):
        settings = {**EXCLUSIVE, "default_tax": [{"account": "旅費交通費", "tax_category": cat}]}
        pid = confirmed_page(client, settings=settings)
        v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
        assert v["errors"] == [] and v["warnings"] == []
        rec = client.post("/api/exports", json={"page_ids": [pid]}).json()
        assert all(r[7] == cat and r[9] == "" for r in _first_rows(client, rec["id"]))


def test_tax_inclusive_never_fills_the_debit_tax_amount(client):
    pid = confirmed_page(client)
    rec = client.post("/api/exports", json={"page_ids": [pid]}).json()
    assert all(r[9] == "" for r in _first_rows(client, rec["id"]))


def test_tax_categories_without_a_rate_are_refused_in_settings(client):
    bad = {**SETTINGS, "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入"}]}
    r = client.put("/api/settings", json=bad)
    assert (
        r.status_code == 422
        and {"path": "default_tax[0].tax_category", "code": "tax_rate_missing"}
        in r.json()["problems"]
    )
    full = {
        **SETTINGS,
        "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入込10％"}],
    }
    r = client.put("/api/settings", json=full)
    assert {"path": "default_tax[0].tax_category", "code": "tax_percent_fullwidth"} in r.json()[
        "problems"
    ]
    assert (
        client.put(
            "/api/settings",
            json={**SETTINGS, "default_tax": [{"account": "旅費交通費", "tax_category": "対象外"}]},
        ).status_code
        == 200
    )


def test_account_names_longer_than_24_bytes_are_refused_in_settings(client):
    s = {
        **SETTINGS,
        "titles": [{"name": "長い科目", "debit_account": "あ" * 13}],
    }  # 全角 13 文字 = 26 バイト
    r = client.put("/api/settings", json=s)
    assert (
        r.status_code == 422
        and {"path": "titles[0].debit_account", "code": "too_long_bytes"} in r.json()["problems"]
    )


def test_unmapped_account_links_back_to_the_cell(client):
    pid = confirmed_page(
        client, settings={**SETTINGS, "payments": [{"name": "現金", "credit_account": "現金"}]}
    )
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    bad = [i for i in v["issues"] if i["code"] == "account_unmapped"]
    cells = {c["id"]: c for c in client.get(f"/api/pages/{pid}").json()["cells"]}
    assert bad and all(cells[i["cell_id"]]["col"] == 1 for i in bad)
    assert all(
        set(i) == {"code", "label", "severity", "page_id", "cell_id", "row"} for i in v["issues"]
    )


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


# ---- 行単位の重複の警告


def test_exporting_the_same_rows_twice_needs_explicit_consent(client):
    pid = confirmed_page(client)
    first = client.post("/api/exports", json={"page_ids": [pid]}).json()
    cand = client.get("/api/exports").json()["candidates"]
    assert [(c["n_rows"], c["n_exported"], c["exported"]) for c in cand] == [(4, 4, True)]
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["duplicates"] == [{"page_id": pid, "n_rows": 4, "export_ids": [first["id"]]}]
    r = client.post("/api/exports", json={"page_ids": [pid]})
    assert r.status_code == 409 and r.json()["error"] == "duplicate_pages"
    # 別の形式でも、同じ行の二重の取り込みになるので警告する
    r = client.post("/api/exports", json={"page_ids": [pid], "format": "smart_torihiki"})
    assert r.status_code == 409 and r.json()["error"] == "duplicate_pages"
    ok = client.post("/api/exports", json={"page_ids": [pid], "accept_duplicates": True})
    assert ok.status_code == 200 and ok.json()["id"] != first["id"]
    assert [e["id"] for e in client.get("/api/exports").json()["exports"]] == [
        ok.json()["id"],
        first["id"],
    ]


def test_only_the_corrected_row_is_new_after_reopening_an_exported_page(client):
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
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["duplicates"][0]["n_rows"] == 3  # 直した 1 行は新しい行、残り 3 行は出力済み
    cand = client.get("/api/exports").json()["candidates"][0]
    assert (cand["n_rows"], cand["n_exported"], cand["exported"]) == (4, 3, False)


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
        "files",
    }
    con = client.app.state.db
    snap = con.execute("SELECT settings_snapshot FROM exports WHERE id = ?", (e["id"],)).fetchone()[
        0
    ]
    assert "旅費交通費" in snap  # 設定のスナップショット(確定値ではない)
    rows = con.execute("SELECT * FROM export_rows WHERE export_id = ?", (e["id"],)).fetchall()
    assert len(rows) == 4 and all(
        len(r["content_sha256"]) == 64 for r in rows
    )  # 行ごとのハッシュだけ
    assert client.get("/api/exports/999/file").status_code == 404
    assert client.get(f"/api/exports/{e['id']}/files/5").status_code == 404


def test_export_file_download_is_confined_to_the_exports_directory(client):
    pid = confirmed_page(client)
    e = client.post("/api/exports", json={"page_ids": [pid]}).json()
    con = client.app.state.db
    con.execute("UPDATE export_files SET name = '../../a.sqlite3' WHERE export_id = ?", (e["id"],))
    assert client.get(f"/api/exports/{e['id']}/file").status_code == 404
