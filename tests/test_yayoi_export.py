"""弥生への出力のゴールデンファイルのテスト(架空データ。docs/yayoi_export.md の仕様)。

バイト列そのもの(cp932・CRLF・25 項目・全行が同じ列数・カンマを含む項目のダブルクォート)を固定する。
弥生が書き出した公開サンプルとの突き合わせは未実施(docs/yayoi_export.md)。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from yayoi_ocr.app import exports
from yayoi_ocr.app.exports import Row, _build_smart, _build_yayoi, _encode, yayoi_fields
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.settings import Settings
from yayoi_ocr.app.store import StoreError


def row(**kw) -> Row:
    base = {
        "page_id": 1,
        "row": 0,
        "date": "2026/11/19",
        "debit": "旅費交通費",
        "debit_sub": "",
        "debit_tax": "課対仕入込10%",
        "credit": "現金",
        "credit_sub": "",
        "payment": "現金",
        "description": "架空の電車代",
        "amount": 3456,
    }
    return Row(**{**base, **kw})


# 25 項目(A〜Y)の期待値(架空データ)
GOLDEN_LINE = (
    "2000,,,2026/11/19,旅費交通費,,,課対仕入込10%,3456,,現金,,,対象外,3456,,架空の電車代,,,0,,,,,no"
)


def test_one_row_is_exactly_the_25_columns_of_the_import_format():
    f = yayoi_fields(row(), Settings())
    assert len(f) == 25
    assert ",".join(f) == GOLDEN_LINE
    # A〜Y の位置(仕様の表のとおり)
    assert (
        f[0] == "2000" and f[3] == "2026/11/19" and f[4] == "旅費交通費" and f[7] == "課対仕入込10%"
    )
    assert (
        f[8] == "3456" and f[9] == "" and f[10] == "現金" and f[13] == "対象外" and f[14] == "3456"
    )
    assert f[16] == "架空の電車代" and f[19] == "0" and f[24] == "no"
    assert [i for i, v in enumerate(f) if v == ""] == [
        1,
        2,
        5,
        6,
        9,
        11,
        12,
        15,
        17,
        18,
        20,
        21,
        22,
        23,
    ]


def test_file_bytes_are_cp932_with_crlf_no_header_and_no_trailing_junk():
    data = _build_yayoi([row(), row(row=1, amount=100, description="2 行目")], Settings())[0].data
    assert data == (
        GOLDEN_LINE
        + "\r\n"
        + GOLDEN_LINE.replace("3456", "100").replace("架空の電車代", "2 行目")
        + "\r\n"
    ).encode("cp932")
    assert data.decode("cp932").count("\r\n") == 2 and b"\n" not in data.replace(b"\r\n", b"")
    assert b"\xef\xbb\xbf" not in data  # UTF-8 の BOM ではない(UTF-8 は不可)
    # cp932 のバイト列そのもの(「旅」= 0x97 0xB7)
    assert "旅".encode("cp932") == b"\x97\xb7" and b"\x97\xb7" in data


def test_items_with_commas_quotes_or_newlines_are_double_quoted_and_empty_items_keep_their_commas():
    f = _build_yayoi([row(description='a,b "c"', credit_sub="A銀行,本店")], Settings())[
        0
    ].data.decode("cp932")
    line = f.split("\r\n")[0]
    assert '"a,b ""c"""' in line and '"A銀行,本店"' in line
    # 引用符を考慮して数えても 25 項目
    import csv
    import io

    assert len(next(csv.reader(io.StringIO(line)))) == 25


def test_a_tax_exempt_business_exports_taxless_categories_only():
    assert yayoi_fields(row(), Settings(taxpayer="exempt"))[7] == "対象外"


def test_description_is_written_as_is_even_if_it_looks_like_a_formula():
    line = _build_yayoi([row(description="=SUM(A1)")], Settings())[0].data.decode("cp932")
    assert ",=SUM(A1)," in line  # ' を付けない(弥生にその文字のまま取り込まれてしまうため)


def test_smart_torihiki_golden_lines_and_file_names_per_payment():
    rows = [
        row(),
        row(row=1, payment="普通預金", amount=200, description="架空の振込"),
        row(row=2, amount=7),
    ]
    files = _build_smart(rows, Settings())
    assert [f.name for f in files] == ["smart_現金_1.csv", "smart_普通預金_1.csv"]
    assert files[
        0
    ].data == "2026/11/19,,3456,架空の電車代\r\n2026/11/19,,7,架空の電車代\r\n".encode("cp932")
    assert files[1].data == "2026/11/19,,200,架空の振込\r\n".encode("cp932")
    assert [f.n_rows for f in files] == [2, 1]
    assert all(
        len(ln.split(",")) == 4 for f in files for ln in f.data.decode("cp932").split("\r\n")[:-1]
    )


def test_smart_torihiki_is_split_below_200kb_with_the_same_column_count_everywhere():
    rows = [row(row=i, description="あ" * 40) for i in range(6000)]  # 1 行 約 100 バイト → 600KB 超
    files = _build_smart(rows, Settings())
    assert len(files) >= 3 and sum(f.n_rows for f in files) == 6000
    assert all(len(f.data) < 200_000 for f in files)
    assert [f.name for f in files] == [f"smart_現金_{i}.csv" for i in range(1, len(files) + 1)]
    cols = {len(ln.split(",")) for f in files for ln in f.data.decode("cp932").split("\r\n")[:-1]}
    assert cols == {4}


def test_file_names_never_contain_characters_windows_forbids():
    assert exports.safe_name('現金/小口:A*B?"C<D>|E') == "現金_小口_A_B__C_D__E"
    assert exports.safe_name("..") == "payment"


# ---- cp932 で表現できない文字(I-4): どの経路でも例外(500)にしない


@pytest.mark.parametrize("bad", ["😀", "𠮷野家", "テスト🈚", "\U0001f4b4"])
def test_unencodable_text_raises_a_user_facing_error_not_an_exception(bad):
    with pytest.raises(StoreError) as e:
        _encode(bad)
    assert e.value.status == 409 and e.value.code == "cp932_unencodable"


@pytest.mark.parametrize("ok", ["髙橋", "﨑", "①㈱℡", "～(全角チルダ)", "〜", "Ⅳ㌔", "ｱｲｳ"])
def test_cp932_dependent_characters_are_encodable(ok):
    assert _encode(ok).decode("cp932") is not None


PORT = 8129
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        yield c


def confirmed_page_with(client, edits: dict[tuple[int, int], str]):
    from _app_helpers import confirm_with
    from yayoi_ocr.app.fake import ACCOUNTS

    client.put(
        "/api/settings",
        json={
            "payments": [{"name": a, "credit_account": a} for a in ACCOUNTS],
            "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
            "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入込10%"}],
        },
    )
    job = client.post(
        "/api/dev/jobs",
        json={
            "pages": 1,
            "seed": 8,
            "error_rate": 0,
            "review_rate": 0,
            "ambiguous_rows": [],
            "fill_rows": 4,
        },
    ).json()
    pid = job["pages"][0]["id"]
    client.put(f"/api/pages/{pid}/title", json={"title": "旅費交通費"})
    confirm_with(client, pid, {})
    con = client.app.state.db
    # 確定後に DB へ直接入れる(画面の検証をすり抜けた値でも、出力の検証が弾くことを確かめる)
    for (r, c), v in edits.items():
        con.execute(
            "UPDATE cells SET confirmed_value = ? WHERE page_id = ? AND row = ? AND col = ?",
            (v, pid, r, c),
        )
    return pid


@pytest.mark.parametrize("bad", ["😀", "𠮷野家"])
@pytest.mark.parametrize("col", [1, 2])  # 科目・摘要
def test_unencodable_cells_block_the_export_and_point_to_the_cell(client, bad, col):
    pid = confirmed_page_with(client, {(1, col): "テスト" + bad})
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    errs = [i for i in v["errors"] if i["code"] == "cp932_unencodable"]
    cell = next(
        c
        for c in client.get(f"/api/pages/{pid}").json()["cells"]
        if c["row"] == 1 and c["col"] == col
    )
    assert [(i["row"], i["cell_id"]) for i in errs] == [(1, cell["id"])]
    for fmt in ("yayoi_import", "smart_torihiki"):
        r = client.post("/api/exports", json={"page_ids": [pid], "format": fmt})
        assert r.status_code == 409 and r.json()["error"] == "validation_failed"  # 500 にならない


def test_formula_like_descriptions_only_warn_and_the_value_is_unchanged(client):
    pid = confirmed_page_with(client, {(0, 2): "=1+1", (1, 2): "-メモ", (2, 2): "@x", (3, 2): "+8"})
    v = client.post("/api/exports/validate", json={"page_ids": [pid]}).json()
    assert v["errors"] == [] and [w["code"] for w in v["warnings"]].count("formula_like") == 4
    assert all(w["cell_id"] is not None and w["row"] is not None for w in v["warnings"])
    e = client.post("/api/exports", json={"page_ids": [pid]}).json()
    text = client.get(f"/api/exports/{e['id']}/file").content.decode("cp932")
    assert (
        ",=1+1," in text and ",-メモ," in text and ",@x," in text and ",+8," in text
    )  # 値はそのまま
