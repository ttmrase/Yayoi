"""確認画面の API(状態遷移・検証・元に戻す・確定の条件)のテスト。合成した認識結果(`fake.py`)だけを使う。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import look_and_confirm
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.validate import validate_value

PORT = 8124
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_client(tmp_path, dev=True):
    app = create_app(tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=dev)
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH)


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def new_job(client, **kw):
    r = client.post("/api/dev/jobs", json={"pages": 2, "seed": 1, **kw})
    assert r.status_code == 200
    return r.json()


def cells_of(client, page_id):
    return client.get(f"/api/pages/{page_id}").json()["cells"]


def pass_all_review(client, page_id):
    for c in cells_of(client, page_id):
        if c["review"] and "dash_digit_ambiguity" not in c["reasons"]:
            r = look_and_confirm(client, page_id, c["id"])
            assert r.status_code == 200, r.text


def test_dev_endpoint_exists_only_in_dev_mode(tmp_path):
    with make_client(tmp_path, dev=False) as c:
        assert c.post("/api/dev/jobs", json={}).status_code in (404, 405)


def test_page_has_cells_summary_and_neighbours(client):
    job = new_job(client)
    p1, p2 = (p["id"] for p in job["pages"])
    page = client.get(f"/api/pages/{p1}").json()
    assert len(page["cells"]) == 29 * 4 and page["n_pages"] == 2
    assert page["prev_page_id"] is None and page["next_page_id"] == p2
    s = page["summary"]
    assert s["review_total"] > 0 and s["review_passed"] == 0 and not s["can_confirm"]
    assert any(c["expansion"] and c["expansion"]["kind"] == "ditto" for c in page["cells"])
    jobs = client.get("/api/jobs").json()
    assert jobs[0]["n_pages"] == 2 and jobs[0]["n_confirmed"] == 0


def test_cannot_confirm_while_review_cells_are_unpassed_or_ambiguity_is_open(client):
    page_id = new_job(client)["pages"][0]["id"]
    r = client.post(f"/api/pages/{page_id}/confirm", json={})
    assert r.status_code == 409 and r.json()["error"] == "cannot_confirm"
    assert r.json()["summary"]["unpassed"]  # サーバでも検証している(UI の無効化に頼らない)
    pass_all_review(client, page_id)
    s = client.get(f"/api/pages/{page_id}").json()["summary"]
    assert s["ambiguous"] and not s["can_confirm"]  # 10 倍違いが未選択
    amb = s["ambiguous"][0]
    # Enter(confirm)では進めない
    r = client.patch(f"/api/pages/{page_id}/cells/{amb}", json={"action": "confirm"})
    assert r.status_code == 409 and r.json()["error"] == "ambiguity_unresolved"
    # 通過だけでも解決にならない
    client.patch(f"/api/pages/{page_id}/cells/{amb}", json={"action": "pass"})
    assert client.get(f"/api/pages/{page_id}").json()["summary"]["ambiguous"] == [amb]
    # 候補(Alt+数字)で選ぶと解決
    r = client.patch(
        f"/api/pages/{page_id}/cells/{amb}", json={"action": "candidate", "candidate_index": 1}
    )
    assert r.status_code == 200 and not r.json()["summary"]["ambiguous"]
    cell = next(c for c in r.json()["cells"] if c["id"] == amb)
    assert cell["value"] == "500" and cell["edit_kind"] == "candidate"


def test_full_flow_confirms_page_and_job(client):
    job = new_job(client)
    for i, p in enumerate(job["pages"]):
        pid = p["id"]
        pass_all_review(client, pid)
        amb = client.get(f"/api/pages/{pid}").json()["summary"]["ambiguous"]
        for a in amb:
            client.patch(
                f"/api/pages/{pid}/cells/{a}", json={"action": "candidate", "candidate_index": 1}
            )
        # 誤りを正解に直す(上書き)
        for c in cells_of(client, pid):
            t = p["truth"][str(c["id"])]
            if c["value"] != t:
                client.patch(
                    f"/api/pages/{pid}/cells/{c['id']}", json={"action": "edit", "value": t}
                )
        # 参照元を直すと、それを参照する〃は再展開されて要確認に戻る → もう一度通す
        for cid in client.get(f"/api/pages/{pid}").json()["summary"]["unpassed"]:
            look_and_confirm(client, pid, cid)
        r = client.post(f"/api/pages/{pid}/confirm", json={"review_seconds": 12.5})
        assert r.status_code == 200 and r.json()["status"] == "confirmed", r.text
        got = {str(c["id"]): c["value"] for c in r.json()["cells"]}
        assert got == p["truth"]  # 確定値が正解になった
        assert (
            client.post(f"/api/pages/{pid}/confirm", json={}).status_code == 409
        )  # 確定済みは変更不可
        assert (
            client.patch(f"/api/pages/{pid}/cells/{c['id']}", json={"action": "pass"}).status_code
            == 409
        )
        assert client.get(f"/api/jobs/{job['job_id']}").json()["status"] == (
            "done" if i == 1 else "ready"
        )
    # 再オープンできる
    pid = job["pages"][0]["id"]
    assert client.post(f"/api/pages/{pid}/reopen").json()["status"] == "reviewing"


def test_edit_undo_redo_and_new_edit_clears_redo(client):
    page_id = new_job(client)["pages"][0]["id"]
    c = next(
        c for c in cells_of(client, page_id) if c["column"] == "amount" and c["kind"] == "text"
    )
    url = f"/api/pages/{page_id}/cells/{c['id']}"
    orig = c["value"]
    r = client.patch(url, json={"action": "edit", "value": "777"}).json()
    assert next(x for x in r["cells"] if x["id"] == c["id"])["value"] == "777" and r["can_undo"]
    u = client.post(f"/api/pages/{page_id}/undo").json()
    assert (
        u["cell_id"] == c["id"]
        and next(x for x in u["page"]["cells"] if x["id"] == c["id"])["value"] == orig
    )
    assert u["page"]["can_redo"]
    rd = client.post(f"/api/pages/{page_id}/redo").json()
    assert next(x for x in rd["page"]["cells"] if x["id"] == c["id"])["value"] == "777"
    client.post(f"/api/pages/{page_id}/undo")
    client.patch(url, json={"action": "edit", "value": "888"})  # 新しい編集でやり直しは消える
    assert not client.get(f"/api/pages/{page_id}").json()["can_redo"]
    # 何も無いときの取り消しは何もしない
    for _ in range(5):
        client.post(f"/api/pages/{page_id}/undo")
    assert client.post(f"/api/pages/{page_id}/undo").json()["cell_id"] is None


def test_editing_a_ditto_cell_breaks_the_link_and_editing_the_source_reexpands(client):
    page_id = new_job(client, error_rate=0.0)["pages"][0]["id"]
    cells = cells_of(client, page_id)
    ditto = next(c for c in cells if c["expansion"] and c["expansion"]["kind"] == "ditto")
    src_row, src_col = ditto["expansion"]["source"]
    src = next(c for c in cells if c["row"] == src_row and c["col"] == src_col)
    # 参照元を書き換えると、〃は再展開され、要確認・未通過に戻る
    client.patch(f"/api/pages/{page_id}/cells/{ditto['id']}", json={"action": "confirm"})
    r = client.patch(
        f"/api/pages/{page_id}/cells/{src['id']}", json={"action": "edit", "value": "変更後"}
    ).json()
    d = next(c for c in r["cells"] if c["id"] == ditto["id"])
    assert (
        d["value"] == "変更後"
        and d["review"]
        and not d["passed"]
        and "ditto_source_changed" in d["reasons"]
    )
    # 〃のセルを自分で書き換えたら、参照元との結びつきが外れる
    r = client.patch(
        f"/api/pages/{page_id}/cells/{ditto['id']}", json={"action": "edit", "value": "別の値"}
    ).json()
    d = next(c for c in r["cells"] if c["id"] == ditto["id"])
    assert d["edit_kind"] == "expanded_edit" and d["expansion"] is None
    r = client.patch(
        f"/api/pages/{page_id}/cells/{src['id']}", json={"action": "edit", "value": "さらに変更"}
    ).json()
    assert next(c for c in r["cells"] if c["id"] == ditto["id"])["value"] == "別の値"


def test_validation_blocks_confirmation(client):
    page_id = new_job(client, review_rate=0.0, error_rate=0.0, ambiguous_rows=[])["pages"][0]["id"]
    cells = cells_of(client, page_id)
    assert client.get(f"/api/pages/{page_id}").json()["summary"][
        "can_confirm"
    ]  # 要確認が無く誤りも無ければ確定できる
    amount = next(c for c in cells if c["column"] == "amount" and c["kind"] == "text")
    desc = next(c for c in cells if c["column"] == "description" and c["kind"] == "text")
    date = next(c for c in cells if c["column"] == "date" and c["kind"] == "text")
    for cell, bad, code in (
        (amount, "abc", "amount_grammar"),
        (desc, "😀", "cp932_unencodable"),
        (desc, "あ" * 40, "too_long"),
        (date, "13.45", "date_grammar"),
        (date, "8.21", "year_missing"),
    ):
        r = client.patch(
            f"/api/pages/{page_id}/cells/{cell['id']}", json={"action": "edit", "value": bad}
        ).json()
        assert code in {e["code"] for e in r["summary"]["errors"]}, code
        assert not r["summary"]["can_confirm"]
        assert client.post(f"/api/pages/{page_id}/confirm", json={}).status_code == 409
        client.post(f"/api/pages/{page_id}/undo")


def test_validate_value_rules():
    assert validate_value("amount", "1200") == [] and validate_value("amount", "0") == [
        "amount_grammar"
    ]
    assert validate_value("date", "R8.9.1") == [] and validate_value("date", "9.1") == [
        "year_missing"
    ]
    assert (
        validate_value("account", "") == ["account_empty"]
        and validate_value("account", "", False) == []
    )
    assert (
        validate_value("description", "") == [] and validate_value("description", "郵便料金") == []
    )


def test_blank_cells_can_be_filled_in(client):
    # 認識が「空欄」と読んだ(実際は書いてある)セルを、利用者が埋められる
    page_id = new_job(client)["pages"][0]["id"]
    blank = next(
        c
        for c in cells_of(client, page_id)
        if c["kind"] == "blank" and c["column"] == "description"
    )
    r = client.patch(
        f"/api/pages/{page_id}/cells/{blank['id']}",
        json={"action": "edit", "value": "追記した摘要"},
    ).json()
    assert next(c for c in r["cells"] if c["id"] == blank["id"])["value"] == "追記した摘要"


def test_cell_image_requires_auth_and_belongs_to_the_page(client, tmp_path):
    job = new_job(client)
    p1, p2 = (p["id"] for p in job["pages"])
    cell = cells_of(client, p1)[0]
    r = client.get(f"/api/pages/{p1}/cells/{cell['id']}/image")
    assert (
        r.status_code == 200
        and r.headers["content-type"] == "image/png"
        and r.content[:4] == b"\x89PNG"
    )
    assert (
        client.get(f"/api/pages/{p2}/cells/{cell['id']}/image").status_code == 404
    )  # 別ページのセルは返さない
    assert client.get(f"/api/pages/{p1}/cells/999999/image").status_code == 404
    with make_client(tmp_path / "other", dev=True) as other:
        assert other.get(f"/api/pages/{p1}/cells/{cell['id']}/image").status_code == 404


def test_layout_approval_clears_only_layout_reasons(client):
    page_id = new_job(client)["pages"][0]["id"]
    page = client.get(f"/api/pages/{page_id}").json()
    r = client.post(f"/api/pages/{page_id}/layout/approve").json()
    assert r["layout_approved"] and r["layout_status"] == "ok"
    # 認識由来の理由(低信頼など)は残る
    assert sum(c["review"] for c in r["cells"]) == sum(c["review"] for c in page["cells"])


def test_position_is_saved_for_resume(client):
    job = new_job(client)
    p2 = job["pages"][1]["id"]
    assert client.put(
        f"/api/jobs/{job['job_id']}/position", json={"page_id": p2, "cell": 17}
    ).json() == {"ok": True}
    j = client.get(f"/api/jobs/{job['job_id']}").json()
    assert j["last_page_id"] == p2 and j["last_cell"] == 17


def test_unknown_action_and_missing_resources(client):
    page_id = new_job(client)["pages"][0]["id"]
    cell = cells_of(client, page_id)[0]
    assert (
        client.patch(
            f"/api/pages/{page_id}/cells/{cell['id']}", json={"action": "explode"}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/api/pages/{page_id}/cells/{cell['id']}", json={"action": "edit"}
        ).status_code
        == 422
    )
    assert client.get("/api/pages/999999").status_code == 404
    assert client.get("/api/jobs/999999").status_code == 404


def test_enter_spam_does_not_pass_review_cells(client, monkeypatch):
    """連打(キーリピート)で要確認を飛ばせない: 値を直さない Enter は、前の通過から一定時間たたないと「通過」に数えない。"""
    from yayoi_ocr.app import store

    job = new_job(client, error_rate=0, review_rate=1, ambiguous_rows=[])
    page_id = job["pages"][0]["id"]
    review = [c for c in cells_of(client, page_id) if c["review"]]
    clock = [1000.0]
    monkeypatch.setattr(store.time, "monotonic", lambda: clock[0])
    store._LAST_LOOK.clear()
    passed = []
    for c in review[:4]:  # 10 ms 間隔の連打
        clock[0] += 0.01
        r = client.patch(
            f"/api/pages/{page_id}/cells/{c['id']}", json={"action": "confirm", "value": c["value"]}
        )
        passed.append(next(x for x in r.json()["cells"] if x["id"] == c["id"])["passed"])
    assert passed == [True, False, False, False]  # 最初の 1 つだけ(その後は間隔が足りない)
    # 十分な間隔(人が画像を見る時間)があれば通る
    for c in review[4:8]:
        clock[0] += 0.3
        r = client.patch(
            f"/api/pages/{page_id}/cells/{c['id']}", json={"action": "confirm", "value": c["value"]}
        )
        assert next(x for x in r.json()["cells"] if x["id"] == c["id"])["passed"]
    # 飛ばされたセルが残るので、ページは確定できない
    assert client.post(f"/api/pages/{page_id}/confirm", json={}).status_code == 409


def test_explicit_input_is_not_rate_limited(client, monkeypatch):
    from yayoi_ocr.app import store

    job = new_job(client, error_rate=0, review_rate=1, ambiguous_rows=[])
    page_id = job["pages"][0]["id"]
    review = [c for c in cells_of(client, page_id) if c["review"] and c["kind"] == "text"][:3]
    monkeypatch.setattr(store.time, "monotonic", lambda: 1000.0)  # 時間が止まっていても
    store._LAST_LOOK.clear()
    for c in review:  # 値を入力して確認した(値を直した)セルは、すぐに通る
        r = client.patch(
            f"/api/pages/{page_id}/cells/{c['id']}",
            json={"action": "confirm", "value": c["value"] + "0"},
        )
        assert next(x for x in r.json()["cells"] if x["id"] == c["id"])["passed"]
