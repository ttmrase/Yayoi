"""アプリ層のテストで共有する補助(合成ジョブの確認・確定)。"""

from __future__ import annotations

from yayoi_ocr.app.store import CHOICE_REASONS


def needs_choice(cell: dict) -> bool:
    return bool(CHOICE_REASONS & set(cell["reasons"]))


def look_and_confirm(client, page_id: int, cell_id: int, **extra):
    """画面と同じ順: 250 ms 以上見て「通過」(pass)→ Enter(confirm)。連打だと通過に数えない(store.MIN_LOOK_S)。"""
    client.patch(f"/api/pages/{page_id}/cells/{cell_id}", json={"action": "pass"})
    return client.patch(
        f"/api/pages/{page_id}/cells/{cell_id}", json={"action": "confirm", **extra}
    )


def confirm_with(client, page_id, values: dict[tuple[int, int], str]):
    """指定したセルの値を直して、残りの要確認を確認済みにして確定する(合成ジョブ)。"""
    for c in client.get(f"/api/pages/{page_id}").json()["cells"]:
        v = values.get((c["row"], c["col"]))
        if v is not None:
            assert (
                client.patch(
                    f"/api/pages/{page_id}/cells/{c['id']}", json={"action": "edit", "value": v}
                ).status_code
                == 200
            )
    for c in client.get(f"/api/pages/{page_id}").json()["cells"]:
        if c["review"] and not needs_choice(c):
            look_and_confirm(client, page_id, c["id"])
    for c in client.get(f"/api/pages/{page_id}").json()["cells"]:
        if needs_choice(c):
            client.patch(
                f"/api/pages/{page_id}/cells/{c['id']}",
                json={"action": "candidate", "candidate_index": 0},
            )
            look_and_confirm(client, page_id, c["id"])
    r = client.post(f"/api/pages/{page_id}/confirm", json={})
    assert r.status_code == 200, r.text
