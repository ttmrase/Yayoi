"""アプリ層のテストで共有する補助(合成ジョブの確認・確定)。"""

from __future__ import annotations


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
        if c["review"] and "dash_digit_ambiguity" not in c["reasons"]:
            client.patch(f"/api/pages/{page_id}/cells/{c['id']}", json={"action": "confirm"})
    for c in client.get(f"/api/pages/{page_id}").json()["cells"]:
        if "dash_digit_ambiguity" in c["reasons"]:
            client.patch(
                f"/api/pages/{page_id}/cells/{c['id']}",
                json={"action": "candidate", "candidate_index": 0},
            )
            client.patch(f"/api/pages/{page_id}/cells/{c['id']}", json={"action": "confirm"})
    r = client.post(f"/api/pages/{page_id}/confirm", json={})
    assert r.status_code == 200, r.text
