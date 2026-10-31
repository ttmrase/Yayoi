"""Phase 4 基盤 2a: 世代(重み + 確信度)の管理。一覧・使用中の世代・ロールバック・リセット・認識への受け渡し・v9 の移行。値(摘要・金額)は一切使わない。"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app import db, gens, store
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend

PORT = 8131
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "a.sqlite3")
    db.migrate(c)
    return c


def models(tmp_path):
    return tmp_path / "models"


def write_weights(tmp_path, profile: int, number: int, with_confidence: bool = True):
    d = models(tmp_path) / str(profile) / gens.gen_dir_name(number)
    d.mkdir(parents=True)
    for f in gens.WEIGHT_FILES:
        (d / f).write_bytes(b"w")
    if with_confidence:
        (d / gens.CONFIDENCE_FILE).write_text("{}")
    return d


def add_trained(con, tmp_path, number=1, *, adopt=True, meta=None, files=True):
    if files:
        write_weights(tmp_path, 1, number)
    return gens.register(
        con,
        models(tmp_path),
        profile_id=1,
        number=number,
        parent_id=gens.active(con)["id"],
        meta=meta or {"n_train_pages": 7, "n_eval_pages": 2},
        adopt=adopt,
        reject_reason=None if adopt else "accuracy_dropped",
    )


# ---- スキーマ(v9)


def test_fresh_db_has_one_active_shipped_generation(con):
    rows = con.execute("SELECT * FROM generations").fetchall()
    assert len(rows) == 1
    g = rows[0]
    assert (g["kind"], g["number"], g["status"], g["dir"]) == ("shipped", 0, "active", None)
    assert con.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(training_runs)")}
    assert {"trigger", "status", "n_new_cells", "cutoff", "generation_id", "reason"} <= cols


def test_migration_v9_marks_old_generation_shipped_and_backfills_pages(tmp_path):
    c = sqlite3.connect(tmp_path / "old.sqlite3", isolation_level=None)
    c.row_factory = sqlite3.Row
    for i in range(8):  # v1〜v8 まで(Phase 5 までの DB)
        c.executescript("BEGIN;" + db.MIGRATIONS[i] + f"PRAGMA user_version = {i + 1};COMMIT;")
    c.execute("INSERT INTO profiles (id, name, created_at) VALUES (1, '共通', 1.0)")
    c.execute("INSERT INTO generations (profile_id, created_at, status) VALUES (1, 1.0, 'active')")
    c.execute(
        "INSERT INTO jobs (name, profile_id, status, created_at) VALUES ('j', 1, 'ready', 1.0)"
    )
    c.execute("INSERT INTO pages (job_id, idx, image_sha256, status) VALUES (1, 0, 'x', 'review')")
    db.migrate(c)
    g = c.execute("SELECT * FROM generations").fetchone()
    assert (g["kind"], g["number"], g["status"]) == ("shipped", 0, "active")
    assert c.execute("SELECT generation FROM pages").fetchone()[0] == g["id"]  # 0 ではなく世代の ID
    assert c.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == 1


# ---- 世代の登録・使用中・認識に使う重み


def test_weights_for_is_shipped_until_a_generation_is_adopted(con, tmp_path):
    w = gens.weights_for(con, models(tmp_path))
    assert w.dir is None and w.confidence is None and w.number == 0
    gid = add_trained(con, tmp_path)
    w = gens.weights_for(con, models(tmp_path))
    assert (w.generation_id, w.number) == (gid, 1)
    assert w.dir == models(tmp_path) / "1" / "gen_0001"
    assert w.confidence == w.dir / "confidence.json"
    # 世代 0 は過去(retired)になる
    assert gens.shipped(con)["status"] == "retired"


def test_a_rejected_generation_does_not_change_the_active_one_and_keeps_the_reason(con, tmp_path):
    gid = add_trained(con, tmp_path, adopt=False, files=False)
    assert gens.active(con)["kind"] == "shipped"
    row = con.execute("SELECT * FROM generations WHERE id = ?", (gid,)).fetchone()
    assert row["status"] == "rejected" and row["reject_reason"] == "accuracy_dropped"
    d = next(g for g in gens.list_generations(con, models(tmp_path)) if g["id"] == gid)
    assert not d["restorable"] and d["reject_reason"] == "accuracy_dropped"


def test_weights_for_falls_back_to_shipped_when_the_weights_are_missing(con, tmp_path):
    add_trained(con, tmp_path)
    d = models(tmp_path) / "1" / "gen_0001"
    (d / gens.WEIGHT_FILES[0]).unlink()
    w = gens.weights_for(con, models(tmp_path))
    assert w.dir is None and w.number == 0  # 認識を止めない


def test_meta_is_stored_without_values_and_listed(con, tmp_path):
    meta = {
        "n_train_pages": 8,
        "n_eval_pages": 2,
        "train_page_ids": [3, 4, 5],
        "eval": {"parent": {"accuracy": 0.9}, "candidate": {"accuracy": 0.95}},
    }
    gid = add_trained(con, tmp_path, meta=meta)
    d = next(g for g in gens.list_generations(con, models(tmp_path)) if g["id"] == gid)
    assert d["n_train_pages"] == 8 and d["n_eval_pages"] == 2
    assert d["eval"]["candidate"]["accuracy"] == 0.95
    assert json.loads(
        con.execute("SELECT meta FROM generations WHERE id = ?", (gid,)).fetchone()[0]
    )["train_page_ids"] == [3, 4, 5]


# ---- ロールバック


def test_rollback_to_shipped_and_forward_again(con, tmp_path):
    g1 = add_trained(con, tmp_path)
    s = gens.shipped(con)["id"]
    out = gens.activate(con, models(tmp_path), s)
    assert out["status"] == "active" and gens.active(con)["id"] == s
    assert (
        con.execute("SELECT status FROM generations WHERE id = ?", (g1,)).fetchone()[0] == "retired"
    )
    w = gens.weights_for(con, models(tmp_path))
    assert w.dir is None
    gens.activate(con, models(tmp_path), g1)  # また世代 1 へ
    assert gens.weights_for(con, models(tmp_path)).number == 1
    assert (
        con.execute("SELECT COUNT(*) FROM generations WHERE status = 'active'").fetchone()[0] == 1
    )


def test_rollback_is_refused_for_rejected_discarded_or_weightless_generations(con, tmp_path):
    rejected = add_trained(con, tmp_path, 1, adopt=False, files=False)
    with pytest.raises(store.StoreError) as e:
        gens.activate(con, models(tmp_path), rejected)
    assert e.value.status == 409 and e.value.code == "generation_not_restorable"
    g2 = add_trained(con, tmp_path, 2)
    for f in (models(tmp_path) / "1" / "gen_0002").iterdir():
        f.unlink()
    gens.activate(con, models(tmp_path), gens.shipped(con)["id"])
    with pytest.raises(store.StoreError) as e:
        gens.activate(con, models(tmp_path), g2)
    assert e.value.code == "generation_weights_missing"
    with pytest.raises(store.StoreError) as e:
        gens.activate(con, models(tmp_path), 999)
    assert e.value.status == 404


# ---- リセット


def test_reset_returns_to_shipped_and_removes_only_generation_dirs(con, tmp_path):
    add_trained(con, tmp_path, 1)
    add_trained(con, tmp_path, 2)
    keep = tmp_path / "images" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("x")
    assert gens.reset(con, models(tmp_path)) == 2
    assert gens.active(con)["kind"] == "shipped"
    assert not (models(tmp_path) / "1" / "gen_0001").exists()
    assert not (models(tmp_path) / "1" / "gen_0002").exists()
    assert keep.exists()
    states = {g["number"]: g["status"] for g in gens.list_generations(con, models(tmp_path))}
    assert states == {0: "active", 1: "discarded", 2: "discarded"}
    assert gens.next_number(con) == 3  # 番号は使い回さない
    assert gens.reset(con, models(tmp_path)) == 0  # 何度でも呼べる


def test_reset_never_deletes_outside_the_models_directory(con, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "x.txt").write_text("x")
    gid = add_trained(con, tmp_path, 1)
    con.execute("UPDATE generations SET dir = '../victim' WHERE id = ?", (gid,))
    gens.reset(con, models(tmp_path))
    assert (victim / "x.txt").exists()


# ---- API


@pytest.fixture
def app_client(tmp_path):
    app = create_app(
        tmp_path / "a.sqlite3",
        TOKEN,
        allowed_hosts_for(PORT),
        dev=True,
        backend=FakeBackend(error_rate=0.0),
    )
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        c.app_ = app
        yield c


def test_generations_api_lists_rolls_back_and_resets(app_client, tmp_path):
    c = app_client
    con = c.app_.state.db
    listing = c.get("/api/generations").json()
    assert [g["number"] for g in listing["generations"]] == [0]
    assert listing["generations"][0]["restorable"] and listing["active_id"]
    g1 = add_trained(con, tmp_path)
    listing = c.get("/api/generations").json()
    assert listing["active_id"] == g1 and [g["number"] for g in listing["generations"]] == [0, 1]
    learning = c.get("/api/learning").json()
    assert learning["can_rollback"] and learning["phase"] == 4
    assert learning["generations"][1]["observed"] is None  # まだこの世代で認識したページが無い
    s = listing["generations"][0]["id"]
    assert c.post(f"/api/generations/{s}/activate").json()["status"] == "active"
    assert c.get("/api/generations").json()["active_id"] == s
    assert c.post("/api/generations/999/activate").status_code == 404
    r = c.post("/api/generations/reset", json={})
    assert r.status_code == 200 and r.json()["discarded"] == 1
    assert not (models(tmp_path) / "1" / "gen_0001").exists()
    assert not c.get("/api/learning").json()["can_rollback"]


def test_reset_can_keep_or_delete_the_confirmed_learning_data(app_client, tmp_path):
    c = app_client
    con = c.app_.state.db
    job = c.post("/api/dev/jobs", json={"pages": 1, "seed": 5, "fill_rows": 6}).json()
    p = job["pages"][0]["id"]
    confirm_with(c, p, {})
    n_crops = con.execute("SELECT COUNT(*) FROM cells WHERE crop_path IS NOT NULL").fetchone()[0]
    assert n_crops > 0
    add_trained(con, tmp_path)
    # 確定データを残す: 切り出し画像と学習状態は残り、重みだけ出荷時に戻る
    r = c.post("/api/generations/reset", json={"delete_learning_data": False}).json()
    assert r["discarded"] == 1 and r["deleted_images"] == 0
    assert (
        con.execute("SELECT COUNT(*) FROM cells WHERE crop_path IS NOT NULL").fetchone()[0]
        == n_crops
    )
    # 確定データも削除する
    add_trained(con, tmp_path, 2)
    r = c.post("/api/generations/reset", json={"delete_learning_data": True}).json()
    assert r["discarded"] == 1 and r["deleted_images"] > 0
    assert con.execute("SELECT COUNT(*) FROM cells WHERE crop_path IS NOT NULL").fetchone()[0] == 0
    assert c.get(f"/api/pages/{p}").json()["status"] == "confirmed"  # 確定した値は残る


def test_observed_results_are_grouped_by_the_generation_that_recognized_the_page(
    app_client, tmp_path
):
    c = app_client
    con = c.app_.state.db
    p0 = c.post("/api/dev/jobs", json={"pages": 1, "seed": 6, "fill_rows": 6}).json()["pages"][0][
        "id"
    ]
    confirm_with(c, p0, {})
    g1 = add_trained(con, tmp_path)
    p1 = c.post("/api/dev/jobs", json={"pages": 1, "seed": 7, "fill_rows": 6}).json()["pages"][0][
        "id"
    ]
    confirm_with(c, p1, {})
    assert con.execute("SELECT generation FROM pages WHERE id = ?", (p1,)).fetchone()[0] == g1
    learning = c.get("/api/learning").json()
    by_id = {g["id"]: g for g in learning["generations"]}
    shipped_id = gens.shipped(con)["id"]
    assert by_id[shipped_id]["observed"]["n_pages"] == 1
    assert by_id[g1]["observed"]["n_pages"] == 1 and by_id[g1]["n_recognized"] == 1
    for g in learning["generations"]:
        assert all(not isinstance(v, str) for v in (g["observed"] or {}).values())


# ---- 認識への受け渡し


class Recording(FakeBackend):
    def __init__(self):
        super().__init__(error_rate=0.0)
        self.payloads = []

    def process(self, payload):
        self.payloads.append(payload)
        return super().process(payload)


def test_the_runner_recognizes_with_the_active_generation_and_records_its_id(tmp_path):
    from test_app_hardening import png_bytes, upload

    backend = Recording()
    app = create_app(
        tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=True, backend=backend
    )
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        con = app.state.db

        def run():
            up = upload(c, "a.png", png_bytes()).json()
            job = c.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
            app.state.runner.run_job(job["job_id"])
            gen = con.execute(
                "SELECT generation FROM pages WHERE id = ?", (job["page_ids"][0],)
            ).fetchone()[0]
            return backend.payloads[-1], gen

        pl, gen = run()
        assert pl.weights_dir is None and gen == gens.shipped(con)["id"]
        g1 = add_trained(con, tmp_path)
        pl, gen = run()
        assert pl.weights_dir == str(models(tmp_path) / "1" / "gen_0001") and gen == g1
        assert pl.confidence.endswith("confidence.json")
        c.post(f"/api/generations/{gens.shipped(con)['id']}/activate")  # ロールバック
        pl, gen = run()
        assert pl.weights_dir is None and gen == gens.shipped(con)["id"]
