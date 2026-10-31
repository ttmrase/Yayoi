"""取り込み(アップロード → ジョブ → 非同期の認識 → 確認待ち)のテスト。認識は差し替え可能なバックエンド(合成)を使う。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from yayoi_ocr.app import uploads
from yayoi_ocr.app.fake import fake_page
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend, PageOutcome, PagePayload, worker_threads

PORT = 8125
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Recording:
    """呼ばれた入力を記録し、指定したページでは失敗するバックエンド。"""

    def __init__(self, fail_keys: set[str] | None = None, last_year: int | None = 8) -> None:
        self.calls: list[PagePayload] = []
        self.fail_keys = fail_keys or set()
        self.last_year = last_year
        self.closed = False

    def process(self, payload: PagePayload) -> PageOutcome:
        self.calls.append(payload)
        if payload.page_key in self.fail_keys:
            raise RuntimeError("boom")
        spec, _ = fake_page(len(self.calls), fill_rows=3)
        return PageOutcome(spec, last_year=self.last_year)

    def close(self) -> None:
        self.closed = True


def make_app(tmp_path, backend):
    return create_app(
        tmp_path / "a.sqlite3", TOKEN, allowed_hosts_for(PORT), dev=False, backend=backend
    )


def client_of(app):
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH)


def png_bytes(v: int = 255) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((60, 40, 3), v, np.uint8))
    assert ok
    return buf.tobytes()


def upload(c, name: str, data: bytes):
    return c.post("/api/uploads", params={"name": name}, content=data)


def tiff_bytes(tmp_path, n: int) -> bytes:
    frames = [Image.fromarray(np.full((40, 30, 3), 20 * i, np.uint8)) for i in range(n)]
    p = tmp_path / "multi.tif"
    frames[0].save(p, save_all=True, append_images=frames[1:])
    return p.read_bytes()


def create_job(c, upload_ids, **kw):
    r = c.post("/api/jobs", json={"upload_ids": upload_ids, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_requires_token(tmp_path):
    app = make_app(tmp_path, Recording())
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        assert (
            c.post("/api/uploads", params={"name": "a.png"}, content=png_bytes()).status_code == 401
        )
        assert c.post("/api/jobs", json={"upload_ids": ["x"]}).status_code == 401


def test_upload_validates_type_content_and_size(tmp_path, monkeypatch):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        assert upload(c, "memo.txt", b"hello").status_code == 415
        assert upload(c, "empty.png", b"").status_code == 422
        r = upload(c, "broken.png", b"not an image at all")
        assert r.status_code == 422 and r.json()["error"] == "unreadable_file"
        monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 100)
        assert upload(c, "big.png", png_bytes()).status_code == 413
        # 失敗したアップロードのファイルは残らない
        assert list((tmp_path / "uploads").iterdir()) == []


def test_upload_never_uses_the_given_name_as_a_path(tmp_path):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        r = upload(c, "../../etc/evil.png", png_bytes())
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "evil.png" and body["n_pages"] == 1
        stored = list((tmp_path / "uploads").rglob("*"))
        files = [p for p in stored if p.is_file()]
        assert [p.name for p in files] == ["original.png"]
        assert body["id"] in files[0].parts
        assert not (tmp_path.parent / "etc").exists()


def test_multipage_tiff_makes_one_page_per_frame(tmp_path):
    rec = Recording()
    app = make_app(tmp_path, rec)
    with client_of(app) as c:
        up = upload(c, "multi.tif", tiff_bytes(tmp_path, 3)).json()
        assert up["n_pages"] == 3
        job = create_job(c, [up["id"]])
        assert len(job["page_ids"]) == 3
        app.state.runner.run_job(job["job_id"])
        j = c.get(f"/api/jobs/{job['job_id']}").json()
        assert [p["status"] for p in j["pages"]] == ["review"] * 3 and j["status"] == "ready"
        assert [x.source_index for x in rec.calls] == [0, 1, 2]


def test_pages_are_processing_until_the_runner_fills_them(tmp_path):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        up = upload(c, "a.png", png_bytes()).json()
        job = create_job(c, [up["id"]], name="テスト")
        jid, pid = job["job_id"], job["page_ids"][0]
        lst = c.get("/api/jobs").json()
        assert lst[0]["name"] == "テスト" and lst[0]["status"] == "processing"
        assert lst[0]["n_processing"] == 1
        page = c.get(f"/api/pages/{pid}").json()
        assert page["status"] == "processing" and page["cells"] == []
        # 処理中のページは編集・確定できない
        assert c.post(f"/api/pages/{pid}/confirm", json={}).status_code == 409
        app.state.runner.run_job(jid)
        page = c.get(f"/api/pages/{pid}").json()
        assert page["status"] == "review" and len(page["cells"]) == 29 * 4
        assert c.get(f"/api/jobs/{jid}").json()["status"] == "ready"
        # 切り出し画像が取れる
        first = next(x for x in page["cells"] if x["has_image"])
        img = c.get(f"/api/pages/{pid}/cells/{first['id']}/image")
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_year_is_carried_from_the_previous_page(tmp_path):
    rec = Recording(last_year=9)
    app = make_app(tmp_path, rec)
    with client_of(app) as c:
        a = upload(c, "a.png", png_bytes(10)).json()
        b = upload(c, "b.png", png_bytes(20)).json()
        job = create_job(c, [a["id"], b["id"]])
        app.state.runner.run_job(job["job_id"])
        assert [p.prev_year for p in rec.calls] == [None, 9]


def test_a_failed_page_does_not_stop_the_job_and_can_be_retried(tmp_path):
    rec = Recording(fail_keys=set())
    app = make_app(tmp_path, rec)
    with client_of(app) as c:
        ups = [upload(c, f"{i}.png", png_bytes(i * 30)).json() for i in range(3)]
        rec.fail_keys = {f"{ups[1]['id']}:0"}
        job = create_job(c, [u["id"] for u in ups])
        app.state.runner.run_job(job["job_id"])
        j = c.get(f"/api/jobs/{job['job_id']}").json()
        assert [p["status"] for p in j["pages"]] == ["review", "failed", "review"]
        assert j["pages"][1]["error"] == "RuntimeError"  # 例外の型名だけ(値や文面は残さない)
        assert c.get("/api/jobs").json()[0]["n_failed"] == 1
        failed = j["pages"][1]["id"]
        assert c.patch(f"/api/pages/{failed}/cells/1", json={"action": "pass"}).status_code == 409
        # 再試行(今度は成功する)
        assert (
            c.post(f"/api/pages/{j['pages'][0]['id']}/retry").status_code == 409
        )  # 失敗でないページ
        rec.fail_keys = set()
        assert c.post(f"/api/pages/{failed}/retry").status_code == 200
        app.state.runner.run_job(job["job_id"])
        j = c.get(f"/api/jobs/{job['job_id']}").json()
        assert [p["status"] for p in j["pages"]] == ["review"] * 3 and j["status"] == "ready"


def test_unfinished_recognition_resumes_after_restart(tmp_path):
    rec1 = Recording()
    app1 = make_app(tmp_path, rec1)
    with client_of(app1) as c:
        up = upload(c, "a.png", png_bytes()).json()
        job = create_job(c, [up["id"]])  # 認識前にアプリが終わった想定(run_job を呼ばない)
    from yayoi_ocr.app.server import close_app

    close_app(app1)
    rec2 = Recording()
    app2 = make_app(tmp_path, rec2)
    app2.state.runner.start()
    try:
        assert app2.state.runner.wait_idle(10)
        with client_of(app2) as c:
            j = c.get(f"/api/jobs/{job['job_id']}").json()
            assert j["pages"][0]["status"] == "review" and len(rec2.calls) == 1
    finally:
        close_app(app2)
    assert rec2.closed


def test_job_needs_known_uploads_and_a_backend(tmp_path):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        r = c.post("/api/jobs", json={"upload_ids": ["nope"]})
        assert r.status_code == 422 and r.json()["error"] == "upload_not_found"
        assert c.post("/api/jobs", json={"upload_ids": []}).status_code == 422
    plain = create_app(tmp_path / "b.sqlite3", TOKEN, allowed_hosts_for(PORT))
    with client_of(plain) as c:
        up = upload(c, "a.png", png_bytes()).json()
        r = c.post("/api/jobs", json={"upload_ids": [up["id"]]})
        assert r.status_code == 503 and r.json()["error"] == "recognition_disabled"


def test_unused_upload_can_be_cancelled_but_a_used_one_cannot(tmp_path):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        a = upload(c, "a.png", png_bytes(1)).json()
        b = upload(c, "b.png", png_bytes(2)).json()
        create_job(c, [a["id"]])
        assert c.delete(f"/api/uploads/{a['id']}").status_code == 409
        assert c.delete(f"/api/uploads/{b['id']}").status_code == 200
        assert not (tmp_path / "uploads" / b["id"]).exists()
        assert c.delete(f"/api/uploads/{b['id']}").status_code == 404


def test_settings_feed_the_recognizer_vocabulary(tmp_path):
    rec = Recording()
    app = make_app(tmp_path, rec)
    with client_of(app) as c:
        r = c.put(
            "/api/settings",
            json={
                "payments": [
                    {"name": "現金", "credit_account": "現金"},
                    {"name": "普通預金", "credit_account": "普通預金", "credit_sub": "A銀行"},
                ],
                "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
            },
        )
        assert r.status_code == 200 and r.json()["missing"] == []
        up = upload(c, "a.png", png_bytes()).json()
        job = create_job(c, [up["id"]])
        app.state.runner.run_job(job["job_id"])
        assert rec.calls[0].accounts == ("現金", "普通預金")
        assert rec.calls[0].titles == ("旅費交通費",)


def test_fake_backend_is_deterministic_per_page(tmp_path):
    b = FakeBackend()
    p = PagePayload(path="x", source_index=0, page_key="k:0")
    a1, a2 = b.process(p), b.process(p)
    assert [c.resolved_value for c in a1.spec.cells] == [c.resolved_value for c in a2.spec.cells]
    other = b.process(PagePayload(path="x", source_index=1, page_key="k:1"))
    assert [c.resolved_value for c in a1.spec.cells] != [c.resolved_value for c in other.spec.cells]


def test_worker_threads_stays_below_the_core_count(monkeypatch):
    monkeypatch.delenv("YAYOI_TORCH_THREADS", raising=False)
    assert [worker_threads(n) for n in (1, 2, 4, 8, 16, 64)] == [2, 2, 2, 4, 4, 4]
    monkeypatch.setenv("YAYOI_TORCH_THREADS", "3")
    assert worker_threads(64) == 3
    monkeypatch.setenv("YAYOI_TORCH_THREADS", "zero")
    assert worker_threads(8) == 4


# ---- レイアウトの確認・補正(重ね表示・表の隅の指定で切り直す)

GOOD_CORNERS = [[80, 150], [620, 150], [620, 900], [80, 900]]


def ready_page(c, app, backend_name="fake"):
    up = upload(c, "a.png", png_bytes()).json()
    job = create_job(c, [up["id"]])
    app.state.runner.run_job(job["job_id"])
    return job["job_id"], job["page_ids"][0]


def test_layout_overlay_info_and_image_are_served(tmp_path):
    fb = FakeBackend()
    fb.layout_ok = False
    app = make_app(tmp_path, fb)
    with client_of(app) as c:
        _, pid = ready_page(c, app)
        info = c.get(f"/api/pages/{pid}/layout").json()
        assert len(info["info"]["corners"]) == 4 and len(info["info"]["hlines"]) == 30
        assert info["status"] == "needs_review" and not info["approved"] and not info["manual"]
        img = c.get(f"/api/pages/{pid}/layout/image")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert img.content[:2] == b"\xff\xd8"


def test_layout_is_not_available_when_the_recognizer_gave_no_view(tmp_path):
    app = make_app(tmp_path, Recording())
    with client_of(app) as c:
        _, pid = ready_page(c, app)
        assert c.get(f"/api/pages/{pid}/layout").status_code == 404
        assert c.get(f"/api/pages/{pid}/layout/image").status_code == 404
        r = c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS})
        assert r.status_code == 409 and r.json()["error"] == "layout_not_available"


def test_recut_reprocesses_the_page_with_the_given_corners_and_counts_as_approved(tmp_path):
    fb = FakeBackend()
    fb.layout_ok = False
    app = make_app(tmp_path, fb)
    with client_of(app) as c:
        jid, pid = ready_page(c, app)
        before = c.get(f"/api/pages/{pid}").json()
        assert before["layout_status"] == "needs_review"
        assert all("page_needs_review" in x["reasons"] for x in before["cells"])
        r = c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS})
        assert r.status_code == 200
        assert c.get(f"/api/pages/{pid}").json()["status"] == "processing"
        assert c.get(f"/api/jobs/{jid}").json()["status"] == "processing"
        app.state.runner.run_job(jid)
        page = c.get(f"/api/pages/{pid}").json()
        assert (
            page["status"] == "review" and page["layout_approved"] and page["layout_status"] == "ok"
        )
        assert not any("page_needs_review" in x["reasons"] for x in page["cells"])
        info = c.get(f"/api/pages/{pid}/layout").json()
        assert info["manual"] and info["approved"]
        assert info["override"] == [[float(v) for v in p] for p in GOOD_CORNERS]
        assert info["info"]["corners"][0] == [80.0, 150.0]  # 指定した隅が重ね表示に反映される


def test_recut_passes_the_corners_to_the_recognizer(tmp_path):
    rec = Recording()
    app = make_app(tmp_path, rec)
    with client_of(app) as c:
        _jid, pid = ready_page(c, app)
        assert rec.calls[0].corners is None
        # 重ね表示が無い認識器でも、隅の指定そのものは受け付ける条件(layout_info)を満たさないと断る
        assert (
            c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS}).status_code
            == 409
        )


def test_recut_validates_the_corners_and_the_page_state(tmp_path):
    app = make_app(tmp_path, FakeBackend())
    with client_of(app) as c:
        jid, pid = ready_page(c, app)
        bad = [[80, 150], [620, 150], [80, 900], [620, 900]]  # ねじれ
        r = c.post(f"/api/pages/{pid}/layout/recut", json={"corners": bad})
        assert r.status_code == 422 and r.json()["error"] == "bad_corners"
        assert (
            c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS[:3]}).status_code
            == 422
        )
        # 認識中のページは切り直せない / 確定済みも不可(再オープンしてから)
        assert (
            c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS}).status_code
            == 200
        )
        assert (
            c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS}).status_code
            == 409
        )
        app.state.runner.run_job(jid)
        for x in c.get(f"/api/pages/{pid}").json()["cells"]:
            if x["review"]:
                c.patch(f"/api/pages/{pid}/cells/{x['id']}", json={"action": "confirm"})
        for x in c.get(f"/api/pages/{pid}").json()["cells"]:
            c.patch(f"/api/pages/{pid}/cells/{x['id']}", json={"action": "revert"})
        # 確定の条件が満たせないこともあるので、確定済みの状態は DB で作る
        app.state.db.execute("UPDATE pages SET status = 'confirmed' WHERE id = ?", (pid,))
        r = c.post(f"/api/pages/{pid}/layout/recut", json={"corners": GOOD_CORNERS})
        assert r.status_code == 409 and r.json()["error"] == "page_confirmed"
