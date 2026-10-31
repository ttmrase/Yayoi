"""checker のセキュリティレビュー(M-1・L-1〜L-4・I-1)への対応のテスト: アップロードの検査は使い捨ての別プロセス、全ページの寸法、保管方針、例外の記録、CLI。"""

from __future__ import annotations

import io
import os
import struct
import sys
import time
import zlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from _app_helpers import confirm_with
from yayoi_ocr.app import learned, retention, settings, store, uploads
from yayoi_ocr.app.__main__ import dev_only_options, main
from yayoi_ocr.app.fake import ACCOUNTS
from yayoi_ocr.app.runner import JobRunner
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend

PORT = 8128
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_app(tmp_path, backend=None):
    return create_app(
        tmp_path / "a.sqlite3",
        TOKEN,
        allowed_hosts_for(PORT),
        dev=True,
        backend=backend or FakeBackend(error_rate=0.0),
    )


def client_of(app):
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH)


def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", np.full((60, 40, 3), 255, np.uint8))
    assert ok
    return buf.tobytes()


def upload(c, name, data):
    return c.post("/api/uploads", params={"name": name}, content=data)


def png_header_only(w: int, h: int) -> bytes:
    """幅 w・高さ h を名乗るだけの PNG(ヘッダーだけ。デコードすると壊れている)。"""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


# ---- M-1: 利用者のファイルをサーバのプロセスでデコードしない


def test_upload_is_inspected_in_a_separate_process_not_in_the_server(tmp_path, monkeypatch):
    app = make_app(tmp_path)

    def boom(*a, **k):
        raise AssertionError("サーバのプロセスで利用者のファイルをデコードしてはいけない")

    monkeypatch.setattr(Image, "open", boom)  # この(サーバと同じ)プロセスの PIL を使えなくする
    monkeypatch.setattr("yayoi_ocr.ingest.page_count", boom)
    monkeypatch.setattr("yayoi_ocr.ingest._check_size", boom)
    with client_of(app) as c:
        r = upload(c, "a.png", png_bytes())
        assert (
            r.status_code == 200 and r.json()["n_pages"] == 1
        )  # 検査は子プロセス(PIL が使える)で行われた


def test_inspection_has_a_timeout_and_a_hung_decoder_rejects_the_file(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(uploads, "PROBE_TIMEOUT_S", 0.01)  # 子プロセスの起動より短い = 時間切れ
    with client_of(app) as c:
        r = upload(c, "a.png", png_bytes())
        assert r.status_code == 422 and r.json()["error"] == "inspect_timeout"
    assert list((tmp_path / "uploads").iterdir()) == []  # 拒否したファイルは残らない


def test_a_crashing_probe_rejects_the_file_without_taking_the_server_down(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    real_run = uploads.subprocess.run

    def crashing(cmd, **kw):
        return real_run([sys.executable, "-c", "import os; os._exit(139)"], **{**kw, "text": True})

    monkeypatch.setattr(uploads.subprocess, "run", crashing)
    with client_of(app) as c:
        assert upload(c, "a.png", png_bytes()).status_code == 422
        monkeypatch.setattr(uploads.subprocess, "run", real_run)
        assert upload(c, "b.png", png_bytes()).status_code == 200  # サーバは無事


# ---- L-1: 全ページ・全フレームの寸法を登録時に検査する


def test_oversized_dimensions_are_rejected_at_upload(tmp_path):
    app = make_app(tmp_path)
    with client_of(app) as c:
        r = upload(c, "huge.png", png_header_only(20000, 20000))  # 4 億画素
        assert r.status_code == 422 and r.json()["error"] == "unreadable_file"


def test_a_big_second_frame_of_a_multipage_tiff_is_rejected(tmp_path):
    app = make_app(tmp_path)
    small = Image.new("1", (10, 10))
    big = Image.new("1", (12000, 12000))  # 1.44 億画素(上限 1.2 億)
    buf = io.BytesIO()
    small.save(buf, format="TIFF", save_all=True, append_images=[big], compression="tiff_lzw")
    with client_of(app) as c:
        assert upload(c, "multi.tif", buf.getvalue()).status_code == 422


def test_pdf_page_size_times_render_dpi_is_checked(tmp_path):
    pytest.importorskip("pypdfium2")
    pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 14400 14400]>>endobj\n"
        b"trailer<</Root 1 0 R/Size 4>>\n%%EOF\n"
    )
    app = make_app(tmp_path)
    with client_of(app) as c:
        assert upload(c, "big.pdf", pdf).status_code == 422  # 200 インチ四方 x 300 dpi


# ---- L-2: 保管の方針


def old(con, upload_id, age):
    con.execute("UPDATE uploads SET created_at = ? WHERE id = ?", (time.time() - age, upload_id))


def test_unused_uploads_older_than_24h_are_purged_but_used_and_recent_ones_stay(tmp_path):
    app = make_app(tmp_path)
    with client_of(app) as c:
        used = upload(c, "u.png", png_bytes()).json()["id"]
        stale = upload(c, "s.png", png_bytes()).json()["id"]
        fresh = upload(c, "f.png", png_bytes()).json()["id"]
        assert c.post("/api/jobs", json={"upload_ids": [used]}).status_code == 200
        con = app.state.db
        old(con, used, 3 * 86400)
        old(con, stale, 3 * 86400)
        old(con, fresh, 3600)
        orphan = tmp_path / "uploads" / "orphan"  # DB に無い古いフォルダ
        orphan.mkdir()
        os.utime(orphan, (time.time() - 3 * 86400,) * 2)
        outside = tmp_path / "keep_me"
        outside.mkdir()
        n = uploads.purge_unused(con, app.state.db_lock, tmp_path / "uploads")
        assert n == 2
        left = {p.name for p in (tmp_path / "uploads").iterdir()}
        assert left == {used, fresh} and outside.exists()
        assert [r["id"] for r in con.execute("SELECT id FROM uploads ORDER BY id")] == sorted(
            [used, fresh]
        )


def tiff_two_pages() -> bytes:
    frames = [Image.fromarray(np.full((40, 30, 3), v, np.uint8)) for v in (10, 200)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def prepare_two_page_job(c, app):
    assert (
        c.put(
            "/api/settings",
            json={
                "original_retention": "delete_after_export",
                "payments": [{"name": a, "credit_account": a} for a in ACCOUNTS],
                "titles": [{"name": "旅費交通費", "debit_account": "旅費交通費"}],
                "default_tax": [{"account": "旅費交通費", "tax_category": "課対仕入込10%"}],
            },
        ).status_code
        == 200
    )
    up = upload(c, "two.tif", tiff_two_pages()).json()
    job = c.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()
    app.state.runner.run_job(job["job_id"])
    return up["id"], job["page_ids"]


def test_originals_are_deleted_after_export_only_when_every_page_is_exported(tmp_path):
    app = make_app(tmp_path)
    with client_of(app) as c:
        uid, (p1, p2) = prepare_two_page_job(c, app)
        for p in (p1, p2):
            c.put(f"/api/pages/{p}/title", json={"title": "旅費交通費"})
            confirm_with(c, p, {})
        original = tmp_path / "uploads" / uid
        assert original.exists() and (tmp_path / "images" / str(p1) / "view.jpg").exists()
        assert c.post("/api/exports", json={"page_ids": [p1]}).status_code == 200
        assert original.exists()  # 1 ページだけ出力 → まだ残す
        assert c.post("/api/exports", json={"page_ids": [p2]}).status_code == 200
        assert not original.exists()  # 全ページを出力した → 元ファイルを削除
        assert not (tmp_path / "images" / str(p1) / "view.jpg").exists()
        row = app.state.db.execute("SELECT deleted_at FROM uploads WHERE id = ?", (uid,)).fetchone()
        assert row["deleted_at"] is not None
        # 以降、罫線の再切り出しはできない。確定した値・出力の履歴は残る
        assert c.get(f"/api/pages/{p1}/layout").status_code == 404
        assert len(c.get("/api/exports").json()["exports"]) == 2
        assert c.get(f"/api/pages/{p1}").json()["status"] == "confirmed"


def test_default_retention_keeps_the_originals(tmp_path):
    app = make_app(tmp_path)
    with client_of(app) as c:
        uid, (p1, p2) = prepare_two_page_job(c, app)
        s = c.get("/api/settings").json()["settings"]
        s["original_retention"] = "keep"
        assert c.put("/api/settings", json=s).status_code == 200
        for p in (p1, p2):
            c.put(f"/api/pages/{p}/title", json={"title": "旅費交通費"})
            confirm_with(c, p, {})
        assert c.post("/api/exports", json={"page_ids": [p1, p2]}).status_code == 200
        assert (tmp_path / "uploads" / uid).exists()


def test_learning_reset_deletes_confirmed_crops_and_the_learned_state_only(tmp_path):
    app = make_app(tmp_path)
    with client_of(app) as c:
        _uid, (p1, _p2) = prepare_two_page_job(c, app)
        confirm_with(c, p1, {(0, 2): "リセット前の摘要"})
        con = app.state.db
        cfg = settings.load(con)
        assert "リセット前の摘要" in learned.build_state(con, 1, cfg).dictionary
        crop = con.execute(
            "SELECT crop_path FROM cells WHERE page_id = ? AND crop_path IS NOT NULL LIMIT 1", (p1,)
        ).fetchone()["crop_path"]
        assert (tmp_path / "images" / crop).exists()
        # データ領域の外を指す crop_path があっても、外のファイルは消さない
        victim = tmp_path / "victim.txt"
        victim.write_text("x")
        con.execute(
            "UPDATE cells SET crop_path = '../victim.txt' WHERE id = (SELECT id FROM cells WHERE page_id = ? AND crop_path IS NOT NULL LIMIT 1 OFFSET 1)",
            (p1,),
        )
        assert c.get("/api/learning").json()["can_reset"] is True
        r = c.post("/api/learning/reset")
        assert r.status_code == 200 and r.json()["deleted_images"] > 0
        assert not (tmp_path / "images" / crop).exists() and victim.exists()
        assert learned.build_state(con, 1, cfg).dictionary == {}  # 学習状態は戻る
        assert c.get(f"/api/pages/{p1}").json()["status"] == "confirmed"  # 確定した値は残る
        assert all(not x["has_image"] for x in c.get(f"/api/pages/{p1}").json()["cells"])
        # リセット後に確定したページは、また学習に使われる
        retention.reset_learning(
            con, app.state.db_lock, tmp_path / "images", now=time.time() - 1000
        )
        assert "リセット前の摘要" in learned.build_state(con, 1, cfg).dictionary


# ---- L-3: 想定外の例外を握りつぶさず、型名を記録する


def test_an_unexpected_error_in_a_job_is_recorded_by_type_name_and_the_thread_survives(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    runner: JobRunner = app.state.runner
    with client_of(app) as c:
        up = upload(c, "a.png", png_bytes()).json()
        job = c.post("/api/jobs", json={"upload_ids": [up["id"]]}).json()

        def boom(con, job_id):
            raise RuntimeError("値を含む文面でも記録されない")

        monkeypatch.setattr(store, "processing_pages", boom)
        runner.start()
        try:
            assert runner.wait_idle(10)
        finally:
            monkeypatch.undo()
        j = c.get(f"/api/jobs/{job['job_id']}").json()
        assert j["error"] == "RuntimeError" and j["pages"][0]["status"] == "failed"
        assert j["pages"][0]["error"] == "RuntimeError" and "値" not in str(j)
        # スレッドは生きていて、再試行で次の処理ができる
        assert c.post(f"/api/pages/{job['page_ids'][0]}/retry").status_code == 200
        assert runner.wait_idle(10)
        assert c.get(f"/api/jobs/{job['job_id']}").json()["pages"][0]["status"] == "review"
    runner.stop()


# ---- L-4: 開発専用の引数は --dev が無いと受け付けない


def test_dev_only_cli_options_need_dev(capsys):
    import argparse

    from yayoi_ocr.app.__main__ import add_arguments

    ap = argparse.ArgumentParser()
    add_arguments(ap)
    assert dev_only_options(ap.parse_args([])) == []
    assert dev_only_options(ap.parse_args(["--token", "x"])) == ["--token"]
    assert set(
        dev_only_options(ap.parse_args(["--port", "9", "--no-window", "--backend", "fake"]))
    ) == {
        "--port",
        "--no-window",
        "--backend",
    }
    assert (
        dev_only_options(
            ap.parse_args(["--dev", "--token", "x", "--port", "9", "--backend", "fake"])
        )
        == []
    )
    assert main(["--token", "x"]) == 2  # サーバは起動せずに終わる
    assert "開発専用" in capsys.readouterr().err
