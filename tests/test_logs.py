"""ログ: 回転・保存期間・例外は型名だけ。そして、一通りの操作の後で、入力した値・ファイル名・設定の名前・パスがログに無いこと(合成データ)。"""

from __future__ import annotations

import logging
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from test_app_hardening import png_bytes, upload
from yayoi_ocr.app import gens, logs
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app
from yayoi_ocr.app.worker import FakeBackend

PORT = 8134
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def _clean_logging():
    yield
    logs.teardown()


def read_logs(d) -> str:
    return "".join(
        p.read_text(encoding="utf-8", errors="replace") for p in sorted(d.glob("app.log*"))
    )


def test_exceptions_are_written_as_type_names_only(tmp_path):
    logs.setup(tmp_path / "logs", console=False)
    lg = logging.getLogger("yayoi.test")
    try:
        raise ValueError("秘密の摘要と C:/Users/secret/path")
    except ValueError:
        lg.exception("training_failed run=%d", 3)
    lg.warning("with exc_info", exc_info=RuntimeError("別の秘密"))
    logs.teardown()
    text = read_logs(tmp_path / "logs")
    assert "training_failed run=3" in text and "exception_type=ValueError" in text
    assert "exception_type=RuntimeError" in text
    assert "秘密" not in text and "secret" not in text and "Traceback" not in text


def test_the_log_rotates_and_keeps_a_bounded_number_of_files(tmp_path):
    d = tmp_path / "logs"
    logs.setup(d, max_bytes=2000, backups=3, console=False)
    lg = logging.getLogger("yayoi.test")
    for i in range(500):
        lg.info("line %d %s", i, "x" * 60)
    logs.teardown()
    files = sorted(p.name for p in d.glob("app.log*"))
    assert files == ["app.log", "app.log.1", "app.log.2", "app.log.3"]  # 世代は 3 つまで
    assert all((d / f).stat().st_size <= 2200 for f in files)  # 1 ファイルは上限付近まで


def test_old_log_files_are_removed_at_startup_but_recent_ones_are_kept(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    now = time.time()
    for name, age_days in (("app.log", 1), ("app.log.1", 29), ("app.log.2", 31), ("app.log.3", 90)):
        p = d / name
        p.write_text("x")
        os.utime(p, (now - age_days * 86400, now - age_days * 86400))
    other = d / "notes.txt"  # ログでないファイルには触れない
    other.write_text("keep")
    os.utime(other, (now - 400 * 86400, now - 400 * 86400))
    assert logs.purge_old(d, 30, now=now) == 2
    assert sorted(p.name for p in d.iterdir()) == ["app.log", "app.log.1", "notes.txt"]
    logs.setup(d, console=False)  # 起動時にも呼ばれる
    assert (d / "notes.txt").exists()


def test_setup_twice_does_not_duplicate_lines(tmp_path):
    logs.setup(tmp_path / "logs", console=False)
    logs.setup(tmp_path / "logs", console=False)
    logging.getLogger("yayoi.test").info("once")
    logs.teardown()
    assert read_logs(tmp_path / "logs").count("once") == 1


# ---- 一通りの操作の後で、値がログに無い


def test_no_entered_value_name_or_path_reaches_the_log_after_a_full_workflow(tmp_path):
    secrets = [
        "秘密のジョブ名",
        "秘密のファイル名",
        "秘密の支払手段",
        "秘密のタイトル",
        "秘密の摘要",
    ]
    app = create_app(
        tmp_path / "app.sqlite3",
        TOKEN,
        allowed_hosts_for(PORT),
        dev=True,
        backend=FakeBackend(error_rate=0.0),
    )
    logs.setup(tmp_path / "logs", console=False)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH) as c:
        cur = c.get("/api/settings").json()["settings"]
        cur.update(
            wizard_done=True,
            payments=[{"name": "秘密の支払手段", "credit_account": "現金", "credit_sub": ""}],
            titles=[{"name": "秘密のタイトル", "debit_account": "旅費交通費", "debit_sub": ""}],
            auto_train=False,
        )
        assert c.put("/api/settings", json=cur).status_code == 200
        # 取り込み → 認識 → 確認 → 確定 → 出力
        up = upload(c, "秘密のファイル名.png", png_bytes()).json()
        job = c.post("/api/jobs", json={"upload_ids": [up["id"]], "name": "秘密のジョブ名"}).json()
        app.state.runner.run_job(job["job_id"])
        pid = job["page_ids"][0]
        confirm_with(c, pid, {(0, 2): "秘密の摘要", (1, 2): "秘密の摘要"})
        c.put(f"/api/pages/{pid}/title", json={"title": "秘密のタイトル"})
        c.post("/api/exports", json={"page_ids": [pid], "format": "yayoi_import"})
        # 失敗の経路: 認識の失敗(例外の文言に値が入っていても、型名だけ)
        app.state.runner.backend.fail_next = 1
        up2 = upload(c, "秘密のファイル名2.png", png_bytes()).json()
        job2 = c.post(
            "/api/jobs", json={"upload_ids": [up2["id"]], "name": "秘密のジョブ名2"}
        ).json()
        app.state.runner.run_job(job2["job_id"])
        # 追加学習(採用・失敗)・世代の警告
        for i in range(7):
            j = c.post(
                "/api/dev/jobs",
                json={
                    "pages": 1,
                    "seed": 300 + i,
                    "fill_rows": 29,
                    "error_rate": 0.0,
                    "review_rate": 0.0,
                    "ambiguous_rows": [],
                },
            ).json()
            confirm_with(c, j["pages"][0]["id"], {})
        for mode in ("improve", "fail", "amount_miss"):
            assert c.post("/api/dev/trainer", json={"mode": mode}).status_code == 200
            c.post("/api/training/start")
            app.state.trainer.join(20)
        con = app.state.db
        row = gens.active(con)
        if row["kind"] != "shipped":
            gens.mark_broken(con, row, "pair_mismatch", tmp_path / "models")
        # バックアップの作成・壊れた zip の検査の失敗
        c.post("/api/backup/create", json={})
        app.state.backup.wait(60)
        c.put("/api/backup/upload", content=b"not a zip" * 20)
        app.state.backup.wait(60)
        values = {
            r[0]
            for r in con.execute(
                "SELECT resolved_value FROM cells UNION SELECT confirmed_value FROM cells"
            )
            if r[0] and len(r[0]) >= 3 and not r[0].isdigit()
        }
    logs.teardown()
    text = read_logs(tmp_path / "logs")
    assert text, "ログが空(試験が成り立っていない)"
    for word in [*secrets, str(tmp_path), "test_logs", *values]:
        assert word not in text, f"ログに値・名前・パスが出ています: {word[:6]}…"
    assert "page" in text and "failed" in text  # 失敗は(ID と型名だけで)記録されている
