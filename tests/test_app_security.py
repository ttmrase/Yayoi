"""ローカルサーバの防御(トークン・Host・Origin・CORS・CSP)と DB スキーマ・起動のテスト。"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient

from yayoi_ocr.app import db
from yayoi_ocr.app.launcher import RunningServer
from yayoi_ocr.app.security import CSP, allowed_hosts_for, new_token
from yayoi_ocr.app.server import create_app

PORT = 8123
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "app.sqlite3", TOKEN, allowed_hosts_for(PORT))
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        yield c


def test_api_requires_the_token(client):
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/health", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    r = client.get("/api/health", headers=AUTH)
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/api/profiles", headers=AUTH).json() == [{"id": 1, "name": "共通"}]
    assert client.get("/api/unknown", headers=AUTH).status_code == 404
    assert client.get("/api/unknown").status_code == 401  # 未知のパスでもトークンが先


def test_token_is_not_accepted_in_the_url(client):
    assert client.get(f"/api/health?token={TOKEN}").status_code == 401


def test_static_page_needs_no_token_but_the_data_does(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/api/profiles").status_code == 401


def test_host_header_is_validated(client):
    ok = client.get("/api/health", headers={**AUTH, "Host": f"localhost:{PORT}"})
    assert ok.status_code == 200
    for host in ("evil.example", f"127.0.0.1:{PORT + 1}", "127.0.0.1", f"evil.example:{PORT}"):
        assert client.get("/api/health", headers={**AUTH, "Host": host}).status_code == 400, host
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400  # 画面も同じ


def test_cross_origin_requests_are_rejected_and_cors_headers_never_sent(client):
    r = client.get("/api/health", headers={**AUTH, "Origin": "http://evil.example"})
    assert r.status_code == 403
    same = client.get("/api/health", headers={**AUTH, "Origin": f"http://127.0.0.1:{PORT}"})
    assert same.status_code == 200
    # プリフライトにも応答しない(許可するヘッダーを返さない)
    pre = client.options(
        "/api/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert pre.status_code in (401, 403, 405)  # allow-int-list
    for resp in (r, same, pre, client.get("/"), client.get("/api/health", headers=AUTH)):
        assert not [h for h in resp.headers if h.lower().startswith("access-control-")]


def test_security_headers_on_every_response(client):
    for resp in (
        client.get("/"),
        client.get("/api/health", headers=AUTH),
        client.get("/api/health"),
    ):
        assert resp.headers["content-security-policy"] == CSP
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in CSP and "http" not in CSP  # 外部の読み込み先を許可しない


def test_docs_and_openapi_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path, headers=AUTH).status_code in (404, 405), path


def test_websocket_is_refused(client):
    with pytest.raises(Exception), client.websocket_connect("/api/ws", headers=AUTH):  # noqa: B017  (接続が閉じられる)
        pass


def test_schema_is_created_with_default_profile_and_generation_zero(tmp_path):
    con = db.connect(tmp_path / "x.sqlite3")
    assert db.migrate(con) == len(db.MIGRATIONS)
    assert db.migrate(con) == len(db.MIGRATIONS)  # 冪等(既定のプロファイルを増やさない)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "profiles",
        "jobs",
        "pages",
        "cells",
        "edits",
        "exports",
        "generations",
        "settings",
    } <= tables
    assert [tuple(r) for r in con.execute("SELECT id, name FROM profiles")] == [(1, "共通")]
    assert con.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == 1
    # ジョブはプロファイルを選べる(外部キー)
    con.execute("INSERT INTO jobs (name, profile_id, created_at) VALUES ('j', 1, 0)")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO jobs (name, profile_id, created_at) VALUES ('j', 99, 0)")
    con.close()


def test_newer_database_is_refused(tmp_path):
    con = db.connect(tmp_path / "x.sqlite3")
    con.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 1}")
    with pytest.raises(RuntimeError):
        db.migrate(con)
    con.close()


def test_running_server_binds_to_loopback_with_random_token_and_port(tmp_path):
    import httpx

    srv = RunningServer(tmp_path).start()
    try:
        assert srv.sock.getsockname()[0] == "127.0.0.1" and srv.port > 0
        assert srv.url.endswith(f"#token={srv.token}") and len(srv.token) >= 40
        base = f"http://127.0.0.1:{srv.port}"
        assert httpx.get(f"{base}/api/health").status_code == 401
        ok = httpx.get(f"{base}/api/health", headers={"Authorization": f"Bearer {srv.token}"})
        assert ok.status_code == 200
        # 別のホスト名での到達(DNS rebinding を想定)
        bad = httpx.get(
            f"{base}/api/health",
            headers={"Host": "evil.example", **{"Authorization": f"Bearer {srv.token}"}},
        )
        assert bad.status_code == 400
    finally:
        srv.stop()
    assert RunningServer(tmp_path).token != srv.token  # 起動ごとに変わる
