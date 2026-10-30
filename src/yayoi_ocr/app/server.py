"""FastAPI アプリ(M0: 健全性・プロファイル。以降のマイルストーンで API を足す)。

`create_app` は DB・トークン・許可する Host を引数に取る(テストでも本番でも同じ入口)。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__ as app_version
from . import db
from .security import SecurityMiddleware

STATIC_DIR = Path(__file__).parent / "static"

FALLBACK_HTML = """<!doctype html><html lang="ja"><meta charset="utf-8"><title>yayoi-ocr</title>
<body><p>画面(フロントエンド)がビルドされていません。<code>web/</code> で <code>npm ci && npm run build</code> を実行してください。</p></body></html>"""


def create_app(
    db_path: str | Path,
    token: str,
    allowed_hosts: frozenset[str],
    static_dir: Path | None = None,
) -> FastAPI:
    con = db.connect(db_path)
    db.migrate(con)
    lock = threading.RLock()  # 1 接続を複数スレッドで共有するので直列化する

    app = FastAPI(title="yayoi-ocr", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db = con
    app.state.db_lock = lock

    @app.get("/api/health")
    def health() -> dict:
        with lock:
            version = con.execute("PRAGMA user_version").fetchone()[0]
        return {"ok": True, "version": app_version, "schema": version}

    @app.get("/api/profiles")
    def profiles() -> list[dict]:
        with lock:
            rows = con.execute("SELECT id, name FROM profiles ORDER BY id").fetchall()
        return [{"id": r["id"], "name": r["name"]} for r in rows]

    root = static_dir if static_dir is not None else STATIC_DIR
    if (root / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(root), html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return FALLBACK_HTML

    app.add_middleware(SecurityMiddleware, token=token, allowed_hosts=allowed_hosts)
    return app


def close_app(app: FastAPI) -> None:
    con: sqlite3.Connection = app.state.db
    con.close()
