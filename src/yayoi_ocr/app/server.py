"""FastAPI アプリ(M0: 健全性・プロファイル。以降のマイルストーンで API を足す)。

`create_app` は DB・トークン・許可する Host を引数に取る(テストでも本番でも同じ入口)。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__ as app_version
from . import db, store
from .paths import AppPaths
from .security import SecurityMiddleware

STATIC_DIR = Path(__file__).parent / "static"

FALLBACK_HTML = """<!doctype html><html lang="ja"><meta charset="utf-8"><title>yayoi-ocr</title>
<body><p>画面(フロントエンド)がビルドされていません。<code>web/</code> で <code>npm ci && npm run build</code> を実行してください。</p></body></html>"""


class CellPatch(BaseModel):
    action: Literal["pass", "edit", "confirm", "candidate", "revert"]
    value: str | None = Field(default=None, max_length=400)
    candidate_index: int | None = None


class ConfirmBody(BaseModel):
    review_seconds: float = 0.0


class PositionBody(BaseModel):
    page_id: int
    cell: int | None = None


class DevJobBody(BaseModel):
    pages: int = Field(default=2, ge=1, le=10)
    seed: int = 0
    error_rate: float = 0.25
    review_rate: float = 0.9
    ambiguous_rows: list[int] = Field(default_factory=lambda: [3])
    fill_rows: int = Field(default=12, ge=1, le=29)


def create_app(
    db_path: str | Path,
    token: str,
    allowed_hosts: frozenset[str],
    static_dir: Path | None = None,
    images_dir: Path | None = None,
    dev: bool = False,
) -> FastAPI:
    db_path = Path(db_path)
    images = images_dir if images_dir is not None else AppPaths(db_path.parent).images
    images.mkdir(parents=True, exist_ok=True)
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
        return {"ok": True, "version": app_version, "schema": version, "dev": dev}

    @app.exception_handler(store.StoreError)
    async def store_error(_: Request, exc: store.StoreError) -> JSONResponse:
        return JSONResponse({"error": exc.code, **exc.extra}, status_code=exc.status)

    @app.get("/api/profiles")
    def profiles() -> list[dict]:
        with lock:
            rows = con.execute("SELECT id, name FROM profiles ORDER BY id").fetchall()
        return [{"id": r["id"], "name": r["name"]} for r in rows]

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        with lock:
            return store.list_jobs(con)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int) -> dict:
        with lock:
            return store.get_job(con, job_id)

    @app.put("/api/jobs/{job_id}/position")
    def position(job_id: int, body: PositionBody) -> dict:
        with lock:
            store.set_position(con, job_id, body.page_id, body.cell)
        return {"ok": True}

    @app.get("/api/pages/{page_id}")
    def page(page_id: int) -> dict:
        with lock:
            return store.get_page(con, page_id)

    @app.get("/api/pages/{page_id}/cells/{cell_id}/image")
    def cell_image(page_id: int, cell_id: int) -> FileResponse:
        with lock:
            path = store.cell_image_path(con, images, page_id, cell_id)
        return FileResponse(path, media_type="image/png")

    @app.patch("/api/pages/{page_id}/cells/{cell_id}")
    def patch_cell(page_id: int, cell_id: int, body: CellPatch) -> dict:
        with lock:
            return store.patch_cell(
                con, page_id, cell_id, body.action, body.value, body.candidate_index
            )

    @app.post("/api/pages/{page_id}/undo")
    def undo(page_id: int) -> dict:
        with lock:
            pg, cell = store.undo(con, page_id)
        return {"page": pg, "cell_id": cell}

    @app.post("/api/pages/{page_id}/redo")
    def redo(page_id: int) -> dict:
        with lock:
            pg, cell = store.redo(con, page_id)
        return {"page": pg, "cell_id": cell}

    @app.post("/api/pages/{page_id}/confirm")
    def confirm(page_id: int, body: ConfirmBody) -> dict:
        with lock:
            return store.confirm_page(con, page_id, body.review_seconds)

    @app.post("/api/pages/{page_id}/reopen")
    def reopen(page_id: int) -> dict:
        with lock:
            return store.reopen_page(con, page_id)

    @app.post("/api/pages/{page_id}/layout/approve")
    def layout_approve(page_id: int) -> dict:
        with lock:
            return store.approve_layout(con, page_id)

    if dev:  # 開発・E2E 専用: 合成した認識結果のジョブを作る(正解つき)。本番の起動では登録しない

        @app.post("/api/dev/jobs")
        def dev_jobs(body: DevJobBody) -> dict:
            from .fake import fake_page

            specs, truths = [], []
            for i in range(body.pages):
                spec, truth = fake_page(
                    body.seed + i,
                    fill_rows=body.fill_rows,
                    error_rate=body.error_rate,
                    review_rate=body.review_rate,
                    ambiguous_rows=tuple(body.ambiguous_rows),
                )
                specs.append(spec)
                truths.append(truth)
            with lock:
                job_id, page_ids = store.create_job(
                    con, images, f"dev-{body.seed}", db.DEFAULT_PROFILE_ID, specs
                )
                out = []
                for pid, truth in zip(page_ids, truths, strict=True):
                    ids = {
                        (r["row"], r["col"]): r["id"]
                        for r in con.execute(
                            "SELECT id, row, col FROM cells WHERE page_id = ?", (pid,)
                        )
                    }
                    out.append({"id": pid, "truth": {str(ids[k]): v for k, v in truth.items()}})
            return {"job_id": job_id, "pages": out}

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
