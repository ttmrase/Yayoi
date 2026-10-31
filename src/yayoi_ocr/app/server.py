"""FastAPI アプリ(M0: 健全性・プロファイル。以降のマイルストーンで API を足す)。

`create_app` は DB・トークン・許可する Host を引数に取る(テストでも本番でも同じ入口)。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__ as app_version
from . import db, exports, metrics, retention, store, uploads
from . import settings as app_settings
from .paths import AppPaths
from .runner import JobRunner
from .security import SecurityMiddleware
from .worker import Backend, FakeBackend

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


class JobBody(BaseModel):
    name: str = Field(default="", max_length=200)
    upload_ids: list[str] = Field(min_length=1, max_length=50)
    profile_id: int = db.DEFAULT_PROFILE_ID


class DevFakeBody(BaseModel):
    delay: float = Field(default=0.0, ge=0, le=30)
    fail_next: int = Field(default=0, ge=0, le=100)
    layout_ok: bool = True  # False: 隅の指定が無いページはレイアウト要確認で返す


class ExportBody(BaseModel):
    page_ids: list[int] = Field(min_length=1, max_length=2000)
    format: str = exports.DEFAULT_FORMAT
    accept_duplicates: bool = False


class TitleBody(BaseModel):
    title: str = Field(max_length=200)


class RecutBody(BaseModel):
    corners: list[list[float]] = Field(
        min_length=4, max_length=4
    )  # 表の 4 隅(元画像 px。左上・右上・右下・左下)


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
    backend: Backend | None = None,
    uploads_dir: Path | None = None,
) -> FastAPI:
    """`backend` が None のときは認識を行わない(ジョブの作成は 503。M0・M1 のテストと合成ジョブだけ)。"""
    db_path = Path(db_path)
    images = images_dir if images_dir is not None else AppPaths(db_path.parent).images
    images.mkdir(parents=True, exist_ok=True)
    uploads_root = uploads_dir if uploads_dir is not None else AppPaths(db_path.parent).uploads
    uploads_root.mkdir(parents=True, exist_ok=True)
    exports_root = AppPaths(db_path.parent).exports
    exports_root.mkdir(parents=True, exist_ok=True)
    con = db.connect(db_path)
    db.migrate(con)
    lock = threading.RLock()  # 1 接続を複数スレッドで共有するので直列化する

    app = FastAPI(title="yayoi-ocr", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db = con
    app.state.db_lock = lock
    runner = JobRunner(con, lock, images, uploads_root, backend) if backend is not None else None
    app.state.runner = runner  # 起動は launcher(テストは run_job を直接呼ぶ)
    app.state.purge_unused = lambda: uploads.purge_unused(
        con, lock, uploads_root
    )  # 起動時に launcher が呼ぶ

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

    @app.get("/api/settings")
    def settings_get() -> dict:
        with lock:
            cur = app_settings.load(con)
        return {"settings": cur.model_dump(), "missing": app_settings.readiness(cur)}

    @app.put("/api/settings")
    def settings_put(body: app_settings.Settings) -> dict:
        with lock:
            app_settings.save(con, body)
        return {"settings": body.model_dump(), "missing": app_settings.readiness(body)}

    @app.get("/api/learning")
    def learning(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        with lock:
            rows = metrics.page_metrics(con, profile_id)
            return {
                "profile_id": profile_id,
                "generations": metrics.generations(con, profile_id),
                "pages": rows,
                "summary": metrics.summary(rows),
                # Phase 4 まで: 世代は出荷時(世代 0)だけなので、戻す・リセットは無効
                "can_rollback": False,
                "can_reset": any(
                    r["n_text"] for r in rows
                ),  # 確定したページがあれば、学習データ(切り出し画像・学習状態)を削除できる
                "phase": 3,
            }

    @app.get("/api/learning/report")
    def learning_report(profile_id: int = db.DEFAULT_PROFILE_ID) -> JSONResponse:
        """精度レポート(統計だけ。値・ファイル名・画像は含まない)。ダウンロードして開発者に送る。"""
        with lock:
            rep = metrics.report(con, profile_id, app_version, time.time())
        return JSONResponse(
            rep, headers={"Content-Disposition": 'attachment; filename="accuracy_report.json"'}
        )

    @app.post("/api/learning/reset")
    def learning_reset() -> dict:
        """学習データの削除(確定したセルの切り出し画像 + 学習状態)。確定した値・出力の履歴は残る。"""
        return retention.reset_learning(con, lock, images)

    @app.get("/api/exports")
    def exports_list() -> dict:
        with lock:
            return {
                "exports": exports.list_exports(con),
                "formats": [
                    {"id": f.id, "label": f.label, "notes": list(f.notes)}
                    for f in exports.FORMATS.values()
                ],
                "candidates": exports.candidates(con),
            }

    @app.post("/api/exports/validate")
    def exports_validate(body: ExportBody) -> dict:
        with lock:
            return exports.validate(con, body.page_ids, body.format)

    @app.post("/api/exports")
    def exports_run(body: ExportBody) -> dict:
        with lock:
            rec = exports.run_export(
                con, exports_root, body.page_ids, body.format, body.accept_duplicates
            )
            keep = app_settings.load(con).original_retention
        if keep == "delete_after_export":  # 設定「元画像の保管: 出力後に削除」
            retention.delete_originals(con, lock, uploads_root, images, body.page_ids)
        return rec

    @app.get("/api/exports/{export_id}/file")
    def exports_file(export_id: int) -> FileResponse:
        return exports_file_n(export_id, 0)

    @app.get("/api/exports/{export_id}/files/{idx}")
    def exports_file_n(export_id: int, idx: int) -> FileResponse:
        with lock:
            path = exports.export_file(con, exports_root, export_id, idx)
        return FileResponse(path, media_type="text/csv; charset=Shift_JIS", filename=path.name)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        with lock:
            return store.list_jobs(con)

    @app.post("/api/uploads")
    async def upload(request: Request, name: str = "") -> dict:
        return await uploads.save_upload(con, lock, uploads_root, name, request.stream())

    @app.delete("/api/uploads/{upload_id}")
    def upload_delete(upload_id: str) -> dict:
        uploads.delete_upload(con, lock, uploads_root, upload_id)
        return {"ok": True}

    @app.post("/api/jobs")
    def job_create(body: JobBody) -> dict:
        if runner is None:
            raise store.StoreError(503, "recognition_disabled")
        with lock:
            pages = uploads.pages_for_job(con, body.upload_ids)
            first = con.execute(
                "SELECT name FROM uploads WHERE id = ?", (body.upload_ids[0],)
            ).fetchone()["name"]
            name = body.name.strip() or Path(first).stem
            job_id, page_ids = store.create_pending_job(
                con, name, body.profile_id, [store.PendingPage(u, i) for u, i in pages]
            )
        runner.submit(job_id)
        return {"job_id": job_id, "page_ids": page_ids}

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

    @app.put("/api/pages/{page_id}/title")
    def page_title(page_id: int, body: TitleBody) -> dict:
        with lock:
            return store.set_title(con, page_id, body.title)

    @app.post("/api/pages/{page_id}/retry")
    def page_retry(page_id: int) -> dict:
        if runner is None:
            raise store.StoreError(503, "recognition_disabled")
        with lock:
            job_id = store.retry_page(con, page_id)
        runner.submit(job_id)
        return {"ok": True}

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

    @app.get("/api/pages/{page_id}/layout")
    def layout_info(page_id: int) -> dict:
        with lock:
            return store.get_layout(con, page_id)

    @app.get("/api/pages/{page_id}/layout/image")
    def layout_image(page_id: int) -> FileResponse:
        with lock:
            path = store.layout_view_path(con, images, page_id)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/pages/{page_id}/layout/recut")
    def layout_recut(page_id: int, body: RecutBody) -> dict:
        if runner is None:
            raise store.StoreError(503, "recognition_disabled")
        with lock:
            job_id = store.request_recut(con, page_id, body.corners)
        runner.submit(job_id)
        return {"ok": True}

    @app.post("/api/pages/{page_id}/layout/approve")
    def layout_approve(page_id: int) -> dict:
        with lock:
            return store.approve_layout(con, page_id)

    if dev:  # 開発・E2E 専用: 合成した認識結果のジョブを作る(正解つき)。本番の起動では登録しない

        @app.post("/api/dev/fake")
        def dev_fake(body: DevFakeBody) -> dict:
            if not isinstance(backend, FakeBackend):
                raise store.StoreError(409, "not_fake_backend")
            backend.delay, backend.fail_next = body.delay, body.fail_next
            backend.layout_ok = body.layout_ok
            return {"ok": True}

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
    runner: JobRunner | None = app.state.runner
    if runner is not None:
        runner.stop()
    con: sqlite3.Connection = app.state.db
    con.close()
