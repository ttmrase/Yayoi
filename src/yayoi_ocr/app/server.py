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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__ as app_version
from . import db, exports, gens, metrics, monitor, retention, rowtax, store, tax, uploads
from . import settings as app_settings
from .paths import AppPaths
from .runner import JobRunner
from .security import SecurityMiddleware
from .trainers import MockTrainer, SubprocessTrainer, Trainer
from .training import TrainManager
from .worker import Backend, FakeBackend, RealBackend

STATIC_DIR = Path(__file__).parent / "static"

FALLBACK_HTML = """<!doctype html><html lang="ja"><meta charset="utf-8"><title>yayoi-ocr</title>
<body><p>画面(フロントエンド)がビルドされていません。<code>web/</code> で <code>npm ci && npm run build</code> を実行してください。</p></body></html>"""


class CellPatch(BaseModel):
    action: Literal["pass", "edit", "confirm", "candidate", "revert"]
    value: str | None = Field(default=None, max_length=400)
    candidate_index: int | None = None


class DevTrainerBody(BaseModel):
    mode: Literal[
        "improve",
        "worsen",
        "flat",
        "fail",
        "amount_miss",
        "overconfident",
        "calib_worse",
        "suggest",
        "promote_prev",
    ] = "improve"
    delay: float = Field(default=0.0, ge=0.0, le=30.0)
    auto: bool | None = (
        None  # False: 設定 auto_train が有効でも自動では始めない(共有サーバの E2E 用)
    )


class GenerationResetBody(BaseModel):
    delete_learning_data: bool = (
        False  # True: 切り出し画像・学習状態も削除する(False: 確定データは残す)
    )


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


class RowTaxBody(BaseModel):
    tax_category: str | None = Field(default=None, max_length=40)  # None = 行の指定を外す


class MemoryBody(BaseModel):
    tax_category: str = Field(max_length=40)


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
    trainer: Trainer | None = None,
) -> FastAPI:
    """`backend` が None のときは認識を行わない(ジョブの作成は 503。M0・M1 のテストと合成ジョブだけ)。
    `trainer` が None のときは、`backend` に合わせて選ぶ(合成の認識 → MockTrainer、本物の認識 → 別プロセスの学習器、認識なし → 追加学習なし)。"""
    db_path = Path(db_path)
    images = images_dir if images_dir is not None else AppPaths(db_path.parent).images
    images.mkdir(parents=True, exist_ok=True)
    uploads_root = uploads_dir if uploads_dir is not None else AppPaths(db_path.parent).uploads
    uploads_root.mkdir(parents=True, exist_ok=True)
    exports_root = AppPaths(db_path.parent).exports
    exports_root.mkdir(parents=True, exist_ok=True)
    models_root = AppPaths(db_path.parent).models  # 利用者のデータから作った世代の置き場
    models_root.mkdir(parents=True, exist_ok=True)
    con = db.connect(db_path)
    db.migrate(con)
    lock = threading.RLock()  # 1 接続を複数スレッドで共有するので直列化する

    app = FastAPI(title="yayoi-ocr", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db = con
    app.state.db_lock = lock
    paths = AppPaths(db_path.parent)
    paths.pagedata.mkdir(parents=True, exist_ok=True)
    runner = (
        JobRunner(con, lock, images, uploads_root, backend, models_root, paths.pagedata)
        if backend is not None
        else None
    )
    if trainer is None:
        trainer = (
            MockTrainer()
            if isinstance(backend, FakeBackend)
            else SubprocessTrainer()
            if isinstance(backend, RealBackend)
            else None
        )
    trainer_mgr = TrainManager(con, lock, paths, trainer)
    trainer_mgr.mark_interrupted()  # 前回、学習の途中でアプリが閉じていたら「中断」にする
    app.state.trainer = trainer_mgr
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
        return {
            "settings": cur.model_dump(),
            "missing": app_settings.readiness(cur),
            "tax_options": tax.all_options(),
        }

    @app.put("/api/settings")
    def settings_put(body: app_settings.Settings) -> dict:
        with lock:
            app_settings.save(con, body)
        return {
            "settings": body.model_dump(),
            "missing": app_settings.readiness(body),
            "tax_options": tax.all_options(),
        }

    @app.get("/api/learning")
    def learning(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        with lock:
            rows = metrics.page_metrics(con, profile_id)
            observed = metrics.by_generation(rows)
            glist = gens.list_generations(con, models_root, profile_id)
            for g in glist:
                g["observed"] = observed.get(g["id"])  # この世代で認識して確定したページの実績
            return {
                "profile_id": profile_id,
                "generations": glist,
                "active_id": next((g["id"] for g in glist if g["status"] == "active"), None),
                "notices": gens.notices(con, profile_id),
                "suggest_effect": metrics.suggest_effect(con, profile_id),
                "suggest_mode": app_settings.load(con).suggest_mode,
                "pages": rows,
                "summary": metrics.summary(rows),
                # 戻せるのは、重みが残っている使用中でない世代(世代 0 を含む)
                "can_rollback": any(g["restorable"] and g["status"] != "active" for g in glist),
                "can_reset": any(r["n_text"] for r in rows)
                or any(g["kind"] == "trained" and g["status"] in gens.RESTORABLE for g in glist),
                "phase": 4,
            }

    @app.get("/api/generations")
    def generations_list(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        with lock:
            glist = gens.list_generations(con, models_root, profile_id)
        return {
            "generations": glist,
            "active_id": next((g["id"] for g in glist if g["status"] == "active"), None),
        }

    @app.post("/api/generations/reset")
    def generations_reset(
        body: GenerationResetBody, profile_id: int = db.DEFAULT_PROFILE_ID
    ) -> dict:
        """出荷時の重み(世代 0)に戻す。利用者のデータから作った世代の重みは消す。確定した値・出力の履歴・設定は残る。
        `delete_learning_data` が真なら、切り出し画像・学習状態も削除する(偽なら、確定データは次の追加学習の元として残す)。"""
        if trainer_mgr.running:  # 学習中は重みも学習データも触れない(止めてから)
            raise store.StoreError(409, "training_running")
        with lock:
            discarded = gens.reset(con, models_root, profile_id)
        out: dict = {"discarded": discarded, "deleted_images": 0}
        if body.delete_learning_data:
            out.update(retention.reset_learning(con, lock, images, pagedata_dir=paths.pagedata))
        return out

    @app.post("/api/generations/{gen_id}/activate")
    def generations_activate(gen_id: int) -> dict:
        """この世代に戻す(ロールバック)。以降に認識するページから使う。すでに認識したページは変わらない。"""
        with lock:
            return gens.activate(con, models_root, gen_id)

    @app.get("/api/notices")
    def notices_list(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        """利用者に知らせること(使えなくなった世代: 重みと確信度の組が壊れた / 安全のため停止した)。確認(ack)するまで画面の上部に出す。"""
        with lock:
            return {"notices": gens.notices(con, profile_id)}

    @app.post("/api/notices/{gen_id}/ack")
    def notices_ack(gen_id: int) -> dict:
        with lock:
            gens.acknowledge(con, gen_id)
        return {"ok": True}

    @app.get("/api/training")
    def training_status(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        return trainer_mgr.status(profile_id)

    @app.post("/api/training/start")
    def training_start(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        """今すぐ学習する(手動。しきい値は見ない)。始められなければ理由のコード(training_running / not_enough_pages …)。"""
        return {"run_id": trainer_mgr.start(profile_id, "manual")}

    @app.post("/api/training/cancel")
    def training_cancel() -> dict:
        return {"cancelled": trainer_mgr.cancel()}

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
        if trainer_mgr.running:
            raise store.StoreError(409, "training_running")
        return retention.reset_learning(con, lock, images, pagedata_dir=paths.pagedata)

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

    @app.get("/api/exports/sample")
    def exports_sample(format: str = exports.DEFAULT_FORMAT, date: str = "") -> Response:
        """弥生の取り込み試験用の架空の 2 行(設定のとおりの科目・補助科目・税区分)。履歴には残さない。"""
        with lock:
            of = exports.sample_file(con, format, date or time.strftime("%Y/%m/%d"))
        return Response(
            of.data,
            media_type="text/csv; charset=Shift_JIS",
            headers={"Content-Disposition": f'attachment; filename="{of.name}"'},
        )

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

    @app.put("/api/pages/{page_id}/rows/{row}/tax")
    def row_tax(page_id: int, row: int, body: RowTaxBody) -> dict:
        with lock:
            rowtax.set_row_tax(con, page_id, row, body.tax_category)
            return store.get_page(con, page_id)

    @app.get("/api/tax-memory")
    def tax_memory(profile_id: int = db.DEFAULT_PROFILE_ID) -> dict:
        with lock:
            return rowtax.memory_list(con, profile_id)

    @app.put("/api/tax-memory/{mem_id}")
    def tax_memory_put(mem_id: int, body: MemoryBody) -> dict:
        with lock:
            rowtax.memory_update(con, mem_id, body.tax_category)
            return rowtax.memory_list(con)

    @app.delete("/api/tax-memory/{mem_id}")
    def tax_memory_delete(mem_id: int) -> dict:
        with lock:
            rowtax.memory_delete(con, mem_id)
            return rowtax.memory_list(con)

    @app.post("/api/tax-memory/clear")
    def tax_memory_clear() -> dict:
        with lock:
            rowtax.memory_clear(con)
            return rowtax.memory_list(con)

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
            out = store.confirm_page(con, page_id, body.review_seconds)
            # 採用した世代で認識したページなら、観測した見逃しを数えて、基準に当たれば自動で親の世代に戻す
            halted = monitor.check_after_confirm(con, models_root, page_id)
        if halted is not None:
            out["generation_halted"] = {"number": halted["number"], "code": halted["reject_reason"]}
        trainer_mgr.notify_confirmed()  # 新しい確定セルがたまっていれば、バックグラウンドの追加学習を始める
        return out

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

        @app.post("/api/dev/trainer")
        def dev_trainer(body: DevTrainerBody) -> dict:
            if not isinstance(trainer, MockTrainer):
                raise store.StoreError(409, "not_mock_trainer")
            trainer.mode, trainer.delay = body.mode, body.delay
            if body.auto is not None:
                trainer_mgr.auto_allowed = body.auto
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
    app.state.trainer.close()  # 学習中なら止める(結果は使わない。次の起動で「中断」と記録される)
    runner: JobRunner | None = app.state.runner
    if runner is not None:
        runner.stop()
    con: sqlite3.Connection = app.state.db
    con.close()
