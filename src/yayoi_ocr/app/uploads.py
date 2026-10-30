"""ファイルの受け取り(アップロード API)。

利用者が指したパスをサーバが読む入口は作らない(任意ファイルの読み出しを防ぐ)。画面がファイルの中身を送り、
サーバはアプリのデータ置き場の `uploads/<ランダムな ID>/` に保存する。利用者のファイル名は表示用に DB に持つだけで、
パスには使わない。保存したあとでページ数を数え、読めないファイルはここで断る。
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath, PureWindowsPath

from starlette.concurrency import run_in_threadpool

from ..ingest import SUPPORTED_SUFFIXES, page_count
from .db import transaction
from .store import StoreError

MAX_UPLOAD_BYTES = 400 * 1024 * 1024
MAX_PAGES_PER_UPLOAD = 300
MAX_JOB_PAGES = 600


def display_name(raw: str) -> str:
    """表示用のファイル名(ディレクトリ部分を落とし、長さを制限する)。パスには使わない。"""
    name = PureWindowsPath(PurePosixPath(raw).name).name.strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:200] or "upload"


async def save_upload(
    con: sqlite3.Connection,
    lock: threading.RLock,
    uploads_dir: Path,
    raw_name: str,
    body: AsyncIterator[bytes],
) -> dict:
    name = display_name(raw_name)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise StoreError(415, "unsupported_type", {"supported": sorted(SUPPORTED_SUFFIXES)})
    upload_id = uuid.uuid4().hex
    folder = uploads_dir / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"original{suffix}"
    h = hashlib.sha256()
    size = 0
    try:
        with open(path, "wb") as fh:
            async for chunk in body:
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise StoreError(413, "too_large", {"max_bytes": MAX_UPLOAD_BYTES})
                h.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise StoreError(422, "empty_file")
        await run_in_threadpool(
            _register, con, lock, uploads_dir, path, upload_id, name, size, h.hexdigest()
        )
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    with lock:
        n_pages = int(
            con.execute("SELECT n_pages FROM uploads WHERE id = ?", (upload_id,)).fetchone()[0]
        )
    return {"id": upload_id, "name": name, "size": size, "n_pages": n_pages}


def _register(
    con: sqlite3.Connection,
    lock: threading.RLock,
    uploads_dir: Path,
    path: Path,
    upload_id: str,
    name: str,
    size: int,
    sha256: str,
) -> None:
    try:
        n_pages = page_count(path)
    except Exception as exc:
        raise StoreError(422, "unreadable_file") from exc
    if not 1 <= n_pages <= MAX_PAGES_PER_UPLOAD:
        raise StoreError(422, "bad_page_count", {"max_pages": MAX_PAGES_PER_UPLOAD})
    with lock, transaction(con):
        con.execute(
            "INSERT INTO uploads (id, name, path, size, sha256, n_pages, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                upload_id,
                name,
                str(path.relative_to(uploads_dir)),
                size,
                sha256,
                n_pages,
                time.time(),
            ),
        )


def delete_upload(
    con: sqlite3.Connection, lock: threading.RLock, uploads_dir: Path, upload_id: str
) -> None:
    """まだジョブに使われていないアップロードを捨てる(取り消し)。"""
    with lock, transaction(con):
        r = con.execute("SELECT path FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if r is None:
            raise StoreError(404, "upload_not_found")
        used = con.execute(
            "SELECT 1 FROM pages WHERE upload_id = ? LIMIT 1", (upload_id,)
        ).fetchone()
        if used is not None:
            raise StoreError(409, "upload_in_use")
        con.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    shutil.rmtree(uploads_dir / upload_id, ignore_errors=True)


def pages_for_job(con: sqlite3.Connection, upload_ids: list[str]) -> list[tuple[str, int]]:
    """ジョブのページ(アップロードの順 × ファイル内のページ順)。"""
    out: list[tuple[str, int]] = []
    for uid in upload_ids:
        r = con.execute("SELECT n_pages FROM uploads WHERE id = ?", (uid,)).fetchone()
        if r is None:
            raise StoreError(422, "upload_not_found")
        out.extend((uid, i) for i in range(int(r["n_pages"])))
    if len(out) > MAX_JOB_PAGES:
        raise StoreError(422, "too_many_pages", {"max_pages": MAX_JOB_PAGES})
    return out
