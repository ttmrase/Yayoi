"""ファイルの受け取り(アップロード API)。

利用者が指したパスをサーバが読む入口は作らない(任意ファイルの読み出しを防ぐ)。画面がファイルの中身を送り、
サーバはアプリのデータ置き場の `uploads/<ランダムな ID>/` に保存する。利用者のファイル名は表示用に DB に持つだけで、
パスには使わない。保存したあとで、ページ数と**全ページの寸法**を調べ、読めないファイル・大きすぎるページはここで断る。
**この検査(PDF・画像のデコーダ)は、使い捨ての別プロセスで、タイムアウトつきで実行する**(`inspect_file`)。トークンと DB を持つサーバのプロセスでは、
利用者が用意したファイルをデコードしない(壊れた/悪意あるファイルで暴走・クラッシュしても本体は影響を受けない)。
どのジョブにも使われなかったアップロードは、起動時に一定時間(既定 24 時間)を過ぎていれば削除する(`purge_unused`)。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath, PureWindowsPath

from starlette.concurrency import run_in_threadpool

from ..ingest import SUPPORTED_SUFFIXES
from .db import transaction
from .store import StoreError

MAX_UPLOAD_BYTES = 400 * 1024 * 1024
MAX_PAGES_PER_UPLOAD = 300
MAX_JOB_PAGES = 600
PROBE_TIMEOUT_S = float(
    os.environ.get("YAYOI_PROBE_TIMEOUT", "30")
)  # テスト用に環境変数で変えられる
UNUSED_UPLOAD_MAX_AGE_S = 24 * 3600  # どのジョブにも使われなかったアップロードを残す時間


def display_name(raw: str) -> str:
    """表示用のファイル名(ディレクトリ部分を落とし、長さを制限する)。パスには使わない。"""
    name = PureWindowsPath(PurePosixPath(raw).name).name.strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:200] or "upload"


def inspect_file(path: Path, timeout: float | None = None) -> dict:
    """ページ数と全ページの寸法を、使い捨ての別プロセスで調べる(`probe.py`)。失敗・タイムアウトは StoreError(422)。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "yayoi_ocr.app.probe", str(path)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S if timeout is None else timeout,
            cwd=str(path.parent),
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # run が子プロセスを kill 済み
        raise StoreError(422, "inspect_timeout") from exc
    except OSError as exc:
        raise StoreError(500, "inspect_unavailable") from exc
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except ValueError:
        info = {}
    if proc.returncode != 0 or "n_pages" not in info:
        raise StoreError(422, "unreadable_file", {"reason": str(info.get("error", "crashed"))[:40]})
    return info


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
                # 書き込み・ハッシュはスレッドプールで(イベントループを止めない)
                await run_in_threadpool(_write_chunk, fh, h, chunk)
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


def _write_chunk(fh, h, chunk: bytes) -> None:
    h.update(chunk)
    fh.write(chunk)


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
    n_pages = int(inspect_file(path)["n_pages"])
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


def _inside(root: Path, p: Path) -> bool:
    try:
        return root.resolve() in p.resolve().parents
    except OSError:
        return False


def purge_unused(
    con: sqlite3.Connection,
    lock: threading.RLock,
    uploads_dir: Path,
    max_age_s: float = UNUSED_UPLOAD_MAX_AGE_S,
    now: float | None = None,
) -> int:
    """どのジョブにも使われず、`max_age_s` を過ぎたアップロード(と、DB に無い孤児のフォルダ)を削除する。削除した件数を返す。
    **データ領域(uploads ディレクトリ)の中だけ**を対象にする。"""
    now = time.time() if now is None else now
    n = 0
    with lock, transaction(con):
        rows = con.execute(
            "SELECT id FROM uploads u WHERE created_at < ?"
            " AND NOT EXISTS (SELECT 1 FROM pages p WHERE p.upload_id = u.id)",
            (now - max_age_s,),
        ).fetchall()
        for r in rows:
            con.execute("DELETE FROM uploads WHERE id = ?", (r["id"],))
    for r in rows:
        d = uploads_dir / r["id"]
        if _inside(uploads_dir, d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    with lock:
        known = {r["id"] for r in con.execute("SELECT id FROM uploads")}
    if uploads_dir.is_dir():
        for d in uploads_dir.iterdir():  # DB に無い孤児(途中で終わったアップロードなど)
            if d.is_dir() and d.name not in known and _inside(uploads_dir, d):
                try:
                    old = d.stat().st_mtime < now - max_age_s
                except OSError:
                    continue
                if old:
                    shutil.rmtree(d, ignore_errors=True)
                    n += 1
    return n
