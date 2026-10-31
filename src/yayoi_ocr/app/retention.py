"""実データ(取り込んだ元画像・切り出し画像)の保管と削除の方針。**削除はデータ領域(uploads / images ディレクトリ)の中だけ**を対象にする。

- どのジョブにも使われなかったアップロード: 起動時に 24 時間を過ぎていれば削除(`uploads.purge_unused`)。
- ジョブの元画像: 設定「元画像の保管」(`keep` が既定 / `delete_after_export`)。後者は、そのファイルの全ページが出力済みになったら、元ファイルと
  確認画面用の縮小画像(`view.jpg`)を削除する(以降、そのページの罫線の再切り出しはできない)。
- 確定したセルの切り出し画像は学習データ: 学習画面の「リセット」で削除できる(`reset_learning`)。確定した値・出力の履歴は残す。
ログに値・ファイル名は出さない。
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from pathlib import Path

from .db import transaction

KEY_LEARNED_SINCE = "learned_since"


def _inside(root: Path, p: Path) -> bool:
    try:
        return root.resolve() in p.resolve().parents
    except OSError:
        return False


def _unlink(root: Path, rel: str | None) -> bool:
    if not rel:
        return False
    p = root / rel
    if not _inside(root, p):
        return False
    try:
        p.unlink()
    except FileNotFoundError:
        return False
    return True


def delete_originals(
    con: sqlite3.Connection,
    lock: threading.RLock,
    uploads_dir: Path,
    images_dir: Path,
    exported_page_ids: list[int],
) -> int:
    """出力したページを含むアップロードのうち、**全ページが出力済み**のものの元ファイルと縮小画像を削除する。削除したアップロードの数を返す。"""
    if not exported_page_ids:
        return 0
    n = 0
    with lock:
        uploads = {
            r["upload_id"]
            for r in con.execute(
                f"SELECT DISTINCT upload_id FROM pages WHERE id IN ({','.join('?' * len(exported_page_ids))})",
                exported_page_ids,
            )
            if r["upload_id"]
        }
        for uid in uploads:
            row = con.execute(
                "SELECT path, deleted_at FROM uploads WHERE id = ?", (uid,)
            ).fetchone()
            if row is None or row["deleted_at"] is not None:
                continue
            pending = con.execute(
                "SELECT COUNT(*) FROM pages p WHERE p.upload_id = ? AND NOT EXISTS"
                " (SELECT 1 FROM export_pages e WHERE e.page_id = p.id)",
                (uid,),
            ).fetchone()[0]
            if pending:
                continue
            views = [
                r["view_path"]
                for r in con.execute(
                    "SELECT view_path FROM pages WHERE upload_id = ? AND view_path IS NOT NULL",
                    (uid,),
                )
            ]
            with transaction(con):
                con.execute("UPDATE uploads SET deleted_at = ? WHERE id = ?", (time.time(), uid))
                con.execute(
                    "UPDATE pages SET view_path = NULL, layout_info = NULL WHERE upload_id = ?",
                    (uid,),
                )
            d = uploads_dir / uid
            if _inside(uploads_dir, d):
                shutil.rmtree(d, ignore_errors=True)
            for v in views:
                _unlink(images_dir, v)
            n += 1
    return n


def reset_learning(
    con: sqlite3.Connection, lock: threading.RLock, images_dir: Path, now: float | None = None
) -> dict:
    """学習データの削除: 確定したページのセルの切り出し画像を削除し、学習状態(辞書・科目の語彙・金額の履歴)を作り直しの起点まで戻す。
    確定した値・出力の履歴・設定は残す。"""
    now = time.time() if now is None else now
    n_images = 0
    with lock:
        rows = con.execute(
            "SELECT c.id, c.crop_path FROM cells c JOIN pages p ON p.id = c.page_id"
            " WHERE p.status = 'confirmed' AND c.crop_path IS NOT NULL"
        ).fetchall()
        with transaction(con):
            con.execute(
                "UPDATE cells SET crop_path = NULL WHERE id IN"
                " (SELECT c.id FROM cells c JOIN pages p ON p.id = c.page_id WHERE p.status = 'confirmed')"
            )
            con.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (KEY_LEARNED_SINCE, repr(now)),
            )
    for r in rows:
        n_images += _unlink(images_dir, r["crop_path"])
    return {"deleted_images": n_images, "learned_since": now}


def learned_since(con: sqlite3.Connection) -> float:
    r = con.execute("SELECT value FROM settings WHERE key = ?", (KEY_LEARNED_SINCE,)).fetchone()
    try:
        return float(r["value"]) if r else 0.0
    except ValueError:
        return 0.0
