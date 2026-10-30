"""SQLite のスキーマ(docs/phase3_design.md §2)。実データの値を含むので、置き場所はリポジトリの外(`paths.app_data_dir`)。

スキーマのバージョンは `PRAGMA user_version`。新しい版は `MIGRATIONS` に SQL を足す(古い DB は起動時に順に移行する)。
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_PROFILE_ID = 1
DEFAULT_PROFILE_NAME = "共通"

MIGRATIONS: list[str] = [
    # --- v1
    """
    CREATE TABLE profiles (          -- 書き手のプロファイル(既定は「共通」1 つ。世代・メモリ・較正は Phase 4 でプロファイルごとに持つ)
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created_at REAL NOT NULL
    );
    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        profile_id INTEGER NOT NULL REFERENCES profiles(id),
        status TEXT NOT NULL DEFAULT 'processing',   -- processing / ready / done
        created_at REAL NOT NULL,
        last_page_id INTEGER, last_cell INTEGER      -- 中断・再開の位置
    );
    CREATE TABLE pages (
        id INTEGER PRIMARY KEY,
        job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        image_sha256 TEXT,
        image_path TEXT,
        status TEXT NOT NULL DEFAULT 'processing',   -- processing / review / reviewing / confirmed
        layout_status TEXT NOT NULL DEFAULT 'ok',
        layout_reasons TEXT NOT NULL DEFAULT '[]',
        layout_approved INTEGER NOT NULL DEFAULT 0,
        generation INTEGER NOT NULL DEFAULT 0,       -- 認識に使った重みの世代
        review_seconds REAL NOT NULL DEFAULT 0,      -- 確認にかかった時間(指標。値の内容は記録しない)
        confirmed_at REAL,
        revision INTEGER NOT NULL DEFAULT 0,
        UNIQUE (job_id, idx)
    );
    CREATE TABLE cells (
        id INTEGER PRIMARY KEY,
        page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
        row INTEGER NOT NULL, col INTEGER NOT NULL,
        column_kind TEXT NOT NULL,                   -- date / account / description / amount
        kind TEXT NOT NULL,                          -- blank / text / ditto / arrow / wave
        recognized_text TEXT NOT NULL DEFAULT '',    -- R1 の第 1 候補
        resolved_value TEXT NOT NULL DEFAULT '',     -- 認識が出した展開後の値
        confirmed_value TEXT,                        -- 確定値(未確認は NULL)
        edit_kind TEXT NOT NULL DEFAULT 'none',      -- none / unchanged / corrected / candidate / expanded_edit
        expansion TEXT,                              -- JSON: {kind: ditto|year|arrow|wave, source: [row, col]}
        reasons TEXT NOT NULL DEFAULT '[]',          -- 要確認の理由コード
        candidates TEXT NOT NULL DEFAULT '[]',       -- JSON: [[値, 事後確率], ...]
        review INTEGER NOT NULL DEFAULT 0,           -- 要確認フラグ
        passed INTEGER NOT NULL DEFAULT 0,           -- 通過済み(フォーカス + 画像表示済みが一定時間)
        crop_path TEXT,
        UNIQUE (page_id, row, col)
    );
    CREATE TABLE edits (                             -- 元に戻す・やり直し用
        id INTEGER PRIMARY KEY,
        page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
        cell_id INTEGER NOT NULL REFERENCES cells(id) ON DELETE CASCADE,
        before_value TEXT, after_value TEXT,
        before_edit_kind TEXT, after_edit_kind TEXT,
        undone INTEGER NOT NULL DEFAULT 0,
        ts REAL NOT NULL
    );
    CREATE TABLE exports (
        id INTEGER PRIMARY KEY,
        format TEXT NOT NULL,
        created_at REAL NOT NULL,
        n_pages INTEGER NOT NULL, n_rows INTEGER NOT NULL,
        file_sha256 TEXT,
        settings_snapshot TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE export_pages (                      -- 重複出力の警告用(ページ画像 + 確定値のハッシュ)
        export_id INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
        page_id INTEGER NOT NULL REFERENCES pages(id),
        content_sha256 TEXT NOT NULL,
        PRIMARY KEY (export_id, page_id)
    );
    CREATE TABLE generations (                       -- Phase 4 の重みの世代(Phase 3 は世代 0 だけ)
        id INTEGER PRIMARY KEY,
        profile_id INTEGER NOT NULL REFERENCES profiles(id),
        parent_id INTEGER,
        created_at REAL NOT NULL,
        n_pages INTEGER NOT NULL DEFAULT 0,
        score REAL,
        status TEXT NOT NULL DEFAULT 'active'        -- active / retired / rejected
    );
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE INDEX idx_pages_job ON pages(job_id, idx);
    CREATE INDEX idx_cells_page ON cells(page_id, row, col);
    CREATE INDEX idx_edits_page ON edits(page_id, id);
    """,
]


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(
        str(path), check_same_thread=False, isolation_level=None
    )  # 明示的にトランザクション
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def migrate(con: sqlite3.Connection) -> int:
    """DB を最新のスキーマにして、版を返す。既定のプロファイル(「共通」)と世代 0 を作る。"""
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        raise RuntimeError(
            f"DB のスキーマ版 {version} がこのアプリより新しい(アプリを更新してください)"
        )
    for i in range(version, len(MIGRATIONS)):
        con.executescript("BEGIN;" + MIGRATIONS[i] + f"PRAGMA user_version = {i + 1};COMMIT;")
    if con.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0:
        now = time.time()
        con.execute(
            "INSERT INTO profiles (id, name, created_at) VALUES (?, ?, ?)",
            (DEFAULT_PROFILE_ID, DEFAULT_PROFILE_NAME, now),
        )
        con.execute(
            "INSERT INTO generations (profile_id, created_at, status) VALUES (?, ?, 'active')",
            (DEFAULT_PROFILE_ID, now),
        )
    return con.execute("PRAGMA user_version").fetchone()[0]


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")
