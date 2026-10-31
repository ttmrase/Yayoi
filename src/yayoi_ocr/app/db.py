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
    # --- v2: 取り込み(M2)。アップロードしたファイル・ページの元画像の位置・認識の失敗
    """
    CREATE TABLE uploads (            -- 受け取ったファイル(アプリのデータ置き場の中。利用者が指したパスは読まない)
        id TEXT PRIMARY KEY,          -- ランダムな ID(ファイル名にも使う。利用者のファイル名は使わない)
        name TEXT NOT NULL,           -- 表示用の元のファイル名
        path TEXT NOT NULL,           -- uploads ディレクトリからの相対パス
        size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        n_pages INTEGER NOT NULL,
        created_at REAL NOT NULL
    );
    ALTER TABLE pages ADD COLUMN upload_id TEXT REFERENCES uploads(id);
    ALTER TABLE pages ADD COLUMN source_index INTEGER NOT NULL DEFAULT 0;  -- アップロード内のページ番号(0 始まり)
    ALTER TABLE pages ADD COLUMN last_year INTEGER;                        -- このページの最後の行の年(次のページが継承する)
    ALTER TABLE pages ADD COLUMN error TEXT;                               -- 認識の失敗(例外の型名だけ。値は入れない)
    ALTER TABLE pages ADD COLUMN recognized_at REAL;
    ALTER TABLE pages ADD COLUMN recognize_seconds REAL;
    """,
    # --- v3: レイアウトの確認・補正(M3)。元画像の縮小版(重ね表示用)・検出した罫線・利用者が指定した表の隅
    """
    ALTER TABLE pages ADD COLUMN view_path TEXT;          -- 確認画面に出す元画像(縮小した JPEG)。images ディレクトリからの相対パス
    ALTER TABLE pages ADD COLUMN view_scale REAL;         -- 表示 px / 元画像 px
    ALTER TABLE pages ADD COLUMN layout_info TEXT;        -- JSON: 元画像の大きさ・検出した表の 4 隅・罫線の折れ線(元画像 px)
    ALTER TABLE pages ADD COLUMN layout_override TEXT;    -- JSON: 利用者が指定した表の 4 隅(元画像 px)。再切り出しで使う
    """,
    # --- v4: ページのタイトル(借方科目の元)・出力ファイル(M4)
    """
    ALTER TABLE pages ADD COLUMN title TEXT;              -- 利用者が確認した(または入力した)ページのタイトル(帳票の上部の勘定科目名)
    ALTER TABLE pages ADD COLUMN title_recognized TEXT;   -- 認識したタイトル(語彙があるときの第 1 候補。無ければ NULL)
    ALTER TABLE exports ADD COLUMN file_name TEXT;        -- アプリのデータ置き場の exports ディレクトリの中のファイル名
    ALTER TABLE exports ADD COLUMN format_label TEXT;
    """,
    # --- v5: ジョブ全体の失敗の型名(runner の想定外の例外。値は入れない)
    """
    ALTER TABLE jobs ADD COLUMN error TEXT;
    """,
    # --- v6: 元画像の保管方針(出力後に削除した印)
    """
    ALTER TABLE uploads ADD COLUMN deleted_at REAL;
    """,
    # --- v7: 出力の行単位の履歴と、出力ファイル(1 回の出力が複数ファイルになり得る)
    """
    CREATE TABLE export_rows (        -- 行ごとの出力済みの記録(同じ行の再出力を警告する)。値は載せない(ハッシュだけ)
        export_id INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
        page_id INTEGER NOT NULL REFERENCES pages(id),
        row INTEGER NOT NULL,
        content_sha256 TEXT NOT NULL,
        PRIMARY KEY (export_id, page_id, row)
    );
    CREATE INDEX idx_export_rows_hash ON export_rows(content_sha256);
    CREATE TABLE export_files (
        export_id INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        name TEXT NOT NULL,           -- exports ディレクトリの中のファイル名
        sha256 TEXT NOT NULL,
        n_rows INTEGER NOT NULL,
        payment TEXT,                 -- スマート取引取込: 支払手段ごとのファイル
        PRIMARY KEY (export_id, idx)
    );
    """,
    # --- v8: 行ごとの税区分(確認画面での上書き)と、摘要ごとの税区分の記憶
    """
    CREATE TABLE row_tax (            -- 行の借方税区分。行が無ければ科目ごとの既定(または記憶からの提案)
        page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
        row INTEGER NOT NULL,
        tax_category TEXT NOT NULL,
        source TEXT NOT NULL,         -- 'user' = 利用者が選んだ / 'suggested' = 確定のときに記憶からの提案を固定した
        PRIMARY KEY (page_id, row)
    );
    CREATE TABLE tax_memory (         -- 摘要(正規化したもの)→ 税区分。プロファイルごと。設定画面で編集・削除できる
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER NOT NULL REFERENCES profiles(id),
        description_key TEXT NOT NULL,
        display TEXT NOT NULL,        -- 画面に出す摘要(最後に使った表記)
        tax_category TEXT NOT NULL,
        n_used INTEGER NOT NULL DEFAULT 1,
        updated_at REAL NOT NULL,
        UNIQUE (profile_id, description_key)
    );
    """,
    # --- v9: Phase 4 の世代(重み + 確信度)の管理と、バックグラウンドの追加学習の実行履歴
    """
    ALTER TABLE generations ADD COLUMN number INTEGER NOT NULL DEFAULT 0;        -- 書き手ごとの通し番号(0 = 出荷時の重み)
    ALTER TABLE generations ADD COLUMN kind TEXT NOT NULL DEFAULT 'trained';     -- shipped / trained
    ALTER TABLE generations ADD COLUMN dir TEXT;                                 -- models/<プロファイル>/gen_XXXX(データ領域からの相対。shipped は NULL)
    ALTER TABLE generations ADD COLUMN meta TEXT;                                -- JSON(元の世代・学習/評価ページの ID・評価の値…。摘要・金額などの値は入れない)
    ALTER TABLE generations ADD COLUMN reject_reason TEXT;                       -- 不採用の理由コード
    ALTER TABLE generations ADD COLUMN activated_at REAL;                        -- 使用中になった時刻
    UPDATE generations SET kind = 'shipped', number = 0, activated_at = created_at;
    UPDATE pages SET generation = COALESCE((SELECT g.id FROM generations g JOIN jobs j ON j.profile_id = g.profile_id
                                             WHERE j.id = pages.job_id AND g.kind = 'shipped'), 0) WHERE generation = 0;
    CREATE TABLE training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER NOT NULL REFERENCES profiles(id),
        trigger TEXT NOT NULL,             -- auto / manual
        status TEXT NOT NULL,              -- running / adopted / rejected / deferred(評価が足りず見送り)/ failed / interrupted
        started_at REAL NOT NULL,
        finished_at REAL,
        n_new_cells INTEGER,
        n_train_pages INTEGER,
        n_eval_pages INTEGER,
        cutoff REAL,                       -- この実行が見たページの最後の確定時刻(次の「新しい確定セル」の起点)
        generation_id INTEGER,             -- 採用・不採用のとき、作った世代
        reason TEXT,                       -- 不採用・見送り・失敗の理由コード(値は入れない)
        stage TEXT,
        progress REAL
    );
    """,
    # --- v10: 世代は (プロファイル, 番号) で一意(並行する学習で同じ番号が二重にできない)。使えなくなった世代の警告を確認済みにできる
    """
    CREATE UNIQUE INDEX ux_generations_profile_number ON generations(profile_id, number);
    ALTER TABLE generations ADD COLUMN ack INTEGER NOT NULL DEFAULT 0;           -- 使えなくなった世代(broken / halted)の警告を利用者が確認した
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
            "INSERT INTO generations (profile_id, created_at, status, kind, number, activated_at)"
            " VALUES (?, ?, 'active', 'shipped', 0, ?)",
            (DEFAULT_PROFILE_ID, now, now),
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
