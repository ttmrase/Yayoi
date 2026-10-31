"""世代(重み + 確信度)の管理: 一覧・使用中の世代・ロールバック・リセット。設計は `docs/phase4_design.md`。

- 世代 0(`kind = 'shipped'`)は出荷時の重み(配布物の `models/`)。ディレクトリを持たない(`dir` は NULL)。
- 世代 1 以降(`kind = 'trained'`)は、`<データ領域>/models/<プロファイル ID>/gen_XXXX/` に重み(幅 256・幅 384)・`confidence.json`・`meta.json` を持つ。
- 使用中(`active`)の世代は書き手(プロファイル)ごとに 1 つ。新しく認識するページはそれで認識し、`pages.generation` に世代の ID を記録する。
- 状態: active(使用中)/ retired(過去。重みがあり、戻せる)/ rejected(不採用。重みは消す)/ discarded(リセットで捨てた。重みは消す)。
- メタデータ(`meta`)には、ページ ID・件数・割合・理由コードだけを入れる。摘要・金額などの値は入れない。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .db import DEFAULT_PROFILE_ID, transaction
from .store import StoreError

WEIGHT_FILES = ("parseq-ndl-24x256-30-tiny.pt", "parseq-ndl-24x384-50-tiny.pt")
CONFIDENCE_FILE = "confidence.json"
META_FILE = "meta.json"
RESTORABLE = ("active", "retired")


@dataclass(frozen=True)
class Weights:
    """認識に使う重み。`dir` が None なら出荷時の重み(既定の選び方)。"""

    generation_id: int
    number: int
    dir: Path | None = None
    confidence: Path | None = None


def gen_dir_name(number: int) -> str:
    return f"gen_{number:04d}"


def shipped(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> sqlite3.Row:
    r = con.execute(
        "SELECT * FROM generations WHERE profile_id = ? AND kind = 'shipped' ORDER BY id LIMIT 1",
        (profile_id,),
    ).fetchone()
    if r is None:
        raise StoreError(404, "generation_not_found")
    return r


def active(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> sqlite3.Row:
    r = con.execute(
        "SELECT * FROM generations WHERE profile_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
        (profile_id,),
    ).fetchone()
    return r if r is not None else shipped(con, profile_id)


def _has_weights(models_root: Path, row: sqlite3.Row) -> bool:
    if row["kind"] == "shipped":
        return True
    if not row["dir"]:
        return False
    d = models_root.parent / row["dir"]
    return all((d / f).is_file() for f in WEIGHT_FILES)


def weights_for(
    con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID
) -> Weights:
    """新しく認識するページに使う重み。使用中の世代の重みが見つからなければ、出荷時の重みに戻す(認識を止めない)。"""
    row = active(con, profile_id)
    if row["kind"] == "shipped" or not _has_weights(models_root, row):
        s = shipped(con, profile_id)
        return Weights(s["id"], 0)
    d = models_root.parent / row["dir"]
    conf = d / CONFIDENCE_FILE
    return Weights(row["id"], row["number"], d, conf if conf.is_file() else None)


def _dict(models_root: Path, r: sqlite3.Row, n_recognized: int | None = None) -> dict:
    meta = json.loads(r["meta"]) if r["meta"] else {}
    return {
        "id": r["id"],
        "number": r["number"],
        "kind": r["kind"],
        "parent_id": r["parent_id"],
        "created_at": r["created_at"],
        "activated_at": r["activated_at"],
        "status": r["status"],
        "reject_reason": r["reject_reason"],
        "n_train_pages": meta.get("n_train_pages", r["n_pages"]),
        "n_eval_pages": meta.get("n_eval_pages"),
        "eval": meta.get(
            "eval"
        ),  # {parent: {...}, candidate: {...}} 正解率・要確認率・見逃し率・評価セル数
        "recalibration": meta.get("recalibration"),
        "restorable": r["status"] in RESTORABLE and _has_weights(models_root, r),
        "n_recognized": n_recognized,
    }


def list_generations(
    con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID
) -> list[dict]:
    rows = con.execute(
        "SELECT g.*, (SELECT COUNT(*) FROM pages p JOIN jobs j ON j.id = p.job_id"
        "   WHERE j.profile_id = g.profile_id AND p.generation = g.id) AS n_recognized"
        " FROM generations g WHERE g.profile_id = ? ORDER BY g.number, g.id",
        (profile_id,),
    ).fetchall()
    return [_dict(models_root, r, r["n_recognized"]) for r in rows]


def activate(con: sqlite3.Connection, models_root: Path, gen_id: int) -> dict:
    """この世代を使用中にする(ロールバック)。重みが残っている世代(過去の世代・出荷時)にだけできる。以降に認識するページから使う。"""
    with transaction(con):
        r = con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        if r is None:
            raise StoreError(404, "generation_not_found")
        if r["status"] not in RESTORABLE:
            raise StoreError(409, "generation_not_restorable", {"status": r["status"]})
        if not _has_weights(models_root, r):
            raise StoreError(409, "generation_weights_missing")
        now = time.time()
        con.execute(
            "UPDATE generations SET status = 'retired' WHERE profile_id = ? AND status = 'active' AND id != ?",
            (r["profile_id"], gen_id),
        )
        if r["status"] != "active":
            con.execute(
                "UPDATE generations SET status = 'active', activated_at = ? WHERE id = ?",
                (now, gen_id),
            )
    return _dict(
        models_root, con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    )


def _remove_dir(models_root: Path, r: sqlite3.Row) -> None:
    if not r["dir"]:
        return
    d = (models_root.parent / r["dir"]).resolve()
    if models_root.resolve() in d.parents and d.is_dir():  # データ領域の models の中だけを消す
        shutil.rmtree(d, ignore_errors=True)


def reset(con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID) -> int:
    """出荷時の重みに戻す: 世代 0 を使用中にし、利用者のデータから作った世代の重みを消す(行は「リセットで捨てた」として残す)。消した世代数を返す。"""
    n = 0
    with transaction(con):
        s = shipped(con, profile_id)
        now = time.time()
        rows = con.execute(
            "SELECT * FROM generations WHERE profile_id = ? AND kind = 'trained' AND status IN ('active', 'retired')",
            (profile_id,),
        ).fetchall()
        for r in rows:
            _remove_dir(models_root, r)
            con.execute("UPDATE generations SET status = 'discarded' WHERE id = ?", (r["id"],))
            n += 1
        con.execute(
            "UPDATE generations SET status = 'active', activated_at = ? WHERE id = ?",
            (now, s["id"]),
        )
    return n


def next_number(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> int:
    return int(
        con.execute(
            "SELECT COALESCE(MAX(number), 0) + 1 FROM generations WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()[0]
    )


def register(
    con: sqlite3.Connection,
    models_root: Path,
    *,
    profile_id: int,
    number: int,
    parent_id: int | None,
    meta: dict,
    adopt: bool,
    reject_reason: str | None = None,
    keep_dir: bool = True,
) -> int:
    """学習で作った世代を登録する。採用なら使用中にし、前の世代は過去(retired)にする。不採用なら rejected で行だけ残し、重みは消す。"""
    rel = f"{models_root.name}/{profile_id}/{gen_dir_name(number)}"
    with transaction(con):
        now = time.time()
        cur = con.execute(
            "INSERT INTO generations (profile_id, parent_id, created_at, n_pages, score, status, number, kind, dir, meta,"
            " reject_reason, activated_at) VALUES (?, ?, ?, ?, NULL, ?, ?, 'trained', ?, ?, ?, ?)",
            (
                profile_id,
                parent_id,
                now,
                int(meta.get("n_train_pages", 0)),
                "active" if adopt else "rejected",
                number,
                rel if keep_dir else None,
                json.dumps(meta, ensure_ascii=False),
                None if adopt else reject_reason,
                now if adopt else None,
            ),
        )
        gid = int(cur.lastrowid)
        if adopt:
            con.execute(
                "UPDATE generations SET status = 'retired' WHERE profile_id = ? AND status = 'active' AND id != ?",
                (profile_id, gid),
            )
    return gid
