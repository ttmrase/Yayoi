"""ジョブ・ページ・セルの読み書き(確認画面の状態遷移)。サーバの API はこの関数を呼ぶだけ。

セルの状態: 未確認 → 通過(`passed`)→ 確認済み/修正済み(`edit_kind`)。確定値 `confirmed_value` が NULL のセルは、
利用者がまだ値を確定していない(表示は認識値)。
元に戻す・やり直しは `edits`(ページ内で共通の履歴)。新しい編集をしたらやり直しの履歴は捨てる。
実データの値はログに出さない(ここではログ出力をしない)。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db import transaction
from .validate import ERROR_LABELS, validate_value

EDIT_KINDS = ("none", "unchanged", "corrected", "candidate", "expanded_edit")
R_AMBIGUITY = "dash_digit_ambiguity"
R_DITTO_SOURCE_CHANGED = "ditto_source_changed"


class StoreError(Exception):
    def __init__(self, status: int, code: str, extra: dict | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.extra = extra or {}


@dataclass
class CellSpec:
    """取り込むセル 1 つ(認識結果)。`crop` は PNG のバイト列。"""

    row: int
    col: int
    column: str
    kind: str
    recognized_text: str = ""
    resolved_value: str = ""
    expansion: dict | None = None
    reasons: list[str] = field(default_factory=list)
    candidates: list[tuple[str, float]] = field(default_factory=list)
    review: bool = False
    crop: bytes | None = None


LAYOUT_REASON_CODES = (
    "page_needs_review",
    "layout_uncertain",
    "band_clipped",
    "ruled_lines_low",
    "table_cut",
)


@dataclass
class PageSpec:
    cells: list[CellSpec]
    layout_status: str = "ok"
    layout_reasons: list[str] = field(default_factory=list)
    image_sha256: str | None = None
    view_jpeg: bytes | None = None  # 確認画面に出す元画像(縮小した JPEG)
    view_scale: float | None = None  # 表示 px / 元画像 px
    layout_info: dict | None = None  # 検出した表の 4 隅・罫線(元画像 px)
    approved: bool = False  # 利用者が表の隅を指定した(レイアウト由来の要確認は付けない)
    title_recognized: str | None = None  # 認識したタイトル(語彙があるときの第 1 候補)


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


# ------------------------------------------------------------------ ジョブの作成


def _insert_cells(
    con: sqlite3.Connection, images_dir: Path, page_id: int, cells: list[CellSpec]
) -> None:
    for c in cells:
        cur = con.execute(
            "INSERT INTO cells (page_id, row, col, column_kind, kind, recognized_text, resolved_value,"
            " expansion, reasons, candidates, review) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                page_id, c.row, c.col, c.column, c.kind, c.recognized_text, c.resolved_value,
                _j(c.expansion) if c.expansion else None, _j(c.reasons),
                _j([[v, p] for v, p in c.candidates]), int(c.review),
            ),
        )  # fmt: skip
        if c.crop is not None:
            path = images_dir / str(page_id) / f"{int(cur.lastrowid)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(c.crop)
            con.execute(
                "UPDATE cells SET crop_path = ? WHERE id = ?",
                (str(path.relative_to(images_dir)), int(cur.lastrowid)),
            )


def create_job(
    con: sqlite3.Connection, images_dir: Path, name: str, profile_id: int, pages: list[PageSpec]
) -> tuple[int, list[int]]:
    """認識済みのページからジョブを作る(テスト・開発用の合成ジョブ)。返り値は (ジョブ ID, ページ ID のリスト)。"""
    page_ids: list[int] = []
    with transaction(con):
        cur = con.execute(
            "INSERT INTO jobs (name, profile_id, status, created_at) VALUES (?, ?, 'ready', ?)",
            (name, profile_id, time.time()),
        )
        job_id = int(cur.lastrowid)
        for idx, spec in enumerate(pages):
            cur = con.execute(
                "INSERT INTO pages (job_id, idx, image_sha256, status, layout_status, layout_reasons)"
                " VALUES (?, ?, ?, 'review', ?, ?)",
                (job_id, idx, spec.image_sha256, spec.layout_status, _j(spec.layout_reasons)),
            )
            page_id = int(cur.lastrowid)
            page_ids.append(page_id)
            _insert_cells(con, images_dir, page_id, spec.cells)
    return job_id, page_ids


@dataclass
class PendingPage:
    """これから認識するページ(アップロードしたファイルの何ページ目か)。"""

    upload_id: str
    source_index: int


def create_pending_job(
    con: sqlite3.Connection, name: str, profile_id: int, pages: list[PendingPage]
) -> tuple[int, list[int]]:
    """認識前のジョブを作る。ページは 'processing'(runner が順に認識して `fill_page` で埋める)。"""
    if not pages:
        raise StoreError(422, "no_pages")
    page_ids: list[int] = []
    with transaction(con):
        if con.execute("SELECT 1 FROM profiles WHERE id = ?", (profile_id,)).fetchone() is None:
            raise StoreError(422, "profile_not_found")
        cur = con.execute(
            "INSERT INTO jobs (name, profile_id, status, created_at) VALUES (?, ?, 'processing', ?)",
            (name, profile_id, time.time()),
        )
        job_id = int(cur.lastrowid)
        for idx, pg in enumerate(pages):
            up = con.execute("SELECT sha256 FROM uploads WHERE id = ?", (pg.upload_id,)).fetchone()
            if up is None:
                raise StoreError(422, "upload_not_found")
            cur = con.execute(
                "INSERT INTO pages (job_id, idx, image_sha256, status, upload_id, source_index)"
                " VALUES (?, ?, ?, 'processing', ?, ?)",
                (job_id, idx, up["sha256"], pg.upload_id, pg.source_index),
            )
            page_ids.append(int(cur.lastrowid))
    return job_id, page_ids


def _refresh_job_status(con: sqlite3.Connection, job_id: int) -> None:
    n = con.execute(
        "SELECT SUM(status = 'processing') AS proc, SUM(status != 'confirmed') AS open_"
        " FROM pages WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    status = "processing" if n["proc"] else "done" if not n["open_"] else "ready"
    con.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))


def fill_page(
    con: sqlite3.Connection,
    images_dir: Path,
    page_id: int,
    spec: PageSpec,
    last_year: int | None = None,
    seconds: float | None = None,
) -> None:
    """認識結果をページに書き込んで 'review' にする(再実行しても同じ結果になるよう、既存のセルは作り直す)。"""
    with transaction(con):
        p = con.execute("SELECT job_id, status FROM pages WHERE id = ?", (page_id,)).fetchone()
        if p is None:
            raise StoreError(404, "page_not_found")
        if p["status"] != "processing":
            raise StoreError(409, "page_not_processing")
        con.execute("DELETE FROM cells WHERE page_id = ?", (page_id,))
        if spec.approved:  # 利用者が表の隅を指定した: レイアウト由来の要確認は外す(承認と同じ)
            for c in spec.cells:
                kept = [x for x in c.reasons if x not in LAYOUT_REASON_CODES]
                if kept != c.reasons:
                    c.reasons, c.review = kept, bool(kept)
        _insert_cells(con, images_dir, page_id, spec.cells)
        view_path = None
        if spec.view_jpeg is not None:
            vp = images_dir / str(page_id) / "view.jpg"
            vp.parent.mkdir(parents=True, exist_ok=True)
            vp.write_bytes(spec.view_jpeg)
            view_path = str(vp.relative_to(images_dir))
        con.execute(
            "UPDATE pages SET status = 'review', layout_status = ?, layout_reasons = ?, layout_approved = ?,"
            " last_year = ?, error = NULL, recognized_at = ?, recognize_seconds = ?,"
            " view_path = COALESCE(?, view_path), view_scale = COALESCE(?, view_scale),"
            " layout_info = COALESCE(?, layout_info), title_recognized = ?, revision = revision + 1"
            " WHERE id = ?",
            (
                "ok" if spec.approved else spec.layout_status,
                _j([] if spec.approved else spec.layout_reasons),
                int(spec.approved),
                last_year,
                time.time(),
                seconds,
                view_path,
                spec.view_scale,
                None if spec.layout_info is None else _j(spec.layout_info),
                spec.title_recognized,
                page_id,
            ),
        )
        _refresh_job_status(con, p["job_id"])


def fail_page(con: sqlite3.Connection, page_id: int, code: str) -> None:
    """認識に失敗したページ(例外の型名だけを残す。値は残さない)。"""
    with transaction(con):
        p = con.execute("SELECT job_id FROM pages WHERE id = ?", (page_id,)).fetchone()
        if p is None:
            return
        con.execute(
            "UPDATE pages SET status = 'failed', error = ? WHERE id = ? AND status = 'processing'",
            (code[:80], page_id),
        )
        _refresh_job_status(con, p["job_id"])


def retry_page(con: sqlite3.Connection, page_id: int) -> int:
    """失敗したページをもう一度認識に回す。ジョブ ID を返す。"""
    with transaction(con):
        p = con.execute("SELECT job_id, status FROM pages WHERE id = ?", (page_id,)).fetchone()
        if p is None:
            raise StoreError(404, "page_not_found")
        if p["status"] != "failed":
            raise StoreError(409, "page_not_failed")
        con.execute("UPDATE pages SET status = 'processing', error = NULL WHERE id = ?", (page_id,))
        _refresh_job_status(con, p["job_id"])
    return int(p["job_id"])


def processing_pages(con: sqlite3.Connection, job_id: int) -> list[sqlite3.Row]:
    """認識待ちのページ(アップロード順)。runner が 1 つずつ取り出す。"""
    return con.execute(
        "SELECT p.id, p.idx, p.upload_id, p.source_index, p.layout_override, u.path AS upload_path"
        " FROM pages p JOIN uploads u ON u.id = p.upload_id"
        " WHERE p.job_id = ? AND p.status = 'processing' ORDER BY p.idx",
        (job_id,),
    ).fetchall()


def previous_year(con: sqlite3.Connection, job_id: int, idx: int) -> int | None:
    """直前までのページで最後に分かった年(前のページの最後の行の年。年を書かない行が継承する)。"""
    r = con.execute(
        "SELECT last_year FROM pages WHERE job_id = ? AND idx < ? AND last_year IS NOT NULL"
        " ORDER BY idx DESC LIMIT 1",
        (job_id, idx),
    ).fetchone()
    return None if r is None else int(r["last_year"])


# ------------------------------------------------------------------ 読み出し


def list_jobs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT j.id, j.name, j.status, j.profile_id, j.created_at, j.last_page_id, j.last_cell,"
        " (SELECT COUNT(*) FROM pages p WHERE p.job_id = j.id) AS n_pages,"
        " (SELECT COUNT(*) FROM pages p WHERE p.job_id = j.id AND p.status = 'confirmed') AS n_confirmed,"
        " (SELECT COUNT(*) FROM pages p WHERE p.job_id = j.id AND p.status = 'processing') AS n_processing,"
        " (SELECT COUNT(*) FROM pages p WHERE p.job_id = j.id AND p.status = 'failed') AS n_failed"
        " FROM jobs j ORDER BY j.id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_job(con: sqlite3.Connection, job_id: int) -> dict:
    r = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if r is None:
        raise StoreError(404, "job_not_found")
    pages = con.execute(
        "SELECT id, idx, status, layout_status, error FROM pages WHERE job_id = ? ORDER BY idx",
        (job_id,),
    ).fetchall()
    out = dict(r)
    out["pages"] = [dict(p) for p in pages]
    return out


def _cell_dict(r: sqlite3.Row) -> dict:
    confirmed = r["confirmed_value"]
    return {
        "id": r["id"], "row": r["row"], "col": r["col"], "column": r["column_kind"], "kind": r["kind"],
        "recognized_text": r["recognized_text"], "resolved_value": r["resolved_value"],
        "value": r["resolved_value"] if confirmed is None else confirmed,
        "edit_kind": r["edit_kind"],
        "expansion": json.loads(r["expansion"]) if r["expansion"] else None,
        "reasons": json.loads(r["reasons"]),
        "candidates": [{"value": v, "p": p} for v, p in json.loads(r["candidates"])],
        "review": bool(r["review"]), "passed": bool(r["passed"]), "has_image": r["crop_path"] is not None,
    }  # fmt: skip


def get_page(con: sqlite3.Connection, page_id: int) -> dict:
    p = con.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    if p is None:
        raise StoreError(404, "page_not_found")
    cells = [
        _cell_dict(r)
        for r in con.execute("SELECT * FROM cells WHERE page_id = ? ORDER BY row, col", (page_id,))
    ]
    siblings = con.execute(
        "SELECT id, idx FROM pages WHERE job_id = ? ORDER BY idx", (p["job_id"],)
    ).fetchall()
    ids = [s["id"] for s in siblings]
    pos = ids.index(page_id)
    return {
        "id": p["id"], "job_id": p["job_id"], "idx": p["idx"], "n_pages": len(ids),
        "prev_page_id": ids[pos - 1] if pos > 0 else None,
        "next_page_id": ids[pos + 1] if pos + 1 < len(ids) else None,
        "status": p["status"], "error": p["error"], "layout_status": p["layout_status"],
        "title": p["title"], "title_recognized": p["title_recognized"],
        "title_suggestion": _title_suggestion(con, p),
        "layout_reasons": json.loads(p["layout_reasons"]), "layout_approved": bool(p["layout_approved"]),
        "revision": p["revision"], "cells": cells, "summary": summarize(cells),
        "can_undo": _has_edit(con, page_id, undone=False), "can_redo": _has_edit(con, page_id, undone=True),
    }  # fmt: skip


def _title_suggestion(con: sqlite3.Connection, p: sqlite3.Row) -> str | None:
    """タイトルの候補: 認識した値 > 同じジョブの直前のページのタイトル(同じ科目の続きのことが多い)。"""
    if p["title_recognized"]:
        return p["title_recognized"]
    r = con.execute(
        "SELECT title FROM pages WHERE job_id = ? AND idx < ? AND title IS NOT NULL AND title != ''"
        " ORDER BY idx DESC LIMIT 1",
        (p["job_id"], p["idx"]),
    ).fetchone()
    return None if r is None else r["title"]


def set_title(con: sqlite3.Connection, page_id: int, title: str) -> dict:
    title = title.strip()
    if len(title) > 40:
        raise StoreError(422, "title_too_long")
    if title:
        try:
            title.encode("cp932")
        except UnicodeEncodeError as exc:
            raise StoreError(422, "cp932_unencodable") from exc
    with transaction(con):
        _require_open(con, page_id)
        con.execute(
            "UPDATE pages SET title = ?, revision = revision + 1 WHERE id = ?",
            (title or None, page_id),
        )
    return get_page(con, page_id)


def cell_image_path(con: sqlite3.Connection, images_dir: Path, page_id: int, cell_id: int) -> Path:
    r = con.execute(
        "SELECT crop_path FROM cells WHERE id = ? AND page_id = ?", (cell_id, page_id)
    ).fetchone()
    if r is None or r["crop_path"] is None:
        raise StoreError(404, "image_not_found")
    path = (images_dir / r["crop_path"]).resolve()
    if images_dir.resolve() not in path.parents:  # パスの外へ出さない
        raise StoreError(404, "image_not_found")
    return path


# ------------------------------------------------------------------ 要約・検証


def summarize(cells: list[dict]) -> dict:
    """確定できるか(通過していない要確認・検証エラー・未選択の 10 倍違い)。値の内容は含めない。"""
    nonblank_rows = {c["row"] for c in cells if c["value"].strip()}
    review = [c for c in cells if c["review"] or R_AMBIGUITY in c["reasons"]]
    errors, ambiguous = [], []
    for c in cells:
        for code in validate_value(c["column"], c["value"], c["row"] in nonblank_rows):
            errors.append({"cell_id": c["id"], "row": c["row"], "col": c["col"], "code": code,
                           "label": ERROR_LABELS[code]})  # fmt: skip
        if R_AMBIGUITY in c["reasons"] and c["edit_kind"] not in ("candidate", "corrected"):
            ambiguous.append(c["id"])
    unpassed = [c["id"] for c in review if not c["passed"]]
    return {
        "review_total": len(review),
        "review_passed": len(review) - len(unpassed),
        "unpassed": unpassed,
        "errors": errors,
        "ambiguous": ambiguous,
        "can_confirm": not unpassed and not errors and not ambiguous,
    }


# ------------------------------------------------------------------ 編集


def _has_edit(con: sqlite3.Connection, page_id: int, undone: bool) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM edits WHERE page_id = ? AND undone = ? LIMIT 1", (page_id, int(undone))
        ).fetchone()
        is not None
    )


def _bump(con: sqlite3.Connection, page_id: int) -> None:
    con.execute("UPDATE pages SET revision = revision + 1, status = CASE WHEN status = 'review' THEN 'reviewing' ELSE status END WHERE id = ?", (page_id,))  # fmt: skip


def _require_open(con: sqlite3.Connection, page_id: int) -> sqlite3.Row:
    p = con.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    if p is None:
        raise StoreError(404, "page_not_found")
    if p["status"] == "confirmed":
        raise StoreError(409, "page_confirmed")
    if p["status"] in ("processing", "failed"):
        raise StoreError(409, "page_not_ready")
    return p


def _cell(con: sqlite3.Connection, page_id: int, cell_id: int) -> sqlite3.Row:
    r = con.execute(
        "SELECT * FROM cells WHERE id = ? AND page_id = ?", (cell_id, page_id)
    ).fetchone()
    if r is None:
        raise StoreError(404, "cell_not_found")
    return r


def _record_edit(
    con, page_id: int, before: sqlite3.Row, after_value: str | None, after_kind: str
) -> None:
    con.execute(
        "DELETE FROM edits WHERE page_id = ? AND undone = 1", (page_id,)
    )  # やり直しの履歴を捨てる
    con.execute(
        "INSERT INTO edits (page_id, cell_id, before_value, after_value, before_edit_kind, after_edit_kind, ts)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (page_id, before["id"], before["confirmed_value"], after_value, before["edit_kind"], after_kind, time.time()),
    )  # fmt: skip


def _propagate_ditto(con: sqlite3.Connection, page_id: int, src: sqlite3.Row, value: str) -> None:
    """参照元の値が変わったら、それを参照する〃のセルを再展開して要確認に戻す(利用者が書き換え済みのセルは除く)。"""
    for d in con.execute(
        "SELECT * FROM cells WHERE page_id = ? AND expansion IS NOT NULL", (page_id,)
    ).fetchall():
        exp = json.loads(d["expansion"])
        if exp.get("kind") != "ditto" or exp.get("source") != [src["row"], src["col"]]:
            continue
        if d["edit_kind"] in ("corrected", "candidate", "expanded_edit"):
            continue
        reasons = json.loads(d["reasons"])
        if R_DITTO_SOURCE_CHANGED not in reasons:
            reasons.insert(0, R_DITTO_SOURCE_CHANGED)  # 先頭(画面では先頭の理由から表示する)
        con.execute(
            "UPDATE cells SET resolved_value = ?, confirmed_value = NULL, edit_kind = 'none', review = 1,"
            " passed = 0, reasons = ? WHERE id = ?",
            (value, _j(reasons), d["id"]),
        )
        _propagate_ditto(con, page_id, d, value)


def _set_value(con, page_id: int, cell: sqlite3.Row, value: str, kind: str) -> None:
    """値を確定値にする(元に戻す用の履歴つき)。展開済みのセルは参照元との結びつきを外す。"""
    if cell["expansion"] and kind in ("corrected", "candidate") and value != cell["resolved_value"]:
        kind = "expanded_edit"
    if cell["confirmed_value"] == value and cell["edit_kind"] == kind:
        return
    _record_edit(con, page_id, cell, value, kind)
    con.execute(
        "UPDATE cells SET confirmed_value = ?, edit_kind = ? WHERE id = ?",
        (value, kind, cell["id"]),
    )
    if kind == "expanded_edit":
        con.execute("UPDATE cells SET expansion = NULL WHERE id = ?", (cell["id"],))
    shown_before = (
        cell["resolved_value"] if cell["confirmed_value"] is None else cell["confirmed_value"]
    )
    if value != shown_before:
        _propagate_ditto(con, page_id, cell, value)


def patch_cell(
    con: sqlite3.Connection, page_id: int, cell_id: int, action: str, value: str | None = None,
    candidate_index: int | None = None,
) -> dict:  # fmt: skip
    """セルへの操作。action: pass(通過)/ edit(下書きの上書き)/ confirm(Enter)/ candidate(Alt+数字)/ revert(Esc)。"""
    with transaction(con):
        _require_open(con, page_id)
        cell = _cell(con, page_id, cell_id)
        if action == "pass":
            con.execute("UPDATE cells SET passed = 1 WHERE id = ?", (cell_id,))
        elif action == "edit":
            if value is None:
                raise StoreError(422, "value_required")
            same = value == cell["resolved_value"]
            _set_value(con, page_id, cell, value, "unchanged" if same else "corrected")
            _bump(con, page_id)
        elif action == "confirm":
            # 末尾「-」の両仮説で 10 倍違う金額は、利用者が明示的に選ぶ(候補)か、値を入力するまで確認済みにしない
            explicit = cell["edit_kind"] in ("candidate", "corrected") or (
                value is not None and value != cell["resolved_value"]
            )
            if R_AMBIGUITY in json.loads(cell["reasons"]) and not explicit:
                raise StoreError(409, "ambiguity_unresolved")
            if value is None:
                value = (
                    cell["resolved_value"]
                    if cell["confirmed_value"] is None
                    else cell["confirmed_value"]
                )
            kind = cell["edit_kind"] if cell["confirmed_value"] == value and cell["edit_kind"] != "none" else (
                "unchanged" if value == cell["resolved_value"] else "corrected")  # fmt: skip
            _set_value(con, page_id, cell, value, kind)
            con.execute("UPDATE cells SET passed = 1 WHERE id = ?", (cell_id,))
            _bump(con, page_id)
        elif action == "candidate":
            cands = json.loads(cell["candidates"])
            if candidate_index is None or not 0 <= candidate_index < len(cands):
                raise StoreError(422, "bad_candidate")
            _set_value(con, page_id, cell, cands[candidate_index][0], "candidate")
            con.execute("UPDATE cells SET passed = 1 WHERE id = ?", (cell_id,))
            _bump(con, page_id)
        elif action == "revert":
            if cell["confirmed_value"] is not None:
                _record_edit(con, page_id, cell, None, "none")
                con.execute(
                    "UPDATE cells SET confirmed_value = NULL, edit_kind = 'none' WHERE id = ?",
                    (cell_id,),
                )
                _bump(con, page_id)
        else:
            raise StoreError(422, "bad_action")
    return get_page(con, page_id)


def undo(con: sqlite3.Connection, page_id: int) -> tuple[dict, int | None]:
    """直前の編集を取り消す。返り値は (ページ, 戻したセルの ID)。"""
    with transaction(con):
        _require_open(con, page_id)
        e = con.execute("SELECT * FROM edits WHERE page_id = ? AND undone = 0 ORDER BY id DESC LIMIT 1", (page_id,)).fetchone()  # fmt: skip
        if e is None:
            return get_page(con, page_id), None
        con.execute("UPDATE cells SET confirmed_value = ?, edit_kind = ? WHERE id = ?", (e["before_value"], e["before_edit_kind"], e["cell_id"]))  # fmt: skip
        con.execute("UPDATE edits SET undone = 1 WHERE id = ?", (e["id"],))
        _bump(con, page_id)
    return get_page(con, page_id), int(e["cell_id"])


def redo(con: sqlite3.Connection, page_id: int) -> tuple[dict, int | None]:
    with transaction(con):
        _require_open(con, page_id)
        e = con.execute("SELECT * FROM edits WHERE page_id = ? AND undone = 1 ORDER BY id ASC LIMIT 1", (page_id,)).fetchone()  # fmt: skip
        if e is None:
            return get_page(con, page_id), None
        con.execute("UPDATE cells SET confirmed_value = ?, edit_kind = ? WHERE id = ?", (e["after_value"], e["after_edit_kind"], e["cell_id"]))  # fmt: skip
        con.execute("UPDATE edits SET undone = 0 WHERE id = ?", (e["id"],))
        _bump(con, page_id)
    return get_page(con, page_id), int(e["cell_id"])


# ------------------------------------------------------------------ ページの確定・位置・レイアウト


def confirm_page(con: sqlite3.Connection, page_id: int, review_seconds: float = 0.0) -> dict:
    """確定の条件をサーバでも検証する(UI の無効化に頼らない)。満たさなければ 409 とその理由。"""
    with transaction(con):
        p = _require_open(con, page_id)
        cells = [
            _cell_dict(r)
            for r in con.execute(
                "SELECT * FROM cells WHERE page_id = ? ORDER BY row, col", (page_id,)
            )
        ]
        s = summarize(cells)
        if not s["can_confirm"]:
            raise StoreError(409, "cannot_confirm", {"summary": s})
        con.execute(
            "UPDATE cells SET confirmed_value = resolved_value, edit_kind = 'unchanged' WHERE page_id = ? AND confirmed_value IS NULL",
            (page_id,),
        )  # 要確認でない(確認済みでもない)セルは、そのまま確定する
        con.execute(
            "UPDATE pages SET status = 'confirmed', confirmed_at = ?, review_seconds = review_seconds + ?,"
            " revision = revision + 1 WHERE id = ?",
            (time.time(), max(0.0, float(review_seconds)), page_id),
        )
        pending = con.execute(
            "SELECT COUNT(*) FROM pages WHERE job_id = ? AND status != 'confirmed'", (p["job_id"],)
        ).fetchone()[0]
        con.execute(
            "UPDATE jobs SET status = ? WHERE id = ?",
            ("done" if pending == 0 else "ready", p["job_id"]),
        )
    return get_page(con, page_id)


def reopen_page(con: sqlite3.Connection, page_id: int) -> dict:
    with transaction(con):
        p = con.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        if p is None:
            raise StoreError(404, "page_not_found")
        con.execute(
            "UPDATE pages SET status = 'reviewing', confirmed_at = NULL WHERE id = ?", (page_id,)
        )
        con.execute("UPDATE jobs SET status = 'ready' WHERE id = ?", (p["job_id"],))
    return get_page(con, page_id)


def set_position(con: sqlite3.Connection, job_id: int, page_id: int, cell: int | None) -> None:
    with transaction(con):
        con.execute(
            "UPDATE jobs SET last_page_id = ?, last_cell = ? WHERE id = ?", (page_id, cell, job_id)
        )


def approve_layout(con: sqlite3.Connection, page_id: int) -> dict:
    """「境界は合っている」の承認。レイアウト由来の要確認(ページ・セル)だけを外す。"""
    with transaction(con):
        _require_open(con, page_id)
        con.execute(
            "UPDATE pages SET layout_approved = 1, layout_status = 'ok' WHERE id = ?", (page_id,)
        )
        layout_codes = LAYOUT_REASON_CODES
        for c in con.execute(
            "SELECT id, reasons FROM cells WHERE page_id = ?", (page_id,)
        ).fetchall():
            reasons = json.loads(c["reasons"])
            kept = [x for x in reasons if x not in layout_codes]
            if kept != reasons:
                con.execute(
                    "UPDATE cells SET reasons = ?, review = ? WHERE id = ?",
                    (_j(kept), int(bool(kept)), c["id"]),
                )
    return get_page(con, page_id)


# ------------------------------------------------------------------ レイアウトの確認・補正(重ね表示・表の隅の指定)


def get_layout(con: sqlite3.Connection, page_id: int) -> dict:
    """重ね表示に必要な情報(元画像の大きさ・検出した表の隅と罫線)。画像の中身は /layout/image で取る。"""
    p = con.execute(
        "SELECT layout_info, view_path, view_scale, layout_status, layout_reasons, layout_approved,"
        " layout_override, status FROM pages WHERE id = ?",
        (page_id,),
    ).fetchone()
    if p is None:
        raise StoreError(404, "page_not_found")
    if p["layout_info"] is None or p["view_path"] is None:
        raise StoreError(404, "layout_not_available")
    return {
        "info": json.loads(p["layout_info"]),
        "view_scale": p["view_scale"],
        "status": p["layout_status"],
        "reasons": json.loads(p["layout_reasons"]),
        "approved": bool(p["layout_approved"]),
        "manual": p["layout_override"] is not None,
        "override": None if p["layout_override"] is None else json.loads(p["layout_override"]),
    }


def layout_view_path(con: sqlite3.Connection, images_dir: Path, page_id: int) -> Path:
    r = con.execute("SELECT view_path FROM pages WHERE id = ?", (page_id,)).fetchone()
    if r is None or r["view_path"] is None:
        raise StoreError(404, "layout_not_available")
    path = (images_dir / r["view_path"]).resolve()
    if images_dir.resolve() not in path.parents:
        raise StoreError(404, "layout_not_available")
    return path


def request_recut(con: sqlite3.Connection, page_id: int, corners: list[list[float]]) -> int:
    """利用者が指定した表の隅でページを切り直して認識し直す(このページの修正は破棄される)。ジョブ ID を返す。"""
    from ..layout.pipeline import validate_corners

    with transaction(con):
        p = _require_open(con, page_id)
        info = con.execute("SELECT layout_info FROM pages WHERE id = ?", (page_id,)).fetchone()[
            "layout_info"
        ]
        if info is None:
            raise StoreError(409, "layout_not_available")
        size = json.loads(info)
        try:
            pts = validate_corners(corners, (int(size["height"]), int(size["width"])))
        except ValueError as exc:
            raise StoreError(422, "bad_corners", {"detail": str(exc)}) from exc
        con.execute(
            "UPDATE pages SET status = 'processing', error = NULL, layout_override = ?,"
            " revision = revision + 1 WHERE id = ?",
            (_j(pts.round(1).tolist()), page_id),
        )
        _refresh_job_status(con, p["job_id"])
    return int(p["job_id"])
