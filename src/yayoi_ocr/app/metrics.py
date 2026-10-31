"""学習画面の指標(確認画面の実績から)。値の内容は一切含めない(件数・割合・時間だけ)。

指標は確定済みページの確認の実績: ページごとの確認時間、非空欄セルのうち要確認だった割合、利用者が直したセルの割合、
**要確認でないのに利用者が直したセル**(見逃しの観測された下限。要確認でないセルは見ずに確定するので、直された分しか見えない)。
世代(Phase 4 の追加学習の採否)は、Phase 3 では世代 0(出荷時の重み)だけ。ロールバック・リセットは世代 0 だけのとき意味がないので無効。
"""

from __future__ import annotations

import sqlite3

CORRECTED = ("corrected", "candidate", "expanded_edit")


def page_metrics(con: sqlite3.Connection, profile_id: int) -> list[dict]:
    """確定済みページ(確定した順)の指標。"""
    out: list[dict] = []
    pages = con.execute(
        "SELECT p.id, p.generation, p.review_seconds, p.confirmed_at, p.recognize_seconds"
        " FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE j.profile_id = ? AND p.status = 'confirmed' ORDER BY p.confirmed_at, p.id",
        (profile_id,),
    ).fetchall()
    for pg in pages:
        c = con.execute(
            "SELECT"
            " SUM(kind != 'blank') AS n_text,"
            " SUM(kind != 'blank' AND review = 1) AS n_review,"
            " SUM(kind != 'blank' AND edit_kind IN ('corrected','candidate','expanded_edit')) AS n_corrected,"
            " SUM(kind != 'blank' AND review = 0 AND edit_kind IN ('corrected','candidate','expanded_edit')) AS n_unflagged_corrected,"
            " SUM(kind != 'blank' AND review = 1 AND edit_kind IN ('unchanged')) AS n_review_unchanged"
            " FROM cells WHERE page_id = ?",
            (pg["id"],),
        ).fetchone()
        n_text = int(c["n_text"] or 0)
        out.append(
            {
                "page_id": pg["id"],
                "generation": pg["generation"],
                "confirmed_at": pg["confirmed_at"],
                "review_seconds": pg["review_seconds"],
                "n_text": n_text,
                "n_review": int(c["n_review"] or 0),
                "n_corrected": int(c["n_corrected"] or 0),
                "n_unflagged_corrected": int(c["n_unflagged_corrected"] or 0),
                "review_rate": (c["n_review"] or 0) / n_text if n_text else None,
                "correction_rate": (c["n_corrected"] or 0) / n_text if n_text else None,
            }
        )
    return out


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    n_text = sum(r["n_text"] for r in rows)
    secs = [r["review_seconds"] for r in rows if r["review_seconds"] > 0]
    return {
        "n_pages": n,
        "n_cells": n_text,
        "review_rate": sum(r["n_review"] for r in rows) / n_text if n_text else None,
        "correction_rate": sum(r["n_corrected"] for r in rows) / n_text if n_text else None,
        "unflagged_corrected": sum(r["n_unflagged_corrected"] for r in rows),
        "median_review_seconds": sorted(secs)[len(secs) // 2] if secs else None,
    }


def generations(con: sqlite3.Connection, profile_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT g.id, g.parent_id, g.created_at, g.n_pages, g.score, g.status,"
        " (SELECT COUNT(*) FROM pages p JOIN jobs j ON j.id = p.job_id"
        "   WHERE j.profile_id = g.profile_id AND p.generation = g.id) AS n_recognized"
        " FROM generations g WHERE g.profile_id = ? ORDER BY g.id",
        (profile_id,),
    ).fetchall()
    return [dict(r) for r in rows]
