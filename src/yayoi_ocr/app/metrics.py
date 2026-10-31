"""学習画面の指標(確認画面の実績から)。値の内容は一切含めない(件数・割合・時間だけ)。

指標は確定済みページの確認の実績: ページごとの確認時間、非空欄セルのうち要確認だった割合、利用者が直したセルの割合、
**要確認でないのに利用者が直したセル**(見逃しの観測された下限。要確認でないセルは見ずに確定するので、直された分しか見えない)。
世代(Phase 4 の追加学習の採否)は、Phase 3 では世代 0(出荷時の重み)だけ。ロールバック・リセットは世代 0 だけのとき意味がないので無効。
"""

from __future__ import annotations

import json
import sqlite3

from .settings import load as load_settings

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


def by_generation(rows: list[dict]) -> dict[int, dict]:
    """確定したページを、認識に使った世代ごとに集計する(世代の実績)。件数と割合だけ(値は含まない)。
    見逃しの観測 = 要確認でないのに利用者が直したセルの、非空欄セルに対する割合(実際の見逃しはこれ以上あり得る)。"""
    acc: dict[int, dict[str, int]] = {}
    for r in rows:
        a = acc.setdefault(
            r["generation"],
            {
                "n_pages": 0,
                "n_text": 0,
                "n_review": 0,
                "n_corrected": 0,
                "n_unflagged_corrected": 0,
            },
        )
        a["n_pages"] += 1
        for k in ("n_text", "n_review", "n_corrected", "n_unflagged_corrected"):
            a[k] += r[k]
    out: dict[int, dict] = {}
    for gid, a in acc.items():
        n = a["n_text"]
        out[gid] = {
            **a,
            "review_rate": a["n_review"] / n if n else None,
            "correction_rate": a["n_corrected"] / n if n else None,
            "miss_rate": a["n_unflagged_corrected"] / n if n else None,
        }
    return out


def report(con: sqlite3.Connection, profile_id: int, app_version: str, now: float) -> dict:
    """試用版の精度レポート(**統計だけ**。セルの値・タイトル・科目・ファイル名・画像は一切含まない)。開発者に送って精度を見るためのもの。
    含めるのは、件数・割合・時間・理由コード(要確認の理由の種類ごとの件数)・設定の種類(件数と選択肢のコード)だけ。"""
    rows = page_metrics(con, profile_id)
    per_col: dict[str, dict[str, int]] = {}
    for r in con.execute(
        "SELECT c.column_kind AS col,"
        " SUM(c.kind != 'blank') AS n_text,"
        " SUM(c.kind != 'blank' AND c.review = 1) AS n_review,"
        " SUM(c.kind != 'blank' AND c.edit_kind IN ('corrected','candidate','expanded_edit')) AS n_corrected,"
        " SUM(c.kind != 'blank' AND c.review = 0 AND c.edit_kind IN ('corrected','candidate','expanded_edit')) AS n_unflagged_corrected"
        " FROM cells c JOIN pages p ON p.id = c.page_id JOIN jobs j ON j.id = p.job_id"
        " WHERE p.status = 'confirmed' AND j.profile_id = ? GROUP BY c.column_kind",
        (profile_id,),
    ):
        per_col[r["col"]] = {
            k: int(r[k] or 0)
            for k in ("n_text", "n_review", "n_corrected", "n_unflagged_corrected")
        }
    reasons: dict[str, int] = {}
    for r in con.execute(
        "SELECT c.reasons FROM cells c JOIN pages p ON p.id = c.page_id JOIN jobs j ON j.id = p.job_id"
        " WHERE p.status = 'confirmed' AND j.profile_id = ? AND c.review = 1",
        (profile_id,),
    ):
        for code in json.loads(r["reasons"]):
            reasons[code] = reasons.get(code, 0) + 1
    edit_kinds = {
        r["edit_kind"]: int(r["n"])
        for r in con.execute(
            "SELECT c.edit_kind, COUNT(*) AS n FROM cells c JOIN pages p ON p.id = c.page_id JOIN jobs j ON j.id = p.job_id"
            " WHERE p.status = 'confirmed' AND j.profile_id = ? AND c.kind != 'blank' GROUP BY c.edit_kind",
            (profile_id,),
        )
    }
    cfg = load_settings(con)
    n_exp = con.execute("SELECT COUNT(*), COALESCE(SUM(n_rows), 0) FROM exports").fetchone()
    return {
        "schema": 1,
        "app_version": app_version,
        "generated_at": now,
        "summary": summary(rows),
        "per_column": per_col,
        "review_reasons": dict(sorted(reasons.items())),
        "edit_kinds": edit_kinds,
        "pages": [
            {
                "order": i + 1,
                "generation": r["generation"],
                "n_text": r["n_text"],
                "n_review": r["n_review"],
                "n_corrected": r["n_corrected"],
                "n_unflagged_corrected": r["n_unflagged_corrected"],
                "review_seconds": round(r["review_seconds"], 1),
            }
            for i, r in enumerate(rows)
        ],
        "settings": {
            "n_payments": len(cfg.payments),
            "n_titles": len(cfg.titles),
            "taxpayer": cfg.taxpayer,
            "tax_scheme": cfg.tax_scheme,
            "tax_method": cfg.tax_method,
        },
        "exports": {"n_exports": int(n_exp[0]), "n_rows": int(n_exp[1])},
    }
