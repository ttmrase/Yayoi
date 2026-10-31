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


_COUNT_KEYS = (
    "n_cells",
    "n_text",
    "n_correct",
    "n_review",
    "n_miss",
    "accuracy",
    "review_rate",
    "miss_rate",
)


_CALIB_KEYS = ("n", "nll", "brier", "expected_miss", "observed_miss")


def _counts(d: dict | None) -> dict | None:
    return None if not d else {k: d[k] for k in _COUNT_KEYS if k in d}


def _side(d: dict | None) -> dict | None:
    """親または新世代の評価成績(全体 + 列ごと。件数・割合・較正の質だけ)。"""
    if not d:
        return None
    out = _counts(d) or {}
    if d.get("calib"):
        out["calib"] = {k: d["calib"].get(k) for k in _CALIB_KEYS}
    out["by_column"] = {
        col: {
            **(_counts(v) or {}),
            **({"calib": {k: v["calib"].get(k) for k in _CALIB_KEYS}} if v.get("calib") else {}),
        }
        for col, v in (d.get("by_column") or {}).items()
    }
    return out


def generation_report(con: sqlite3.Connection, profile_id: int, rows: list[dict]) -> list[dict]:
    """世代ごとの要約(件数・割合・コードだけ。ページ ID・値は含めない)。評価の値は採否のときの親と新世代の比較、実績はその世代で認識して確定したページでの値。"""
    obs = by_generation(rows)
    out = []
    for g in con.execute(
        "SELECT id, number, kind, status, reject_reason, meta FROM generations WHERE profile_id = ?"
        " ORDER BY number, id",
        (profile_id,),
    ):
        meta = json.loads(g["meta"]) if g["meta"] else {}
        ev = meta.get("eval") or {}
        recal = (meta.get("recalibration") or {}).get("columns", {})
        train = meta.get("train") or {}
        out.append(
            {
                "number": g["number"],
                "kind": g["kind"],
                "status": g["status"],
                "reject_reason": g["reject_reason"],
                "trigger": meta.get("trigger"),
                "n_train_pages": meta.get("n_train_pages"),
                "n_eval_pages": meta.get("n_eval_pages"),
                "train": {
                    k: train.get(k)
                    for k in (
                        "samples",
                        "rehearsal",
                        "epochs",
                        "seconds_total",
                        "device",
                        "threads",
                    )
                },
                "eval": {
                    k: _side(ev.get(k))
                    for k in ("parent", "candidate", "candidate_global", "candidate_final")
                },
                "recalibration": {
                    col: {
                        k: c.get(k)
                        for k in ("tau_before", "tau_after", "k_before", "k_after", "action")
                    }
                    for col, c in recal.items()
                },
                "halt": {
                    k: (meta.get("halt") or {}).get(k)
                    for k in ("code", "k_total", "k_amount", "n_cells", "n_pages")
                }
                if meta.get("halt")
                else None,
                "observed": obs.get(g["id"]),
            }
        )
    return out


def training_report(con: sqlite3.Connection, profile_id: int, limit: int = 20) -> list[dict]:
    return [
        {
            "trigger": r["trigger"],
            "status": r["status"],
            "reason": r["reason"],
            "n_new_cells": r["n_new_cells"],
            "n_train_pages": r["n_train_pages"],
            "n_eval_pages": r["n_eval_pages"],
            "seconds": None
            if r["finished_at"] is None
            else round(r["finished_at"] - r["started_at"], 1),
        }
        for r in con.execute(
            "SELECT * FROM training_runs WHERE profile_id = ? ORDER BY id DESC LIMIT ?",
            (profile_id, limit),
        )
    ]


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
    gen_number = {
        g["id"]: g["number"]
        for g in con.execute(
            "SELECT id, number FROM generations WHERE profile_id = ?", (profile_id,)
        )
    }
    n_exp = con.execute("SELECT COUNT(*), COALESCE(SUM(n_rows), 0) FROM exports").fetchone()
    return {
        "schema": 2,
        "app_version": app_version,
        "generated_at": now,
        "summary": summary(rows),
        "per_column": per_col,
        "review_reasons": dict(sorted(reasons.items())),
        "edit_kinds": edit_kinds,
        "generations": generation_report(con, profile_id, rows),
        "training_runs": training_report(con, profile_id),
        "pages": [
            {
                "order": i + 1,
                "generation": gen_number.get(r["generation"]),
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
            "auto_train": cfg.auto_train,
            "train_threshold_cells": cfg.train_threshold_cells,
        },
        "exports": {"n_exports": int(n_exp[0]), "n_rows": int(n_exp[1])},
    }
