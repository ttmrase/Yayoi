"""世代(重み + 確信度)の管理: 一覧・使用中の世代・ロールバック・リセット・安全のための停止。設計は `docs/phase4_design.md`。

- 世代 0(`kind = 'shipped'`)は出荷時の重み(配布物の `models/`)。ディレクトリを持たない(`dir` は NULL)。
- 世代 1 以降(`kind = 'trained'`)は、`<データ領域>/models/<プロファイル ID>/gen_XXXX/` に重み(幅 256・幅 384)・`confidence.json`・`meta.json` を持つ。
  `dir` は**この形に厳密に一致するものだけ**を扱う(それ以外は、読み込みも削除もしない。出荷時の重みが同じデータ領域に同居するため)。
- **重みと確信度は必ず同じ世代の組で使う**(見逃し率の保証に直結する)。重みファイルの SHA256 を `meta.json` と `confidence.json` の両方に記録し、読み込みのたびに照合する。
  欠け・不一致があれば、その世代は使わず(状態を `broken`、理由コードを `reject_reason` に記録)、出荷時の組へ戻す。黙って戻さず、学習画面に警告を出す。
- 使用中(`active`)の世代は書き手(プロファイル)ごとに 1 つ。新しく認識するページはそれで認識し、`pages.generation` に世代の ID を記録する。
- 状態: active(使用中)/ retired(過去。重みがあり、戻せる)/ rejected(不採用。重みは消す)/ discarded(リセットで捨てた。重みは消す)/
  broken(重みと確信度の組が壊れていて使えない)/ halted(採用後の監視で見逃しが見つかり、安全のため停止した。戻せない)/
  **suggest(提案モード)**: 安全基準には届かない(金額の見逃し・見逃しの悪化・自信過剰・較正の悪化)が、評価用ページで親より正解率が有意に高い世代。
  親が使用中の間だけ、認識のたびに親と両方で推論し、**入力済みの値を提案モードの世代の値にする**。要確認の判定は「親 OR 提案モード OR 値が違う」なので、要確認でないセルの値は親と同じ
  (見逃しは親の見逃しの部分集合)。提案モードの世代は 1 つだけ(新しい結果で置き換える)。
- メタデータ(`meta`)には、ページ ID・件数・割合・理由コードだけを入れる(`clean_meta` が、数値・コード形式の短い文字列・キーだけを通す。日本語や空白を含む値は構造的に入らない)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .db import DEFAULT_PROFILE_ID, transaction
from .store import StoreError

log = logging.getLogger("yayoi.gens")

WEIGHT_FILES = ("parseq-ndl-24x256-30-tiny.pt", "parseq-ndl-24x384-50-tiny.pt")
CONFIDENCE_FILE = "confidence.json"
META_FILE = "meta.json"
RESTORABLE = ("active", "retired")
UNUSABLE = ("broken", "halted")  # 学習画面と通知で警告する状態

# 理由コード(学習画面の文言は web/src/trainingText.ts と対応させる)
R_WEIGHTS_MISSING = "weights_missing"
R_CONFIDENCE_MISSING = "confidence_missing"
R_META_MISSING = "meta_missing"
R_PAIR_MISMATCH = "pair_mismatch"
R_DIR_INVALID = "dir_invalid"


@dataclass(frozen=True)
class Weights:
    """認識に使う重み。`dir` が None なら出荷時の重み(既定の選び方)。"""

    generation_id: int
    number: int
    dir: Path | None = None
    confidence: Path | None = None
    suggest: Weights | None = None  # 提案モードの世代の重みと確信度(使用中の世代が親のときだけ)


def gen_dir_name(number: int) -> str:
    return f"gen_{number:04d}"


def expected_dir(profile_id: int, number: int) -> str:
    """世代の置き場(データ領域からの相対)。DB の `dir` はこれに厳密に一致しなければならない。"""
    return f"models/{int(profile_id)}/{gen_dir_name(int(number))}"


def gen_path(models_root: Path, row: sqlite3.Row) -> Path | None:
    """世代の置き場の絶対パス。`dir` が決まった形でなければ None(読み込みも削除もしない)。"""
    if row["kind"] == "shipped" or not row["dir"]:
        return None
    if row["dir"] != expected_dir(row["profile_id"], row["number"]):
        return None
    return models_root.parent / row["dir"]


# ------------------------------------------------------------------ メタデータの許可リスト

_CODE = re.compile(r"^[A-Za-z0-9_.:+\-]{0,64}$")
_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def clean_meta(x: object, depth: int = 0) -> object:
    """メタデータに入れてよいもの(数値・真偽・コード形式の短い文字列・それらの配列と辞書)だけを通す。日本語や空白を含む文字列は ValueError。
    数字だけの文字列や日付の形([A-Za-z0-9_.:+-] だけの文字列)は通るので、呼び出し側は、値(金額・日付)を文字列で渡さないこと(今の呼び出しは、アプリが決めたキーと件数・割合・コードだけ)。"""
    if depth > 6:
        raise ValueError("meta_too_deep")
    if x is None or isinstance(x, bool | int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, str):
        if _CODE.match(x):
            return x
        raise ValueError("meta_not_a_code")
    if isinstance(x, list | tuple):
        if len(x) > 5000:
            raise ValueError("meta_too_long")
        return [clean_meta(v, depth + 1) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            ks = str(k)
            if not _KEY.match(ks):
                raise ValueError("meta_bad_key")
            out[ks] = clean_meta(v, depth + 1)
        return out
    raise ValueError("meta_bad_type")


# ------------------------------------------------------------------ 取得


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


# ------------------------------------------------------------------ 重みと確信度の組の照合

_VERIFIED: dict[tuple, str | None] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_hashes(d: Path) -> dict[str, str]:
    return {f: _sha256(d / f) for f in WEIGHT_FILES}


def verify_pair(models_root: Path, row: sqlite3.Row) -> str | None:
    """世代の重みと確信度が、記録された組と一致するか。使えるなら None、使えないなら理由コード。出荷時の世代は検査しない(配布物の検証は MANIFEST)。
    ファイルの大きさと更新時刻が変わらない限り、結果を覚えておく(認識のたびにハッシュを計算しない)。"""
    if row["kind"] == "shipped":
        return None
    d = gen_path(models_root, row)
    if d is None:
        return R_DIR_INVALID
    weights = [d / f for f in WEIGHT_FILES]
    if not all(f.is_file() for f in weights):
        return R_WEIGHTS_MISSING
    conf, meta = d / CONFIDENCE_FILE, d / META_FILE
    if not conf.is_file():
        return R_CONFIDENCE_MISSING
    if not meta.is_file():
        return R_META_MISSING
    sig = tuple((f.stat().st_size, f.stat().st_mtime_ns) for f in [*weights, conf, meta])
    key = (str(d), sig)
    if key in _VERIFIED:
        return _VERIFIED[key]
    try:
        actual = weights_hashes(d)
        recorded_c = json.loads(conf.read_text(encoding="utf-8"))["weights_sha256"]
        recorded_m = json.loads(meta.read_text(encoding="utf-8"))["weights_sha256"]
        result = None if actual == recorded_c == recorded_m else R_PAIR_MISMATCH
    except (OSError, ValueError, KeyError, TypeError):
        result = R_PAIR_MISMATCH
    if len(_VERIFIED) > 64:
        _VERIFIED.clear()
    _VERIFIED[key] = result
    return result


def _has_weights(models_root: Path, row: sqlite3.Row) -> bool:
    return verify_pair(models_root, row) is None


def _activate_row(
    con: sqlite3.Connection, row: sqlite3.Row, models_root: Path | None = None
) -> None:
    """この世代を使用中にし、ほかの使用中の世代を過去にする(トランザクションの中で呼ぶ)。親が使用中でなくなる提案モードの世代は片づける。"""
    supersede_suggestions(
        con, models_root, row["profile_id"], keep_parent=row["id"], except_id=row["id"]
    )
    con.execute(
        "UPDATE generations SET status = 'retired' WHERE profile_id = ? AND status = 'active' AND id != ?",
        (row["profile_id"], row["id"]),
    )
    if row["status"] != "active":
        con.execute(
            "UPDATE generations SET status = 'active', activated_at = ? WHERE id = ?",
            (time.time(), row["id"]),
        )


def _with_meta(row: sqlite3.Row, key: str, value: dict) -> str:
    try:
        meta = json.loads(row["meta"]) if row["meta"] else {}
    except ValueError:
        meta = {}
    meta[key] = clean_meta(value)
    return json.dumps(meta, ensure_ascii=False)


def mark_broken(
    con: sqlite3.Connection, row: sqlite3.Row, reason: str, models_root: Path | None = None
) -> None:
    """重みと確信度の組が使えない世代を `broken` にし、使用中だったら出荷時の組へ戻す(理由コードを残す。学習画面に警告が出る)。"""
    log.warning("generation_invalid number=%s reason=%s", row["number"], reason)
    with transaction(con):
        cur = con.execute("SELECT * FROM generations WHERE id = ?", (row["id"],)).fetchone()
        if cur is None or cur["status"] in UNUSABLE:
            return
        was_active = cur["status"] == "active"
        con.execute(
            "UPDATE generations SET status = 'broken', reject_reason = ?, ack = 0, meta = ? WHERE id = ?",
            (reason, _with_meta(cur, "broken", {"code": reason, "at": time.time()}), cur["id"]),
        )
        if was_active:
            _activate_row(con, shipped(con, cur["profile_id"]), models_root)


def weights_for(
    con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID
) -> Weights:
    """新しく認識するページに使う重み(と確信度)。使用中の世代の組が壊れていたら、出荷時の組に戻す(認識を止めない。戻したことは状態と学習画面の警告に残す)。
    使用中の世代を親とする提案モードの世代があり、その組が正しければ `.suggest` に入れる(組が壊れていれば `broken` にして、提案なしで認識する)。"""
    row = active(con, profile_id)
    if row["kind"] != "shipped":
        reason = verify_pair(models_root, row)
        if reason is not None:
            mark_broken(con, row, reason, models_root)
            s = shipped(con, profile_id)
            return Weights(s["id"], 0, suggest=_suggestion(con, models_root, profile_id, s["id"]))
        d = gen_path(models_root, row)
        assert d is not None
        return Weights(
            row["id"],
            row["number"],
            d,
            d / CONFIDENCE_FILE,
            _suggestion(con, models_root, profile_id, row["id"]),
        )
    s = shipped(con, profile_id)
    return Weights(s["id"], 0, suggest=_suggestion(con, models_root, profile_id, s["id"]))


def check_all(con: sqlite3.Connection, models_root: Path) -> dict:
    """全世代の重みと確信度の組を照合する(バックアップからの復元の後など)。使用中・提案モードの世代は `weights_for` と同じ扱い
    (壊れていれば出荷時の組へ戻し、警告を出す)。過去の世代は、壊れていれば `broken` にする(戻せなくなる)。返り値: 件数だけ。"""
    for p in con.execute("SELECT id FROM profiles").fetchall():
        weights_for(con, models_root, p["id"])
    checked = broken = 0
    for r in con.execute(
        "SELECT * FROM generations WHERE kind != 'shipped' AND status IN ('active', 'retired', 'suggest')"
    ).fetchall():
        checked += 1
        reason = verify_pair(models_root, r)
        if reason is not None:
            mark_broken(con, r, reason, models_root)
            broken += 1
    return {"checked": checked, "broken": broken}


def _suggestion(
    con: sqlite3.Connection, models_root: Path, profile_id: int, parent_id: int
) -> Weights | None:
    r = con.execute(
        "SELECT * FROM generations WHERE profile_id = ? AND status = 'suggest' AND parent_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (profile_id, parent_id),
    ).fetchone()
    if r is None:
        return None
    reason = verify_pair(models_root, r)
    if reason is not None:
        mark_broken(con, r, reason, models_root)
        return None
    d = gen_path(models_root, r)
    assert d is not None
    return Weights(r["id"], r["number"], d, d / CONFIDENCE_FILE)


def supersede_suggestions(
    con: sqlite3.Connection,
    models_root: Path | None,
    profile_id: int,
    keep_parent: int | None = None,
    except_id: int | None = None,
) -> int:
    """提案モードの世代を片づける(置き換えられた / 親が使用中でなくなった)。`keep_parent` の世代を親とするもの、`except_id`(昇格する世代)は残す。
    重みは消し、行は `rejected`(理由 superseded)にする。"""
    n = 0
    for r in con.execute(
        "SELECT * FROM generations WHERE profile_id = ? AND status = 'suggest'", (profile_id,)
    ).fetchall():
        if (keep_parent is not None and r["parent_id"] == keep_parent) or r["id"] == except_id:
            continue
        if models_root is not None:
            _remove_dir(models_root, r)
        con.execute(
            "UPDATE generations SET status = 'rejected', reject_reason = 'superseded' WHERE id = ?",
            (r["id"],),
        )
        n += 1
    return n


def recognized_count(con: sqlite3.Connection, gen_id: int) -> int:
    """この世代で認識したページの数(採用直後のウォームアップの判定に使う)。"""
    return int(
        con.execute("SELECT COUNT(*) FROM pages WHERE generation = ?", (gen_id,)).fetchone()[0]
    )


# ------------------------------------------------------------------ 一覧・通知


def _meta_of(r: sqlite3.Row) -> dict:
    try:
        return json.loads(r["meta"]) if r["meta"] else {}
    except ValueError:
        return {}


def _dict(models_root: Path, r: sqlite3.Row, n_recognized: int | None = None) -> dict:
    meta = _meta_of(r)
    return {
        "id": r["id"],
        "number": r["number"],
        "kind": r["kind"],
        "parent_id": r["parent_id"],
        "created_at": r["created_at"],
        "activated_at": r["activated_at"],
        "status": r["status"],
        "reject_reason": r["reject_reason"],
        "ack": bool(r["ack"]),
        "n_train_pages": meta.get("n_train_pages", r["n_pages"]),
        "n_eval_pages": meta.get("n_eval_pages"),
        "eval": meta.get(
            "eval"
        ),  # {parent: {...}, candidate: {...}} 正解率・要確認率・見逃し率・評価セル数
        "recalibration": meta.get("recalibration"),
        "halt": meta.get("halt"),
        "suggest": meta.get("suggest"),
        "restorable": r["status"] in RESTORABLE and _has_weights(models_root, r),
        "n_recognized": n_recognized,
    }


def list_generations(
    con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID
) -> list[dict]:
    rows = con.execute(
        "SELECT g.*, (SELECT COUNT(*) FROM pages p JOIN jobs j ON j.id = p.job_id"
        "   WHERE j.profile_id = g.profile_id AND p.generation = g.id) AS n_recognized,"
        " (SELECT COUNT(*) FROM pages p JOIN jobs j ON j.id = p.job_id"
        "   WHERE j.profile_id = g.profile_id AND p.suggest_generation = g.id) AS n_suggested"
        " FROM generations g WHERE g.profile_id = ? ORDER BY g.number, g.id",
        (profile_id,),
    ).fetchall()
    active_id = next((r["id"] for r in rows if r["status"] == "active"), None)
    out = []
    for r in rows:
        d = _dict(models_root, r, r["n_recognized"])
        d["n_suggested"] = r["n_suggested"]  # この世代の提案モードで認識したページ数
        d["suggest_in_effect"] = r["status"] == "suggest" and r["parent_id"] == active_id
        out.append(d)
    return out


def notices(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> list[dict]:
    """利用者に知らせるべきこと: 使えなくなった世代(重みと確信度の組が壊れた / 安全のため停止した)で、まだ確認されていないもの。"""
    out = []
    for r in con.execute(
        "SELECT * FROM generations WHERE profile_id = ? AND status IN ('broken', 'halted') AND ack = 0"
        " ORDER BY id",
        (profile_id,),
    ):
        meta = _meta_of(r)
        info = meta.get("halt") or meta.get("broken") or {}
        out.append(
            {
                "id": r["id"],
                "number": r["number"],
                "status": r["status"],
                "code": r["reject_reason"],
                "at": info.get("at"),
            }
        )
    return out


def acknowledge(con: sqlite3.Connection, gen_id: int) -> None:
    with transaction(con):
        cur = con.execute("UPDATE generations SET ack = 1 WHERE id = ?", (gen_id,))
        if cur.rowcount == 0:
            raise StoreError(404, "generation_not_found")


def halt_cutoff(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> float:
    """安全のため停止した世代がある場合、その時点までに確定したデータの最後の時刻(次の追加学習は、これより後に確定したデータで判断する)。"""
    best = 0.0
    for r in con.execute(
        "SELECT meta FROM generations WHERE profile_id = ? AND status = 'halted'", (profile_id,)
    ):
        try:
            h = (json.loads(r["meta"]) if r["meta"] else {}).get("halt") or {}
        except ValueError:
            continue
        best = max(best, float(h.get("cutoff") or 0.0))
    return best


# ------------------------------------------------------------------ ロールバック・リセット・停止


def activate(con: sqlite3.Connection, models_root: Path, gen_id: int) -> dict:
    """この世代を使用中にする(ロールバック)。重みと確信度の組が正しく残っている世代(過去の世代・出荷時)にだけできる。以降に認識するページから使う。"""
    with transaction(con):
        r = con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        if r is None:
            raise StoreError(404, "generation_not_found")
        if r["status"] not in RESTORABLE:
            raise StoreError(409, "generation_not_restorable", {"status": r["status"]})
        reason = verify_pair(models_root, r)
        if reason is not None:
            raise StoreError(409, "generation_weights_invalid", {"reason": reason})
        _activate_row(con, r, models_root)
    return _dict(
        models_root, con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    )


def _remove_dir(models_root: Path, r: sqlite3.Row) -> None:
    d = gen_path(models_root, r)  # 決まった形の置き場だけを消す(出荷時の重みや他の世代には触れない)
    if d is not None and d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def reset(con: sqlite3.Connection, models_root: Path, profile_id: int = DEFAULT_PROFILE_ID) -> int:
    """出荷時の重みに戻す: 世代 0 を使用中にし、利用者のデータから作った世代の重みを消す(行は「リセットで捨てた」として残す)。消した世代数を返す。"""
    n = 0
    with transaction(con):
        s = shipped(con, profile_id)
        now = time.time()
        rows = con.execute(
            "SELECT * FROM generations WHERE profile_id = ? AND kind = 'trained'"
            " AND status IN ('active', 'retired', 'broken', 'halted', 'suggest')",
            (profile_id,),
        ).fetchall()
        for r in rows:
            _remove_dir(models_root, r)
            con.execute(
                "UPDATE generations SET status = 'discarded', ack = 1 WHERE id = ?", (r["id"],)
            )
            n += 1
        con.execute(
            "UPDATE generations SET status = 'active', activated_at = ? WHERE id = ?",
            (now, s["id"]),
        )
    return n


def _restorable_ancestor(con: sqlite3.Connection, models_root: Path, r: sqlite3.Row) -> sqlite3.Row:
    """親の世代をたどって、戻せる最初の世代(最後は出荷時)。"""
    seen = {r["id"]}
    pid = r["parent_id"]
    while pid is not None and pid not in seen:
        seen.add(pid)
        p = con.execute("SELECT * FROM generations WHERE id = ?", (pid,)).fetchone()
        if p is None:
            break
        if p["kind"] == "shipped" or (
            p["status"] in RESTORABLE and verify_pair(models_root, p) is None
        ):
            return p
        pid = p["parent_id"]
    return shipped(con, r["profile_id"])


def halt(
    con: sqlite3.Connection, models_root: Path, gen_id: int, code: str, info: dict | None = None
) -> dict:
    """採用した世代を安全のため停止し(`halted`。戻せない)、使用中だったら親の世代(戻せなければさらにその親、最後は出荷時)に戻す。
    `info` は件数・割合・時刻だけ(世代のメタに残る)。`info["cutoff"]` は、次の追加学習が「これより後に確定したデータ」で判断するための起点。"""
    with transaction(con):
        r = con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        if r is None or r["kind"] != "trained":
            raise StoreError(404, "generation_not_found")
        if r["status"] in UNUSABLE:
            return _dict(models_root, r)
        was_active = r["status"] == "active"
        con.execute(
            "UPDATE generations SET status = 'halted', reject_reason = ?, ack = 0, meta = ? WHERE id = ?",
            (
                code,
                _with_meta(r, "halt", {**(info or {}), "code": code, "at": time.time()}),
                gen_id,
            ),
        )
        if was_active:
            _activate_row(con, _restorable_ancestor(con, models_root, r), models_root)
    log.warning("generation_halted number=%s code=%s", r["number"], code)
    return _dict(
        models_root, con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    )


def next_number(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> int:
    return int(
        con.execute(
            "SELECT COALESCE(MAX(number), 0) + 1 FROM generations WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()[0]
    )


# ------------------------------------------------------------------ 登録


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
    suggest: bool = False,
) -> int:
    """学習で作った世代を登録する。採用なら使用中にし、前の世代は過去(retired)にする。不採用なら rejected で行だけ残し、重みは消す。
    `suggest` なら提案モード(重みを残す。`reject_reason` に、採用基準に届かなかった安全側の理由を残す)。提案モードの世代は 1 つだけ。
    `(profile_id, number)` は一意(DB の制約)。メタデータは `clean_meta` を通ったものだけ。"""
    meta = clean_meta(meta)  # type: ignore[assignment]
    with transaction(con):
        now = time.time()
        cur = con.execute(
            "INSERT INTO generations (profile_id, parent_id, created_at, n_pages, score, status, number, kind, dir, meta,"
            " reject_reason, activated_at) VALUES (?, ?, ?, ?, NULL, ?, ?, 'trained', ?, ?, ?, ?)",
            (
                profile_id,
                parent_id,
                now,
                int(meta.get("n_train_pages", 0)),  # type: ignore[attr-defined]
                "active" if adopt else "suggest" if suggest else "rejected",
                number,
                expected_dir(profile_id, number) if keep_dir else None,
                json.dumps(meta, ensure_ascii=False),
                None if adopt else reject_reason,
                now if adopt or suggest else None,
            ),
        )
        gid = int(cur.lastrowid)
        if adopt:
            con.execute(
                "UPDATE generations SET status = 'retired' WHERE profile_id = ? AND status = 'active' AND id != ?",
                (profile_id, gid),
            )
            supersede_suggestions(
                con, models_root, profile_id
            )  # 親が変わるので、提案モードの世代は片づける
        elif suggest:  # 提案モードは 1 つだけ。前のものは置き換える(自分は残す)
            for r in con.execute(
                "SELECT * FROM generations WHERE profile_id = ? AND status = 'suggest' AND id != ?",
                (profile_id, gid),
            ).fetchall():
                _remove_dir(models_root, r)
                con.execute(
                    "UPDATE generations SET status = 'rejected', reject_reason = 'superseded' WHERE id = ?",
                    (r["id"],),
                )
    return gid


def promote(
    con: sqlite3.Connection, models_root: Path, gen_id: int, info: dict | None = None
) -> dict:
    """提案モードの世代が、あとで採用基準をすべて満たしたとき、通常の採用にする(使用中にし、親は過去の世代にする)。"""
    with transaction(con):
        r = con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        if r is None or r["status"] != "suggest":
            raise StoreError(409, "generation_not_suggest")
        reason = verify_pair(models_root, r)
        if reason is not None:
            raise StoreError(409, "generation_weights_invalid", {"reason": reason})
        con.execute(
            "UPDATE generations SET reject_reason = NULL, meta = ? WHERE id = ?",
            (_with_meta(r, "promoted", {**(info or {}), "at": time.time()}), gen_id),
        )
        _activate_row(con, r, models_root)
    log.info("generation_promoted number=%s", r["number"])
    return _dict(
        models_root, con.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    )


def _inside(root: Path, p: Path) -> bool:
    try:
        return root.resolve() in p.resolve().parents
    except OSError:
        return False


def install(
    con: sqlite3.Connection,
    models_root: Path,
    *,
    profile_id: int,
    number: int,
    parent_id: int | None,
    meta: dict,
    weights_dir: Path,
    confidence: Path,
    run_dir: Path,
    suggest: bool = False,
    reject_reason: str | None = None,
) -> int:
    """採用する世代(`suggest` なら提案モードの世代)の重み・確信度・メタデータを置き場(`gen_XXXX`)に置いて、使用中として登録する。
    学習器が返したパス(`weights_dir`・`confidence`)は `run_dir` の内側でなければならない。重みの SHA256 を `meta.json` と `confidence.json` の両方に記録する(組の照合用)。
    途中で失敗したら置き場を残さない。"""
    for p in (weights_dir, confidence):
        if not _inside(run_dir, p):
            raise ValueError("result_path_outside_run_dir")
    dest = models_root / str(profile_id) / gen_dir_name(number)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    placed = False
    try:
        for f in WEIGHT_FILES:
            shutil.copy2(weights_dir / f, tmp / f)
        hashes = weights_hashes(tmp)
        conf = json.loads(confidence.read_text(encoding="utf-8"))
        if not isinstance(conf, dict):
            raise ValueError("confidence_format")
        conf["weights_sha256"] = hashes
        (tmp / CONFIDENCE_FILE).write_text(
            json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        meta = clean_meta({**meta, "weights_sha256": hashes})  # type: ignore[assignment]
        (tmp / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
        placed = True
        return register(
            con,
            models_root,
            profile_id=profile_id,
            number=number,
            parent_id=parent_id,
            meta=meta,
            adopt=not suggest,
            suggest=suggest,
            reject_reason=reject_reason,
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if placed:
            shutil.rmtree(dest, ignore_errors=True)
        raise
