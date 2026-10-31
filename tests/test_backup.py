"""バックアップと復元(ユーザー要件 B5): zip の検証(悪意のある zip・壊れた zip の拒否)、往復(バックアップ → リセット → 復元 → 同じ状態)、入れ替えの失敗時の巻き戻し。
合成データだけを使う。zip の解析は別プロセス(`backup_worker`)で行う(サーバのプロセスでは解かない)。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from _app_helpers import confirm_with
from yayoi_ocr.app import backup, gens
from yayoi_ocr.app import db as appdb
from yayoi_ocr.app.backup import BackupError, Limits
from yayoi_ocr.app.security import allowed_hosts_for, new_token
from yayoi_ocr.app.server import close_app, create_app
from yayoi_ocr.app.worker import FakeBackend

PORT = 8133
TOKEN = new_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------- zip を組み立てる補助


def db_bytes(tmp_path: Path, mutate=None, name: str = "x.sqlite3") -> bytes:
    p = tmp_path / name
    con = appdb.connect(p)
    appdb.migrate(con)
    if mutate is not None:
        mutate(con)
    con.close()
    return p.read_bytes()


def build_zip(
    path: Path,
    files: dict[str, bytes],
    *,
    manifest: dict | None = None,
    hashlist: str | None = None,
    with_manifest: bool = True,
    with_hashlist: bool = True,
    extra: dict[str, bytes] | None = None,
    schema: int | None = None,
) -> Path:
    """正しい zip を組み立てる。引数で壊す。"""
    listing = "".join(f"{hashlib.sha256(b).hexdigest()} {len(b)} {n}\n" for n, b in files.items())
    text = (hashlist if hashlist is not None else listing).encode()
    m = {
        "format": backup.FORMAT,
        "version": backup.FORMAT_VERSION,
        "created_at": 1.0,
        "app_version": "t",
        "schema": schema if schema is not None else len(appdb.MIGRATIONS),
        "includes_originals": any(n.startswith("uploads/") for n in files),
        "n_files": len(files),
        "total_bytes": sum(len(b) for b in files.values()),
        "n_pages": 0,
        "n_confirmed_pages": 0,
        "n_generations": 0,
        "list_sha256": hashlib.sha256(text).hexdigest(),
    }
    m.update(manifest or {})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in {**files, **(extra or {})}.items():
            zi = zipfile.ZipInfo(n)
            zi.external_attr = (stat.S_IFREG | 0o644) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, b)
        if with_hashlist:
            zf.writestr(backup.HASHLIST, text)
        if with_manifest:
            zf.writestr(backup.MANIFEST, json.dumps(m))
    return path


def good_files(tmp_path: Path) -> dict[str, bytes]:
    return {
        backup.DB_ENTRY: db_bytes(tmp_path),
        "images/1/1.png": b"png-bytes",
        "images/1/view.jpg": b"jpg-bytes",
        "pagedata/1.npz": b"npz",
        "exports/export_0001_20260101_支払.csv": "日付,金額\n".encode(),
    }


def code_of(path: Path, **kw) -> str:
    with pytest.raises(BackupError) as ei:
        backup.inspect_archive(path, **kw)
    return ei.value.code


# ---------------------------------------------------------------- 名前の検査


@pytest.mark.parametrize(
    "name",
    [
        "db/app.sqlite3",
        "images/12/345.png",
        "images/12/view.jpg",
        "pagedata/7.npz",
        "models/1/gen_0001/meta.json",
        "models/1/gen_0001/confidence.json",
        "models/1/gen_0012/parseq-ndl-24x256-30-tiny.pt",
        "uploads/0123456789abcdef0123456789abcdef/original.pdf",
        "exports/export_0001_20260101_現金.csv",
    ],
)
def test_expected_entry_names_are_accepted(name):
    assert backup.entry_problem(name) is None


@pytest.mark.parametrize(
    "name",
    [
        "../evil.txt",
        "images/../../evil.png",
        "/etc/passwd",
        "images\\1\\2.png",
        "C:/Windows/x.png",
        "images//1.png",
        "images/1/2.PNG",
        "images/a/b.png",
        "models/1/gen_0001/evil.pt",
        "models/1/gen_1/meta.json",
        "uploads/xyz/original.pdf",
        "uploads/0123456789abcdef0123456789abcdef/evil.exe",
        "exports/a/b.csv",
        "exports/CON",
        "exports/nul.csv",
        "exports/trailing.",
        "exports/a:b.csv",
        "exports/a\x00b.csv",
        "exports/a*b.csv",
        "somewhere/else.txt",
        "app.sqlite3",
        "",
    ],
)
def test_unexpected_entry_names_are_rejected(name):
    assert backup.entry_problem(name) == "entry_name_invalid"


# ---------------------------------------------------------------- 悪意のある・壊れた zip


def test_a_valid_archive_is_inspected_and_extracted_with_matching_files(tmp_path):
    files = good_files(tmp_path)
    z = build_zip(tmp_path / "ok.zip", files)
    s = backup.inspect_archive(z)
    assert s["n_files"] == len(files) and s["total_bytes"] == sum(map(len, files.values()))
    out = tmp_path / "stage"
    backup.extract_archive(z, out)
    for n, b in files.items():
        assert (out / n).read_bytes() == b
    with pytest.raises(BackupError, match="staging_exists"):
        backup.extract_archive(z, out)  # 既にあるディレクトリには取り出さない


def test_a_file_that_is_not_a_zip_is_rejected(tmp_path):
    p = tmp_path / "no.zip"
    p.write_bytes(b"this is not a zip" * 100)
    assert code_of(p) == "not_a_zip"
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    "bad",
    [
        "../evil.txt",
        "/abs.txt",
        "images\\1\\2.png",
        "C:/x.png",
        "images/1/../../x.png",
        "exports/CON",
    ],
)
def test_entries_that_escape_or_are_not_in_the_allow_list_are_rejected(tmp_path, bad):
    z = build_zip(tmp_path / "a.zip", good_files(tmp_path), extra={bad: b"x"})
    # Windows の zipfile は、名前の `\` を `/` に直して読み書きする(その場合は普通の名前になり、ファイル一覧との不一致で断られる)
    want = {"entry_name_invalid"} | (
        {"file_list_mismatch"} if os.name == "nt" and "\\" in bad else set()
    )
    assert code_of(z) in want
    stage = tmp_path / "stage"
    with pytest.raises(BackupError):
        backup.extract_archive(z, stage)
    assert not stage.exists() and not (tmp_path.parent / "evil.txt").exists()


def test_symlink_directory_encrypted_and_unsupported_entries_are_rejected(tmp_path):
    files = good_files(tmp_path)
    z = tmp_path / "s.zip"
    build_zip(z, files)
    with zipfile.ZipFile(z, "a") as zf:  # シンボリックリンク(中身は領域の外を指す)
        zi = zipfile.ZipInfo("exports/link.csv")
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, "/etc/passwd")
    assert code_of(z) == "entry_type_invalid"
    z2 = tmp_path / "d.zip"
    build_zip(z2, files)
    with zipfile.ZipFile(z2, "a") as zf:
        zf.writestr(zipfile.ZipInfo("exports/dir/"), b"")
    assert code_of(z2) in ("entry_type_invalid", "entry_name_invalid")
    z3 = tmp_path / "bz.zip"
    build_zip(z3, files)
    with zipfile.ZipFile(z3, "a") as zf:
        zf.writestr(zipfile.ZipInfo("exports/b.csv"), b"x", compress_type=zipfile.ZIP_BZIP2)
    assert code_of(z3) == "entry_type_invalid"


def test_names_that_differ_only_by_case_are_duplicates(tmp_path):
    files = {**good_files(tmp_path), "exports/A.csv": b"1", "exports/a.csv": b"2"}
    assert code_of(build_zip(tmp_path / "dup.zip", files)) == "entry_duplicate"


def test_tampered_content_a_missing_or_extra_file_and_a_changed_hash_list_are_rejected(tmp_path):
    files = good_files(tmp_path)
    # 中身を差し替える(一覧のハッシュは元のまま)
    good = build_zip(tmp_path / "g.zip", files)
    listing = "".join(f"{hashlib.sha256(b).hexdigest()} {len(b)} {n}\n" for n, b in files.items())
    tampered = build_zip(
        tmp_path / "t.zip",
        {**files, "images/1/1.png": b"PNG-BYTES"},  # 同じ長さで中身が違う
        hashlist=listing,
    )
    assert code_of(tampered) == "hash_mismatch"
    # 一覧に無い余分なファイル
    assert (
        code_of(build_zip(tmp_path / "e.zip", files, extra={"pagedata/9.npz": b"x"}))
        == "file_list_mismatch"
    )
    # 一覧の改ざん(manifest のハッシュと合わない)
    assert (
        code_of(build_zip(tmp_path / "h.zip", files, hashlist=listing + "# x\n"))
        == "hash_list_mismatch"
    )
    # 一覧にあるのに zip に無いファイル
    miss = build_zip(
        tmp_path / "m.zip",
        {k: v for k, v in files.items() if k != "pagedata/1.npz"},
        hashlist=listing,
        manifest={"n_files": len(files), "total_bytes": sum(map(len, files.values()))},
    )
    assert code_of(miss) == "file_list_mismatch"
    assert backup.inspect_archive(good)["n_files"] == len(files)


def test_manifest_problems_are_rejected(tmp_path):
    files = good_files(tmp_path)
    assert code_of(build_zip(tmp_path / "1.zip", files, with_manifest=False)) == "manifest_missing"
    assert (
        code_of(build_zip(tmp_path / "2.zip", files, manifest={"format": "other"}))
        == "format_unsupported"
    )
    assert (
        code_of(build_zip(tmp_path / "3.zip", files, manifest={"version": 99}))
        == "version_unsupported"
    )
    assert (
        code_of(build_zip(tmp_path / "4.zip", files, manifest={"n_files": "many"}))
        == "manifest_invalid"
    )
    assert (
        code_of(build_zip(tmp_path / "5.zip", files, schema=len(appdb.MIGRATIONS) + 1))
        == "db_too_new"
    )
    no_db = {k: v for k, v in files.items() if k != backup.DB_ENTRY}
    assert code_of(build_zip(tmp_path / "6.zip", no_db)) == "db_missing"
    # 元画像を含まないと書いてあるのに uploads/ がある
    with_up = {**files, "uploads/" + "a" * 32 + "/original.pdf": b"%PDF"}
    assert (
        code_of(build_zip(tmp_path / "7.zip", with_up, manifest={"includes_originals": False}))
        == "file_list_mismatch"
    )


def test_size_count_and_compression_ratio_limits(tmp_path):
    files = good_files(tmp_path)
    z = build_zip(tmp_path / "l.zip", files)
    assert code_of(z, limits=Limits(max_entries=3)) == "too_many_entries"
    assert code_of(z, limits=Limits(max_total_bytes=100)) == "total_too_large"
    assert code_of(z, limits=Limits(max_entry_bytes=10)) == "entry_too_large"
    bomb = build_zip(tmp_path / "bomb.zip", {**files, "pagedata/2.npz": b"\0" * (8 << 20)})
    assert code_of(bomb, limits=Limits(min_ratio_bytes=1 << 20, max_ratio=100)) == "ratio_too_high"
    assert (
        code_of(bomb) == "ratio_too_high"
    )  # 既定の上限でも拒否する(4 MB 以上で圧縮率が 200 倍を超える)
    small = build_zip(tmp_path / "small.zip", {**files, "pagedata/2.npz": b"\0" * (2 << 20)})
    assert (
        backup.inspect_archive(small)["n_files"] == len(files) + 1
    )  # 小さいファイルは(圧縮率が高くても)通す


def test_the_declared_size_is_not_trusted_the_actual_bytes_are_counted(tmp_path):
    files = good_files(tmp_path)
    z = build_zip(tmp_path / "x.zip", files)
    # 一覧の大きさを実際より小さく書き換えた zip(manifest の合計・一覧のハッシュも合わせる)
    lie = {
        n: (hashlib.sha256(b).hexdigest(), len(b) - 1 if n == "pagedata/1.npz" else len(b))
        for n, b in files.items()
    }
    text = "".join(f"{d} {s} {n}\n" for n, (d, s) in lie.items())
    zz = build_zip(
        tmp_path / "lie.zip",
        files,
        hashlist=text,
        manifest={"total_bytes": sum(s for _, s in lie.values())},
    )
    assert code_of(zz) == "size_mismatch"
    assert backup.inspect_archive(z)["n_files"] == len(files)


# ---------------------------------------------------------------- 取り出した DB の検査


def test_a_database_with_extra_schema_objects_or_unsafe_paths_is_rejected(tmp_path):
    def with_trigger(con):
        con.executescript(
            "CREATE TABLE evil (x); CREATE TRIGGER t AFTER INSERT ON settings BEGIN INSERT INTO evil VALUES (1); END;"
        )

    def with_view(con):
        con.executescript("CREATE VIEW v AS SELECT * FROM settings;")

    def with_bad_crop(con):
        con.execute(
            "INSERT INTO jobs (id, name, profile_id, status, created_at) VALUES (1, 'j', 1, 'done', 1.0)"
        )
        con.execute("INSERT INTO pages (id, job_id, idx, status) VALUES (1, 1, 0, 'confirmed')")
        con.execute(
            "INSERT INTO cells (page_id, row, col, column_kind, kind, crop_path) VALUES (1, 0, 0, 'date', 'text', '../../secret.png')"
        )

    def with_bad_gen_dir(con):
        con.execute(
            "UPDATE generations SET dir = '../outside', kind = 'user' WHERE kind = 'shipped'"
        )

    for i, (mut, want) in enumerate(
        [
            (with_trigger, "db_schema_mismatch"),
            (with_view, "db_schema_mismatch"),
            (with_bad_crop, "db_path_invalid"),
            (with_bad_gen_dir, "db_path_invalid"),
        ]
    ):
        files = {**good_files(tmp_path), backup.DB_ENTRY: db_bytes(tmp_path, mut, f"m{i}.sqlite3")}
        z = build_zip(tmp_path / f"d{i}.zip", files)
        stage = tmp_path / f"stage{i}"
        with pytest.raises(BackupError) as ei:
            backup.extract_archive(z, stage)
        assert ei.value.code == want, i
        assert not stage.exists()  # 失敗したら取り出したものを消す


def test_database_paths_with_windows_separators_are_accepted_but_traversal_is_not(tmp_path):
    def with_paths(sep: str, first: str):
        def mut(con):
            con.execute(
                "INSERT INTO jobs (id, name, profile_id, status, created_at) VALUES (1, 'j', 1, 'done', 1.0)"
            )
            con.execute(
                "INSERT INTO pages (id, job_id, idx, status, view_path) VALUES (1, 1, 0, 'confirmed', ?)",
                (f"1{sep}view.jpg",),
            )
            con.execute(
                "INSERT INTO cells (page_id, row, col, column_kind, kind, crop_path) VALUES (1, 0, 0, 'date', 'text', ?)",
                (first,),
            )

        return mut

    for i, (sep, first, ok) in enumerate(
        [
            ("/", "1/1.png", True),
            ("\\", "1\\1.png", True),  # Windows では DB にこの形で入る(os.sep)
            ("\\", "..\\..\\secret.png", False),
            ("/", "C:/Windows/x.png", False),
            ("/", "1/1.png\n", False),
        ]
    ):
        path = tmp_path / f"p{i}.sqlite3"
        path.write_bytes(db_bytes(tmp_path, with_paths(sep, first), f"q{i}.sqlite3"))
        if ok:
            assert backup.check_db(path) == len(appdb.MIGRATIONS)
        else:
            with pytest.raises(BackupError) as ei:
                backup.check_db(path)
            assert ei.value.code == "db_path_invalid", (i, first)


def test_a_corrupted_database_is_rejected(tmp_path):
    raw = bytearray(db_bytes(tmp_path))
    raw[100:140] = b"\xff" * 40
    files = {**good_files(tmp_path), backup.DB_ENTRY: bytes(raw)}
    with pytest.raises(BackupError) as ei:
        backup.extract_archive(build_zip(tmp_path / "c.zip", files), tmp_path / "stage")
    assert ei.value.code in ("db_invalid", "db_schema_mismatch")
    garbage = {**good_files(tmp_path), backup.DB_ENTRY: b"not a database" * 50}
    with pytest.raises(BackupError) as ei:
        backup.extract_archive(build_zip(tmp_path / "g.zip", garbage), tmp_path / "stage2")
    assert ei.value.code == "db_invalid"


# ---------------------------------------------------------------- 入れ替え(起動時)


def make_root(tmp_path: Path, tag: str) -> Path:
    r = tmp_path / "root"
    (r / "images" / "1").mkdir(parents=True)
    (r / "images" / "1" / "1.png").write_bytes(tag.encode())
    (r / "pagedata").mkdir()
    (r / "pagedata" / "1.npz").write_bytes(tag.encode())
    (r / "app.sqlite3").write_bytes(("db-" + tag).encode())
    return r


def stage(root: Path, tag: str) -> None:
    s = root / backup.PENDING_DIR / backup.STAGING
    (s / "db").mkdir(parents=True)
    (s / "db" / "app.sqlite3").write_bytes(("db-" + tag).encode())
    (s / "images" / "2").mkdir(parents=True)
    (s / "images" / "2" / "2.png").write_bytes(tag.encode())
    backup.write_pending(root, {"n_pages": 3})


def test_apply_pending_swaps_the_data_and_keeps_the_old_data_in_undo(tmp_path):
    root = make_root(tmp_path, "old")
    stage(root, "new")
    res = backup.apply_pending(root)
    assert res and res["ok"] and res["summary"] == {"n_pages": 3}
    assert (root / "app.sqlite3").read_bytes() == b"db-new"
    assert (root / "images" / "2" / "2.png").read_bytes() == b"new"
    assert not (root / "images" / "1").exists()  # 古い切り出し画像は復元後の領域に残らない
    assert not (root / "pagedata" / "1.npz").exists()
    undo = root / backup.UNDO_DIR
    assert (undo / "app.sqlite3").read_bytes() == b"db-old"
    assert (undo / "images" / "1" / "1.png").read_bytes() == b"old"
    assert backup.read_pending(root) is None and not (root / backup.PENDING_DIR).exists()
    assert backup.read_result(root)["ok"] is True
    assert backup.undo_info(root)["bytes"] > 0
    assert backup.delete_undo(root) and backup.undo_info(root) is None
    assert backup.apply_pending(root) is None  # もう準備はない


def test_a_failure_in_the_middle_of_the_swap_restores_the_old_data(tmp_path, monkeypatch):
    root = make_root(tmp_path, "old")
    stage(root, "new")
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:  # 4 回目(新しいデータを置く途中)で失敗する
            raise OSError("boom")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    res = backup.apply_pending(root)
    monkeypatch.setattr(os, "replace", real)
    assert res and res["ok"] is False and res["code"] == "error_OSError"
    assert (root / "app.sqlite3").read_bytes() == b"db-old"  # 元のまま
    assert (root / "images" / "1" / "1.png").read_bytes() == b"old"
    assert not (root / "images" / "2").exists()
    assert not (root / backup.UNDO_DIR).exists()


def test_a_transient_permission_error_is_retried_and_the_swap_succeeds(tmp_path, monkeypatch):
    """Windows では、ウイルス対策ソフトなどが一時的にファイルを掴むと PermissionError になる。数回の再試行で通れば入れ替わる。"""
    root = make_root(tmp_path, "old")
    stage(root, "new")
    monkeypatch.setattr(backup, "REPLACE_DELAY_S", 0.0)
    real = os.replace
    fails = {"left": 3}

    def flaky(src, dst):
        if fails["left"] > 0:
            fails["left"] -= 1
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    res = backup.apply_pending(root)
    monkeypatch.setattr(os, "replace", real)
    assert res and res["ok"] is True and fails["left"] == 0
    assert (root / "app.sqlite3").read_bytes() == b"db-new"


def test_a_persistent_permission_error_rolls_back_and_names_the_entry(tmp_path, monkeypatch):
    root = make_root(tmp_path, "old")
    stage(root, "new")
    monkeypatch.setattr(backup, "REPLACE_DELAY_S", 0.0)
    monkeypatch.setattr(backup, "REPLACE_ATTEMPTS", 3)
    real = os.replace

    def locked(src, dst):
        if str(src).endswith("pagedata"):  # pagedata だけがずっと掴まれている
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", locked)
    res = backup.apply_pending(root)
    monkeypatch.setattr(os, "replace", real)
    assert res and res["ok"] is False and res["code"] == "error_PermissionError"
    assert res["entry"] == "pagedata"  # 固定の名前だけ(値・パスは含まない)
    assert (root / "app.sqlite3").read_bytes() == b"db-old"  # 元のまま
    assert (root / "images" / "1" / "1.png").read_bytes() == b"old"
    assert (root / "pagedata" / "1.npz").read_bytes() == b"old"
    assert not (root / backup.UNDO_DIR).exists()


def test_a_pending_mark_without_staged_data_is_ignored_and_cleared(tmp_path):
    root = make_root(tmp_path, "old")
    (root / backup.PENDING_DIR).mkdir()
    backup.write_pending(root, {})
    res = backup.apply_pending(root)
    assert res and res["ok"] is False and res["code"] == "staging_missing"
    assert (root / "app.sqlite3").read_bytes() == b"db-old"
    assert backup.read_pending(root) is None


# ---------------------------------------------------------------- 往復(アプリ全体)


def make(tmp_path: Path):
    app = create_app(
        tmp_path / "app.sqlite3",
        TOKEN,
        allowed_hosts_for(PORT),
        dev=True,
        backend=FakeBackend(error_rate=0.0),
    )
    return app, TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers=AUTH)


def confirm_pages(c, n: int) -> None:
    for i in range(n):
        job = c.post(
            "/api/dev/jobs",
            json={
                "pages": 1,
                "seed": 70 + i,
                "fill_rows": 29,
                "error_rate": 0.0,
                "review_rate": 0.0,
                "ambiguous_rows": [],
            },
        ).json()
        confirm_with(c, job["pages"][0]["id"], {})


def dump(con) -> dict:
    out = {}
    for (t,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        out[t] = sorted(repr(tuple(r)) for r in con.execute(f"SELECT * FROM {t}"))
    return out


def file_hashes(root: Path) -> dict[str, str]:
    out = {}
    for top in backup.DATA_DIRS:
        for dirpath, _d, names in os.walk(root / top):
            for n in names:
                p = Path(dirpath) / n
                out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def adopt_a_generation(c) -> int:
    """モックの学習器で世代 1 を採用させる(重みと確信度の組を持つ世代を作る)。"""
    c.put(
        "/api/settings",
        json={
            **c.get("/api/settings").json()["settings"],
            "auto_train": False,
            "train_threshold_cells": 50,
        },
    )
    assert c.post("/api/dev/trainer", json={"mode": "improve"}).status_code == 200
    assert c.post("/api/training/start").status_code == 200
    c.app.state.trainer.join(20)
    assert c.get("/api/training").json()["last"]["status"] == "adopted"
    return gens.active(c.app.state.db)["id"]


def wait(c) -> dict:
    c.app.state.backup.wait(120)
    return c.get("/api/backup").json()


def test_backup_reset_restore_returns_to_the_same_state(tmp_path):
    app, c = make(tmp_path)
    with c:
        c.app = app
        confirm_pages(c, 7)
        # 設定・摘要ごとの税区分の記憶・世代(重みと確信度の組)・行ごとの税区分
        cur = c.get("/api/settings").json()["settings"]
        c.put(
            "/api/settings",
            json={
                **cur,
                "wizard_done": True,
                "payments": [{"name": "現金", "credit_account": "現金", "credit_sub": ""}],
            },
        )
        gid = adopt_a_generation(c)
        before_db = dump(app.state.db)
        before_files = file_hashes(tmp_path)
        assert any(k.startswith("models/") for k in before_files) and any(
            k.startswith("images/") for k in before_files
        )

        # バックアップを作る(進捗 → できた → ダウンロードの合言葉は 1 回だけ使える)
        assert c.post("/api/backup/create", json={"include_originals": False}).status_code == 200
        st = wait(c)
        assert st["error"] is None and st["created"]["bytes"] > 0
        assert (
            st["created"]["summary"]["n_confirmed_pages"] == 7
            and st["created"]["summary"]["includes_originals"] is False
        )
        url = c.post("/api/backup/download-link").json()["url"]
        r = c.get(url, headers={})  # トークンなし(リンクのダウンロード)
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert "attachment" in r.headers["content-disposition"]
        blob = r.content
        assert c.get(url, headers={}).status_code == 404  # 合言葉は 1 回で失効する
        assert c.get("/download/backup/not-a-real-nonce", headers={}).status_code == 404

        # リセット(切り出し画像・認識の入力・世代を消す)→ 状態が変わる
        assert (
            c.post("/api/generations/reset", json={"delete_learning_data": True}).status_code == 200
        )
        assert file_hashes(tmp_path) != before_files and dump(app.state.db) != before_db

        # 復元: zip を送る → 検査の要約 → 実行(取り出して検査)→ 次の起動で入れ替わる
        assert c.put("/api/backup/upload", content=blob).status_code == 200
        st = wait(c)
        assert st["error"] is None and st["incoming"]["n_confirmed_pages"] == 7
        assert c.post("/api/backup/restore").status_code == 200
        st = wait(c)
        assert st["error"] is None and st["pending"] is not None and st["incoming"] is None
        assert not (tmp_path / backup.PENDING_DIR / backup.INCOMING).exists()
    # アプリを閉じて(DB の接続を閉じる。TestClient の with は close_app を呼ばない)、開き直す(launcher と同じ順: apply_pending → create_app)
    close_app(app)
    res = backup.apply_pending(tmp_path)
    assert res and res["ok"]
    app2, c2 = make(tmp_path)
    with c2:
        assert dump(app2.state.db) == before_db
        assert file_hashes(tmp_path) == before_files
        # 世代の重みと確信度の組の照合を通っていて、そのまま使える
        w = gens.weights_for(app2.state.db, tmp_path / "models")
        assert w.generation_id == gid and w.dir is not None
        st = c2.get("/api/backup").json()
        assert (
            st["result"]["ok"] is True
            and st["result"]["pairs"]["broken"] == 0
            and st["result"]["pairs"]["checked"] >= 1
        )
        assert st["undo"] is not None  # 復元前のデータは退避されている
        assert c2.post("/api/backup/ack").json()["result"] is None
        assert c2.delete("/api/backup/undo").json()["undo"] is None
    close_app(app2)


def test_restore_is_refused_while_recognition_or_training_is_running(tmp_path):
    app, c = make(tmp_path)
    with c:
        confirm_pages(c, 1)
        assert c.post("/api/backup/create", json={}).status_code == 200
        blob = (tmp_path / "backups" / wait(c)["created"]["name"]).read_bytes()
        app.state.backup.busy = lambda: True
        r = c.put("/api/backup/upload", content=blob)
        assert r.status_code == 409 and r.json()["error"] == "busy"
        app.state.backup.busy = lambda: False
        assert c.put("/api/backup/upload", content=blob).status_code == 200
        wait(c)
        app.state.backup.busy = lambda: True
        r = c.post("/api/backup/restore")
        assert r.status_code == 409 and r.json()["error"] == "busy"


def test_a_bad_upload_is_rejected_with_a_code_and_leaves_nothing_behind(tmp_path):
    _app, c = make(tmp_path)
    with c:
        assert c.put("/api/backup/upload", content=b"").status_code == 422
        assert (
            c.put("/api/backup/upload", content=b"PK\x03\x04 broken" * 10).status_code == 200
        )  # 受け取って、検査で失敗する
        st = wait(c)
        assert st["error"] == "not_a_zip" and st["incoming"] is None
        assert not (tmp_path / backup.PENDING_DIR).exists()
        assert c.post("/api/backup/restore").status_code == 409  # 検査済みの zip がない


def test_the_status_and_errors_contain_no_cell_values_or_paths(tmp_path):
    _app, c = make(tmp_path)
    with c:
        confirm_pages(c, 1)
        secret = c.get("/api/pages/1").json()["cells"][0]["value"]
        c.post("/api/backup/create", json={})
        st = wait(c)
        text = json.dumps(st, ensure_ascii=False)
        assert str(tmp_path) not in text
        if secret:
            assert secret not in text
        c.put("/api/backup/upload", content=b"junk")
        text = json.dumps(wait(c), ensure_ascii=False)
        assert str(tmp_path) not in text


def test_a_second_operation_while_one_is_running_is_refused(tmp_path):
    app, c = make(tmp_path)
    with c:
        mgr = app.state.backup
        with mgr._mu:
            mgr._state["running"] = True
        assert c.post("/api/backup/create", json={}).status_code == 409
        with mgr._mu:
            mgr._state["running"] = False


def test_stale_download_files_are_removed_at_startup_but_a_staged_restore_is_kept(tmp_path):
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups" / "yayoi-backup-old.zip").write_bytes(b"x")
    stage(tmp_path, "new")
    _app, c = make(tmp_path)
    with c:
        assert not (tmp_path / "backups").exists()
        assert backup.read_pending(tmp_path) is not None
        assert c.get("/api/backup").json()["pending"]["summary"] == {"n_pages": 3}
        assert c.post("/api/backup/cancel").json()["pending"] is None
        assert not (tmp_path / backup.PENDING_DIR).exists()
