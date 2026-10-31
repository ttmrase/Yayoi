"""バックアップと復元(ユーザー要件 B5)。データ領域を 1 つの zip にし、zip から戻す。

zip の構成(`FORMAT_VERSION` 1):
  manifest.json   形式・版・作成日時・アプリの版・DB のスキーマ版・元画像を含むか・件数・ファイル一覧のハッシュ(小さい JSON)
  files.sha256    1 行 1 ファイル `sha256 サイズ パス`(ファイルが多くても、行ごとに読める)
  db/app.sqlite3  DB。**SQLite の backup API の整合の取れたスナップショット**(動いている DB のファイルをそのままコピーしない)
  images/ pagedata/ models/ exports/ [uploads/]   切り出し画像・認識の入力・世代の重み/確信度/メタ・出力ファイル・(任意で)元画像
含めないもの: 追加学習の作業ディレクトリ(training/)・復元の作業用ディレクトリ。

**zip は信用しない**(利用者が別の場所から持ってきたファイルかもしれない)。`inspect` / `extract` は、`backup_worker`(使い捨ての別プロセス、`python -I`)で動かし、サーバのプロセスでは zip を解かない:
  - 名前は許可リスト(決まった形のパスだけ)。`..`・絶対パス・`\\`・ドライブ文字・Windows の予約名・大文字小文字だけ違う重複・シンボリックリンク・ディレクトリ・暗号化・未対応の圧縮を拒否する。
  - 個数・1 ファイルの大きさ・合計の大きさ・圧縮率(zip 爆弾)に上限を持つ。大きさは申告を信用せず、読んだ実際のバイト数でも数える。
  - manifest とファイル一覧のハッシュ、全ファイルのハッシュと大きさを照合する(欠け・余分・改変を拒否)。
  - 取り出した DB は、読み取り専用で開いて **整合検査**・**スキーマが現在の版のスキーマと完全に一致すること**(余分なテーブル・ビュー・トリガーを拒否)・
    **ファイルの置き場を指す列が決まった形であること**(`../` で領域の外を読ませない)を確かめる。
復元は、確認のあと「準備」(zip を作業ディレクトリに取り出して検証)までをサーバが行い、**入れ替えは次の起動の最初**(DB を開く前)に `apply_pending` が行う
(DB を開いたまま入れ替えない)。入れ替えの前に、いまのデータを `restore-undo/` へ**移して退避**し、失敗したら元に戻す。復元の後は、世代の重みと確信度の組を全部照合する(`gens.check_all`)。
ログ・結果に実データの値は出さない(件数・大きさ・コードだけ)。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import time
import unicodedata
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import db as appdb
from .gens import CONFIDENCE_FILE, META_FILE, WEIGHT_FILES, expected_dir

FORMAT = "yayoi-backup"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
HASHLIST = "files.sha256"
DB_ENTRY = "db/app.sqlite3"
DATA_DIRS = (
    "images",
    "pagedata",
    "models",
    "exports",
    "uploads",
)  # 領域の中の、バックアップと復元の対象
DB_FILES = ("app.sqlite3", "app.sqlite3-wal", "app.sqlite3-shm")
CHUNK = 1 << 20

PENDING_DIR = "restore"  # 復元の作業用(incoming.zip・staging/・pending.json)
PENDING_FILE = "pending.json"
INCOMING = "incoming.zip"
STAGING = "staging"
UNDO_DIR = "restore-undo"  # 入れ替え前のデータの退避先
RESULT_FILE = "restore-result.json"  # 直近の復元の結果(次の起動で画面に出す)
BACKUPS_DIR = "backups"  # 作成したバックアップの一時置き場(ダウンロードのあと消す)

# 既に圧縮されている形式は、zip でも圧縮しない(時間の無駄)
_STORED_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz", ".pt", ".pdf", ".zip", ".tif", ".tiff"}


class BackupError(Exception):
    """理由コードだけを持つ(メッセージに実データを入れない)。"""

    def __init__(self, code: str, **extra: object) -> None:
        super().__init__(code)
        self.code = code
        self.extra = extra


@dataclass(frozen=True)
class Limits:
    max_entries: int = 2_000_000
    max_total_bytes: int = 64 << 30
    max_entry_bytes: int = 16 << 30
    max_ratio: int = 200  # 圧縮率の上限(大きさが min_ratio_bytes 以上のファイルだけ。小さい JSON は高く圧縮されうる)
    min_ratio_bytes: int = 4 << 20
    max_manifest_bytes: int = 1 << 20
    max_hashlist_bytes: int = 512 << 20
    max_db_bytes: int = 16 << 30


DEFAULT_LIMITS = Limits()

Progress = Callable[[str, int, int], None]


def _noop(_stage: str, _done: int, _total: int) -> None:
    return None


# ---------------------------------------------------------------------- 名前の検査

_FORBIDDEN = set('<>:"|?*\\')
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_GEN_FILES = (*WEIGHT_FILES, CONFIDENCE_FILE, META_FILE)
_PATTERNS = (
    re.compile(r"^db/app\.sqlite3$"),
    re.compile(r"^images/\d{1,12}/(\d{1,12}\.png|view\.jpg)$"),
    re.compile(r"^pagedata/\d{1,12}\.npz$"),
    re.compile(r"^models/\d{1,6}/gen_\d{4}/(" + "|".join(re.escape(f) for f in _GEN_FILES) + r")$"),
    re.compile(r"^uploads/[0-9a-f]{32}/original\.[a-z0-9]{1,8}$"),
)


def _component_bad(c: str) -> bool:
    if not c or c in (".", "..") or len(c) > 200 or c[-1] in ". ":
        return True
    if any(ord(ch) < 32 or ord(ch) == 0x7F or ch in _FORBIDDEN for ch in c):
        return True
    if c.split(".")[0].upper() in _RESERVED:
        return True
    try:
        c.encode("utf-8")
    except UnicodeEncodeError:  # 孤立したサロゲートなど
        return True
    return False


def entry_problem(name: object) -> str | None:
    """zip の中の名前(`/` 区切り)が許可された形でなければ理由コード、よければ None。manifest とハッシュ一覧は呼び出し側で別に扱う。"""
    if not isinstance(name, str) or not name or len(name) > 400 or "\x00" in name:
        return "entry_name_invalid"
    if name.startswith("/") or "\\" in name or ":" in name:
        return "entry_name_invalid"
    parts = name.split("/")
    if any(_component_bad(p) for p in parts):
        return "entry_name_invalid"
    if parts[0] == "exports":
        return (
            None if len(parts) == 2 else "entry_name_invalid"
        )  # exports/<ファイル名> だけ(日本語の名前もある)
    if any(p.fullmatch(name) for p in _PATTERNS):
        return None
    return "entry_name_invalid"


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


# ---------------------------------------------------------------------- 作成


def _walk(root: Path, top: str) -> Iterator[tuple[str, Path, int]]:
    base = root / top
    if not base.is_dir() or base.is_symlink():
        return
    stack = [base]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in sorted(it, key=lambda x: x.name):
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    rel = Path(e.path).relative_to(root).as_posix()
                    yield rel, Path(e.path), e.stat(follow_symlinks=False).st_size


def _copy_into(
    zf: zipfile.ZipFile, name: str, src: Path, mtime: float, progress: Callable[[int], None]
) -> tuple[int, str]:
    """ファイルを zip に書きながらハッシュを計算する。返り値: (書いたバイト数, sha256)。読んだ実際の中身のハッシュを記録する。"""
    zi = zipfile.ZipInfo(name, date_time=time.gmtime(max(mtime, 315532800))[:6])
    zi.compress_type = (
        zipfile.ZIP_STORED
        if Path(name).suffix.lower() in _STORED_SUFFIXES
        else zipfile.ZIP_DEFLATED
    )
    zi.external_attr = (stat.S_IFREG | 0o644) << 16
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, zf.open(zi, "w", force_zip64=True) as fo:
        while chunk := fi.read(CHUNK):
            fo.write(chunk)
            h.update(chunk)
            n += len(chunk)
            progress(len(chunk))
    return n, h.hexdigest()


def snapshot_counts(db_path: Path) -> dict:
    """スナップショットの DB から、件数だけを数える(manifest 用。値は読まない)。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:

        def one(q: str) -> int:
            return int(con.execute(q).fetchone()[0])

        return {
            "schema": one("PRAGMA user_version"),
            "n_pages": one("SELECT COUNT(*) FROM pages"),
            "n_confirmed_pages": one("SELECT COUNT(*) FROM pages WHERE status = 'confirmed'"),
            "n_generations": one("SELECT COUNT(*) FROM generations WHERE kind != 'shipped'"),
        }
    finally:
        con.close()


def create_archive(
    root: Path,
    snapshot: Path,
    out: Path,
    *,
    include_originals: bool,
    app_version: str,
    progress: Progress = _noop,
) -> dict:
    """`root`(データ領域)と DB のスナップショット `snapshot` から、バックアップの zip `out` を作る。返り値は manifest の要約(件数・大きさ)。"""
    tops = tuple(t for t in DATA_DIRS if include_originals or t != "uploads")
    files: list[tuple[str, Path, int]] = [(DB_ENTRY, snapshot, snapshot.stat().st_size)]
    for t in tops:
        for rel, path, size in _walk(root, t):
            if (
                entry_problem(rel) is None
            ):  # 決まった形でない余計なファイルは入れない(復元の検証で拒否されるため)
                files.append((rel, path, size))
    total = sum(s for _, _, s in files)
    counts = snapshot_counts(snapshot)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    listing: list[tuple[str, int, str]] = []
    done = 0
    progress("archive", 0, total)
    try:
        with zipfile.ZipFile(
            part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as zf:

            def tick(n: int) -> None:
                nonlocal done
                done += n
                progress("archive", done, total)

            for rel, path, _size in files:
                try:
                    n, digest = _copy_into(zf, rel, path, path.stat().st_mtime, tick)
                except FileNotFoundError:  # 取り込み・削除の最中に消えたファイルは入れない
                    continue
                listing.append((rel, n, digest))
            text = "".join(f"{d} {n} {rel}\n" for rel, n, d in listing).encode("utf-8")
            zf.writestr(HASHLIST, text)
            manifest = {
                "format": FORMAT,
                "version": FORMAT_VERSION,
                "created_at": time.time(),
                "app_version": app_version,
                "schema": counts["schema"],
                "includes_originals": include_originals,
                "n_files": len(listing),
                "total_bytes": sum(n for _, n, _ in listing),
                "n_pages": counts["n_pages"],
                "n_confirmed_pages": counts["n_confirmed_pages"],
                "n_generations": counts["n_generations"],
                "list_sha256": hashlib.sha256(text).hexdigest(),
            }
            zf.writestr(
                MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1)
            )  # 最後(途中で止まった zip には manifest がない)
        os.replace(part, out)
    except BaseException:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    progress("archive", total, total)
    return {**manifest, "zip_bytes": out.stat().st_size}


# ---------------------------------------------------------------------- 検査・取り出し


def _check_infos(zf: zipfile.ZipFile, limits: Limits) -> dict[str, zipfile.ZipInfo]:
    infos = zf.infolist()
    if len(infos) > limits.max_entries:
        raise BackupError("too_many_entries")
    out: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    total = 0
    for zi in infos:
        name = zi.filename
        if name not in (MANIFEST, HASHLIST):
            code = entry_problem(name)
            if code:
                raise BackupError(code)
        if zi.is_dir():
            raise BackupError("entry_type_invalid")
        if zi.flag_bits & 0x1:  # 暗号化
            raise BackupError("entry_type_invalid")
        if zi.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise BackupError("entry_type_invalid")
        kind = stat.S_IFMT(
            zi.external_attr >> 16
        )  # 種類の情報が無い(0)zip もある。あれば普通のファイルだけ許す
        if kind not in (0, stat.S_IFREG):  # シンボリックリンク・デバイスなど
            raise BackupError("entry_type_invalid")
        key = _fold(name)
        if key in seen or name in out:
            raise BackupError("entry_duplicate")
        seen.add(key)
        if zi.file_size > limits.max_entry_bytes:
            raise BackupError("entry_too_large")
        if zi.file_size >= limits.min_ratio_bytes and zi.file_size > limits.max_ratio * max(
            1, zi.compress_size
        ):
            raise BackupError("ratio_too_high")
        total += zi.file_size
        if total > limits.max_total_bytes:
            raise BackupError("total_too_large")
        out[name] = zi
    return out


def _read_small(zf: zipfile.ZipFile, zi: zipfile.ZipInfo, limit: int) -> bytes:
    if zi.file_size > limit:
        raise BackupError("manifest_invalid")
    with zf.open(zi) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise BackupError("manifest_invalid")
    return data


def _manifest(zf: zipfile.ZipFile, infos: dict[str, zipfile.ZipInfo], limits: Limits) -> dict:
    zi = infos.get(MANIFEST)
    if zi is None or HASHLIST not in infos:
        raise BackupError("manifest_missing")
    try:
        m = json.loads(_read_small(zf, zi, limits.max_manifest_bytes))
    except (ValueError, UnicodeDecodeError) as exc:
        raise BackupError("manifest_invalid") from exc
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise BackupError("format_unsupported")
    if m.get("version") != FORMAT_VERSION:
        raise BackupError("version_unsupported")
    schema = m.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise BackupError("manifest_invalid")
    if schema > len(appdb.MIGRATIONS):
        raise BackupError("db_too_new")
    for k in ("n_files", "total_bytes", "n_pages", "n_confirmed_pages", "n_generations"):
        if not isinstance(m.get(k), int) or isinstance(m.get(k), bool) or m[k] < 0:
            raise BackupError("manifest_invalid")
    if not isinstance(m.get("list_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", m["list_sha256"]
    ):
        raise BackupError("manifest_invalid")
    if not isinstance(m.get("includes_originals"), bool) or not isinstance(
        m.get("created_at"), (int, float)
    ):
        raise BackupError("manifest_invalid")
    if not isinstance(m.get("app_version"), str) or len(m["app_version"]) > 40:
        raise BackupError("manifest_invalid")
    return m


_LINE = re.compile(r"^([0-9a-f]{64}) (\d{1,15}) (.{1,400})$")


def _hashlist(
    zf: zipfile.ZipFile, zi: zipfile.ZipInfo, manifest: dict, limits: Limits
) -> dict[str, tuple[int, str]]:
    if zi.file_size > limits.max_hashlist_bytes:
        raise BackupError("hash_list_mismatch")
    h = hashlib.sha256()
    out: dict[str, tuple[int, str]] = {}
    with zf.open(zi) as fh:
        for raw in fh:
            h.update(raw)
            try:
                line = raw.decode("utf-8").rstrip("\n")
            except UnicodeDecodeError as exc:
                raise BackupError("hash_list_mismatch") from exc
            m = _LINE.match(line)
            if not m or m.group(3) in out:
                raise BackupError("hash_list_mismatch")
            out[m.group(3)] = (int(m.group(2)), m.group(1))
    if h.hexdigest() != manifest["list_sha256"] or len(out) != manifest["n_files"]:
        raise BackupError("hash_list_mismatch")
    return out


def _open_checked(
    path: Path, limits: Limits
) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo], dict, dict[str, tuple[int, str]]]:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError) as exc:
        raise BackupError("not_a_zip") from exc
    try:
        infos = _check_infos(zf, limits)
        manifest = _manifest(zf, infos, limits)
        listed = _hashlist(zf, infos[HASHLIST], manifest, limits)
        data_names = {n for n in infos if n not in (MANIFEST, HASHLIST)}
        if data_names != set(listed):
            raise BackupError("file_list_mismatch")
        for n, (size, _digest) in listed.items():
            if infos[n].file_size != size:
                raise BackupError("size_mismatch")
        if DB_ENTRY not in listed:
            raise BackupError("db_missing")
        if sum(s for s, _ in listed.values()) != manifest["total_bytes"]:
            raise BackupError("size_mismatch")
        if not manifest["includes_originals"] and any(n.startswith("uploads/") for n in listed):
            raise BackupError("file_list_mismatch")
        return zf, infos, manifest, listed
    except BaseException:
        zf.close()
        raise


def _stream(
    zf: zipfile.ZipFile,
    infos: dict[str, zipfile.ZipInfo],
    listed: dict[str, tuple[int, str]],
    progress: Progress,
    stage: str,
    open_sink: Callable[[str], object] | None = None,
) -> None:
    """全ファイルを読み、大きさとハッシュを照合する(`open_sink` があれば、書き出しながら)。"""
    total = sum(s for s, _ in listed.values())
    done = 0
    progress(stage, 0, total)
    for name in sorted(listed):
        size, digest = listed[name]
        h = hashlib.sha256()
        n = 0
        sink = open_sink(name) if open_sink else None
        try:
            with zf.open(infos[name]) as src:
                while chunk := src.read(CHUNK):
                    n += len(chunk)
                    if n > size:
                        raise BackupError("size_mismatch")
                    h.update(chunk)
                    if sink is not None:
                        sink.write(chunk)  # type: ignore[attr-defined]
                    done += len(chunk)
                    progress(stage, done, total)
        except zipfile.BadZipFile as exc:  # CRC の不一致など
            raise BackupError("hash_mismatch") from exc
        finally:
            if sink is not None:
                sink.close()  # type: ignore[attr-defined]
        if n != size or h.hexdigest() != digest:
            raise BackupError("hash_mismatch")
    progress(stage, total, total)


def _summary(manifest: dict, listed: dict[str, tuple[int, str]], zip_bytes: int) -> dict:
    return {
        "created_at": manifest["created_at"],
        "app_version": manifest["app_version"],
        "schema": manifest["schema"],
        "includes_originals": manifest["includes_originals"],
        "n_files": manifest["n_files"],
        "total_bytes": manifest["total_bytes"],
        "zip_bytes": zip_bytes,
        "n_pages": manifest["n_pages"],
        "n_confirmed_pages": manifest["n_confirmed_pages"],
        "n_generations": manifest["n_generations"],
        "n_images": sum(1 for n in listed if n.startswith("images/")),
        "n_pagedata": sum(1 for n in listed if n.startswith("pagedata/")),
    }


def inspect_archive(
    path: Path, *, limits: Limits = DEFAULT_LIMITS, progress: Progress = _noop
) -> dict:
    """zip を検査する(全ファイルのハッシュまで)。返り値は画面に出す要約(件数・大きさ・日時。値は含まない)。問題があれば BackupError。"""
    zf, infos, manifest, listed = _open_checked(path, limits)
    try:
        _stream(zf, infos, listed, progress, "verify")
        return _summary(manifest, listed, path.stat().st_size)
    finally:
        zf.close()


def extract_archive(
    path: Path, staging: Path, *, limits: Limits = DEFAULT_LIMITS, progress: Progress = _noop
) -> dict:
    """zip を `staging`(空の新しいディレクトリ)に取り出しながら検査し、取り出した DB も検査する。失敗したら `staging` を消す。"""
    if staging.exists():
        raise BackupError("staging_exists")
    zf, infos, manifest, listed = _open_checked(path, limits)
    try:
        need = manifest["total_bytes"] + (256 << 20)
        parent = staging.parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < need:
            raise BackupError("disk_full", need_bytes=need)
        staging.mkdir()
        root = staging.resolve()

        def open_sink(name: str):
            dest = staging / name
            if root not in dest.resolve().parents:  # 名前は検査済みだが、念のため
                raise BackupError("entry_name_invalid")
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644
            )
            return os.fdopen(fd, "wb")

        _stream(zf, infos, listed, progress, "extract", open_sink)
        progress("check_db", 0, 1)
        check_db(staging / DB_ENTRY, limits=limits)
        progress("check_db", 1, 1)
        return _summary(manifest, listed, path.stat().st_size)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        zf.close()


# ---------------------------------------------------------------------- DB の検査

# DB に入るファイルの置き場は「領域からの相対パス」で、区切りは OS による(Windows では `\`)。許す文字は英数字・`_`・`-` だけなので、`..`・ドライブ文字・絶対パスは通らない
_SEP = r"[/\\]"
_SAFE_REL = re.compile(rf"^[A-Za-z0-9_\-]+({_SEP}[A-Za-z0-9_\-]+)*\.[A-Za-z0-9]{{1,8}}$")
_RE_CROP = re.compile(rf"^\d{{1,12}}{_SEP}\d{{1,12}}\.png$")
_RE_VIEW = re.compile(rf"^\d{{1,12}}{_SEP}view\.jpg$")
_RE_UPLOAD = re.compile(rf"^[0-9a-f]{{32}}{_SEP}original\.[a-z0-9]{{1,8}}$")
# (テーブル, 列, 許可する形)。NULL は許す
_PATH_COLUMNS = (
    ("cells", "crop_path", _RE_CROP),
    ("pages", "view_path", _RE_VIEW),
    ("pages", "image_path", _SAFE_REL),
    ("uploads", "path", _RE_UPLOAD),
)


def _schema_rows(con: sqlite3.Connection) -> set[tuple]:
    return {
        (t, n, tbl, s)
        for t, n, tbl, s in con.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }


def _reference_schema(version: int) -> set[tuple]:
    ref = sqlite3.connect(":memory:")
    try:
        for i in range(version):
            ref.executescript(appdb.MIGRATIONS[i])
        return _schema_rows(ref)
    finally:
        ref.close()


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        r[1] == column for r in con.execute(f"PRAGMA table_info({table})")
    )  # table は固定の名前


def check_db(path: Path, *, limits: Limits = DEFAULT_LIMITS) -> int:
    """取り出した DB を、読み取り専用で検査する。返り値はスキーマの版。問題があれば BackupError。"""
    if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > limits.max_db_bytes:
        raise BackupError("db_invalid")
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as exc:
        raise BackupError("db_invalid") from exc
    try:
        con.execute("PRAGMA trusted_schema = OFF")
        try:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                raise BackupError("db_invalid")
            if version > len(appdb.MIGRATIONS):
                raise BackupError("db_too_new")
            ok = con.execute("PRAGMA integrity_check").fetchall()
            if [r[0] for r in ok] != ["ok"]:
                raise BackupError("db_invalid")
            if _schema_rows(con) != _reference_schema(version):
                raise BackupError(
                    "db_schema_mismatch"
                )  # 余分なテーブル・ビュー・トリガー、または別のスキーマ
            for table, column, rx in _PATH_COLUMNS:
                if not _has_column(con, table, column):
                    continue
                for (value,) in con.execute(
                    f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
                ):
                    if not isinstance(value, str) or not rx.fullmatch(value):
                        raise BackupError("db_path_invalid")
            for table, column in (("exports", "file_name"), ("export_files", "name")):
                if not _has_column(con, table, column):
                    continue
                for (value,) in con.execute(
                    f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
                ):
                    if not isinstance(value, str) or _component_bad(value) or "/" in value:
                        raise BackupError("db_path_invalid")
            if _has_column(con, "generations", "dir"):
                for pid, number, kind, d in con.execute(
                    "SELECT profile_id, number, kind, dir FROM generations"
                ):
                    if kind != "shipped" and d is not None and d != expected_dir(pid, number):
                        raise BackupError("db_path_invalid")
        except sqlite3.Error as exc:
            raise BackupError("db_invalid") from exc
        return version
    finally:
        con.close()


# ---------------------------------------------------------------------- 復元の入れ替え(起動時)


def pending_path(root: Path) -> Path:
    return root / PENDING_DIR / PENDING_FILE


def read_pending(root: Path) -> dict | None:
    p = pending_path(root)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(d, dict)
        or d.get("staging") != STAGING
        or not isinstance(d.get("summary"), dict)
    ):
        return None
    return d


def write_pending(root: Path, summary: dict) -> None:
    p = pending_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps({"staging": STAGING, "at": time.time(), "summary": summary}), encoding="utf-8"
    )
    os.replace(tmp, p)


def cancel_pending(root: Path) -> None:
    """復元の準備(取り出したデータ・受け取った zip・印)を消す。"""
    shutil.rmtree(root / PENDING_DIR, ignore_errors=True)


def _movable(root: Path) -> list[str]:
    return [n for n in (*DB_FILES, *DATA_DIRS) if (root / n).exists()]


def apply_pending(root: Path) -> dict | None:
    """復元の準備ができていれば、いまのデータを `restore-undo/` へ移して退避し、取り出したデータに入れ替える。**DB を開く前**に呼ぶ。
    結果は `restore-result.json` に残す(画面に出す)。失敗したら元に戻す。準備がなければ何もしない(None)。"""
    pend = read_pending(root)
    if pend is None:
        return None
    staging = root / PENDING_DIR / STAGING
    result: dict
    if not (staging / DB_ENTRY).is_file():
        result = {"ok": False, "code": "staging_missing"}
    else:
        undo = root / UNDO_DIR
        shutil.rmtree(undo, ignore_errors=True)  # 退避は直近の 1 回分だけ
        undo.mkdir(parents=True)
        moved: list[str] = []
        placed: list[str] = []
        try:
            for n in _movable(root):
                os.replace(root / n, undo / n)
                moved.append(n)
            os.replace(staging / DB_ENTRY, root / "app.sqlite3")
            placed.append("app.sqlite3")
            for t in DATA_DIRS:
                if (staging / t).exists():
                    os.replace(staging / t, root / t)
                    placed.append(t)
            result = {"ok": True, "summary": pend["summary"], "undo_bytes": dir_bytes(undo)}
        except BaseException as exc:
            for n in placed:  # 置いたものを片づけて、退避したものを戻す
                p = root / n
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    with contextlib.suppress(OSError):
                        p.unlink()
            for n in moved:
                with contextlib.suppress(OSError):
                    os.replace(undo / n, root / n)
            shutil.rmtree(undo, ignore_errors=True)
            result = {"ok": False, "code": f"error_{type(exc).__name__}"}
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
    result["at"] = time.time()
    (root / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")
    cancel_pending(root)
    return result


def read_result(root: Path) -> dict | None:
    try:
        d = json.loads((root / RESULT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def record_pairs(root: Path, pairs: dict) -> None:
    """復元の後の、世代の重みと確信度の組の照合の結果(件数)を、復元の結果に足す。"""
    r = read_result(root)
    if r is None:
        return
    r["pairs"] = pairs
    (root / RESULT_FILE).write_text(json.dumps(r), encoding="utf-8")


def clear_result(root: Path) -> None:
    with contextlib.suppress(OSError):
        (root / RESULT_FILE).unlink()


def dir_bytes(p: Path) -> int:
    n = 0
    for dirpath, _dirs, files in os.walk(p):
        for f in files:
            with contextlib.suppress(OSError):
                n += os.lstat(os.path.join(dirpath, f)).st_size
    return n


def undo_info(root: Path) -> dict | None:
    d = root / UNDO_DIR
    if not d.is_dir():
        return None
    return {"bytes": dir_bytes(d), "at": d.stat().st_mtime}


def delete_undo(root: Path) -> bool:
    d = root / UNDO_DIR
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True
