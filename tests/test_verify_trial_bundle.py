"""試用版の束の検査(scripts/verify_trial_bundle.py): 正しい束は通り、ハッシュの食い違い・出所の表にない重み・lock にない wheel・画像・拒否リストの語は落ちる(合成の小さな束)。"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "verify_trial_bundle", ROOT / "scripts" / "verify_trial_bundle.py"
)
vtb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vtb)

VERSION = "9.9.9"
DENY = (["ひみつの語"], [])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wheel_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("pkg/__init__.py", text)
    return buf.getvalue()


def make_bundle(
    tmp_path: Path,
    *,
    extra: dict[str, bytes] | None = None,
    models: dict[str, bytes] | None = None,
    doc_hash_override: dict[str, str] | None = None,
    foo_in_lock: bool = True,
    side_sha: str | None = None,
) -> dict:
    models = models or {"models/a.bin": b"AAAA", "models/adapted/b.bin": b"BBBB"}
    foo = wheel_bytes("foo")
    lock = f"foo==1.0 \\\n    --hash=sha256:{sha(foo)}\n" if foo_in_lock else "bar==2.0 \\\n"
    py = b"python archive"
    files: dict[str, bytes] = {
        **models,
        f"wheels/yayoi_ocr-{VERSION}-py3-none-any.whl": wheel_bytes("app code"),
        "wheels/foo-1.0-py3-none-any.whl": foo,
        "requirements-lock.txt": lock.encode(),
        "python-standalone.tar.gz": py,
        "python-standalone.pin": (
            f"tag=1\nversion=3.11.1\nfile=cpython-3.11.1-x.tar.gz\nsha256={sha(py)}\n"
        ).encode(),
        "manual.html": "<p>手引き</p>".encode(),
        **(extra or {}),
    }
    files["MANIFEST.sha256"] = "".join(
        f"{sha(d)}  {n}\n" for n, d in sorted(files.items())
    ).encode()
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    digest = sha(zip_path.read_bytes())
    (tmp_path / "bundle.zip.sha256").write_text(f"{side_sha or digest}  bundle.zip\n")
    table = {n: sha(d) for n, d in (models or {}).items()}
    table.update(doc_hash_override or {})
    prov = tmp_path / "prov.md"
    prov.write_text(
        "| ファイル | 由来 | SHA256 |\n|---|---|---|\n"
        + "".join(f"| `{n}` | x | y | z | `{h}` |\n" for n, h in table.items()),
        encoding="utf-8",
    )
    lockfile = tmp_path / "lock.txt"
    lockfile.write_text(lock, encoding="utf-8")
    return {"zip_path": zip_path, "provenance": prov, "locks": (lockfile,)}


def run(b: dict) -> dict[str, bool]:
    results, listing = vtb.verify(
        b["zip_path"], provenance=b["provenance"], locks=b["locks"], deny=DENY, version=VERSION
    )
    assert listing and all(len(h) == 64 for _p, _s, h in listing)
    return {c: ok for c, ok, _t in results}


def test_a_correct_bundle_passes_every_check(tmp_path):
    assert run(make_bundle(tmp_path)) == dict.fromkeys(
        ["B1", "B2", "B3", "B4", "B5", "B6", "B7"], True
    )


def test_the_provenance_table_is_parsed_from_the_real_document():
    table = vtb.provenance_table(
        (ROOT / "docs" / "model_provenance.md").read_text(encoding="utf-8")
    )
    assert len(table) == 9 and all(k.startswith("models/") for k in table)
    assert "models/adapted/confidence.json" in table


def test_a_zip_that_differs_from_the_recorded_sha256_fails(tmp_path):
    assert run(make_bundle(tmp_path, side_sha="0" * 64))["B1"] is False


def test_a_weight_that_does_not_match_the_provenance_table_fails_b4_only(tmp_path):
    r = run(make_bundle(tmp_path, doc_hash_override={"models/a.bin": "1" * 64}))
    assert r["B4"] is False and r["B3"] is True and r["B1"] is True


def test_a_weight_that_is_not_listed_in_the_provenance_table_fails(tmp_path):
    b = make_bundle(tmp_path, models={"models/a.bin": b"AAAA", "models/experiment.pt": b"X"})
    b["provenance"].write_text(
        f"| `models/a.bin` | x | y | z | `{sha(b'AAAA')}` |\n", encoding="utf-8"
    )  # 出所の表には a.bin だけ
    assert run(b)["B4"] is False


def test_a_wheel_that_is_not_in_the_lock_fails(tmp_path):
    assert run(make_bundle(tmp_path, foo_in_lock=False))["B5"] is False


def test_an_image_or_a_denylisted_word_outside_the_wheels_fails(tmp_path):
    assert run(make_bundle(tmp_path, extra={"licenses/scan.png": b"\x89PNG"}))["B7"] is False
    assert (
        run(make_bundle(tmp_path, extra={"notes.txt": "これは ひみつの語 です".encode()}))["B7"]
        is False
    )


def test_a_file_that_the_manifest_does_not_list_fails_b3(tmp_path):
    b = make_bundle(tmp_path)
    with zipfile.ZipFile(b["zip_path"], "a") as z:
        z.writestr("stowaway.txt", b"x")
    side = b["zip_path"].with_name("bundle.zip.sha256")
    side.write_text(f"{sha(b['zip_path'].read_bytes())}  bundle.zip\n")
    assert run(b)["B3"] is False


@pytest.mark.parametrize("name", ["../evil.txt", "/abs.txt"])
def test_an_unsafe_entry_name_fails_b2(tmp_path, name):
    b = make_bundle(tmp_path, extra={name: b"x"})
    assert run(b)["B2"] is False
