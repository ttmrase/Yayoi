"""ライセンス表示: 主な利用物の帰属表示(NDLOCR-Lite の CC BY 4.0 など)、lock と wheel の METADATA からの一覧の生成、API。"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from yayoi_ocr.app import licenses

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen_licenses", ROOT / "scripts/gen_licenses.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def make_wheel(
    d: Path, name: str, version: str, metadata_lines: list[str], files: dict[str, str]
) -> None:
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(d / f"{name.replace('-', '_')}-{version}-py3-none-any.whl", "w") as zf:
        zf.writestr(
            f"{dist}/METADATA",
            "\n".join(
                ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}", *metadata_lines]
            )
            + "\n",
        )
        for rel, text in files.items():
            zf.writestr(f"{dist}/{rel}", text)
        zf.writestr(f"{name}/__init__.py", "")


def test_the_attribution_for_ndlocr_lite_and_parseq_is_present_with_license_source_and_changes():
    main = {m["name"]: m for m in licenses.MAIN}
    ndl = next(m for k, m in main.items() if "NDLOCR-Lite" in k)
    assert ndl["license"] == "CC BY 4.0" and "国立国会図書館" in ndl["owner"]
    assert (
        "github.com/ndl-lab/ndlocr-lite" in ndl["source"]
        and "変換" in ndl["changes"]
        and "追加学習" in ndl["changes"]
    )
    parseq = next(m for m in main.values() if "baudm/parseq" in m["source"])
    assert (
        "Apache" in parseq["license"]
        and "baudm/parseq" in parseq["source"]
        and "取り込み" in parseq["changes"]
    )
    assert any("Python" in k for k in main) and any("WebView2" in k for k in main)
    for m in licenses.MAIN:
        assert m["license"] and m["source"] and m["changes"] and m["owner"]


def test_license_is_read_from_expression_short_license_or_classifiers(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "alpha==1.0 \\\n    --hash=sha256:aa\nbeta==2.0 \\\n    --hash=sha256:bb\ngamma==3.0 \\\n    --hash=sha256:cc\ndelta==4.0 \\\n    --hash=sha256:dd\n",
        encoding="utf-8",
    )
    wheels = tmp_path / "w"
    wheels.mkdir()
    make_wheel(
        wheels, "alpha", "1.0", ["License-Expression: MIT"], {"licenses/LICENSE": "MIT text"}
    )
    make_wheel(
        wheels,
        "beta",
        "2.0",
        ["License: BSD-3-Clause"],
        {"LICENSE.txt": "BSD text", "NOTICE": "notice"},
    )
    make_wheel(
        wheels,
        "gamma",
        "3.0",
        ["License: " + "x" * 300, "Classifier: License :: OSI Approved :: Apache Software License"],
        {},
    )
    # delta は wheel が無い → 要確認
    texts = tmp_path / "texts"
    comps = gen.pip_components(wheels, [lock], texts)
    by = {c["name"]: c for c in comps}
    assert by["alpha"]["license"] == "MIT" and by["alpha"]["version"] == "1.0"
    assert by["beta"]["license"] == "BSD-3-Clause" and by["beta"]["texts"] == [
        "LICENSE.txt",
        "NOTICE",
    ]
    assert by["gamma"]["license"] == "Apache Software License"  # 長い License は使わず分類子から
    assert by["delta"]["license"] == gen.UNKNOWN
    assert (texts / "pip/alpha-1.0/LICENSE").read_text() == "MIT text"
    assert (texts / "pip/beta-2.0/NOTICE").read_text() == "notice"


def test_npm_components_include_only_production_dependencies(tmp_path):
    lock = tmp_path / "package-lock.json"
    lock.write_text(
        json.dumps(
            {
                "packages": {
                    "": {"name": "web"},
                    "node_modules/react": {"version": "18.3.1", "license": "MIT"},
                    "node_modules/typescript": {
                        "version": "5.0.0",
                        "license": "Apache-2.0",
                        "dev": True,
                    },
                    "node_modules/@scope/pkg": {"version": "1.0.0", "license": "ISC"},
                }
            }
        ),
        encoding="utf-8",
    )
    mods = tmp_path / "node_modules"
    (mods / "react").mkdir(parents=True)
    (mods / "react" / "LICENSE").write_text("react license")
    texts = tmp_path / "texts"
    comps = gen.npm_components(lock, mods, texts)
    assert [c["name"] for c in comps] == ["@scope/pkg", "react"]
    assert (texts / "npm/react-18.3.1/LICENSE").read_text() == "react license"


def test_the_real_locks_parse_and_the_generated_json_roundtrips(tmp_path):
    out = tmp_path / "licenses.json"
    data = gen.generate(None, out)  # wheel が無くても lock の一覧は作れる(ライセンスは要確認)
    names = {c["name"].lower() for c in data["components"]}
    assert {"torch", "numpy", "fastapi", "react"} <= names
    assert json.loads(out.read_text(encoding="utf-8"))["generated"] is True


def test_load_falls_back_to_the_fixed_entries_when_nothing_was_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(licenses, "DATA", tmp_path / "missing.json")
    d = licenses.load()
    assert d["generated"] is False and d["components"] == [] and d["main"]
    (tmp_path / "ok.json").write_text(
        json.dumps(
            {
                "generated": True,
                "components": [
                    {"name": "x", "version": "1", "license": "MIT", "kind": "python", "texts": []}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(licenses, "DATA", tmp_path / "ok.json")
    d = licenses.load()
    assert d["generated"] is True and d["components"][0]["name"] == "x"
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(licenses, "DATA", tmp_path / "bad.json")
    assert licenses.load()["generated"] is False


def test_the_api_serves_the_licenses(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("cv2")
    from fastapi.testclient import TestClient

    from yayoi_ocr.app.security import allowed_hosts_for, new_token
    from yayoi_ocr.app.server import create_app

    token = new_token()
    app = create_app(tmp_path / "app.sqlite3", token, allowed_hosts_for(8135), dev=True)
    with TestClient(
        app, base_url="http://127.0.0.1:8135", headers={"Authorization": f"Bearer {token}"}
    ) as c:
        d = c.get("/api/licenses").json()
        assert d["main"] and "components" in d and "generated" in d
