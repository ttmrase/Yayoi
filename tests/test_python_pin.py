"""同梱する Python の固定(requirements/python-standalone.txt)の形式を見張る。取得・SHA256 の検証は scripts/fetch_python_standalone.py(開発機で実行)。"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "fetch_python_standalone", ROOT / "scripts" / "fetch_python_standalone.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_pin_is_consistent():
    pin = _load().read_pin()
    assert re.fullmatch(r"\d{8}", pin["tag"])
    assert re.fullmatch(r"3\.11\.\d+", pin["version"])  # Python 3.11(依存の lock は cp311)
    assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    # ファイル名: Windows x86_64 の install_only(_stripped)で、tag と version がファイル名と一致する
    assert pin["file"].startswith(f"cpython-{pin['version']}+{pin['tag']}-x86_64-pc-windows-msvc-")
    assert pin["file"].endswith(("install_only.tar.gz", "install_only_stripped.tar.gz"))


def test_setup_script_checks_the_pin_and_the_installer_signature():
    ps1 = (ROOT / "scripts" / "windows" / "setup.ps1").read_bytes().decode("utf-8-sig")
    assert "python-standalone.pin" in ps1 and "Get-FileHash $bundled" in ps1
    # 予備のインストーラは、署名者と署名の状態を確かめてから実行する
    assert "Get-AuthenticodeSignature" in ps1 and "Python Software Foundation" in ps1
    assert ps1.index("Get-AuthenticodeSignature") < ps1.index("Start-Process -FilePath $installer")
