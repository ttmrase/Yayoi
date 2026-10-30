"""scripts/check_no_private_data.py のテスト(画像はテスト中に合成し、一時リポジトリでのみ扱う)。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_no_private_data.py"

spec = importlib.util.spec_from_file_location("check_no_private_data", SCRIPT)
checker = importlib.util.module_from_spec(spec)
sys.modules["check_no_private_data"] = checker
spec.loader.exec_module(checker)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, text=True
    )


def test_clean_text_passes(repo: Path):
    (repo / "a.py").write_text("print('hello')\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    assert _run(repo).returncode == 0


def test_staged_image_fails(repo: Path):
    arr = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(repo / "scan.jpg")
    _git(repo, "add", "-f", "scan.jpg")
    res = _run(repo)
    assert res.returncode == 1
    assert "scan.jpg" in res.stderr


def test_disguised_image_is_detected_by_magic_bytes(repo: Path):
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(repo / "notes.txt", format="PNG")
    _git(repo, "add", "-f", "notes.txt")
    assert _run(repo).returncode == 1


def test_private_dir_and_pdf_fail(repo: Path):
    (repo / "ground_truth").mkdir()
    (repo / "ground_truth" / "x.json").write_text("{}", encoding="utf-8")
    (repo / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    _git(repo, "add", "-f", "ground_truth/x.json", "doc.pdf")
    res = _run(repo)
    assert res.returncode == 1
    assert "ground_truth/x.json" in res.stderr and "doc.pdf" in res.stderr


def test_large_binary_fails_small_text_ok():
    big = b"\x00" * (checker.MAX_BINARY_BYTES + 1)
    assert checker.check_blob("data.bin", big[:65536], len(big))
    assert not checker.check_blob("data.txt", b"abc", 3)


def test_unpushed_history_detects_removed_image(repo: Path):
    (repo / "a.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "base")
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(repo / "leak.png")
    _git(repo, "add", "-f", "leak.png")
    _git(repo, "commit", "-qm", "oops")
    _git(repo, "rm", "-q", "leak.png")
    _git(repo, "commit", "-qm", "remove")
    # 最終ツリーには無いが、未 push の履歴には残っている → 検出される
    assert _run(repo, "--all").returncode == 0
    assert _run(repo, "--unpushed").returncode == 1


def test_this_repository_is_clean():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--all"], cwd=ROOT, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
