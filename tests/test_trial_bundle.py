from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")  # make_trial_bundle は認識エンジン(torch)の定数を読む

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_trial_bundle.py"


@pytest.fixture(scope="module")
def bundle():
    spec = importlib.util.spec_from_file_location("make_trial_bundle", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_models(root: Path, names: list[str]) -> None:
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"w")


def test_collect_models_copies_only_the_allow_list(bundle, tmp_path):
    src = tmp_path / "models"
    _write_models(src, bundle.MODEL_NAMES)
    _write_models(src, [f"adapted/{n}" for n in bundle.ADAPTED_NAMES])
    # 許可リストに無いもの(評価の画像・正解・学習データ・別の確信度モデル)は入れない
    _write_models(
        src, ["eval/sample.jpg", "truth.json", "confidence_ft1.json", "adapted/train.npz"]
    )
    dest = tmp_path / "stage" / "models"
    copied = bundle.collect_models(src, dest)
    names = sorted(p.relative_to(dest).as_posix() for p in copied)
    expected = sorted([*bundle.MODEL_NAMES, *(f"adapted/{n}" for n in bundle.ADAPTED_NAMES)])
    assert names == expected
    assert {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()} == set(expected)


def test_collect_models_adapted_is_optional_but_base_is_required(bundle, tmp_path):
    src = tmp_path / "models"
    _write_models(src, bundle.MODEL_NAMES)
    assert bundle.collect_models(src, tmp_path / "ok")  # adapted が無くても作れる
    (src / bundle.MODEL_NAMES[0]).unlink()
    with pytest.raises(SystemExit):
        bundle.collect_models(src, tmp_path / "ng")


def test_refuses_to_write_inside_the_repo(bundle):
    repo = Path(__file__).resolve().parent.parent
    with pytest.raises(PermissionError):
        bundle.assert_safe_output_dir(repo / "dist")


def test_write_crlf_normalizes_line_endings(bundle, tmp_path):
    src = tmp_path / "a.txt"
    src.write_bytes(b"a\nb\r\nc\n")
    dest = tmp_path / "b.txt"
    bundle.write_crlf(src, dest)
    assert dest.read_bytes() == b"a\r\nb\r\nc\r\n"
