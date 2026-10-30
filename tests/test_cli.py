from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from yayoi_ocr.cli import main
from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.synth.generator import SynthConfig, generate_sample

ROOT = Path(__file__).resolve().parents[1]


def test_output_inside_repository_is_refused():
    with pytest.raises(PermissionError):
        assert_safe_output_dir(ROOT / "out_debug")
    assert main(["layout", "x.png", "--out", str(ROOT / "debug_out")]) == 2


def test_private_data_dir_is_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("YAYOI_PRIVATE_DATA", str(tmp_path / "priv"))
    assert (
        assert_safe_output_dir(tmp_path / "priv" / "debug")
        == (tmp_path / "priv" / "debug").resolve()
    )


@pytest.mark.needs_font
def test_layout_command_writes_outputs(tmp_path):
    s = generate_sample(SynthConfig(seed=41, dpi=110))
    img = tmp_path / "in.png"
    cv2.imwrite(str(img), s.image)
    out = tmp_path / "out"
    code = main(["layout", str(img), "--out", str(out)])
    assert code in (0, 3)
    for name in ("overlay.png", "rectified.png", "layout.json", "title_color.png"):
        assert (out / name).exists(), name
    assert len(list((out / "cells" / "binary").glob("*.png"))) == 29 * 4
    data = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    assert len(data["page"]["rows"]) == 29


@pytest.mark.needs_font
def test_synth_command_writes_image_and_ground_truth(tmp_path):
    from yayoi_ocr.groundtruth import load_ground_truth

    out = tmp_path / "synth"
    assert main(["synth", "--out", str(out), "-n", "2", "--writer", "5", "--clean"]) == 0
    imgs = sorted(out.glob("*.png"))
    assert len(imgs) == 2
    gt = load_ground_truth(out / f"{imgs[0].stem}.json")
    assert len(gt.rows) == 29 and gt.image == imgs[0].name
