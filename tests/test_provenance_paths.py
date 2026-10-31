"""実サンプルの置き場を読むコードの経路が増えていないことを見張る(docs/model_provenance.md の検査 C1)。

学習・較正・評価のスクリプトやアプリが、`YAYOI_PRIVATE_DATA/samples`(実サンプル)を読むようになったら失敗する。
読めるのは config.py の定義と tests/test_samples.py(レイアウトの構造チェック)だけ。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "model_provenance.py"


def _load():
    spec = importlib.util.spec_from_file_location("model_provenance", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_only_allowed_files_reference_the_samples_dir():
    assert _load().check_samples_paths() == []


def test_the_check_detects_a_new_reader(tmp_path, monkeypatch):
    m = _load()
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "yayoi_ocr").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts" / "train_x.py").write_text(
        "from yayoi_ocr.config import samples_dir\n", encoding="utf-8"
    )
    (root / "src" / "yayoi_ocr" / "config.py").write_text(
        "def samples_dir(): ...\n", encoding="utf-8"
    )
    monkeypatch.setattr(m, "ROOT", root)
    assert m.check_samples_paths() == ["scripts/train_x.py"]
