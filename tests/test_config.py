from __future__ import annotations

from yayoi_ocr import config


def test_models_dir_defaults_under_private_data(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ENV_MODELS_DIR, raising=False)
    monkeypatch.setenv(config.ENV_PRIVATE_DATA, str(tmp_path))
    assert config.models_dir() == tmp_path.resolve() / "models"


def test_models_dir_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_PRIVATE_DATA, str(tmp_path / "private"))
    monkeypatch.setenv(config.ENV_MODELS_DIR, str(tmp_path / "bundle"))
    assert config.models_dir() == (tmp_path / "bundle").resolve()
