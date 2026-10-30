"""確認 UI とローカルサーバ(Phase 3)。

構成: pywebview のウィンドウ → 127.0.0.1 だけにバインドした FastAPI(起動ごとのランダムなトークンで保護)→ SQLite。
設計は docs/phase3_design.md。実データの値(認識値・確定値・ファイル名)はログに出さない。
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """遅延 import(FastAPI が入っていない環境で `yayoi_ocr.app` の import を失敗させない)。"""
    from .server import create_app as _create

    return _create(*args, **kwargs)
