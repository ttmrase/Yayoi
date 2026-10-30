"""アプリのデータ置き場(DB・取り込んだ画像・切り出し画像)。リポジトリの外。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import private_data_dir

ENV_APP_DATA = "YAYOI_APP_DATA"


def app_data_dir() -> Path:
    """`YAYOI_APP_DATA` > Windows の `%LOCALAPPDATA%\\yayoi-ocr` > `YAYOI_PRIVATE_DATA/app`(開発)。"""
    env = os.environ.get(ENV_APP_DATA)
    if env:
        return Path(env).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local:
        return Path(local) / "yayoi-ocr"
    return private_data_dir() / "app"


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "app.sqlite3"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    def ensure(self) -> AppPaths:
        for d in (self.root, self.images, self.uploads):
            d.mkdir(parents=True, exist_ok=True)
        return self
