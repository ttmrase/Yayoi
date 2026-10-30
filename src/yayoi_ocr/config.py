"""実行環境の設定。実データの置き場所(リポジトリの外)を一元管理する。

実データ(サンプル画像・切り出し画像・正解データ・学習データ・モデル)は
`YAYOI_PRIVATE_DATA`(既定 ~/yayoi_private)配下にのみ置く。リポジトリは public。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PRIVATE_DATA = "YAYOI_PRIVATE_DATA"
DEFAULT_PRIVATE_DATA = "~/yayoi_private"


def private_data_dir() -> Path:
    """実データ置き場(存在しなくてもよい)。"""
    return Path(os.environ.get(ENV_PRIVATE_DATA, DEFAULT_PRIVATE_DATA)).expanduser().resolve()


def samples_dir() -> Path:
    return private_data_dir() / "samples"


def fonts_dir() -> Path:
    """合成帳票用の手書き風フォント置き場(scripts/fetch_fonts.py の取得先。コミットしない)。"""
    return private_data_dir() / "fonts"


def debug_dir() -> Path:
    return private_data_dir() / "debug"


def _find_git_root(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def assert_safe_output_dir(path: str | Path) -> Path:
    """出力先が git 管理下(= 誤コミットされうる場所)でないことを確認して絶対パスを返す。

    切り出し画像・重ね描き画像など実データ由来の出力は、リポジトリの外にだけ書く。
    ただし `YAYOI_PRIVATE_DATA` 配下は(たとえ git 管理下でも)許可する。
    """
    resolved = Path(path).expanduser().resolve()
    private = private_data_dir()
    if resolved == private or private in resolved.parents:
        return resolved
    root = _find_git_root(resolved)
    if root is not None:
        raise PermissionError(
            f"出力先 {resolved} は git リポジトリ({root})の内側です。"
            f"実データ由来の出力は {ENV_PRIVATE_DATA}({private})配下など、"
            "リポジトリの外に指定してください。"
        )
    return resolved
