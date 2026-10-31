"""アプリのログ(ファイルに残す・古いものは消す)。**実データの値(摘要・金額・ファイル名・パス)は、ログに出さない。**

- 出力先: データ領域の `logs/app.log`。1 ファイル最大 1 MiB で回転し、世代は 5 つまで(合計で最大 約 6 MiB)。
  起動のたびに、更新から 30 日を過ぎたログのファイルを消す(`purge_old`)。
- 出すもの: ID・件数・コード・例外の**型名**だけ(呼び出し側が守る)。サーバ(uvicorn)の未処理の例外も同じ(`launcher` が uvicorn の既定のログ設定を外してこの設定に任せるので、入力値を含む検証エラーの文面がコンソールに出ない)。守られなかった場合の備えとして、**例外のメッセージとトレースバックは書かない**
  (`ValueFreeFormatter`。`log.exception(...)` や `exc_info=True` があっても、型名だけになる。メッセージにはパスや値が入りうるため)。
- 保証は試験で行う(`tests/test_logs.py`): 一通りの操作(取り込み・認識・確認・確定・出力・追加学習・バックアップ・失敗)の後で、入力した値・ファイル名・設定の名前・パスがログに無い。
- 標準エラー(最小化したコンソール)にも WARNING 以上を出す(困ったときに見える)。ログは利用者の PC の中だけ。外へ送らない。
"""

from __future__ import annotations

import contextlib
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "app.log"
MAX_BYTES = 1 << 20
BACKUPS = 5
RETENTION_DAYS = 30
FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOGGERS = (
    "yayoi",
    "yayoi_ocr",
    "uvicorn",
)  # アプリのログ(`yayoi.training` / `yayoi.gens` / `yayoi_ocr.app`)と、サーバ(uvicorn.error の未処理の例外)
_MARK = "_yayoi_log_handler"


class ValueFreeFormatter(logging.Formatter):
    """例外は型名だけを書く(メッセージ・トレースバックは書かない)。"""

    def formatException(self, ei) -> str:  # noqa: N802 (logging.Formatter の名前)
        return f"exception_type={ei[0].__name__ if ei and ei[0] else 'unknown'}"

    def formatStack(self, stack_info: str) -> str:  # noqa: N802
        return ""


def purge_old(logs_dir: Path, days: int = RETENTION_DAYS, now: float | None = None) -> int:
    """更新から `days` 日を過ぎたログ(`app.log` と回転した `app.log.N`)を消す。返り値は消した数。"""
    cutoff = (now if now is not None else time.time()) - days * 86400
    n = 0
    for p in logs_dir.glob(f"{LOG_NAME}*"):
        with contextlib.suppress(OSError):
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
    return n


def setup(
    logs_dir: Path,
    *,
    max_bytes: int = MAX_BYTES,
    backups: int = BACKUPS,
    retention_days: int = RETENTION_DAYS,
    console: bool = True,
) -> list[logging.Handler]:
    """ログの出力先を設定する(二重に呼んでも、前の設定を外してから付け直す)。返り値は付けたハンドラー。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    purge_old(logs_dir, retention_days)
    teardown()
    fmt = ValueFreeFormatter(FORMAT)
    handlers: list[logging.Handler] = []
    fh = RotatingFileHandler(
        logs_dir / LOG_NAME, maxBytes=max_bytes, backupCount=backups, encoding="utf-8", delay=True
    )
    fh.setLevel(logging.INFO)
    handlers.append(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        handlers.append(sh)
    for h in handlers:
        h.setFormatter(fmt)
        setattr(h, _MARK, True)
    for name in LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        for h in handlers:
            lg.addHandler(h)
    return handlers


def teardown() -> None:
    """`setup` が付けたハンドラーを外して閉じる。"""
    for name in LOGGERS:
        lg = logging.getLogger(name)
        for h in [h for h in lg.handlers if getattr(h, _MARK, False)]:
            lg.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
