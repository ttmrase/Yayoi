"""アプリの起動: 127.0.0.1 のランダムなポートで FastAPI を立て、pywebview のウィンドウで開く。

トークンは URL の **フラグメント**(`#token=…`)で渡す(サーバには送られず、ログにも残らない)。
`--no-window` は開発・E2E 用: URL を表示して待つ。`--token` / `--port` の固定は開発・E2E 専用(通常は指定しない)。
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import uvicorn

from . import backup, logs
from .paths import AppPaths, app_data_dir
from .security import allowed_hosts_for, new_token
from .server import close_app, create_app
from .worker import make_backend


def _bind_localhost(port: int = 0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if port and os.name != "nt":  # 固定ポート(開発・E2E)の再起動で TIME_WAIT に引っかからないように
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))  # 127.0.0.1 だけ(0.0.0.0 にしない)
    sock.listen(64)
    return sock


class RunningServer:
    def __init__(
        self,
        data_dir: Path | None = None,
        port: int = 0,
        token: str | None = None,
        dev: bool = False,
        backend: str = "real",
    ) -> None:
        paths = AppPaths(data_dir or app_data_dir())
        if paths.root.is_dir():
            backup.apply_pending(
                paths.root
            )  # バックアップからの復元の準備ができていれば、DB を開く前に入れ替える
        self.paths = paths.ensure()
        logs.setup(self.paths.logs)  # logs/app.log(回転・30 日で削除。値は出さない)
        self.sock = _bind_localhost(port)
        self.port = self.sock.getsockname()[1]
        self.token = token or new_token()
        self.app = create_app(
            self.paths.db,
            self.token,
            allowed_hosts_for(self.port),
            images_dir=self.paths.images,
            dev=dev,
            backend=make_backend(backend),
            uploads_dir=self.paths.uploads,
        )
        config = uvicorn.Config(self.app, log_level="warning", access_log=False, lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=lambda: self.server.run(sockets=[self.sock]), name="yayoi-server", daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/#token={self.token}"

    def start(self) -> RunningServer:
        self.app.state.purge_unused()  # どのジョブにも使われず 24 時間を過ぎたアップロードを削除
        runner = self.app.state.runner
        if runner is not None:
            runner.start()  # 前回中断した認識の続きもここで始まる
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("サーバの起動に失敗しました")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        close_app(self.app)
        self.sock.close()
        logs.teardown()


def configure_webview(webview) -> None:
    """pywebview の設定。WebView2 のダウンロード(出力の CSV・精度レポート・バックアップ)は、`ALLOW_DOWNLOADS` が False(既定)だと
    **何も表示されずに取り消される**。True にすると、保存先を選ぶ「名前を付けて保存」ダイアログが出る。画面は自分のサーバ(127.0.0.1)の内容だけで、外部のページは開かない(CSP)。"""
    webview.settings["ALLOW_DOWNLOADS"] = True


def run(
    data_dir: Path | None = None,
    window: bool = True,
    port: int = 0,
    token: str | None = None,
    dev: bool = False,
    backend: str = "real",
) -> int:
    srv = RunningServer(data_dir, port, token, dev, backend).start()
    try:
        if window:
            try:
                import webview  # pywebview(Windows は WebView2)
            except ImportError:
                print(
                    "pywebview が入っていません(`pip install yayoi-ocr[desktop]`)。ブラウザで開けます:"
                )
                print(srv.url)
                window = False
            else:
                configure_webview(webview)
                webview.create_window(
                    "yayoi-ocr", srv.url, width=1440, height=900, min_size=(1100, 700)
                )
                webview.start()
                return 0
        print(srv.url, flush=True)  # 開発用。トークンを含むので通常の起動ではここに来ない
        while srv.thread.is_alive():
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        srv.stop()
