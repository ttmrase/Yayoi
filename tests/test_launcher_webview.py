"""pywebview(WebView2)の設定: ダウンロード(出力の CSV・精度レポート・バックアップ)は、`ALLOW_DOWNLOADS` が True でないと何も出ずに取り消される。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from yayoi_ocr.app.launcher import configure_webview


class _ImmutableSettings(dict):
    """pywebview の `ImmutableDict` と同じく、既存のキーだけ変更できる。"""

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(key)
        super().__setitem__(key, value)


class _FakeWebview:
    settings = _ImmutableSettings(
        {"ALLOW_DOWNLOADS": False, "OPEN_EXTERNAL_LINKS_IN_BROWSER": True}
    )


def test_downloads_are_enabled_so_that_the_save_dialog_appears():
    wv = _FakeWebview()
    configure_webview(wv)
    assert wv.settings["ALLOW_DOWNLOADS"] is True
