"""ローカルサーバの防御(docs/phase3_design.md §9)。

- 127.0.0.1 だけにバインドするのは起動側(launcher)の責任。ここでは **Host ヘッダー** を検証する(DNS rebinding 対策)。
- `/api/` は起動ごとのランダムなトークン(`Authorization: Bearer`)を必須にする。静的ファイル(画面そのもの)にはデータが無いので不要。
- **CORS は許可しない**: `Access-Control-*` を一切返さない(同一オリジンで配信するので不要)。`Origin` が付いていて同一オリジンでなければ拒否する。
- 全ての応答に CSP などのセキュリティヘッダーを付ける。外部リソース(CDN・Web フォント)は一切読み込まない。
"""

from __future__ import annotations

import hmac
import secrets

from starlette.types import ASGIApp, Message, Receive, Scope, Send

CSP = (
    "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    b"content-security-policy": CSP.encode(),
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"cache-control": b"no-store",
}


def new_token() -> str:
    return secrets.token_urlsafe(32)


def allowed_hosts_for(port: int) -> frozenset[str]:
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


class SecurityMiddleware:
    """Host 検証 → Origin 検証 → トークン検証(/api/ のみ)→ 応答ヘッダーの付与・CORS ヘッダーの除去。"""

    def __init__(self, app: ASGIApp, token: str, allowed_hosts: frozenset[str]) -> None:
        self.app = app
        self.token = token
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # 使わない(進捗は fetch のストリーム)。閉じる
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope["headers"]}
        host = headers.get(b"host", b"").decode("latin-1").lower()
        if host not in self.allowed_hosts:
            await self._reject(send, 400, "bad host")
            return
        origin = headers.get(b"origin")
        if origin is not None and origin.decode("latin-1").lower() not in {
            f"http://{h}" for h in self.allowed_hosts
        }:
            await self._reject(send, 403, "bad origin")
            return
        if scope["path"].startswith("/api/") or scope["path"] == "/api":
            auth = headers.get(b"authorization", b"").decode("latin-1")
            scheme, _, supplied = auth.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                supplied.encode(), self.token.encode()
            ):
                await self._reject(send, 401, "unauthorized")
                return

        async def send_wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                kept = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if not k.lower().startswith(b"access-control-")
                    and k.lower() not in SECURITY_HEADERS
                ]
                kept += list(SECURITY_HEADERS.items())
                message = {**message, "headers": kept}
            await send(message)

        await self.app(scope, receive, send_wrapped)

    async def _reject(self, send: Send, status: int, text: str) -> None:
        body = text.encode()
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
        ]
        headers += list(SECURITY_HEADERS.items())
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
