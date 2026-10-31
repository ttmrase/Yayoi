#!/usr/bin/env python3
"""試用版の導入先(または開発環境)で、アプリを**開発用でないモード**で起動し、HTTP で一通り動かす煙試験。Windows の CI から使う(`.github/workflows/ci.yml` の trial-windows)。

    python scripts/trial_smoke.py --python <導入先の venv の python> --models <重みの置き場> [--data-dir DIR] [--timeout 900]

この script 自身は、アプリを導入した環境ではなく、リポジトリの開発環境で動かす(合成ページの生成に、日本語フォントと `yayoi_ocr.synth` を使うため)。アプリは別プロセス
(`--python` の Python)で起動する。起動は本番と同じ `RunningServer`(`dev=False`・本物の認識・ランダムなポートとトークン)で、**ウィンドウだけを開かない**
(`--no-window` は `--dev` が必要で、トークンを表示してしまうため、本番の起動経路では使えない)。

確かめること(実データは使わない。合成の 1 ページだけ。値はログに出さない):
  1. 127.0.0.1 だけで待ち受ける(この PC の他のアドレスから繋がらない)・開発用の API が無い・セキュリティヘッダー
  2. 合成ページの取り込み → 本物の認識(重みは基本の重みだけでもよい)→ 要確認の処理 → 確定
  3. 設定 → 出力前の検証 → 弥生インポート形式の出力 → **cp932 のバイト列**(25 項目・CRLF・BOM なし・行数・日本語の保たれ方)
  4. バックアップの作成 → 保存(合言葉つきの URL)→ 検査(別プロセス)
  5. アプリの出力(標準出力・標準エラー)に、入力した値が出ていない
IME・高 DPI・GPU・弥生への実際の取り込みは、実機のチェックリスト(docs/windows_checklist.md)。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

SERVER = """
import sys, time
from pathlib import Path
from yayoi_ocr.app.launcher import RunningServer
s = RunningServer(Path(sys.argv[1]), dev=False, backend="real").start()
print(s.url, flush=True)
while s.thread.is_alive():
    time.sleep(0.5)
"""

# 入力する値(合成。cp932 で表せる日本語を含める)
ROWS = [
    ("R8.4.1", "現金", "事務用品費", "1500"),
    ("R8.4.2", "現金", "新聞代", "2500"),
    ("R8.4.3", "現金", "旅費交通費", "3500"),
]
TITLE = "旅費交通費"
SECRET_WORDS = [r[2] for r in ROWS]  # アプリの出力に出てはいけない


class SmokeError(Exception):
    pass


STATE: dict[str, Path] = {}  # 失敗したとき、アプリの標準エラーの末尾を出すために覚えておく


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


class Client:
    def __init__(self, base: str, token: str) -> None:
        self.base, self.token = base, token

    def call(
        self,
        method: str,
        path: str,
        body: object = None,
        *,
        raw: bytes | None = None,
        auth: bool = True,
        timeout: float = 60,
    ):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if auth:
            req.add_header("Authorization", f"Bearer {self.token}")
        if raw is not None:
            req.add_header("Content-Type", "application/octet-stream")
        elif body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def json(
        self, method: str, path: str, body: object = None, *, ok: tuple[int, ...] = (200,), **kw
    ):
        status, _h, data = self.call(method, path, body, **kw)
        if status not in ok:
            code = ""
            with contextlib.suppress(ValueError):
                code = str(json.loads(data).get("error", ""))
            raise SmokeError(f"{method} {path} -> {status} {code}")
        return json.loads(data) if data else {}


def start_server(
    python: str, models: Path, data_dir: Path, log_dir: Path
) -> tuple[subprocess.Popen, str, str, Path, Path]:
    import os

    env = {**os.environ, "YAYOI_MODELS_DIR": str(models), "PYTHONUTF8": "1"}
    out_path, err_path = log_dir / "server.out", log_dir / "server.err"
    out, err = open(out_path, "w", encoding="utf-8"), open(err_path, "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(
        [python, "-c", SERVER, str(data_dir)], stdout=out, stderr=err, env=env, text=True
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SmokeError(f"アプリが起動直後に終了しました(終了コード {proc.returncode})")
        text = out_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("http://127.0.0.1:") and "#token=" in line:
                url, token = line.strip().split("#token=", 1)
                return proc, url.rstrip("/"), token, out_path, err_path
        time.sleep(0.3)
    raise SmokeError("アプリの URL が表示されませんでした")


def synthetic_page_png() -> bytes:
    import cv2

    from yayoi_ocr.synth.generator import SynthConfig, generate_sample

    s = generate_sample(SynthConfig(seed=424242, fill_rows=6))
    ok, buf = cv2.imencode(".png", s.image)
    if not ok:
        raise SmokeError("合成ページを作れませんでした")
    return buf.tobytes()


def wait_for(c: Client, fn, what: str, timeout: float, every: float = 1.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(every)
    raise SmokeError(f"時間切れ: {what}")


def check_network(port: int) -> None:
    """127.0.0.1 以外のアドレスからは繋がらない。"""
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        host_ip = "127.0.0.1"
    if host_ip.startswith("127."):
        log("この PC の他のアドレスが見つからないので、外部からの接続の確認は省略")
        return
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect((host_ip, port))
    except OSError:
        log(f"OK 127.0.0.1 だけで待ち受けている({host_ip}:… には繋がらない)")
        return
    finally:
        s.close()
    raise SmokeError(f"{host_ip}:{port} に繋がってしまいました(0.0.0.0 で待ち受けている)")


def run(args: argparse.Namespace) -> None:
    work = Path(tempfile.mkdtemp(prefix="yayoi-smoke-"))
    data_dir = args.data_dir or work / "data"
    proc, base, token, out_path, err_path = start_server(args.python, args.models, data_dir, work)
    STATE["err"] = err_path
    port = int(base.rsplit(":", 1)[1])
    c = Client(base, token)
    try:
        # ---- 1. 起動の状態
        h = c.json("GET", "/api/health")
        if h.get("dev") is not False:
            raise SmokeError("開発用のモードで起動しています")
        log(f"OK 起動(版 {h['version']}、スキーマ {h['schema']}、dev=False)")
        if c.call("GET", "/api/health", auth=False)[0] != 401:
            raise SmokeError("トークンなしで /api/ に繋がりました")
        status, headers, body = c.call("GET", "/", auth=False)
        if status != 200 or (b"<div" not in body.lower() and b"root" not in body.lower()):
            raise SmokeError(f"画面が配信されません({status})")
        hl = {k.lower(): v for k, v in headers.items()}
        if "content-security-policy" not in hl or "access-control-allow-origin" in hl:
            raise SmokeError("セキュリティヘッダーがおかしい")
        if c.call("POST", "/api/dev/jobs", {"pages": 1})[0] not in (404, 405):
            raise SmokeError("開発用の API が有効です")
        check_network(port)

        # ---- 2. 取り込み → 認識 → 確定
        png = synthetic_page_png()
        up = c.json("POST", "/api/uploads?name=smoke.png", raw=png, timeout=120)
        job = c.json("POST", "/api/jobs", {"upload_ids": [up["id"]], "name": "smoke"})
        pid = job["page_ids"][0]
        t0 = time.time()

        def recognized():
            p = c.json("GET", f"/api/pages/{pid}")
            if p["status"] == "failed" or p.get("error"):
                raise SmokeError(f"認識に失敗しました({p.get('error')})")
            return p if p["status"] in ("review", "reviewing") and p["cells"] else None

        page = wait_for(c, recognized, "認識", args.timeout, every=2.0)
        log(f"OK 認識({time.time() - t0:.0f} 秒、セル {len(page['cells'])})")
        if len(page["cells"]) != 116:
            raise SmokeError(f"セル数が 116 ではありません({len(page['cells'])})")

        want = {(r, col): v for r, row in enumerate(ROWS) for col, v in enumerate(row)}
        for cell in page["cells"]:
            val = want.get((cell["row"], cell["col"]), "")
            if cell["review"] or cell["reasons"]:
                c.json("PATCH", f"/api/pages/{pid}/cells/{cell['id']}", {"action": "pass"})
            c.json(
                "PATCH", f"/api/pages/{pid}/cells/{cell['id']}", {"action": "edit", "value": val}
            )
        page = c.json("GET", f"/api/pages/{pid}")
        for cid in list(
            page["summary"]["ambiguous"]
        ):  # 明示の選択が要るセル(まれ): 候補の先頭を選ぶ
            c.json(
                "PATCH",
                f"/api/pages/{pid}/cells/{cid}",
                {"action": "candidate", "candidate_index": 0},
            )
        page = c.json("PUT", f"/api/pages/{pid}/title", {"title": TITLE})
        s = page["summary"]
        if not s["can_confirm"]:
            raise SmokeError(
                f"確定できません(未通過 {len(s['unpassed'])}・誤り {len(s['errors'])}・未選択 {len(s['ambiguous'])})"
            )
        c.json("POST", f"/api/pages/{pid}/confirm", {})
        log("OK 確認 → 確定")

        # ---- 3. 設定 → 出力 → cp932
        cur = c.json("GET", "/api/settings")["settings"]
        cur.update(
            wizard_done=True,
            taxpayer="exempt",
            payments=[{"name": "現金", "credit_account": "現金", "credit_sub": ""}],
            titles=[{"name": TITLE, "debit_account": TITLE, "debit_sub": ""}],
        )
        c.json("PUT", "/api/settings", cur)
        v = c.json("POST", "/api/exports/validate", {"page_ids": [pid], "format": "yayoi_import"})
        errs = [i for i in v.get("issues", []) if i.get("severity") == "error"]
        if errs:
            raise SmokeError(
                f"出力前の検証に誤りがあります: {sorted({i.get('code', '?') for i in errs})}"
            )
        rec = c.json(
            "POST",
            "/api/exports",
            {"page_ids": [pid], "format": "yayoi_import", "accept_duplicates": False},
        )
        status, _h, raw = c.call("GET", f"/api/exports/{rec['id']}/files/0")
        if status != 200:
            raise SmokeError(f"出力ファイルを取得できません({status})")
        check_cp932(raw)
        log(f"OK 出力 → cp932({len(raw)} バイト、{len(ROWS)} 行)")

        # ---- 4. バックアップ
        c.json("POST", "/api/backup/create", {"include_originals": False})
        st = wait_for(
            c,
            lambda: (lambda s: s if (not s["running"] and (s["created"] or s["error"])) else None)(
                c.json("GET", "/api/backup")
            ),
            "バックアップの作成",
            300,
        )
        if st["error"] or not st["created"]:
            raise SmokeError(f"バックアップの作成に失敗しました({st['error']})")
        link = c.json("POST", "/api/backup/download-link", {})["url"]
        status, _hdr, blob = c.call("GET", link, auth=False)
        if status != 200 or blob[:2] != b"PK":
            raise SmokeError(f"バックアップのダウンロードに失敗しました({status})")
        if c.call("GET", link, auth=False)[0] != 404:
            raise SmokeError("合言葉が 1 回で失効しません")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            if "manifest.json" not in zf.namelist() or "db/app.sqlite3" not in zf.namelist():
                raise SmokeError("バックアップの中身がおかしい")
        c.json("PUT", "/api/backup/upload", raw=blob, timeout=300)
        st = wait_for(
            c,
            lambda: (lambda s: s if (not s["running"] and (s["incoming"] or s["error"])) else None)(
                c.json("GET", "/api/backup")
            ),
            "バックアップの検査",
            300,
        )
        if st["error"] or st["incoming"]["n_confirmed_pages"] != 1:
            raise SmokeError(f"バックアップの検査に失敗しました({st['error']})")
        c.json("POST", "/api/backup/cancel", {})
        log(f"OK バックアップ(作成 → 保存 → 検査。zip {len(blob)} バイト)")
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
    # ---- 5. 出力に値が出ていない
    for p in (out_path, err_path):
        text = p.read_text(encoding="utf-8", errors="replace")
        leaked = [w for w in SECRET_WORDS if w in text]
        if leaked:
            raise SmokeError(f"アプリの出力に入力した値が出ています({p.name})")
    log("OK アプリの標準出力・標準エラーに入力した値は出ていない")


def check_cp932(raw: bytes) -> None:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SmokeError("BOM があります")
    text = raw.decode("cp932")  # デコードできること
    if "\r\n" not in text or "\n" in text.replace("\r\n", ""):
        raise SmokeError("改行が CRLF ではありません")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != len(ROWS):
        raise SmokeError(f"行数が違います({len(rows)} 行)")
    if any(len(r) != 25 for r in rows):
        raise SmokeError(f"項目数が 25 ではありません({sorted({len(r) for r in rows})})")
    body = "\n".join(",".join(r) for r in rows)
    for _d, _a, desc, amount in ROWS:
        if desc not in body or amount not in body:
            raise SmokeError("入力した値が出力に見つかりません")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--python", default=sys.executable, help="アプリを動かす Python(導入先の venv の python)"
    )
    ap.add_argument("--models", type=Path, required=True, help="重みの置き場(YAYOI_MODELS_DIR)")
    ap.add_argument("--data-dir", type=Path, help="アプリのデータ置き場(既定: 一時ディレクトリ)")
    ap.add_argument("--timeout", type=float, default=900, help="認識を待つ秒数")
    args = ap.parse_args()
    try:
        run(args)
    except SmokeError as exc:
        print(f"[smoke] 失敗: {exc}", file=sys.stderr, flush=True)
        err = STATE.get("err")
        if (
            err is not None and err.is_file()
        ):  # 例外の型と位置だけ(値は出さない作り)。調査のため末尾を出す
            tail = err.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print(
                "[smoke] アプリの標準エラー(末尾):\n" + "\n".join(tail), file=sys.stderr, flush=True
            )
        return 1
    log("すべて成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
