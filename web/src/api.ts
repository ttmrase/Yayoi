// API クライアント。トークンは URL のフラグメント(#token=…)から読み、メモリにだけ保持する
// (localStorage / sessionStorage / Cookie には置かない)。読んだらフラグメントを消す。

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: Record<string, unknown> = {},
  ) {
    super(message);
  }
  /** サーバが返すエラーコード(例: ambiguity_unresolved) */
  get code(): string {
    return typeof this.body.error === "string" ? this.body.error : "";
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let body: Record<string, unknown> = {};
  try {
    body = (await res.json()) as Record<string, unknown>;
  } catch {
    /* 本文が JSON でない(401 のテキストなど) */
  }
  return new ApiError(res.status, res.statusText, body);
}

let token: string | null = null;

/** location.hash から token を取り出してメモリに保持し、フラグメントを URL から消す。 */
export function initToken(loc: Location = window.location, hist: History = window.history): boolean {
  const m = /(?:^#|&)token=([^&]+)/.exec(loc.hash);
  if (!m) return token !== null;
  token = decodeURIComponent(m[1]);
  hist.replaceState(null, "", loc.pathname + loc.search);
  return true;
}

export function hasToken(): boolean {
  return token !== null;
}

export function resetTokenForTest(): void {
  token = null;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  if (token === null) throw new ApiError(401, "no token");
  const res = await fetch(path, {
    ...init,
    headers: { ...(init.headers ?? {}), Authorization: `Bearer ${token}` },
    credentials: "omit",
  });
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

/** 画像などのバイナリ。<img src> にはトークンを付けられないので、fetch → blob URL にする。 */
export async function apiBlobUrl(path: string): Promise<string> {
  if (token === null) throw new ApiError(401, "no token");
  const res = await fetch(path, { headers: { Authorization: `Bearer ${token}` }, credentials: "omit" });
  if (!res.ok) throw await errorFrom(res);
  return URL.createObjectURL(await res.blob());
}
