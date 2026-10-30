// API クライアント。トークンは URL のフラグメント(#token=…)から読み、メモリにだけ保持する
// (localStorage / sessionStorage / Cookie には置かない)。読んだらフラグメントを消す。

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
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
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return (await res.json()) as T;
}

/** 画像などのバイナリ。<img src> にはトークンを付けられないので、fetch → blob URL にする。 */
export async function apiBlobUrl(path: string): Promise<string> {
  if (token === null) throw new ApiError(401, "no token");
  const res = await fetch(path, { headers: { Authorization: `Bearer ${token}` }, credentials: "omit" });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return URL.createObjectURL(await res.blob());
}
