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

/** ファイルの中身をそのまま送る(アップロード)。サーバは利用者のパスを読まない。 */
export async function apiUpload<T>(path: string, file: Blob): Promise<T> {
  if (token === null) throw new ApiError(401, "no token");
  const res = await fetch(path, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/octet-stream" },
    body: file,
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

/** ファイルの中身をそのまま PUT する(復元の zip など、大きいファイル用。送った割合を `onProgress`(0〜1)で知らせる)。 */
export function apiPutWithProgress<T>(path: string, file: Blob, onProgress: (ratio: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    if (token === null) {
      reject(new ApiError(401, "no token"));
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", path);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "network"));
    xhr.onload = () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText) as Record<string, unknown>;
      } catch {
        /* 本文が JSON でない */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, xhr.statusText, body));
    };
    xhr.send(file);
  });
}
