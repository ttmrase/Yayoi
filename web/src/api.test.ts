import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, hasToken, initToken, resetTokenForTest } from "./api";

function fakeLocation(hash: string): Location {
  return { hash, pathname: "/", search: "" } as Location;
}

describe("token handling", () => {
  beforeEach(() => resetTokenForTest());

  it("reads the token from the fragment, keeps it in memory and removes the fragment", () => {
    const replaceState = vi.fn();
    expect(initToken(fakeLocation("#token=abc_DEF-123"), { replaceState } as unknown as History)).toBe(true);
    expect(hasToken()).toBe(true);
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
    expect(JSON.stringify(Object.entries(localStorage))).not.toContain("abc_DEF-123");
    expect(JSON.stringify(Object.entries(sessionStorage))).not.toContain("abc_DEF-123");
  });

  it("does nothing without a token", () => {
    expect(initToken(fakeLocation(""), { replaceState: vi.fn() } as unknown as History)).toBe(false);
    expect(hasToken()).toBe(false);
  });

  it("refuses to call the API without a token", async () => {
    await expect(api("/api/health")).rejects.toBeInstanceOf(ApiError);
  });

  it("sends the Bearer header and no credentials", async () => {
    initToken(fakeLocation("#token=t0k"), { replaceState: vi.fn() } as unknown as History);
    const f = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) });
    vi.stubGlobal("fetch", f);
    await api("/api/health");
    const init = f.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer t0k");
    expect(init.credentials).toBe("omit");
    vi.unstubAllGlobals();
  });
});
