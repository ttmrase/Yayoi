import { describe, expect, it } from "vitest";
import { normalizeInput } from "./text";

describe("normalizeInput", () => {
  it("turns full-width digits into half-width and strips amount decorations", () => {
    expect(normalizeInput("amount", "１，２００－")).toBe("1200");
    expect(normalizeInput("amount", " 5 000")).toBe("5000");
    expect(normalizeInput("date", "Ｒ８．９．１")).toBe("R8.9.1");
  });
  it("leaves text columns alone", () => {
    expect(normalizeInput("description", "郵便 料金")).toBe("郵便 料金");
  });
});
