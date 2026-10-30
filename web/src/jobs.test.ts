import { describe, expect, it } from "vitest";
import { importErrorLabel, naturalSort, progressLabel, readyPages, resumeTarget } from "./jobs";
import type { JobDetail, JobSummary } from "./types";

const job = (o: Partial<JobSummary> = {}): JobSummary => ({
  id: 1,
  name: "j",
  status: "ready",
  profile_id: 1,
  n_pages: 4,
  n_confirmed: 0,
  n_processing: 0,
  n_failed: 0,
  last_page_id: null,
  last_cell: null,
  ...o,
});
const detail = (statuses: JobDetail["pages"][number]["status"][]): JobDetail => ({
  ...job({ n_pages: statuses.length }),
  pages: statuses.map((status, i) => ({ id: 10 + i, idx: i, status, layout_status: "ok", error: null })),
});

describe("naturalSort", () => {
  it("数字を数として並べる", () => {
    const names = ["scan10.png", "scan2.png", "scan1.png"].map((name) => ({ name }));
    expect(naturalSort(names).map((f) => f.name)).toEqual(["scan1.png", "scan2.png", "scan10.png"]);
  });
  it("元の配列は変えない", () => {
    const a = [{ name: "b" }, { name: "a" }];
    naturalSort(a);
    expect(a[0].name).toBe("b");
  });
});

describe("進捗の表示", () => {
  it("認識中は終わったページ数(失敗を含む)を出す", () => {
    const j = job({ n_pages: 5, n_processing: 2, n_failed: 1 });
    expect(readyPages(j)).toBe(2);
    expect(progressLabel(j)).toBe("認識中 3 / 5");
  });
  it("認識が終われば失敗数・確認待ち・確定済み", () => {
    expect(progressLabel(job({ n_failed: 1 }))).toBe("失敗 1");
    expect(progressLabel(job())).toBe("確認待ち");
    expect(progressLabel(job({ n_confirmed: 4 }))).toBe("確定済み");
  });
  it("取り込みエラーのコードを日本語にする", () => {
    expect(importErrorLabel("unsupported_type")).toBe("対応していない形式です");
    expect(importErrorLabel("???")).toBe("取り込めませんでした");
  });
});

describe("resumeTarget", () => {
  it("中断した位置のページ(認識済み)を優先する", () => {
    expect(resumeTarget(job({ last_page_id: 12 }), detail(["confirmed", "review", "review"]))).toBe(12);
  });
  it("中断位置のページがまだ認識中なら、最初の未確定のページ", () => {
    expect(resumeTarget(job({ last_page_id: 12 }), detail(["confirmed", "review", "processing"]))).toBe(11);
  });
  it("最初のページが認識中で何も開けないときは null", () => {
    expect(resumeTarget(job(), detail(["processing", "processing"]))).toBeNull();
  });
  it("全部確定済みなら最初のページ", () => {
    expect(resumeTarget(job(), detail(["confirmed", "confirmed"]))).toBe(10);
  });
  it("失敗したページは後回し(他に未確定があればそちら)", () => {
    expect(resumeTarget(job(), detail(["failed", "review"]))).toBe(11);
  });
});
