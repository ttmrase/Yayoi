import { api, apiBlobUrl, apiUpload } from "./api";
import type {
  AppSettings,
  ExportList,
  ExportRecord,
  ExportValidation,
  Generation,
  JobDetail,
  JobSummary,
  LayoutData,
  LearningData,
  Page,
  Pt,
  SettingsResponse,
  TaxMemory,
  TrainingData,
  Upload,
} from "./types";

export const fetchJobs = () => api<JobSummary[]>("/api/jobs");
export const fetchJob = (id: number) => api<JobDetail>(`/api/jobs/${id}`);
export const fetchPage = (id: number) => api<Page>(`/api/pages/${id}`);

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const uploadFile = (file: File) => apiUpload<Upload>(`/api/uploads?name=${encodeURIComponent(file.name)}`, file);
export const deleteUpload = (id: string) => api<{ ok: boolean }>(`/api/uploads/${id}`, { method: "DELETE" });
export const createJob = (uploadIds: string[], name: string) =>
  api<{ job_id: number; page_ids: number[] }>("/api/jobs", json("POST", { upload_ids: uploadIds, name }));
export const retryPage = (pageId: number) => api<{ ok: boolean }>(`/api/pages/${pageId}/retry`, json("POST", {}));

export const fetchSettings = () => api<SettingsResponse>("/api/settings");
export const saveSettings = (s: AppSettings) => api<SettingsResponse>("/api/settings", json("PUT", s));
export const putTitle = (pageId: number, title: string) => api<Page>(`/api/pages/${pageId}/title`, json("PUT", { title }));
export const putRowTax = (pageId: number, row: number, category: string | null) =>
  api<Page>(`/api/pages/${pageId}/rows/${row}/tax`, json("PUT", { tax_category: category }));
export const fetchTaxMemory = () => api<TaxMemory>("/api/tax-memory");
export const putTaxMemory = (id: number, category: string) =>
  api<TaxMemory>(`/api/tax-memory/${id}`, json("PUT", { tax_category: category }));
export const deleteTaxMemory = (id: number) => api<TaxMemory>(`/api/tax-memory/${id}`, { method: "DELETE" });
export const clearTaxMemory = () => api<TaxMemory>("/api/tax-memory/clear", json("POST", {}));
export const fetchExports = () => api<ExportList>("/api/exports");
export const validateExport = (pageIds: number[], format: string) =>
  api<ExportValidation>("/api/exports/validate", json("POST", { page_ids: pageIds, format }));
export const runExport = (pageIds: number[], format: string, acceptDuplicates: boolean) =>
  api<ExportRecord>("/api/exports", json("POST", { page_ids: pageIds, format, accept_duplicates: acceptDuplicates }));
export const exportFile = (id: number, idx = 0) => apiBlobUrl(`/api/exports/${id}/files/${idx}`);
export const exportSample = (format: string, date: string) =>
  apiBlobUrl(`/api/exports/sample?format=${encodeURIComponent(format)}&date=${encodeURIComponent(date)}`);
export const fetchLearning = () => api<LearningData>("/api/learning");
export const learningReport = () => apiBlobUrl("/api/learning/report");
export const fetchTraining = () => api<TrainingData>("/api/training");
export const startTraining = () => api<{ run_id: number }>("/api/training/start", json("POST", {}));
export const cancelTraining = () => api<{ cancelled: boolean }>("/api/training/cancel", json("POST", {}));
export const activateGeneration = (id: number) => api<Generation>(`/api/generations/${id}/activate`, json("POST", {}));
export const resetGenerations = (deleteLearningData: boolean) =>
  api<{ discarded: number; deleted_images: number }>("/api/generations/reset", json("POST", { delete_learning_data: deleteLearningData }));

export type CellAction = "pass" | "edit" | "confirm" | "candidate" | "revert";

export const patchCell = (
  pageId: number,
  cellId: number,
  action: CellAction,
  extra: { value?: string; candidate_index?: number } = {},
) => api<Page>(`/api/pages/${pageId}/cells/${cellId}`, json("PATCH", { action, ...extra }));

export const undoPage = (pageId: number) =>
  api<{ page: Page; cell_id: number | null }>(`/api/pages/${pageId}/undo`, json("POST", {}));
export const redoPage = (pageId: number) =>
  api<{ page: Page; cell_id: number | null }>(`/api/pages/${pageId}/redo`, json("POST", {}));
export const confirmPage = (pageId: number, reviewSeconds: number) =>
  api<Page>(`/api/pages/${pageId}/confirm`, json("POST", { review_seconds: reviewSeconds }));
export const reopenPage = (pageId: number) => api<Page>(`/api/pages/${pageId}/reopen`, json("POST", {}));
export const approveLayout = (pageId: number) =>
  api<Page>(`/api/pages/${pageId}/layout/approve`, json("POST", {}));
export const fetchLayout = (pageId: number) => api<LayoutData>(`/api/pages/${pageId}/layout`);
export const layoutImage = (pageId: number) => apiBlobUrl(`/api/pages/${pageId}/layout/image`);
export const recutPage = (pageId: number, corners: Pt[]) =>
  api<{ ok: boolean }>(`/api/pages/${pageId}/layout/recut`, json("POST", { corners }));
export const putPosition = (jobId: number, pageId: number, cell: number | null) =>
  api<{ ok: boolean }>(`/api/jobs/${jobId}/position`, json("PUT", { page_id: pageId, cell }));
export const cellImage = (pageId: number, cellId: number) =>
  apiBlobUrl(`/api/pages/${pageId}/cells/${cellId}/image`);

export const devCreateJob = (body: Record<string, unknown>) =>
  api<{ job_id: number; pages: { id: number; truth: Record<string, string> }[] }>(
    "/api/dev/jobs",
    json("POST", body),
  );
