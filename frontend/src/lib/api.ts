import type {
  AnalysisReport,
  JobStatus,
  SubmitResponse,
} from "../types/api";

const BASE = `${import.meta.env.VITE_API_BASE_URL ?? ""}/v1`;

export async function submitImage(
  file: File,
  token: string,
): Promise<SubmitResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  return res.json() as Promise<SubmitResponse>;
}

export async function pollJob(
  jobId: string,
  token: string,
): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${jobId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!res.ok) {
    throw new Error(`Poll failed (${res.status})`);
  }

  return res.json() as Promise<JobStatus>;
}

export async function fetchReport(
  jobId: string,
  token: string,
): Promise<AnalysisReport> {
  const res = await fetch(`${BASE}/reports/${jobId}/json`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!res.ok) {
    throw new Error(`Report fetch failed (${res.status})`);
  }

  return res.json() as Promise<AnalysisReport>;
}

export function pdfUrl(jobId: string): string {
  return `${BASE}/reports/${jobId}/pdf`;
}
