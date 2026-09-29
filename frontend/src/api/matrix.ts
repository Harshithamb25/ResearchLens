
import { getProjectPaperUrl } from "./projects";

export type MatrixFieldKey =
  | "methodology"
  | "datasets"
  | "evaluation_metrics"
  | "accuracy_results"
  | "advantages"
  | "limitations"
  | "applications";

export interface MatrixSource {
  paper: string;
  document_id: string;
  page: number;
}

export interface MatrixEntry {
  value: string;
  evidence_text: string;
  source: MatrixSource;
}

export interface MatrixField {
  reported: boolean;
  entries: MatrixEntry[];
}

export interface MatrixRow {
  serial_number: number;
  paper_title: string;
  document_id: string;
  methodology: string;
  datasets: string;
  evaluation_metrics: string;
  accuracy_results: string;
  advantages: string;
  limitations: string;
  applications: string;
  fields: Record<MatrixFieldKey, MatrixField>;
}

export interface PaperMatrix {
  success: boolean;
  project_id: string;
  extraction_mode: string;
  columns: { key: string; label: string }[];
  rows: MatrixRow[];
  paper_count: number;
  note: string;
}

function projectEndpoint(
  projectId: string,
  suffix: string
): string {
  // Reuse the API origin and prefix already configured
  // by the existing project PDF URL helper.
  const pdfUrl = new URL(
    getProjectPaperUrl(projectId, "matrix-placeholder"),
    window.location.origin
  );

  const marker = `/projects/${encodeURIComponent(projectId)}/papers/`;
  const markerIndex = pdfUrl.pathname.indexOf(marker);

  if (markerIndex < 0) {
    throw new Error("Could not determine the ResearchLens API URL.");
  }

  const apiPrefix = pdfUrl.pathname.slice(0, markerIndex);

  return (
    pdfUrl.origin +
    apiPrefix +
    `/projects/${encodeURIComponent(projectId)}/${suffix}`
  );
}

export async function getPaperMatrix(
  projectId: string,
  signal?: AbortSignal
): Promise<PaperMatrix> {
  const response = await fetch(
    projectEndpoint(projectId, "paper-matrix"),
    { signal }
  );

  if (!response.ok) {
    let message = "Could not generate the evidence matrix.";

    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Preserve the fallback message.
    }

    throw new Error(message);
  }

  return (await response.json()) as PaperMatrix;
}

export function getPaperMatrixCsvUrl(
  projectId: string
): string {
  return projectEndpoint(projectId, "paper-matrix.csv");
}