
export type AnalysisStatus =
  | "completed"
  | "partial"
  | "failed"
  | "no_evidence";

export interface EvidenceItem {
  evidence_id?: number | string;
  paper?: string;
  paper_title?: string;
  page?: number | string;
  evidence_text?: string;
  text?: string;
  claim?: string | null;
  dataset?: string | null;
  method?: string | null;
  metric?: string | null;
  conditions?: string | null;
  retrieval_score?: number | null;
  chunk_id?: number | string | null;
  [key: string]: unknown;
}

export interface ResearchTheme {
  theme_id: string;
  theme: string;
  evidence: EvidenceItem[];
  source_count: number;
  cross_paper: boolean;
}

export interface ThemeSourceFinding {
  page?: number | string | null;
  claim?: string | null;
  evidence_text?: string | null;
  chunk_id?: number | string | null;
}

export interface ThemeSourceSummary {
  paper: string;
  findings: ThemeSourceFinding[];
}

export interface ThemeComparison {
  theme_id: string;
  theme: string;
  relationship: string;
  relationship_label: string;
  explanation: string;
  shared_concern?: string | null;
  important_differences?: string[];
  limitations?: string[];
  source_count: number;
  papers: string[];
  source_summaries: ThemeSourceSummary[];
  agreement_established: boolean;
  conflict_established: boolean;
  analysis_method: string;
}

export interface AnalysisFailure {
  group_index: number;
  claim: string;
  candidate_evidence: number;
  reason: string;
}

export interface EvidenceResult {
  question: string;
  status: AnalysisStatus;
  evidence: EvidenceItem[];
  claim_groups: Record<string, unknown>[];
  themes?: ResearchTheme[];
  theme_comparisons?: ThemeComparison[];
  relationships: Record<string, unknown>[];
  audits: Record<string, unknown>[];
  analysis_failures: AnalysisFailure[];
  source_count?: number;
  cross_paper_evidence?: boolean;
  cross_paper_theme_count?: number;
}

export interface ResearchResult {
  question: string;
  query_type: string;
  route: string;
  answer: string | null;
  answer_generated: boolean;
  analysis_status: AnalysisStatus;
  analysis_message: string;
  audit_completed: boolean;
  failed_group_count: number;
  result: EvidenceResult;
}

export interface QueryResponse {
  success: boolean;
  analysis_status: AnalysisStatus;
  audit_completed: boolean;
  data: ResearchResult;
}

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000";

export async function analyzeResearch(
  question: string,
  signal?: AbortSignal
): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: question.trim(),
    }),
    signal,
  });

  if (!response.ok) {
    if (response.status === 422) {
      throw new Error(
        "Please enter a valid research question."
      );
    }

    if (response.status === 400) {
      throw new Error(
        "The research request was rejected."
      );
    }

    throw new Error(
      "ResearchLens could not complete the request. Please try again."
    );
  }

  const data: QueryResponse = await response.json();

  if (!data.success || !data.data?.result) {
    throw new Error(
      "The server returned an invalid research response."
    );
  }

  return data;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}