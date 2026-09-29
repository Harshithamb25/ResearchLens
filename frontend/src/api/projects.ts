
const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000";

/* ============================================================
   TYPES
   ============================================================ */

export interface ResearchProject {
  id: string;
  name: string;
  description: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  document_count: number;
}

export interface ProjectPaper {
  id: string;
  filename: string;
  indexed_chunks: number;
  indexed: boolean;
  created_at: string;
}

interface ProjectListResponse {
  projects: ResearchProject[];
  total: number;
}

interface ProjectPapersResponse {
  success?: boolean;
  project_id?: string;
  papers: ProjectPaper[];
  total?: number;
}

/* ============================================================
   ERROR HANDLING
   ============================================================ */

async function readApiError(
  response: Response
): Promise<string> {
  try {
    const data: unknown = await response.json();

    if (
      typeof data === "object" &&
      data !== null &&
      "detail" in data
    ) {
      const detail = data.detail;

      if (typeof detail === "string") {
        return detail;
      }

      if (Array.isArray(detail)) {
        return detail
          .map((item: unknown) => {
            if (
              typeof item === "object" &&
              item !== null &&
              "msg" in item &&
              typeof item.msg === "string"
            ) {
              return item.msg;
            }

            return "Invalid request.";
          })
          .join(" ");
      }
    }
  } catch {
    // Use the fallback HTTP error message.
  }

  return `Request failed (HTTP ${response.status}).`;
}

/* ============================================================
   PROJECT MANAGEMENT
   ============================================================ */

/**
 * Fetch all research projects.
 *
 * The optional AbortSignal allows React components to
 * cancel outdated requests when they unmount.
 */
export async function getProjects(
  signal?: AbortSignal
): Promise<ResearchProject[]> {
  const response = await fetch(
    `${API_URL}/projects`,
    { signal }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response)
    );
  }

  const data: ProjectListResponse =
    await response.json();

  if (!Array.isArray(data.projects)) {
    throw new Error(
      "Invalid project list returned by the server."
    );
  }

  return data.projects;
}

/**
 * Create a new research project.
 */
export async function createProject(
  name: string,
  description = ""
): Promise<ResearchProject> {
  const response = await fetch(
    `${API_URL}/projects`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: name.trim(),
        description: description.trim(),
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response)
    );
  }

  return response.json();
}

/* ============================================================
   PAPER LIBRARY
   ============================================================ */

/**
 * Fetch papers belonging to one research project.
 */
export async function getProjectPapers(
  projectId: string,
  signal?: AbortSignal
): Promise<ProjectPaper[]> {
  const response = await fetch(
    `${API_URL}/projects/${encodeURIComponent(
      projectId
    )}/papers`,
    { signal }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response)
    );
  }

  const data: ProjectPapersResponse =
    await response.json();

  if (!Array.isArray(data.papers)) {
    throw new Error(
      "The server did not return a valid list of papers."
    );
  }

  return data.papers;
}

/**
 * Upload and index one PDF in the selected project.
 *
 * The browser automatically sets the correct multipart
 * boundary for FormData. Do not manually set Content-Type.
 */
export async function uploadProjectPaper(
  projectId: string,
  file: File
): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_URL}/projects/${encodeURIComponent(
      projectId
    )}/papers/upload`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response)
    );
  }

  return response.json();
}

/* ============================================================
   ORIGINAL PDF ACCESS
   ============================================================ */

/**
 * Return the backend URL for an uploaded PDF.
 *
 * Used by the Paper Library and clickable research
 * citations to open the original source document.
 */
export function getProjectPaperUrl(
  projectId: string,
  documentId: string
): string {
  return (
    `${API_URL}/projects/` +
    `${encodeURIComponent(projectId)}/papers/` +
    `${encodeURIComponent(documentId)}/file`
  );
}