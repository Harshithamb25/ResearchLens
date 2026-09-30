
import type { QueryResponse, AnalysisStatus } from "./research";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000";

export interface ResearchSessionSummary {
  id: string;
  project_id: string;
  question: string;
  analysis_status: AnalysisStatus;
  audit_completed: boolean;
  created_at: string;
}

export interface SavedResearchSession
  extends ResearchSessionSummary {
  response: QueryResponse;
}

interface SavedResearchSessionResponse
  extends SavedResearchSession {
  success: boolean;
}

interface SessionListResponse {
  success: boolean;
  project_id: string;
  sessions: ResearchSessionSummary[];
  total: number;
}

interface DeleteSessionResponse {
  success: boolean;
  message: string;
  session_id: string;
}

async function readApiError(
  response: Response,
  fallback: string
): Promise<Error> {
  try {
    const body: unknown = await response.json();

    if (
      body !== null &&
      typeof body === "object" &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      return new Error(body.detail);
    }
  } catch {
    // Use the fallback message.
  }

  return new Error(
    `${fallback} (HTTP ${response.status}).`
  );
}

export async function getResearchSessions(
  projectId: string,
  signal?: AbortSignal
): Promise<ResearchSessionSummary[]> {
  const response = await fetch(
    `${API_URL}/projects/${encodeURIComponent(
      projectId
    )}/sessions`,
    { signal }
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Could not load research history"
    );
  }

  const data: SessionListResponse =
    await response.json();

  if (
    !data.success ||
    !Array.isArray(data.sessions)
  ) {
    throw new Error(
      "The server returned invalid research history."
    );
  }

  return data.sessions;
}

export async function getResearchSession(
  projectId: string,
  sessionId: string,
  signal?: AbortSignal
): Promise<SavedResearchSession> {
  const response = await fetch(
    `${API_URL}/projects/${encodeURIComponent(
      projectId
    )}/sessions/${encodeURIComponent(sessionId)}`,
    { signal }
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Could not open the research session"
    );
  }

  const data: SavedResearchSessionResponse =
    await response.json();

  if (
    !data.success ||
    data.project_id !== projectId ||
    !data.response?.success ||
    !data.response?.data?.result
  ) {
    throw new Error(
      "The saved research session is incomplete or invalid."
    );
  }

  return data;
}

export async function deleteResearchSession(
  projectId: string,
  sessionId: string
): Promise<void> {
  const response = await fetch(
    `${API_URL}/projects/${encodeURIComponent(
      projectId
    )}/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" }
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Could not delete the research session"
    );
  }

  const data: DeleteSessionResponse =
    await response.json();

  if (!data.success) {
    throw new Error(
      "The server did not confirm session deletion."
    );
  }
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response: QueryResponse | null;
  created_at: string;
}

export interface ProjectConversation {
  id: string;
  project_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

export async function getProjectConversation(
  projectId: string,
  signal?: AbortSignal
): Promise<ProjectConversation> {
  const response = await fetch(
    API_URL + "/projects/" + encodeURIComponent(projectId) + "/conversation",
    { signal }
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Could not load the project conversation"
    );
  }

  const data: {
    success: boolean;
    conversation: ProjectConversation;
  } = await response.json();

  if (!data.success || !data.conversation) {
    throw new Error(
      "The server returned an invalid project conversation."
    );
  }

  return data.conversation;
}
