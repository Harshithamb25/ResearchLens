
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  ExternalLink,
  FileSearch,
  GitCompareArrows,
  History,
  Library,
  LoaderCircle,
  Menu,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Trash2,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";

import ProjectWorkspace from "./ProjectWorkspace";
import ResearchHistory from "./ResearchHistory";
import { getProjectConversation, type SavedResearchSession, type ConversationMessage } from "./api/history";

import {
  getProjects,
  getProjectPapers,
  getProjectPaperUrl,
  updateProject,
  deleteProject,
  type ProjectPaper,
  type ResearchProject,
} from "./api/projects";

import {
  analyzeResearch,
  analyzeDocument,
  type AnalysisStatus,
  type EvidenceItem,
  type QueryResponse,
  type ResearchTheme,
  type ThemeComparison,
} from "./api/research";

import "./App.css";
import "./ResultView.css";
import "./Workbench.css";

/* ============================================================
   TYPES AND CONSTANTS
   ============================================================ */

type Theme = "light" | "dark";
type View = "research" | "library" | "history";
type ResultTab = "synthesis" | "themes" | "evidence";
type InspectorTab = "overview" | "evidence" | "comparisons";

interface PaperLookup {
  projectId: string;
  papers: ProjectPaper[];
}

const documentExamples = [
  "What is the main idea of this document?",
  "Summarize the methodology in simple terms.",
  "What findings or conclusions does it report?",
];

const researchExamples = [
  "Compare the methodologies across the papers.",
  "Which datasets and evaluation metrics are used?",
  "What limitations or research gaps are reported?",
];

const statusLabels: Record<AnalysisStatus, string> = {
  completed: "Claim audit completed",
  partial: "Partially completed",
  failed: "Audit failed",
  no_evidence: "No analyzable evidence",
  not_applicable: "Document grounding completed",
};

/* ============================================================
   EVIDENCE AND CITATION HELPERS
   ============================================================ */

function evidenceSource(item: EvidenceItem): string {
  return String(
    item.paper_title ??
      item.paper ??
      "Source not specified"
  );
}

function evidencePage(item: EvidenceItem): string | null {
  return item.page == null
    ? null
    : `Page ${item.page}`;
}

function evidenceText(item: EvidenceItem): string {
  return String(
    item.evidence_text ??
      item.text ??
      "Original passage text was not included."
  );
}

function evidenceLabel(
  item: EvidenceItem,
  index: number
): string {
  return `Evidence ${item.evidence_id ?? index + 1}`;
}

function evidenceMatches(
  left: EvidenceItem,
  right: EvidenceItem
): boolean {
  if (
    left.chunk_id != null &&
    right.chunk_id != null &&
    left.paper === right.paper
  ) {
    return left.chunk_id === right.chunk_id;
  }

  if (
    left.paper !== right.paper ||
    String(left.page ?? "") !==
      String(right.page ?? "")
  ) {
    return false;
  }

  if (
    left.claim &&
    right.claim &&
    left.claim === right.claim
  ) {
    return true;
  }

  return evidenceText(left) === evidenceText(right);
}

function normalizeFilename(value: string): string {
  return (
    value
      .replace(/\\/g, "/")
      .split("/")
      .pop()
      ?.trim()
      .toLowerCase() ?? ""
  );
}

function validPage(value: unknown): number | null {
  const page = Number(value);

  return Number.isInteger(page) && page >= 1
    ? page
    : null;
}

function citationKey(
  filename: string,
  page: number
): string {
  return `${normalizeFilename(filename)}|${page}`;
}

function escapeMarkdownLinkLabel(value: string): string {
  return value.replace(/([\\[\]])/g, "\\$1");
}

/*
 * Only link citations when:
 * 1. The paper belongs to the active project.
 * 2. The cited page exists in retrieved evidence.
 * 3. The backend has not flagged the citation as invalid.
 */
function buildLinkedSynthesis(
  answer: string,
  evidence: EvidenceItem[],
  papers: ProjectPaper[],
  projectId: string,
  invalidCitationTexts: string[]
): string {
  if (!answer || !projectId || papers.length === 0) {
    return answer;
  }

  const supportedSources = new Set<string>();

  for (const item of evidence) {
    const page = validPage(item.page);

    if (page === null) continue;

    for (const name of [item.paper, item.paper_title]) {
      if (typeof name === "string" && name.trim()) {
        supportedSources.add(citationKey(name, page));
      }
    }
  }

  const invalidCitations = new Set(
    invalidCitationTexts.map((value) =>
      value.trim().toLowerCase()
    )
  );

  const citationPattern =
    /\[([^\]\r\n]+?\.pdf),\s*Page\s+(\d+)\]/gi;

  const protectedPattern =
    /(`[^`\n]*`|\[[^\]\n]*\]\([^)]+\))/g;

  return answer
    .split(protectedPattern)
    .map((segment) => {
      if (
        segment.startsWith("`") ||
        /^\[[^\]\n]*\]\([^)]+\)$/.test(segment)
      ) {
        return segment;
      }

      return segment.replace(
        citationPattern,
        (
          citation,
          filename: string,
          pageText: string
        ) => {
          const page = validPage(pageText);

          if (
            page === null ||
            invalidCitations.has(
              citation.trim().toLowerCase()
            ) ||
            !supportedSources.has(
              citationKey(filename, page)
            )
          ) {
            return citation;
          }

          const paper = papers.find(
            (candidate) =>
              normalizeFilename(candidate.filename) ===
              normalizeFilename(filename)
          );

          if (!paper) return citation;

          const url =
            getProjectPaperUrl(projectId, paper.id) +
            `#page=${page}`;

          const label = escapeMarkdownLinkLabel(
            `${filename}, Page ${page}`
          );

          return `[${label}](${url})`;
        }
      );
    })
    .join("");
}

/* ============================================================
   COMPARISON HELPERS
   ============================================================ */

function comparisonSources(
  comparison: ThemeComparison
): string[] {
  return (comparison.source_summaries ?? []).flatMap(
    (source) =>
      source.findings.map((finding) => {
        const page =
          finding.page == null
            ? ""
            : `, Page ${finding.page}`;

        return `${source.paper}${page}`;
      })
  );
}

function readableComparisonText(
  value: string | null | undefined,
  comparison: ThemeComparison
): string {
  if (!value) return "";

  const sources = comparisonSources(comparison);

  return value.replace(
    /\bS(\d+)\b/g,
    (match, number: string) =>
      sources[Number(number) - 1] ?? match
  );
}

/* ============================================================
   APPLICATION
   ============================================================ */

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem(
      "researchlens-theme"
    );

    if (saved === "light" || saved === "dark") {
      return saved;
    }

    return window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches
      ? "dark"
      : "light";
  });

  const [view, setView] = useState<View>("research");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] =
    useState(true);

  const [resultTab, setResultTab] =
    useState<ResultTab>("synthesis");

  const [inspectorTab, setInspectorTab] =
    useState<InspectorTab>("overview");

  const [projects, setProjects] =
    useState<ResearchProject[]>([]);

  const [projectsLoading, setProjectsLoading] =
    useState(true);
  const [projectMenuId, setProjectMenuId] = useState<string | null>(null);
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [editingProjectName, setEditingProjectName] = useState("");

  const [projectsError, setProjectsError] =
    useState<string | null>(null);

  const [selectedProject, setSelectedProject] =
    useState<ResearchProject | null>(null);

  const [conversationMessages, setConversationMessages] =
    useState<ConversationMessage[]>([]);
  const [conversationLoading, setConversationLoading] =
    useState(false);

  const [projectPapers, setProjectPapers] =
    useState<PaperLookup>({
      projectId: "",
      papers: [],
    });

  const [papersLoading, setPapersLoading] =
    useState(false);

  const [papersError, setPapersError] =
    useState<string | null>(null);

  const [papersVersion, setPapersVersion] =
    useState(0);

  const [historyVersion, setHistoryVersion] =
    useState(0);

  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] =
    useState("");

  const [response, setResponse] =
    useState<QueryResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] =
    useState<string | null>(null);

  const [selectedEvidence, setSelectedEvidence] =
    useState<EvidenceItem | null>(null);

  const [sourceDialogOpen, setSourceDialogOpen] =
    useState(false);

  const requestController =
    useRef<AbortController | null>(null);

  const activeProjectId = selectedProject?.id ?? null;

  const result = response?.data;
  const analysis = result?.result;
  const evidence = analysis?.evidence ?? [];
  const themes = analysis?.themes ?? [];
  const comparisons =
    analysis?.theme_comparisons ?? [];

  const crossPaperThemes = themes.filter(
    (item) => item.cross_paper
  );

  const singlePaperThemes = themes.filter(
    (item) => !item.cross_paper
  );

  const citationValidation =
    result?.citation_validation;

  const sourceCount =
    analysis?.source_count ??
    new Set(
      evidence.map((item) => evidenceSource(item))
    ).size;

  const invalidCitationTexts =
    citationValidation?.invalid_citations.map(
      (item) => item.citation
    ) ?? [];

  const linkedAnswer =
    result?.answer &&
    selectedProject &&
    projectPapers.projectId === selectedProject.id
      ? buildLinkedSynthesis(
          result.answer,
          evidence,
          projectPapers.papers,
          selectedProject.id,
          invalidCitationTexts
        )
      : result?.answer ?? "";

  /* --------------------------------------------------------
     THEME
     -------------------------------------------------------- */

  useEffect(() => {
    document.documentElement.dataset.theme = theme;

    localStorage.setItem(
      "researchlens-theme",
      theme
    );
  }, [theme]);

  function toggleTheme() {
    setTheme((current) =>
      current === "dark" ? "light" : "dark"
    );
  }

  /* --------------------------------------------------------
     INITIAL PROJECT LOADING
     -------------------------------------------------------- */

  useEffect(() => {
    const controller = new AbortController();

    setProjectsLoading(true);

    getProjects(controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return;

        setProjects(items);
        setProjectsError(null);

        setSelectedProject((current) =>
          items.find(
            (project) => project.id === current?.id
          ) ??
          items[0] ??
          null
        );
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;

        setProjectsError(
          caught instanceof Error
            ? caught.message
            : "Could not load research projects."
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setProjectsLoading(false);
        }
      });

    return () => controller.abort();
  }, []);

  async function refreshProjects(
    preferredId?: string
  ) {
    try {
      const items = await getProjects();

      setProjects(items);
      setProjectsError(null);

      const next =
        items.find(
          (project) =>
            project.id ===
            (preferredId ?? selectedProject?.id)
        ) ??
        items[0] ??
        null;

      selectProject(next);
    } catch (caught) {
      setProjectsError(
        caught instanceof Error
          ? caught.message
          : "Could not refresh projects."
      );
    }
  }

  /* --------------------------------------------------------
     CLEAN UP ACTIVE RESEARCH REQUESTS
     -------------------------------------------------------- */

  useEffect(() => {
    return () => {
      requestController.current?.abort();
    };
  }, []);

  /* --------------------------------------------------------
     LOAD ACTIVE PROJECT PDF REFERENCES
     -------------------------------------------------------- */

  useEffect(() => {
    if (!activeProjectId) {
      setProjectPapers({
        projectId: "",
        papers: [],
      });

      setPapersLoading(false);
      return;
    }

    const controller = new AbortController();

    setPapersLoading(true);
    setPapersError(null);

    getProjectPapers(
      activeProjectId,
      controller.signal
    )
      .then((papers) => {
        if (controller.signal.aborted) return;

        setProjectPapers({
          projectId: activeProjectId,
          papers,
        });

        setPapersLoading(false);
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;

        setPapersError(
          caught instanceof Error
            ? caught.message
            : "Could not load citation sources."
        );

        setPapersLoading(false);
      });

    return () => controller.abort();
  }, [activeProjectId, papersVersion]);

  function refreshCitationPapers() {
    setPapersLoading(true);
    setPapersError(null);
    setPapersVersion((value) => value + 1);
  }

  /* --------------------------------------------------------
     ORIGINAL PDF LINKS
     -------------------------------------------------------- */

  function getEvidencePdfUrl(
    item: EvidenceItem
  ): string | null {
    if (
      !selectedProject ||
      projectPapers.projectId !== selectedProject.id
    ) {
      return null;
    }

    const names = [
      item.paper,
      item.paper_title,
    ].filter(
      (value): value is string =>
        typeof value === "string" &&
        value.trim().length > 0
    );

    const paper = projectPapers.papers.find(
      (candidate) =>
        names.some(
          (name) =>
            normalizeFilename(candidate.filename) ===
            normalizeFilename(name)
        )
    );

    if (!paper) return null;

    const url = getProjectPaperUrl(
      selectedProject.id,
      paper.id
    );

    const page = validPage(item.page);

    return page === null
      ? url
      : `${url}#page=${page}`;
  }

  function PdfSourceLink({
    item,
    showUnavailable = false,
  }: {
    item: EvidenceItem;
    showUnavailable?: boolean;
  }) {
    const url = getEvidencePdfUrl(item);

    if (url) {
      return (
        <a
          className="source-pdf-link"
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(event) =>
            event.stopPropagation()
          }
        >
          <ExternalLink size={15} />
          Open original PDF
          {evidencePage(item)
            ? ` — ${evidencePage(item)}`
            : ""}
        </a>
      );
    }

    if (!showUnavailable) return null;

    return (
      <div className="source-pdf-unavailable">
        <p>
          {papersLoading
            ? "Loading PDF links..."
            : papersError
              ? papersError
              : "Original PDF link unavailable."}
        </p>

        <button
          type="button"
          className="source-pdf-retry"
          onClick={refreshCitationPapers}
          disabled={papersLoading}
        >
          <RotateCcw size={14} />
          Refresh PDF links
        </button>
      </div>
    );
  }

  useEffect(() => {
    if (selectedProject?.mode === "document") {
      setResultTab((current) =>
        current === "themes" ? "synthesis" : current
      );
      setInspectorTab((current) =>
        current === "comparisons" ? "overview" : current
      );
    }
  }, [selectedProject?.mode]);

  useEffect(() => {
    const projectId = selectedProject?.id;

    if (!projectId) {
      setConversationMessages([]);
      setConversationLoading(false);
      return;
    }

    const controller = new AbortController();
    setConversationLoading(true);

    getProjectConversation(projectId, controller.signal)
      .then((conversation) => {
        if (!controller.signal.aborted) {
          setConversationMessages(conversation.messages);

          const lastUser = [...conversation.messages]
            .reverse()
            .find((message) => message.role === "user");
          const lastAssistant = [...conversation.messages]
            .reverse()
            .find(
              (message) =>
                message.role === "assistant" &&
                message.response
            );

          if (
            lastUser &&
            lastAssistant?.response
          ) {
            setSubmittedQuestion(lastUser.content);
            setQuestion("");
            setResponse(lastAssistant.response);
            setView("research");
          }
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setConversationMessages([]);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setConversationLoading(false);
        }
      });

    return () => controller.abort();
  }, [selectedProject?.id]);

  /* --------------------------------------------------------
     RESEARCH ACTIONS
     -------------------------------------------------------- */

  async function runResearch(value: string) {
    const trimmed = value.trim();

    if (!trimmed) return;

    if (!selectedProject) {
      setError(
        "Select or create a research project first."
      );
      return;
    }

    if (
      selectedProject.mode === "research" &&
      selectedProject.document_count < 2
    ) {
      setError(
        "Upload at least two indexed papers before using Research Lens."
      );
      return;
    }

    requestController.current?.abort();

    const controller = new AbortController();
    requestController.current = controller;

    setSubmittedQuestion(trimmed);
    setQuestion(trimmed);
    setResponse(null);
    setError(null);
    setSelectedEvidence(null);
    setSourceDialogOpen(false);
    setResultTab("synthesis");
    setInspectorTab("overview");
    setLoading(true);
    setView("research");
    setSidebarOpen(false);

    try {
      const nextResponse =
        selectedProject.mode === "document"
          ? await analyzeDocument(
              selectedProject.id,
              trimmed,
              controller.signal
            )
          : await analyzeResearch(
              selectedProject.id,
              trimmed,
              controller.signal
            );

      if (controller.signal.aborted) return;

      setResponse(nextResponse);
      setConversationMessages((messages) => [
        ...messages,
        {
          id: `${Date.now()}-user`,
          role: "user",
          content: trimmed,
          response: null,
          created_at: new Date().toISOString(),
        },
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: nextResponse.data.answer ?? "",
          response: nextResponse,
          created_at: new Date().toISOString(),
        },
      ]);
      refreshCitationPapers();

      setHistoryVersion(
        (version) => version + 1
      );
    } catch (caught) {
      if (controller.signal.aborted) return;

      setError(
        caught instanceof Error
          ? caught.message
          : "An unexpected error occurred."
      );
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
        requestController.current = null;
      }
    }
  }

  function submitQuestion(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    void runResearch(question);
  }

  function newResearch() {
    requestController.current?.abort();
    requestController.current = null;

    setQuestion("");
    setSubmittedQuestion("");
    setResponse(null);
    setError(null);
    setLoading(false);
    setSelectedEvidence(null);
    setSourceDialogOpen(false);
    setResultTab("synthesis");
    setInspectorTab("overview");
    setView("research");
    setSidebarOpen(false);
  }

  async function saveProjectName(project: ResearchProject) {
    const name = editingProjectName.trim();
    if (!name || name === project.name) {
      setEditingProjectId(null);
      return;
    }

    try {
      const updated = await updateProject(project.id, { name });
      setProjects((items) =>
        items.map((item) => item.id === updated.id ? updated : item)
      );
      if (selectedProject?.id === updated.id) {
        setSelectedProject(updated);
      }
      setEditingProjectId(null);
      setProjectMenuId(null);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not rename the project."
      );
    }
  }

  async function toggleProjectPin(project: ResearchProject) {
    try {
      const updated = await updateProject(project.id, {
        is_pinned: !project.is_pinned,
      });
      setProjects((items) =>
        items
          .map((item) => item.id === updated.id ? updated : item)
          .sort((a, b) => {
            if (a.is_pinned !== b.is_pinned) {
              return a.is_pinned ? -1 : 1;
            }
            return b.updated_at.localeCompare(a.updated_at);
          })
      );
      if (selectedProject?.id === updated.id) {
        setSelectedProject(updated);
      }
      setProjectMenuId(null);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not update the project pin."
      );
    }
  }

  async function removeProject(project: ResearchProject) {
    const confirmed = window.confirm(
      "Delete \"" + project.name + "\"? This removes the project, its conversation, and its uploaded PDFs."
    );
    if (!confirmed) return;

    try {
      await deleteProject(project.id);
      const remaining = projects.filter(
        (item) => item.id !== project.id
      );
      setProjects(remaining);
      setProjectMenuId(null);
      setEditingProjectId(null);

      if (selectedProject?.id === project.id) {
        setSelectedProject(remaining[0] ?? null);
        setSubmittedQuestion("");
        setResponse(null);
        setQuestion("");
      }
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not delete the project."
      );
    }
  }

  async function switchProjectMode(
    mode: "document" | "research"
  ) {
    if (!selectedProject || selectedProject.mode === mode) return;

    try {
      const updated = await updateProject(selectedProject.id, { mode });
      setProjects((items) =>
        items.map((item) =>
          item.id === updated.id ? updated : item
        )
      );
      selectProject(updated);
      if (mode === "document" && view === "history") {
        setView("research");
      }
      setError(null);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not switch the project mode."
      );
    }
  }

  function selectProject(
    project: ResearchProject | null
  ) {
    if (project?.id !== selectedProject?.id) {
      requestController.current?.abort();
      requestController.current = null;

      setQuestion("");
      setSubmittedQuestion("");
      setResponse(null);
      setError(null);
      setLoading(false);
      setSelectedEvidence(null);
      setSourceDialogOpen(false);
      setResultTab("synthesis");
      setInspectorTab("overview");

      setProjectPapers({
        projectId: "",
        papers: [],
      });

      setPapersError(null);
      setPapersLoading(Boolean(project));
    }

    setSelectedProject(project);
  }

  function reopenResearchSession(
    session: SavedResearchSession
  ) {
    requestController.current?.abort();
    requestController.current = null;

    setSubmittedQuestion(session.question);
    setQuestion(session.question);
    setResponse(session.response);
    setLoading(false);
    setError(null);
    setSelectedEvidence(null);
    setSourceDialogOpen(false);
    setResultTab("synthesis");
    setInspectorTab("overview");
    setView("research");
    setSidebarOpen(false);

    refreshCitationPapers();
  }

  function inspectEvidence(item: EvidenceItem) {
    setSelectedEvidence(item);
    setInspectorOpen(true);
    setInspectorTab("evidence");
    setSourceDialogOpen(true);
  }

  /* --------------------------------------------------------
     SHARED RESEARCH COMPONENTS
     -------------------------------------------------------- */

  function ResearchMarkdown({
    content,
  }: {
    content: string;
  }) {
    return (
      <div className="answer-text research-markdown">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className={
                  href?.includes("/papers/") &&
                  href?.includes("#page=")
                    ? "synthesis-citation-link"
                    : undefined
                }
                title={
                  href?.includes("/papers/")
                    ? "Open the cited PDF page"
                    : undefined
                }
              >
                {children}
              </a>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }

  function renderComparison(
    comparison: ThemeComparison
  ) {
    const differences =
      comparison.important_differences ?? [];

    const limitations =
      comparison.limitations ?? [];

    return (
      <div
        className="theme-comparison"
        key={comparison.theme_id}
      >
        <div className="comparison-heading">
          <GitCompareArrows size={17} />
          <span>Cross-paper comparison</span>
        </div>

        <div className="comparison-relationship">
          <span className="comparison-relationship-label">
            {comparison.relationship_label}
          </span>

          <span className="comparison-method">
            {comparison.analysis_method ===
            "Gemini-assisted thematic comparison"
              ? "AI-assisted interpretation"
              : "Conservative comparison"}
          </span>
        </div>

        <p className="comparison-explanation">
          {readableComparisonText(
            comparison.explanation,
            comparison
          )}
        </p>

        {comparison.shared_concern && (
          <div className="comparison-detail">
            <strong>Shared research concern</strong>

            <p>
              {readableComparisonText(
                comparison.shared_concern,
                comparison
              )}
            </p>
          </div>
        )}

        {differences.length > 0 && (
          <div className="comparison-detail">
            <strong>Important differences</strong>

            <ul>
              {differences.map(
                (difference, index) => (
                  <li key={index}>
                    {readableComparisonText(
                      difference,
                      comparison
                    )}
                  </li>
                )
              )}
            </ul>
          </div>
        )}

        {limitations.length > 0 && (
          <div className="comparison-detail comparison-limitations">
            <strong>Interpretation limits</strong>

            <ul>
              {limitations.map(
                (limitation, index) => (
                  <li key={index}>
                    {readableComparisonText(
                      limitation,
                      comparison
                    )}
                  </li>
                )
              )}
            </ul>
          </div>
        )}

        <div className="comparison-footer">
          <ShieldCheck size={15} />

          <span>
            {comparison.agreement_established
              ? "Agreement reported by the comparison."
              : "Method-specific agreement has not been established."}
            {" "}
            Inspect original passages before
            drawing a conclusion.
          </span>
        </div>
      </div>
    );
  }

  function renderTheme(themeItem: ResearchTheme) {
    const comparison = comparisons.find(
      (item) =>
        item.theme_id === themeItem.theme_id
    );

    return (
      <article
        className={`theme-card ${
          themeItem.cross_paper
            ? "theme-card-cross-paper"
            : ""
        }`}
        key={themeItem.theme_id}
      >
        <div className="theme-card-header">
          <div>
            <h3>{themeItem.theme}</h3>

            <span className="theme-source-count">
              {themeItem.source_count} source
              {themeItem.source_count === 1
                ? " paper"
                : " papers"}
            </span>
          </div>

          <span
            className={`theme-badge ${
              themeItem.cross_paper
                ? "theme-badge-cross-paper"
                : ""
            }`}
          >
            {themeItem.cross_paper
              ? "Shared theme"
              : "Single-paper theme"}
          </span>
        </div>

        {comparison &&
          renderComparison(comparison)}

        <div className="theme-findings-heading">
          <BookOpen size={15} />
          <span>Findings from the papers</span>
        </div>

        <div className="theme-findings">
          {themeItem.evidence.map(
            (item, index) => (
              <div
                className="theme-finding"
                key={`${themeItem.theme_id}-${index}`}
              >
                <div className="theme-finding-source">
                  <BookOpen size={15} />

                  <strong>
                    {evidenceSource(item)}
                  </strong>

                  {evidencePage(item) && (
                    <span>
                      {evidencePage(item)}
                    </span>
                  )}
                </div>

                <p>
                  {item.claim ||
                    "No extracted claim available."}
                </p>

                <div className="source-actions">
                  <button
                    className="theme-inspect-button"
                    type="button"
                    onClick={() =>
                      inspectEvidence(item)
                    }
                  >
                    Inspect source passage
                    <ArrowRight size={15} />
                  </button>

                  <PdfSourceLink item={item} />
                </div>
              </div>
            )
          )}
        </div>

        {themeItem.cross_paper &&
          !comparison && (
            <p className="theme-disclaimer">
              These findings address a shared
              research topic. A comparative
              explanation was not available.
            </p>
          )}
      </article>
    );
  }

  function renderEvidenceCard(
    item: EvidenceItem,
    index: number
  ) {
    const selected =
      selectedEvidence !== null &&
      evidenceMatches(selectedEvidence, item);

    return (
      <article
        className={`evidence-card ${
          selected ? "selected" : ""
        }`}
        key={`${item.chunk_id ?? index}-${index}`}
      >
        <button
          type="button"
          className="evidence-card-inspect"
          onClick={() =>
            inspectEvidence(item)
          }
        >
          <div className="evidence-card-top">
            <span className="evidence-id">
              {evidenceLabel(item, index)}
            </span>

            <ChevronRight size={17} />
          </div>

          <strong>
            {evidenceSource(item)}
          </strong>

          {evidencePage(item) && (
            <span className="evidence-page">
              {evidencePage(item)}
            </span>
          )}

          <p>{evidenceText(item)}</p>
        </button>

        <PdfSourceLink item={item} />
      </article>
    );
  }

  function renderCitationValidation() {
    if (!citationValidation) {
      return (
        <div className="wb-inspector-card">
          <strong>Citation validation</strong>
          <p>
            No citation validation report was
            returned for this research session.
          </p>
        </div>
      );
    }

    return (
      <section
        className={`citation-panel citation-${citationValidation.status}`}
        aria-label="Citation validation"
      >
        <div className="citation-panel-heading">
          <ShieldCheck size={20} />

          <div>
            <strong>Citation validation</strong>

            <span className="citation-status">
              {citationValidation.status ===
              "structurally_valid"
                ? "Structurally valid"
                : citationValidation.status ===
                    "review_required"
                  ? "Review required"
                  : "Invalid citations detected"}
            </span>
          </div>
        </div>

        <p className="citation-count">
          {citationValidation.valid_citation_count}
          {" "}of{" "}
          {citationValidation.checked_citation_count}
          {" "}citations valid
        </p>

        <div className="citation-metrics">
          <span>
            {
              citationValidation.invalid_citations
                .length
            }{" "}
            invalid
          </span>

          <span>
            {
              citationValidation.malformed_citations
                .length
            }{" "}
            malformed
          </span>

          <span>
            {
              citationValidation
                .uncited_passage_warnings.length
            }{" "}
            uncited-passage warnings
          </span>
        </div>

        {citationValidation.invalid_citations
          .length > 0 && (
          <details className="citation-issues">
            <summary>
              Invalid references
              <ChevronDown size={15} />
            </summary>

            <ul>
              {citationValidation.invalid_citations.map(
                (item, index) => (
                  <li key={index}>
                    {item.citation}
                  </li>
                )
              )}
            </ul>
          </details>
        )}

        {citationValidation.malformed_citations
          .length > 0 && (
          <details className="citation-issues">
            <summary>
              Malformed references
              <ChevronDown size={15} />
            </summary>

            <ul>
              {citationValidation.malformed_citations.map(
                (item, index) => (
                  <li key={index}>{item}</li>
                )
              )}
            </ul>
          </details>
        )}

        <p className="citation-note">
          Structural validation checks document
          and page references. It does not
          establish that a cited passage supports
          the associated claim.
        </p>
      </section>
    );
  }

  function renderResearchQuestionForm(
    compact = false
  ) {
    return (
      <form
        className={`question-form ${
          compact ? "wb-compact-question" : ""
        }`}
        onSubmit={submitQuestion}
      >
        <label
          className="sr-only"
          htmlFor={
            compact
              ? "compact-research-question"
              : "research-question"
          }
        >
          Workspace question
        </label>

        <textarea
          id={
            compact
              ? "compact-research-question"
              : "research-question"
          }
          placeholder={
            compact
              ? selectedProject?.mode === "document"
                ? "Ask another document question..."
                : "Ask another research question..."
              : selectedProject?.mode === "document"
                ? "What would you like to understand from these documents?"
                : "What would you like to investigate?"
          }
          value={question}
          onChange={(event) =>
            setQuestion(event.target.value)
          }
          rows={compact ? 1 : 3}
        />

        <div className="form-bottom">
          <span>
            <BookOpen size={15} />

            {selectedProject
              ? `${selectedProject.document_count} ${selectedProject.mode === "document" ? "documents" : "papers"} · ${selectedProject.name}`
              : "Select a project"}
          </span>

          <button
            className="submit-button"
            type="submit"
            disabled={
              loading ||
              !question.trim() ||
              !selectedProject ||
              (selectedProject.mode === "research" &&
                selectedProject.document_count < 2)
            }
            aria-label={
              selectedProject?.mode === "document"
                ? "Ask document question"
                : "Analyze research question"
            }
            title={
              selectedProject?.mode === "research" &&
              selectedProject.document_count < 2
                ? "Upload at least two indexed papers"
                : "Send question"
            }
          >
            {loading ? (
              <LoaderCircle
                size={18}
                className="spinning"
              />
            ) : (
              <ArrowRight size={19} />
            )}
          </button>
        </div>
      </form>
    );
  }

  /* ============================================================
     APPLICATION LAYOUT
     ============================================================ */

  return (
    <div className={"app-shell " + (selectedProject?.mode === "document" ? "mode-document" : "mode-research")}>
      {sidebarOpen && (
        <button
          type="button"
          className="mobile-overlay"
          aria-label="Close navigation"
          onClick={() =>
            setSidebarOpen(false)
          }
        />
      )}

      {/* SIDEBAR */}

      <aside
        className={`sidebar ${
          sidebarOpen ? "sidebar-open" : ""
        }`}
      >
        <div className="brand">
          <div className="brand-icon">
            <FileSearch
              size={22}
              strokeWidth={2.2}
            />
          </div>

          <div className="brand-copy">
            <strong>Lens</strong>
            <span>Documents &amp; research</span>
          </div>

          <button
            type="button"
            className="icon-button mobile-close"
            aria-label="Close navigation"
            onClick={() =>
              setSidebarOpen(false)
            }
          >
            <X size={19} />
          </button>
        </div>

        <button
          type="button"
          className="new-research"
          onClick={newResearch}
        >
          <Plus size={18} />
          New conversation
        </button>

        <div className="nav-heading">
          WORKSPACE
        </div>

        <nav
          className="nav-list"
          aria-label="Main navigation"
        >
          <button
            type="button"
            className={`nav-item ${view === "research" ? "active" : ""}`}
            onClick={() => {
              setView("research");
              setSidebarOpen(false);
            }}
          >
            <Search size={19} />
            Conversation
            {view === "research" && (
              <ChevronRight size={16} className="nav-end" />
            )}
          </button>

          <button
            type="button"
            className={`nav-item ${view === "library" ? "active" : ""}`}
            onClick={() => {
              setView("library");
              setSidebarOpen(false);
            }}
          >
            <Library size={19} />
            {selectedProject?.mode === "document"
              ? "Document library"
              : "Paper library"}
          </button>

          {selectedProject?.mode === "research" && (
            <button
              type="button"
              className={`nav-item ${view === "history" ? "active" : ""}`}
              onClick={() => {
                setView("history");
                setSidebarOpen(false);
              }}
            >
              <History size={19} />
              Research history
            </button>
          )}
        </nav>

        {/* PROJECTS */}

        <div className="wb-sidebar-projects">
          <div className="wb-sidebar-projects-heading">
            <span className="nav-heading">
              PROJECTS
            </span>

            <button
              type="button"
              className="icon-button"
              title="Create or manage projects"
              aria-label="Manage projects"
              onClick={() => {
                setView("library");
                setSidebarOpen(false);
              }}
            >
              <Plus size={17} />
            </button>
          </div>

          {projectsLoading ? (
            <p className="wb-sidebar-message">
              <LoaderCircle
                size={15}
                className="spinning"
              />
              Loading projects...
            </p>
          ) : projectsError ? (
            <div className="wb-sidebar-message">
              <AlertCircle size={15} />
              <span>{projectsError}</span>

              <button
                type="button"
                onClick={() =>
                  void refreshProjects()
                }
              >
                Retry
              </button>
            </div>
          ) : projects.length === 0 ? (
            <button
              type="button"
              className="wb-sidebar-empty"
              onClick={() =>
                setView("library")
              }
            >
              <Plus size={16} />
              Create your first project
            </button>
          ) : (
            <div className="wb-project-groups">
              {(["pinned", "recent"] as const).map((group) => {
                const items = projects.filter((project) =>
                  group === "pinned"
                    ? project.is_pinned
                    : !project.is_pinned
                );

                if (items.length === 0) return null;

                return (
                  <div className="wb-project-group" key={group}>
                    <span className="wb-project-group-label">
                      {group === "pinned" ? "PINNED" : "RECENT"}
                    </span>

                    <div className="wb-project-list">
                      {items.map((project) => (
                        <div
                          className={`wb-project-row ${selectedProject?.id === project.id ? "active" : ""}`}
                          key={project.id}
                        >
                          {editingProjectId === project.id ? (
                            <form
                              className="wb-project-rename"
                              onSubmit={(event) => {
                                event.preventDefault();
                                void saveProjectName(project);
                              }}
                            >
                              <input
                                value={editingProjectName}
                                onChange={(event) =>
                                  setEditingProjectName(event.target.value)
                                }
                                autoFocus
                                maxLength={100}
                                aria-label="Project name"
                              />
                            </form>
                          ) : (
                            <button
                              type="button"
                              className={`wb-project-item ${selectedProject?.id === project.id ? "active" : ""}`}
                              onClick={() => {
                                selectProject(project);
                                setView("research");
                                setSidebarOpen(false);
                              }}
                            >
                              {project.is_pinned ? (
                                <Pin size={14} />
                              ) : (
                                <BookOpen size={16} />
                              )}
                              <span title={project.name}>
                                {project.name}
                              </span>
                              {selectedProject?.id === project.id && (
                                <span className="wb-project-active-dot" />
                              )}
                            </button>
                          )}

                          <button
                            type="button"
                            className="wb-project-menu-button"
                            aria-label={"Actions for " + project.name}
                            aria-expanded={projectMenuId === project.id}
                            onClick={() =>
                              setProjectMenuId((current) =>
                                current === project.id ? null : project.id
                              )
                            }
                          >
                            <MoreHorizontal size={16} />
                          </button>

                          {projectMenuId === project.id && (
                            <div className="wb-project-menu" role="menu">
                              <button
                                type="button"
                                onClick={() => {
                                  setEditingProjectId(project.id);
                                  setEditingProjectName(project.name);
                                  setProjectMenuId(null);
                                }}
                              >
                                <Pencil size={14} />
                                Rename
                              </button>

                              <button
                                type="button"
                                onClick={() => void toggleProjectPin(project)}
                              >
                                {project.is_pinned ? (
                                  <PinOff size={14} />
                                ) : (
                                  <Pin size={14} />
                                )}
                                {project.is_pinned ? "Unpin" : "Pin"}
                              </button>

                              <button
                                type="button"
                                className="danger"
                                onClick={() => void removeProject(project)}
                              >
                                <Trash2 size={14} />
                                Delete
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>          )}
        </div>

        <div className="sidebar-spacer" />

        <div className="sidebar-note">
          {selectedProject?.mode === "document" ? (
            <FileSearch size={19} />
          ) : (
            <ShieldCheck size={19} />
          )}

          <div>
            <strong>
              {selectedProject?.mode === "document"
                ? "Grounded answers"
                : "Evidence first"}
            </strong>

            <p>
              {selectedProject?.mode === "document"
                ? "Answers stay anchored to your uploaded documents."
                : "Inspect the sources behind research conclusions."}
            </p>
          </div>
        </div>

        <div className="sidebar-footer">
          <button
            type="button"
            className="nav-item"
            onClick={toggleTheme}
          >
            {theme === "dark" ? (
              <Sun size={19} />
            ) : (
              <Moon size={19} />
            )}

            {theme === "dark"
              ? "Light appearance"
              : "Dark appearance"}
          </button>

          <div className="version">
            Lens · Documents & research
          </div>
        </div>
      </aside>

      {/* MAIN SHELL */}

      <div className="main-shell">
        <header className="topbar wb-topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="icon-button menu-button"
              aria-label="Open navigation"
              onClick={() =>
                setSidebarOpen(true)
              }
            >
              <Menu size={21} />
            </button>

            <div className="wb-topbar-project">
              <span className="section-caption">
                ACTIVE PROJECT
              </span>

              <select
                className="wb-topbar-select"
                aria-label="Active project"
                value={selectedProject?.id ?? ""}
                onChange={(event) => {
                  const next =
                    projects.find(
                      (project) =>
                        project.id ===
                        event.target.value
                    ) ?? null;

                  selectProject(next);
                }}
                disabled={
                  projectsLoading ||
                  projects.length === 0
                }
              >
                {projects.length === 0 && (
                  <option value="">
                    No project selected
                  </option>
                )}

                {projects.map((project) => (
                  <option
                    key={project.id}
                    value={project.id}
                  >
                    {project.name}
                  </option>
                ))}
              </select>

              <span className="wb-project-subtitle">
                {selectedProject
                  ? `${selectedProject.document_count} ${selectedProject.mode === "document" ? "documents" : "papers"} · Active`
                  : "Create a project to begin"}
              </span>

              {selectedProject && (
                <div
                  className="wb-mode-switcher"
                  role="group"
                  aria-label="Workspace mode"
                >
                  <button
                    type="button"
                    className={selectedProject.mode === "document" ? "active" : ""}
                    aria-pressed={selectedProject.mode === "document"}
                    onClick={() => void switchProjectMode("document")}
                    disabled={loading}
                  >
                    Document Lens
                  </button>
                  <button
                    type="button"
                    className={selectedProject.mode === "research" ? "active" : ""}
                    aria-pressed={selectedProject.mode === "research"}
                    onClick={() => void switchProjectMode("research")}
                    disabled={loading}
                  >
                    Research Lens
                  </button>
                </div>
              )}
            </div>
          </div>


          <div className="topbar-actions">
            <button
              type="button"
              className="icon-button"
              title="Refresh projects"
              aria-label="Refresh projects"
              onClick={() =>
                void refreshProjects()
              }
            >
              <RefreshCw size={18} />
            </button>

            <button
              type="button"
              className="icon-button"
              title="Toggle theme"
              aria-label="Toggle theme"
              onClick={toggleTheme}
            >
              {theme === "dark" ? (
                <Sun size={19} />
              ) : (
                <Moon size={19} />
              )}
            </button>

            {view === "research" && (
              <button
                type="button"
                className="icon-button inspector-toggle"
                title={
                  inspectorOpen
                    ? "Hide evidence inspector"
                    : "Show evidence inspector"
                }
                aria-label={
                  inspectorOpen
                    ? "Hide evidence inspector"
                    : "Show evidence inspector"
                }
                onClick={() =>
                  setInspectorOpen(
                    !inspectorOpen
                  )
                }
              >
                {inspectorOpen ? (
                  <PanelRightClose size={20} />
                ) : (
                  <PanelRightOpen size={20} />
                )}
              </button>
            )}
          </div>
        </header>

        {/* RESEARCH PAGE */}

        {view === "research" ? (
          <div className="research-layout">
            <main className="research-main wb-research-main">
              {!submittedQuestion ? (
                <div className="welcome">
                  {conversationLoading && selectedProject && (
                    <div className="wb-conversation-loading" role="status">
                      <LoaderCircle size={16} className="spinning" />
                      Restoring this project's conversation…
                    </div>
                  )}
                  <div className="eyebrow">
                    <Sparkles size={15} />
                    {selectedProject?.mode === "document"
                      ? "YOUR DOCUMENTS, UNDERSTOOD"
                      : "YOUR RESEARCH, UNDERSTOOD"}
                  </div>

                  <h1>
                    {selectedProject?.mode === "document" ? (
                      <>
                        Understand your{" "}
                        <span>documents.</span>
                      </>
                    ) : (
                      <>
                        Explore your{" "}
                        <span>research.</span>
                      </>
                    )}
                  </h1>

                  <p className="welcome-description">
                    {selectedProject?.mode === "document"
                      ? "Ask questions about the uploaded PDFs and keep the answers grounded in their content."
                      : "Ask questions across your research papers, compare findings and inspect the evidence behind each conclusion."}
                  </p>


                  {!selectedProject && (
                    <div className="wb-setup-notice">
                      <AlertCircle size={18} />

                      <div>
                        <strong>
                          Start with a research project
                        </strong>

                        <p>
                          Create a project and upload
                          PDFs to begin.
                        </p>

                        <button
                          type="button"
                          className="secondary-action"
                          onClick={() =>
                            setView("library")
                          }
                        >
                          Open Paper Library
                          <ArrowRight size={15} />
                        </button>
                      </div>
                    </div>
                  )}

                  {selectedProject?.mode === "research" &&
                    selectedProject.document_count <
                      2 && (
                    <div className="wb-setup-notice">
                      <BookOpen size={18} />

                      <div>
                        <strong>
                          Add more research papers
                        </strong>

                        <p>
                          Research Lens requires at
                          least two indexed PDFs.
                        </p>

                        <button
                          type="button"
                          className="secondary-action"
                          onClick={() =>
                            setView("library")
                          }
                        >
                          Add papers
                          <ArrowRight size={15} />
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="suggestions">
                    <span className="section-caption">
                      TRY A QUESTION
                    </span>

                    <div className="suggestion-list">
                      {(selectedProject?.mode === "document"
                        ? documentExamples
                        : researchExamples
                      ).map((example) => (
                        <button
                          type="button"
                          key={example}
                          className="suggestion"
                          onClick={() =>
                            setQuestion(example)
                          }
                        >
                          <span>{example}</span>
                          <ArrowRight size={16} />
                        </button>
                      ))}
                    </div>
                  </div>

                  {selectedProject?.mode === "research" && (
                    <div className="value-strip">
                    <div>
                      <FileSearch size={20} />

                      <strong>
                        Traceable evidence
                      </strong>

                      <span>
                        Paper, page and passage
                        references
                      </span>
                    </div>

                    <div>
                      <ShieldCheck size={20} />

                      <strong>
                        Transparent audits
                      </strong>

                      <span>
                        Clear analysis completion
                        status
                      </span>
                    </div>

                    <div>
                      <Sparkles size={20} />

                      <strong>
                        Cross-paper insight
                      </strong>

                      <span>
                        Shared themes and grounded
                        comparisons
                      </span>
                    </div>
                  </div>
                  )}
                </div>
              ) : (
                <div className="research-results wb-results">

<div className="wb-results-top">
  <button
    type="button"
    className="back-link"
    onClick={newResearch}
  >
    <ArrowLeft size={16} />
    New conversation
  </button>
</div>

                  {/* LOADING */}

                  {loading && (
                    <div
                      className="analysis-loading"
                      role="status"
                    >
                      <LoaderCircle
                        size={27}
                        className="spinning"
                      />

                      <span className="wb-loading-kicker">
                        {selectedProject?.mode === "document"
                          ? "DOCUMENT LENS"
                          : "RESEARCH LENS"}
                      </span>

                      <h2>
                        {selectedProject?.mode === "document"
                          ? "Reading your documents"
                          : "Analyzing your research"}
                      </h2>

                      <p>
                        {selectedProject?.mode === "document"
                          ? "Retrieving relevant passages and preparing a grounded answer."
                          : "Retrieving passages, comparing findings and preparing your evidence audit."}
                      </p>

                      <div className="wb-loading-steps" aria-hidden="true">
                        <span>Retrieve</span>
                        <span>Ground</span>
                        <span>{selectedProject?.mode === "document" ? "Answer" : "Audit"}</span>
                      </div>
                    </div>
                  )}

                  {/* ERROR */}

                  {error && !loading && (
                    <div
                      className="analysis-error"
                      role="alert"
                    >
                      <AlertCircle size={23} />

                      <div>
                        <span className="wb-error-kicker">
                          {selectedProject?.mode === "document"
                            ? "DOCUMENT LENS"
                            : "RESEARCH LENS"}
                        </span>

                        <h2>
                          We couldn't complete that request
                        </h2>

                        <p>{error}</p>

                        <button
                          type="button"
                          className="retry-button"
                          onClick={() =>
                            void runResearch(
                              submittedQuestion
                            )
                          }
                        >
                          <RotateCcw size={16} />
                          Try again
                        </button>
                      </div>
                    </div>
                  )}

                  {/* CONVERSATIONAL RESPONSE */}

                  {result && !loading && (
                    <>
                      {conversationMessages.length > 2 && (
                        <section
                          className="conversation-history"
                          aria-label="Previous conversation"
                        >
                          <div className="result-section-title">
                            <History size={18} />
                            <h2>Conversation</h2>
                          </div>

                          <div className="conversation-thread">
                            {conversationMessages
                              .slice(0, -2)
                              .map((message) => (
                                <article
                                  key={message.id}
                                  className={
                                    message.role === "user"
                                      ? "conversation-message user"
                                      : "conversation-message assistant"
                                  }
                                >
                                  <span className="conversation-role">
                                    {message.role === "user"
                                      ? "You"
                                      : selectedProject?.mode === "document"
                                        ? "Document Lens"
                                        : "Research Lens"}
                                  </span>
                                  <div className="conversation-content">
                                    {message.role === "assistant" ? (
                                      <ResearchMarkdown
                                        content={message.content}
                                      />
                                    ) : (
                                      <p>{message.content}</p>
                                    )}
                                  </div>
                                </article>
                              ))}
                          </div>
                        </section>
                      )}

                      <section className="wb-conversation-turn">
                        <div className="wb-chat-message wb-chat-user">
                          <span className="wb-chat-role">You</span>
                          <p>{submittedQuestion}</p>
                        </div>

                        <div className="wb-chat-message wb-chat-assistant">
                          <div className="wb-chat-assistant-heading">
                            <span className="wb-chat-role">
                              {selectedProject?.mode === "document"
                                ? "Document Lens"
                                : "Research Lens"}
                            </span>
                            <span className="wb-chat-status">
                              <CheckCircle2 size={14} />
                              Grounded response
                            </span>
                          </div>

                          <div className="wb-chat-answer">
                            {result.answer ? (
                              <ResearchMarkdown content={linkedAnswer} />
                            ) : (
                              <p>No answer was generated for this question.</p>
                            )}
                          </div>

                          {!result.audit_completed &&
                            selectedProject?.mode === "research" &&
                            result.answer && (
                            <div className="answer-caution">
                              <AlertCircle size={16} />
                              <span>
                                The claim-level audit was not fully completed.
                                Review the original evidence before relying on
                                the synthesis.
                              </span>
                            </div>
                          )}

                          <div className="wb-chat-meta">
                            {selectedProject?.mode === "document" ? (
                              <>
                                <span>
                                  <BookOpen size={15} />
                                  {evidence.length} evidence passages
                                </span>
                                <span>
                                  <FileSearch size={15} />
                                  {sourceCount} source {sourceCount === 1 ? "document" : "documents"}
                                </span>
                              </>
                            ) : (
                              <>
                                <span>
                                  <ShieldCheck size={15} />
                                  {statusLabels[result.analysis_status]}
                                </span>
                                <span>
                                  <CheckCircle2 size={15} />
                                  {citationValidation
                                    ? `${citationValidation.valid_citation_count} of ${citationValidation.checked_citation_count} valid citations`
                                    : "Citation validation not reported"}
                                </span>
                                <span>
                                  <BookOpen size={15} />
                                  {evidence.length} passages · {sourceCount} source {sourceCount === 1 ? "paper" : "papers"}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      </section>

                      {/* RESULT TABS */}

                      <div
                        className="wb-result-tabs"
                        role="tablist"
                        aria-label="Research results"
                      >
                        <button
                          type="button"
                          role="tab"
                          aria-selected={
                            resultTab ===
                            "synthesis"
                          }
                          className={
                            resultTab ===
                            "synthesis"
                              ? "active"
                              : ""
                          }
                          onClick={() =>
                            setResultTab(
                              "synthesis"
                            )
                          }
                        >
                          <Sparkles size={17} />
                          {selectedProject?.mode === "document" ? "Answer" : "Synthesis"}
                        </button>

                        {selectedProject?.mode === "research" && (
                        <button
                          type="button"
                          role="tab"
                          aria-selected={
                            resultTab ===
                            "themes"
                          }
                          className={
                            resultTab ===
                            "themes"
                              ? "active"
                              : ""
                          }
                          onClick={() =>
                            setResultTab(
                              "themes"
                            )
                          }
                        >
                          <GitCompareArrows
                            size={17}
                          />
                          Themes & Comparisons
                        </button>
                        )}

                        <button
                          type="button"
                          role="tab"
                          aria-selected={
                            resultTab ===
                            "evidence"
                          }
                          className={
                            resultTab ===
                            "evidence"
                              ? "active"
                              : ""
                          }
                          onClick={() =>
                            setResultTab(
                              "evidence"
                            )
                          }
                        >
                          <BookOpen size={17} />
                          {selectedProject?.mode === "document" ? "Sources" : "Evidence"}
                          <span className="wb-tab-count">
                            {evidence.length}
                          </span>
                        </button>
                      </div>

                      {/* SYNTHESIS TAB */}

                      {resultTab ===
                        "synthesis" && (
                        <section className="wb-tab-panel">
                          <div className="result-section-title">
                            <Sparkles size={19} />
                            <h2>
                              {selectedProject?.mode === "document" ? "Document answer" : "Research synthesis"}
                            </h2>
                          </div>

                          <div className="answer-card">
                            {result.answer ? (
                              <ResearchMarkdown
                                content={linkedAnswer}
                              />
                            ) : (
                              <p className="empty-answer">
                                No answer was
                                generated.
                              </p>
                            )}
                          </div>

                          <section className="result-summary">
                            <div>
                              <span className="summary-number">
                                {
                                  evidence.length
                                }
                              </span>

                              <span className="summary-label">
                                Evidence passages
                              </span>
                            </div>

                            <div>
                              <span className="summary-number">
                                {sourceCount}
                              </span>

                              <span className="summary-label">
                                {selectedProject?.mode === "document" ? "Source documents" : "Source papers"}
                              </span>
                            </div>

                            {selectedProject?.mode === "research" && (
                              <>
                            <div>
                              <span className="summary-number">
                                {analysis
                                  ?.claim_groups
                                  .length ?? 0}
                              </span>

                              <span className="summary-label">
                                Claim groups
                              </span>
                            </div>

                            <div>
                              <span className="summary-number">
                                {
                                  comparisons.length
                                }
                              </span>

                              <span className="summary-label">
                                Comparisons
                              </span>
                            </div>
                              </>
                            )}
                          </section>
                        </section>
                      )}

                      {/* THEMES TAB */}

                      {resultTab ===
                        "themes" && (
                        <section className="themes-section wb-tab-panel">
                          <div className="result-section-title">
                            <GitCompareArrows
                              size={19}
                            />

                            <h2>
                              Themes & Comparisons
                            </h2>
                          </div>

                          <p className="themes-intro">
                            Explore shared research
                            concerns, differences
                            between papers and the
                            limits of each comparison.
                          </p>

                          {crossPaperThemes.length >
                          0 ? (
                            <div className="themes-list">
                              {crossPaperThemes.map(
                                renderTheme
                              )}
                            </div>
                          ) : (
                            <div className="themes-empty">
                              <CircleHelp
                                size={19}
                              />

                              <p>
                                No shared themes
                                were identified
                                in the selected
                                evidence.
                              </p>
                            </div>
                          )}

                          {singlePaperThemes.length >
                            0 && (
                            <div className="single-paper-themes">
                              <h3>
                                Other identified
                                themes
                              </h3>

                              <div className="themes-list">
                                {singlePaperThemes.map(
                                  renderTheme
                                )}
                              </div>
                            </div>
                          )}
                        </section>
                      )}

                      {/* EVIDENCE TAB */}

                      {resultTab ===
                        "evidence" && (
                        <section className="evidence-section wb-tab-panel">
                          <div className="result-section-title">
                            <FileSearch size={19} />

                            <h2>
                              Retrieved evidence
                            </h2>
                          </div>

                          {evidence.length ===
                          0 ? (
                            <div className="empty-evidence">
                              No evidence passages
                              were returned.
                            </div>
                          ) : (
                            <div className="evidence-list">
                              {evidence.map(
                                renderEvidenceCard
                              )}
                            </div>
                          )}
                        </section>
                      )}
                    </>
                  )}
                </div>
              )}
            <div className="wb-bottom-composer">
              {renderResearchQuestionForm(Boolean(submittedQuestion))}
            </div>

            </main>

            {/* EVIDENCE INSPECTOR */}

            {selectedProject?.mode === "research" && inspectorOpen && (
              <aside className="inspector wb-inspector">
                <div className="inspector-header">
                  <div>
                    <span className="section-caption">
                      RESEARCH CONTEXT
                    </span>

                    <h2>
                      Evidence Inspector
                    </h2>
                  </div>

                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Close evidence inspector"
                    onClick={() =>
                      setInspectorOpen(false)
                    }
                  >
                    <PanelRightClose
                      size={19}
                    />
                  </button>
                </div>

                <div
                  className="wb-inspector-tabs"
                  role="tablist"
                  aria-label="Evidence inspector"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={
                      inspectorTab ===
                      "overview"
                    }
                    className={
                      inspectorTab ===
                      "overview"
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setInspectorTab(
                        "overview"
                      )
                    }
                  >
                    Overview
                  </button>

                  <button
                    type="button"
                    role="tab"
                    aria-selected={
                      inspectorTab ===
                      "evidence"
                    }
                    className={
                      inspectorTab ===
                      "evidence"
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setInspectorTab(
                        "evidence"
                      )
                    }
                  >
                    Evidence
                    {result
                      ? ` (${evidence.length})`
                      : ""}
                  </button>

                  {selectedProject?.mode === "research" && (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={
                      inspectorTab ===
                      "comparisons"
                    }
                    className={
                      inspectorTab ===
                      "comparisons"
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setInspectorTab(
                        "comparisons"
                      )
                    }
                  >
                    Comparisons
                  </button>
                  )}
                </div>

                <div className="inspector-body">
                  {!result && (
                    <div className="inspector-empty">
                      <div className="inspector-illustration">
                        <FileSearch
                          size={29}
                          strokeWidth={1.5}
                        />
                      </div>

                      <h3>
                        Your evidence,
                        in context
                      </h3>

                      <p>
                        Run a research query
                        to inspect its audit,
                        citations and original
                        source passages here.
                      </p>
                    </div>
                  )}

                  {result &&
                    inspectorTab ===
                      "overview" && (
                    <>
                      <div className="wb-inspector-card">
                        <span className="section-caption">
                          CLAIM AUDIT
                        </span>

                        <div
                          className={`audit-banner audit-${result.analysis_status}`}
                          role="status"
                        >
                          {result.analysis_status ===
                          "completed" ? (
                            <CheckCircle2
                              size={20}
                            />
                          ) : (
                            <AlertCircle
                              size={20}
                            />
                          )}

                          <div>
                            <strong>
                              {
                                statusLabels[
                                  result
                                    .analysis_status
                                ]
                              }
                            </strong>

                            <p>
                              {
                                result.analysis_message
                              }
                            </p>
                          </div>
                        </div>

                        {result.failed_group_count >
                          0 && (
                          <p className="audit-detail">
                            {
                              result.failed_group_count
                            }{" "}
                            claim group(s)
                            could not be
                            analyzed.
                          </p>
                        )}
                      </div>

                      {renderCitationValidation()}

                      <div className="wb-inspector-card">
                        <span className="section-caption">
                          EVIDENCE
                        </span>

                        <div className="guide-row">
                          <BookOpen size={18} />

                          <span>
                            {
                              evidence.length
                            }{" "}
                            retrieved passages
                          </span>
                        </div>

                        <div className="guide-row">
                          <FileSearch size={18} />

                          <span>
                            {sourceCount}
                            {" "}source papers
                          </span>
                        </div>

                        <button
                          type="button"
                          className="wb-inspector-action"
                          onClick={() =>
                            setInspectorTab(
                              "evidence"
                            )
                          }
                        >
                          Inspect evidence
                          <ArrowRight
                            size={15}
                          />
                        </button>
                      </div>
                    </>
                  )}

                  {result &&
                    inspectorTab ===
                      "evidence" && (
                    <div className="wb-inspector-evidence">
                      {selectedEvidence ? (
                        <div className="selected-evidence">
                          <span className="section-caption">
                            SELECTED PASSAGE
                          </span>

                          <h3>
                            {evidenceSource(
                              selectedEvidence
                            )}
                          </h3>

                          {evidencePage(
                            selectedEvidence
                          ) && (
                            <span className="evidence-page">
                              {evidencePage(
                                selectedEvidence
                              )}
                            </span>
                          )}

                          {selectedEvidence.claim && (
                            <div className="inspector-claim">
                              <strong>
                                Extracted finding
                              </strong>

                              <p>
                                {
                                  selectedEvidence.claim
                                }
                              </p>
                            </div>
                          )}

                          <div className="selected-passage">
                            {evidenceText(
                              selectedEvidence
                            )}
                          </div>

                          <PdfSourceLink
                            item={
                              selectedEvidence
                            }
                            showUnavailable
                          />

                          <p className="inspector-disclaimer">
                            Retrieved source
                            text does not by
                            itself establish
                            support for a
                            generated claim.
                          </p>
                        </div>
                      ) : (
                        <p className="wb-inspector-hint">
                          Select a passage
                          to inspect its
                          original source.
                        </p>
                      )}

                      <div className="wb-inspector-evidence-list">
                        {evidence.map(
                          (item, index) => (
                            <button
                              type="button"
                              key={index}
                              className={`wb-inspector-evidence-item ${
                                selectedEvidence &&
                                evidenceMatches(
                                  selectedEvidence,
                                  item
                                )
                                  ? "active"
                                  : ""
                              }`}
                              onClick={() =>
                                setSelectedEvidence(
                                  item
                                )
                              }
                            >
                              <span>
                                {evidenceLabel(
                                  item,
                                  index
                                )}
                              </span>

                              <strong>
                                {evidenceSource(
                                  item
                                )}
                              </strong>

                              <small>
                                {evidencePage(
                                  item
                                ) ??
                                  "Page not specified"}
                              </small>
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  )}

                  {result &&
                    inspectorTab ===
                      "comparisons" && (
                    <div className="wb-inspector-comparisons">
                      <div className="wb-inspector-card">
                        <span className="section-caption">
                          CROSS-PAPER ANALYSIS
                        </span>

                        <strong>
                          {
                            comparisons.length
                          }{" "}
                          comparisons
                        </strong>

                        <p>
                          {
                            crossPaperThemes.length
                          }{" "}
                          shared themes
                          identified.
                        </p>
                      </div>

                      {comparisons.length >
                      0 ? (
                        comparisons.map(
                          (comparison) => (
                            <button
                              type="button"
                              key={
                                comparison.theme_id
                              }
                              className="wb-inspector-comparison-item"
                              onClick={() => {
                                setResultTab(
                                  "themes"
                                );
                              }}
                            >
                              <strong>
                                {
                                  comparison.theme
                                }
                              </strong>

                              <span>
                                {
                                  comparison.relationship_label
                                }
                              </span>

                              <ArrowRight
                                size={15}
                              />
                            </button>
                          )
                        )
                      ) : (
                        <p className="wb-inspector-hint">
                          No cross-paper
                          comparisons were
                          returned.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </aside>
            )}
          </div>
        ) : view === "library" ? (
          /* PAPER LIBRARY */

          <main className="secondary-page wb-library-page">
            <ProjectWorkspace
              selectedProject={selectedProject}
              onSelectProject={(project) => {
                selectProject(project);

                // Refresh sidebar projects and
                // citation sources after library changes.
                void refreshProjects(
                  project?.id
                );

                setPapersVersion(
                  (version) => version + 1
                );
              }}
            />
          </main>
        ) : (
          /* HISTORY */

          <ResearchHistory
            project={selectedProject}
            version={historyVersion}
            onOpen={reopenResearchSession}
            onNew={newResearch}
          />
        )}
      </div>

      {/* ORIGINAL EVIDENCE PASSAGE DIALOG */}

      {sourceDialogOpen &&
        selectedEvidence && (
        <div
          className="wb-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (
              event.target ===
              event.currentTarget
            ) {
              setSourceDialogOpen(false);
            }
          }}
        >
          <section
            className="wb-source-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="source-dialog-heading"
          >
            <header className="wb-dialog-header">
              <div>
                <span className="section-caption">
                  ORIGINAL SOURCE PASSAGE
                </span>

                <h2 id="source-dialog-heading">
                  {evidenceSource(
                    selectedEvidence
                  )}
                </h2>

                {evidencePage(
                  selectedEvidence
                ) && (
                  <span className="evidence-page">
                    {evidencePage(
                      selectedEvidence
                    )}
                  </span>
                )}
              </div>

              <button
                type="button"
                className="icon-button"
                aria-label="Close source passage"
                onClick={() =>
                  setSourceDialogOpen(false)
                }
              >
                <X size={20} />
              </button>
            </header>

            <div className="wb-dialog-body">
              {selectedEvidence.claim && (
                <div className="inspector-claim">
                  <strong>
                    Extracted finding
                  </strong>

                  <p>
                    {
                      selectedEvidence.claim
                    }
                  </p>
                </div>
              )}

              <div className="selected-passage">
                {evidenceText(
                  selectedEvidence
                )}
              </div>

              <PdfSourceLink
                item={selectedEvidence}
                showUnavailable
              />

              <p className="inspector-disclaimer">
                This is retrieved source text.
                Verify that it supports the
                associated claim before using
                it in your research.
              </p>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}