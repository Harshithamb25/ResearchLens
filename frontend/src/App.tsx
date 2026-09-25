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
  ChevronRight,
  CircleHelp,
  FileSearch,
  GitCompareArrows,
  History,
  Library,
  LoaderCircle,
  Menu,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import {
  analyzeResearch,
  type AnalysisStatus,
  type EvidenceItem,
  type QueryResponse,
  type ResearchTheme,
  type ThemeComparison,
} from "./api/research";
import "./App.css";
import "./ResultView.css";

type Theme = "light" | "dark";
type View = "research" | "library" | "history";

const examples = [
  "What limitations and challenges affect AI-based intrusion detection systems?",
  "Compare the methods and datasets used.",
  "What do the papers agree on regarding intrusion detection limitations?",
];

const statusLabels: Record<AnalysisStatus, string> = {
  completed: "Claim audit completed",
  partial: "Partially completed",
  failed: "Audit failed",
  no_evidence: "No analyzable evidence",
};

function evidenceLabel(
  item: EvidenceItem,
  index: number
): string {
  return `Evidence ${item.evidence_id ?? index + 1}`;
}

function evidenceSource(item: EvidenceItem): string {
  return String(
    item.paper_title ?? item.paper ?? "Source not specified"
  );
}

function evidencePage(
  item: EvidenceItem
): string | null {
  return item.page == null
    ? null
    : `Page ${item.page}`;
}

function evidenceText(item: EvidenceItem): string {
  return String(
    item.evidence_text ??
      item.text ??
      "Original passage text was not included in this finding."
  );
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
    String(left.page ?? "") !== String(right.page ?? "")
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

  return (
    evidenceText(left) === evidenceText(right)
  );
}

/**
 * The backend numbers source IDs in the same order
 * as source_summaries: sorted paper names, followed
 * by each paper's findings.
 *
 * Convert S1/S2 references into readable citations.
 */
function comparisonSources(
  comparison: ThemeComparison
): string[] {
  const summaries = comparison.source_summaries ?? [];

  return summaries.flatMap((source) =>
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
    (match, number: string) => {
      const index = Number(number) - 1;
      return sources[index] ?? match;
    }
  );
}

function App() {
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
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] =
    useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] =
    useState(true);

  const [response, setResponse] =
    useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] =
    useState<string | null>(null);
  const [selectedEvidence, setSelectedEvidence] =
    useState<EvidenceItem | null>(null);
  const [sourceDialogOpen, setSourceDialogOpen] = useState(false);

  const requestController =
    useRef<AbortController | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(
      "researchlens-theme",
      theme
    );
  }, [theme]);

  useEffect(() => {
    return () => {
      requestController.current?.abort();
    };
  }, []);

  async function runResearch(value: string) {
    const trimmed = value.trim();

    if (!trimmed) return;

    requestController.current?.abort();

    const controller = new AbortController();
    requestController.current = controller;

    setSubmittedQuestion(trimmed);
    setQuestion(trimmed);
    setResponse(null);
    setError(null);
    setSelectedEvidence(null);
    setLoading(true);
    setView("research");
    setSidebarOpen(false);

    try {
      const nextResponse = await analyzeResearch(
        trimmed,
        controller.signal
      );

      if (!controller.signal.aborted) {
        setResponse(nextResponse);
      }
    } catch (caught) {
      if (!controller.signal.aborted) {
        setError(
          caught instanceof Error
            ? caught.message
            : "An unexpected error occurred."
        );
      }
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
    setView("research");
    setSidebarOpen(false);
  }

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

  const selectedItem = selectedEvidence;

  function inspectEvidence(item: EvidenceItem) {
    setSelectedEvidence(item);
    setInspectorOpen(true);
    // The source dialog is always visible, even when the side inspector
    // is outside the viewport or hidden by responsive layout.
    setSourceDialogOpen(true);
  }

  function renderComparison(
    comparison: ThemeComparison
  ) {
    const differences =
      comparison.important_differences ?? [];
    const limitations =
      comparison.limitations ?? [];

    return (
      <div className="theme-comparison">
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
              {differences.map((difference, index) => (
                <li key={index}>
                  {readableComparisonText(
                    difference,
                    comparison
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {limitations.length > 0 && (
          <div className="comparison-detail comparison-limitations">
            <strong>Interpretation limits</strong>
            <ul>
              {limitations.map((limitation, index) => (
                <li key={index}>
                  {readableComparisonText(
                    limitation,
                    comparison
                  )}
                </li>
              ))}
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
            Inspect the original passages before
            drawing a research conclusion.
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
          {themeItem.evidence.map((item, index) => {
            return (
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

                <button
                  className="theme-inspect-button"
                  type="button"
                  onClick={() => inspectEvidence(item)}
                >
                  Inspect source passage
                  <ArrowRight size={15} />
                </button>
              </div>
            );
          })}
        </div>

        {themeItem.cross_paper &&
          !comparison && (
            <p className="theme-disclaimer">
              These findings address a shared
              research topic. A comparative
              explanation was not available
              for this theme.
            </p>
          )}
      </article>
    );
  }

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-overlay"
          aria-label="Close navigation"
          onClick={() =>
            setSidebarOpen(false)
          }
        />
      )}

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
            <strong>ResearchLens</strong>
            <span>Evidence-aware research</span>
          </div>

          <button
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
          className="new-research"
          onClick={newResearch}
        >
          <Plus size={18} />
          New research
        </button>

        <div className="nav-heading">
          WORKSPACE
        </div>

        <nav
          className="nav-list"
          aria-label="Main navigation"
        >
          <button
            className={`nav-item ${
              view === "research"
                ? "active"
                : ""
            }`}
            onClick={() => {
              setView("research");
              setSidebarOpen(false);
            }}
          >
            <Search size={19} />
            Research

            {view === "research" && (
              <ChevronRight
                size={16}
                className="nav-end"
              />
            )}
          </button>

          <button
            className={`nav-item ${
              view === "library"
                ? "active"
                : ""
            }`}
            onClick={() => {
              setView("library");
              setSidebarOpen(false);
            }}
          >
            <Library size={19} />
            Paper library
          </button>

          <button
            className={`nav-item ${
              view === "history"
                ? "active"
                : ""
            }`}
            onClick={() => {
              setView("history");
              setSidebarOpen(false);
            }}
          >
            <History size={19} />
            Research history
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-note">
          <ShieldCheck size={19} />
          <div>
            <strong>Evidence first</strong>
            <p>
              Inspect the sources behind
              every research conclusion.
            </p>
          </div>
        </div>

        <div className="sidebar-footer">
          <button
            className="nav-item"
            onClick={() =>
              setTheme(
                theme === "dark"
                  ? "light"
                  : "dark"
              )
            }
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
            ResearchLens · Development preview
          </div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button menu-button"
              aria-label="Open navigation"
              onClick={() =>
                setSidebarOpen(true)
              }
            >
              <Menu size={21} />
            </button>

            <div className="breadcrumb">
              <span>Workspace</span>
              <ChevronRight size={15} />
              <strong>
                {view === "research"
                  ? "Research"
                  : view === "library"
                    ? "Paper library"
                    : "Research history"}
              </strong>
            </div>
          </div>

          <div className="topbar-actions">
            <span className="preview-badge">
              <span className="preview-dot" />
              Preview
            </span>

            <button
              className="icon-button"
              aria-label={`Switch to ${
                theme === "dark"
                  ? "light"
                  : "dark"
              } theme`}
              title="Toggle theme"
              onClick={() =>
                setTheme(
                  theme === "dark"
                    ? "light"
                    : "dark"
                )
              }
            >
              {theme === "dark" ? (
                <Sun size={19} />
              ) : (
                <Moon size={19} />
              )}
            </button>

            {view === "research" && (
              <button
                className="icon-button inspector-toggle"
                aria-label={
                  inspectorOpen
                    ? "Hide evidence panel"
                    : "Show evidence panel"
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

        {view === "research" ? (
          <div className="research-layout">
            <main className="research-main">
              {!submittedQuestion ? (
                <div className="welcome">
                  <div className="eyebrow">
                    <Sparkles size={15} />
                    YOUR RESEARCH, UNDERSTOOD
                  </div>

                  <h1>
                    Research with{" "}
                    <span>clarity.</span>
                  </h1>

                  <p className="welcome-description">
                    Ask questions across your
                    research papers. Explore
                    the evidence, compare
                    findings and see where
                    uncertainty remains.
                  </p>

                  <form
                    className="question-form"
                    onSubmit={
                      submitQuestion
                    }
                  >
                    <label
                      className="sr-only"
                      htmlFor="research-question"
                    >
                      Research question
                    </label>

                    <textarea
                      id="research-question"
                      placeholder="What would you like to investigate?"
                      value={question}
                      onChange={(event) =>
                        setQuestion(
                          event.target.value
                        )
                      }
                      rows={3}
                    />

                    <div className="form-bottom">
                      <span>
                        <BookOpen size={15} />
                        Grounded in your
                        paper collection
                      </span>

                      <button
                        className="submit-button"
                        type="submit"
                        disabled={
                          !question.trim()
                        }
                        aria-label="Analyze research question"
                      >
                        <ArrowRight size={20} />
                      </button>
                    </div>
                  </form>

                  <div className="suggestions">
                    <span className="section-caption">
                      TRY A QUESTION
                    </span>

                    <div className="suggestion-list">
                      {examples.map(
                        (example) => (
                          <button
                            key={example}
                            className="suggestion"
                            onClick={() =>
                              setQuestion(
                                example
                              )
                            }
                          >
                            <span>
                              {example}
                            </span>
                            <ArrowRight
                              size={16}
                            />
                          </button>
                        )
                      )}
                    </div>
                  </div>

                  <div className="value-strip">
                    <div>
                      <FileSearch size={20} />
                      <strong>
                        Traceable evidence
                      </strong>
                      <span>
                        Paper, page and
                        passage references
                      </span>
                    </div>

                    <div>
                      <ShieldCheck size={20} />
                      <strong>
                        Transparent audits
                      </strong>
                      <span>
                        Clear analysis
                        completion status
                      </span>
                    </div>

                    <div>
                      <Sparkles size={20} />
                      <strong>
                        Cross-paper insight
                      </strong>
                      <span>
                        Shared themes and
                        grounded comparisons
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="research-results">
                  <button
                    className="back-link"
                    onClick={newResearch}
                  >
                    <ArrowLeft size={17} />
                    New research
                  </button>

                  <div className="result-heading">
                    <span className="section-caption">
                      RESEARCH QUESTION
                    </span>
                    <h1>
                      {submittedQuestion}
                    </h1>
                  </div>

                  {loading && (
                    <div
                      className="analysis-loading"
                      role="status"
                    >
                      <div className="loading-symbol">
                        <LoaderCircle
                          size={26}
                          className="spinning"
                        />
                      </div>

                      <h2>
                        Analyzing your
                        research
                      </h2>

                      <p>
                        Retrieving relevant
                        passages, comparing
                        findings and preparing
                        your evidence audit.
                        This may take a
                        little time.
                      </p>
                    </div>
                  )}

                  {error && !loading && (
                    <div
                      className="analysis-error"
                      role="alert"
                    >
                      <AlertCircle
                        size={23}
                      />

                      <div>
                        <h2>
                          Research request
                          unsuccessful
                        </h2>

                        <p>{error}</p>

                        <button
                          className="retry-button"
                          onClick={() =>
                            void runResearch(
                              submittedQuestion
                            )
                          }
                        >
                          <RotateCcw
                            size={16}
                          />
                          Try again
                        </button>
                      </div>
                    </div>
                  )}

                  {result && !loading && (
                    <>
                      <div
                        className={`audit-banner audit-${result.analysis_status}`}
                        role="status"
                      >
                        <div className="audit-banner-icon">
                          {result.analysis_status ===
                          "completed" ? (
                            <CheckCircle2
                              size={21}
                            />
                          ) : result.analysis_status ===
                            "no_evidence" ? (
                            <CircleHelp
                              size={21}
                            />
                          ) : (
                            <AlertCircle
                              size={21}
                            />
                          )}
                        </div>

                        <div>
                          <strong>
                            {
                              statusLabels[
                                result.analysis_status
                              ]
                            }
                          </strong>

                          <p>
                            {
                              result.analysis_message
                            }
                          </p>

                          {result.failed_group_count >
                            0 && (
                            <span className="audit-detail">
                              {
                                result.failed_group_count
                              }{" "}
                              claim group(s)
                              could not be
                              analyzed.
                            </span>
                          )}
                        </div>
                      </div>

                      <section className="answer-section">
                        <div className="result-section-title">
                          <Sparkles
                            size={19}
                          />
                          <h2>
                            Research synthesis
                          </h2>
                        </div>

                        <div className="answer-card">
                          {result.answer ? (
                            <div className="answer-text research-markdown">
                              <ReactMarkdown>
                                {
                                  result.answer
                                }
                              </ReactMarkdown>
                            </div>
                          ) : (
                            <p className="empty-answer">
                              No answer was
                              generated for
                              this question.
                            </p>
                          )}

                          {!result.audit_completed &&
                            result.answer && (
                              <div className="answer-caution">
                                <AlertCircle
                                  size={16}
                                />
                                <span>
                                  This answer
                                  was generated
                                  from retrieved
                                  passages, but
                                  the claim-level
                                  audit was not
                                  fully completed.
                                </span>
                              </div>
                            )}
                        </div>
                      </section>

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
                            {
                              analysis?.source_count ??
                              new Set(
                                evidence.map(
                                  (item) =>
                                    item.paper
                                )
                              ).size
                            }
                          </span>
                          <span className="summary-label">
                            Source papers
                          </span>
                        </div>

                        <div>
                          <span className="summary-number">
                            {
                              analysis?.claim_groups
                                .length ?? 0
                            }
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
                            Cross-paper comparisons
                          </span>
                        </div>
                      </section>

                      <section className="themes-section">
                        <div className="result-section-title">
                          <GitCompareArrows
                            size={19}
                          />
                          <h2>
                            Cross-paper analysis
                          </h2>
                        </div>

                        <p className="themes-intro">
                          Compare related
                          findings across your
                          papers. Each comparison
                          distinguishes common
                          research concerns from
                          method-specific
                          agreement or
                          contradiction.
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
                              No shared research
                              themes were
                              identified in
                              the selected
                              evidence. This
                              does not mean
                              the papers have
                              no topics in
                              common.
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

                      <section className="evidence-section">
                        <div className="result-section-title">
                          <FileSearch
                            size={19}
                          />
                          <h2>
                            Retrieved evidence
                          </h2>
                        </div>

                        {evidence.length ===
                        0 ? (
                          <div className="empty-evidence">
                            No evidence
                            passages were
                            returned.
                          </div>
                        ) : (
                          <div className="evidence-list">
                            {evidence.map(
                              (
                                item,
                                index
                              ) => (
                                <button
                                  className={`evidence-card ${
                                    selectedEvidence !== null &&
                                    evidenceMatches(selectedEvidence, item)
                                      ? "selected"
                                      : ""
                                  }`}
                                  key={index}
                                  onClick={() => inspectEvidence(item)}
                                >
                                  <div className="evidence-card-top">
                                    <span className="evidence-id">
                                      {evidenceLabel(
                                        item,
                                        index
                                      )}
                                    </span>

                                    <ChevronRight
                                      size={17}
                                    />
                                  </div>

                                  <strong>
                                    {evidenceSource(
                                      item
                                    )}
                                  </strong>

                                  {evidencePage(
                                    item
                                  ) && (
                                    <span className="evidence-page">
                                      {evidencePage(
                                        item
                                      )}
                                    </span>
                                  )}

                                  <p>
                                    {evidenceText(
                                      item
                                    )}
                                  </p>
                                </button>
                              )
                            )}
                          </div>
                        )}
                      </section>
                    </>
                  )}
                </div>
              )}
            </main>

            {inspectorOpen && (
              <aside className="inspector">
                <div className="inspector-header">
                  <div>
                    <span className="section-caption">
                      RESEARCH CONTEXT
                    </span>

                    <h2>
                      Evidence inspector
                    </h2>
                  </div>

                  <button
                    className="icon-button"
                    aria-label="Close evidence inspector"
                    onClick={() =>
                      setInspectorOpen(
                        false
                      )
                    }
                  >
                    <PanelRightClose
                      size={19}
                    />
                  </button>
                </div>

                <div className="inspector-body">
                  {selectedItem ? (
                    <div className="selected-evidence">
                      <span className="section-caption">
                        {(() => {
                          const index = evidence.findIndex(
                            (item) => evidenceMatches(item, selectedItem)
                          );
                          return index >= 0
                            ? evidenceLabel(selectedItem, index)
                            : "Additional thematic evidence";
                        })()}
                      </span>

                      <h3>
                        {evidenceSource(
                          selectedItem
                        )}
                      </h3>

                      {evidencePage(
                        selectedItem
                      ) && (
                        <span className="evidence-page">
                          {evidencePage(
                            selectedItem
                          )}
                        </span>
                      )}

                      {selectedItem.claim && (
                        <div className="inspector-claim">
                          <strong>
                            Extracted finding
                          </strong>

                          <p>
                            {
                              selectedItem.claim
                            }
                          </p>
                        </div>
                      )}

                      <div className="selected-passage">
                        {evidenceText(
                          selectedItem
                        )}
                      </div>

                      <p className="inspector-disclaimer">
                        Retrieved source
                        passage. Its presence
                        does not by itself
                        establish support
                        for the generated
                        answer or thematic
                        interpretation.
                      </p>
                    </div>
                  ) : (
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
                        Select a retrieved
                        passage or thematic
                        finding to inspect
                        its source details
                        here.
                      </p>
                    </div>
                  )}

                  {result && (
                    <div className="inspector-guide">
                      <span className="section-caption">
                        AUDIT OVERVIEW
                      </span>

                      <div className="guide-row">
                        <ShieldCheck
                          size={17}
                        />
                        <span>
                          {
                            statusLabels[
                              result.analysis_status
                            ]
                          }
                        </span>
                      </div>

                      <div className="guide-row">
                        <BookOpen
                          size={17}
                        />
                        <span>
                          {evidence.length}{" "}
                          evidence passage(s)
                        </span>
                      </div>

                      <div className="guide-row">
                        <GitCompareArrows
                          size={17}
                        />
                        <span>
                          {
                            comparisons.length
                          }{" "}
                          cross-paper
                          comparison(s)
                        </span>
                      </div>

                      <div className="guide-row">
                        <CircleHelp
                          size={17}
                        />
                        <span>
                          {
                            result.failed_group_count
                          }{" "}
                          failed claim
                          group(s)
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </aside>
            )}
          </div>
        ) : (
          <main className="secondary-page">
            <div className="secondary-icon">
              {view === "library" ? (
                <Library size={27} />
              ) : (
                <History size={27} />
              )}
            </div>

            <h1>
              {view === "library"
                ? "Paper library"
                : "Research history"}
            </h1>

            <p>
              {view === "library"
                ? "Your indexed research papers and their source details will appear here when we implement library management."
                : "Previous research sessions will appear here when we implement session storage."}
            </p>

            <button
              className="secondary-action"
              onClick={newResearch}
            >
              Start new research
              <ArrowRight size={17} />
            </button>
          </main>
        )}
      </div>

      {sourceDialogOpen && selectedItem && (
        <div
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSourceDialogOpen(false);
            }
          }}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 10000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 20,
            background: "rgba(15, 23, 42, 0.65)",
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="source-dialog-heading"
            style={{
              width: "min(100%, 680px)",
              maxHeight: "min(85vh, 820px)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              border: "1px solid var(--border)",
              borderRadius: 16,
              background: "var(--surface)",
              color: "var(--text)",
              boxShadow: "0 24px 80px rgba(0, 0, 0, 0.28)",
            }}
          >
            <header
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                gap: 16,
                padding: 20,
                borderBottom: "1px solid var(--border)",
              }}
            >
              <div>
                <span className="section-caption">SOURCE PASSAGE</span>
                <h2
                  id="source-dialog-heading"
                  style={{ margin: "8px 0", fontSize: 19 }}
                >
                  {evidenceSource(selectedItem)}
                </h2>
                {evidencePage(selectedItem) && (
                  <span className="evidence-page">
                    {evidencePage(selectedItem)}
                  </span>
                )}
              </div>
              <button
                type="button"
                className="icon-button"
                aria-label="Close source passage"
                onClick={() => setSourceDialogOpen(false)}
              >
                <X size={20} />
              </button>
            </header>
            <div style={{ overflowY: "auto", padding: 20 }}>
              {selectedItem.claim && (
                <div className="inspector-claim">
                  <strong>Extracted finding</strong>
                  <p>{selectedItem.claim}</p>
                </div>
              )}
              <div className="selected-passage">
                {evidenceText(selectedItem)}
              </div>
              <p className="inspector-disclaimer">
                Retrieved source passage. Verify the original paper
                before drawing a conclusion from this finding.
              </p>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;