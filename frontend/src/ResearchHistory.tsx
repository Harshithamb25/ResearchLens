
import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  AlertCircle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Clock3,
  History,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";

import type { ResearchProject } from "./api/projects";

import {
  deleteResearchSession,
  getResearchSession,
  getResearchSessions,
  type ResearchSessionSummary,
  type SavedResearchSession,
} from "./api/history";

import "./ResearchHistory.css";

interface ResearchHistoryProps {
  project: ResearchProject | null;
  version: number;
  onOpen: (session: SavedResearchSession) => void;
  onNew: () => void;
}

const statusLabels: Record<string, string> = {
  completed: "Completed",
  partial: "Partial",
  failed: "Failed",
  no_evidence: "No evidence",
};

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export default function ResearchHistory({
  project,
  version,
  onOpen,
  onNew,
}: ResearchHistoryProps) {
  const [sessions, setSessions] = useState<
    ResearchSessionSummary[]
  >([]);

  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(
    null
  );

  const [openingId, setOpeningId] = useState<
    string | null
  >(null);

  const [deletingId, setDeletingId] = useState<
    string | null
  >(null);

  const [reloadVersion, setReloadVersion] =
    useState(0);

  const [pendingDelete, setPendingDelete] =
    useState<ResearchSessionSummary | null>(null);

  const openController =
    useRef<AbortController | null>(null);

  const projectId = project?.id ?? null;

  useEffect(() => {
    if (!projectId) {
      setSessions([]);
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();

    setSessions([]);
    setError(null);
    setLoading(true);

    getResearchSessions(
      projectId,
      controller.signal
    )
      .then((items) => {
        if (controller.signal.aborted) return;
        setSessions(items);
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;

        setError(
          caught instanceof Error
            ? caught.message
            : "Could not load research history."
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [projectId, version, reloadVersion]);

  useEffect(() => {
    openController.current?.abort();
    setOpeningId(null);
    setPendingDelete(null);
    setSearch("");

    return () => {
      openController.current?.abort();
    };
  }, [projectId]);

  const filteredSessions = sessions.filter(
    (session) =>
      session.question
        .toLowerCase()
        .includes(search.trim().toLowerCase())
  );

  function refresh() {
    setReloadVersion((current) => current + 1);
  }

  async function openSession(
    session: ResearchSessionSummary
  ) {
    if (!projectId || openingId || deletingId) {
      return;
    }

    openController.current?.abort();

    const controller = new AbortController();
    openController.current = controller;

    setOpeningId(session.id);
    setError(null);

    try {
      const saved = await getResearchSession(
        projectId,
        session.id,
        controller.signal
      );

      if (controller.signal.aborted) return;

      onOpen(saved);
    } catch (caught) {
      if (controller.signal.aborted) return;

      setError(
        caught instanceof Error
          ? caught.message
          : "Could not open the saved analysis."
      );
    } finally {
      if (!controller.signal.aborted) {
        setOpeningId(null);
        openController.current = null;
      }
    }
  }

  async function confirmDelete() {
    if (
      !projectId ||
      !pendingDelete ||
      deletingId
    ) {
      return;
    }

    const sessionId = pendingDelete.id;

    setDeletingId(sessionId);
    setError(null);

    try {
      await deleteResearchSession(
        projectId,
        sessionId
      );

      setSessions((current) =>
        current.filter(
          (session) => session.id !== sessionId
        )
      );

      setPendingDelete(null);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not delete the session."
      );
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <main className="secondary-page rh-page">
      <div className="rh-container">
        <header className="rh-header">
          <div>
            <div className="rh-eyebrow">
              <History size={15} />
              RESEARCH WORKSPACE
            </div>

            <h1>Research History</h1>

            <p>
              Revisit your previous questions,
              findings, evidence and citation
              audits.
            </p>
          </div>

          <div className="rh-header-actions">
            <button
              type="button"
              className="rh-secondary-button"
              onClick={refresh}
              disabled={!projectId || loading}
            >
              <RefreshCw
                size={16}
                className={
                  loading ? "spinning" : ""
                }
              />
              Refresh
            </button>

            <button
              type="button"
              className="rh-primary-button"
              onClick={onNew}
            >
              <Plus size={17} />
              New research
            </button>
          </div>
        </header>

        {!project ? (
          <section className="rh-empty">
            <BookOpen size={34} />
            <h2>Select a research project</h2>
            <p>
              Choose a project from the sidebar
              to see its saved research sessions.
            </p>
          </section>
        ) : (
          <>
            <section className="rh-toolbar">
              <div className="rh-project-context">
                <BookOpen size={18} />

                <div>
                  <strong>{project.name}</strong>

                  <span>
                    {sessions.length} saved{" "}
                    {sessions.length === 1
                      ? "session"
                      : "sessions"}
                  </span>
                </div>
              </div>

              <label className="rh-search">
                <Search size={18} />

                <input
                  type="search"
                  value={search}
                  onChange={(event) =>
                    setSearch(event.target.value)
                  }
                  placeholder="Search your questions..."
                  aria-label="Search research history"
                />

                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch("")}
                    aria-label="Clear search"
                  >
                    <X size={16} />
                  </button>
                )}
              </label>
            </section>

            {error && (
              <div className="rh-error" role="alert">
                <AlertCircle size={19} />

                <span>{error}</span>

                <button
                  type="button"
                  onClick={() => setError(null)}
                  aria-label="Dismiss error"
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {loading ? (
              <div
                className="rh-empty"
                role="status"
              >
                <LoaderCircle
                  size={30}
                  className="spinning"
                />
                <h2>Loading saved research</h2>
              </div>
            ) : sessions.length === 0 &&
              !error ? (
              <section className="rh-empty">
                <History size={35} />

                <h2>No saved sessions yet</h2>

                <p>
                  Your completed research
                  requests will appear here
                  after they are saved by
                  the backend.
                </p>

                <button
                  type="button"
                  className="rh-primary-button"
                  onClick={onNew}
                >
                  Start researching
                  <ArrowRight size={16} />
                </button>
              </section>
            ) : filteredSessions.length === 0 ? (
              <section className="rh-empty">
                <Search size={32} />

                <h2>No matching questions</h2>

                <p>
                  Try a different search term.
                </p>

                <button
                  type="button"
                  className="rh-secondary-button"
                  onClick={() => setSearch("")}
                >
                  Clear search
                </button>
              </section>
            ) : (
              <section
                className="rh-session-list"
                aria-label="Saved research sessions"
              >
                {filteredSessions.map((session) => (
                  <article
                    className="rh-session"
                    key={session.id}
                  >
                    <div className="rh-session-main">
                      <div className="rh-session-meta">
                        <span>
                          <Clock3 size={14} />
                          {formatDate(
                            session.created_at
                          )}
                        </span>

                        <span
                          className={`rh-status rh-status-${session.analysis_status}`}
                        >
                          {session.audit_completed ? (
                            <CheckCircle2
                              size={14}
                            />
                          ) : (
                            <AlertCircle
                              size={14}
                            />
                          )}

                          {statusLabels[
                            session.analysis_status
                          ] ??
                            session.analysis_status}
                        </span>
                      </div>

                      <h2>{session.question}</h2>

                      <p>
                        {session.audit_completed
                          ? "Claim-level audit completed"
                          : "Review the analysis status and evidence"}
                      </p>
                    </div>

                    <div className="rh-session-actions">
                      <button
                        type="button"
                        className="rh-open-button"
                        disabled={
                          openingId !== null ||
                          deletingId !== null
                        }
                        onClick={() =>
                          void openSession(session)
                        }
                      >
                        {openingId === session.id ? (
                          <LoaderCircle
                            size={16}
                            className="spinning"
                          />
                        ) : (
                          <ArrowRight size={17} />
                        )}

                        Open analysis
                      </button>

                      <button
                        type="button"
                        className="rh-delete-button"
                        disabled={
                          openingId !== null ||
                          deletingId !== null
                        }
                        onClick={() =>
                          setPendingDelete(session)
                        }
                        title="Delete saved session"
                        aria-label={`Delete: ${session.question}`}
                      >
                        <Trash2 size={17} />
                      </button>
                    </div>
                  </article>
                ))}
              </section>
            )}
          </>
        )}
      </div>

      {pendingDelete && (
        <div
          className="rh-dialog-backdrop"
          role="presentation"
        >
          <section
            className="rh-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="rh-delete-heading"
            aria-describedby="rh-delete-description"
          >
            <div className="rh-dialog-icon">
              <Trash2 size={23} />
            </div>

            <h2 id="rh-delete-heading">
              Delete this research session?
            </h2>

            <p id="rh-delete-description">
              The saved analysis for
              {" "}
              <strong>
                {pendingDelete.question}
              </strong>
              {" "}
              will be permanently deleted.
              Your project and uploaded PDFs
              will not be affected.
            </p>

            <div className="rh-dialog-actions">
              <button
                type="button"
                className="rh-secondary-button"
                disabled={deletingId !== null}
                onClick={() =>
                  setPendingDelete(null)
                }
              >
                Cancel
              </button>

              <button
                type="button"
                className="rh-confirm-delete"
                disabled={deletingId !== null}
                onClick={() =>
                  void confirmDelete()
                }
              >
                {deletingId ? (
                  <LoaderCircle
                    size={16}
                    className="spinning"
                  />
                ) : (
                  <Trash2 size={16} />
                )}

                Delete session
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}