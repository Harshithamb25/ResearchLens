
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";

import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  LoaderCircle,
  Plus,
  RefreshCw,
  UploadCloud,
} from "lucide-react";

import {
  createProject,
  getProjectPaperUrl,
  getProjectPapers,
  getProjects,
  uploadProjectPaper,
  type ProjectPaper,
  type ResearchProject,
} from "./api/projects";

import EvidenceMatrix from "./EvidenceMatrix";
import "./ProjectWorkspace.css";

interface Props {
  selectedProject: ResearchProject | null;
  onSelectProject: (project: ResearchProject | null) => void;
}

interface UploadResult {
  filename: string;
  success: boolean;
  message: string;
}

type LibraryTab = "papers" | "matrix";

export default function ProjectWorkspace({
  selectedProject,
  onSelectProject,
}: Props) {
  const [projects, setProjects] = useState<ResearchProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const [projectName, setProjectName] = useState("");
  const [creatingMode, setCreatingMode] = useState<"document" | "research">("research");
  const [showCreate, setShowCreate] = useState(false);
  const [error, setError] = useState("");

  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);

  const [papers, setPapers] = useState<ProjectPaper[]>([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [papersError, setPapersError] = useState("");
  const [papersVersion, setPapersVersion] = useState(0);
  const [matrixVersion, setMatrixVersion] = useState(0);
  const [libraryTab, setLibraryTab] = useState<LibraryTab>("papers");

  const inputRef = useRef<HTMLInputElement>(null);
  const uploadingRef = useRef(false);
  const selectedProjectRef = useRef(selectedProject);
  selectedProjectRef.current = selectedProject;

  useEffect(() => {
    let active = true;

    getProjects()
      .then((items) => {
        if (!active) return;

        setProjects(items);

        const currentId = selectedProjectRef.current?.id;

        onSelectProject(
          items.find((item) => item.id === currentId) ??
            items[0] ??
            null
        );
      })
      .catch((caught: unknown) => {
        if (!active) return;

        setError(
          caught instanceof Error
            ? caught.message
            : "Could not load projects."
        );
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };

    // Load the initial project list once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const projectId = selectedProject?.id;

    if (!projectId) {
      setPapers([]);
      setPapersError("");
      setPapersLoading(false);
      return;
    }

    const controller = new AbortController();

    setPapers([]);
    setPapersError("");
    setPapersLoading(true);

    getProjectPapers(projectId, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) {
          setPapers(items);
        }
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setPapersError(
            caught instanceof Error
              ? caught.message
              : "Could not load the paper library."
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPapersLoading(false);
        }
      });

    return () => controller.abort();
  }, [selectedProject?.id, papersVersion]);

  async function refreshProjects(preferredId?: string) {
    const items = await getProjects();
    setProjects(items);

    const targetId =
      preferredId ?? selectedProjectRef.current?.id;

    const active =
      items.find((item) => item.id === targetId) ??
      items[0] ??
      null;

    onSelectProject(active);
  }

  async function handleCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const name = projectName.trim();
    if (!name || creating) return;

    setCreating(true);
    setError("");

    try {
      const created = await createProject(name, "", creatingMode);

      setProjectName("");
      setCreatingMode("research");
      setShowCreate(false);
      setUploadResults([]);
      setLibraryTab("papers");

      await refreshProjects(created.id);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not create project."
      );
    } finally {
      setCreating(false);
    }
  }

  async function handleFiles(files: FileList | File[]) {
    const projectId = selectedProject?.id;

    if (!projectId || uploadingRef.current) return;

    const selectedFiles = Array.from(files);

    const pdfs = selectedFiles.filter(
      (file) =>
        file.type === "application/pdf" ||
        file.name.toLowerCase().endsWith(".pdf")
    );

    if (pdfs.length === 0) {
      setError("Select at least one PDF file.");
      return;
    }

    setError(
      pdfs.length < selectedFiles.length
        ? "Non-PDF files were skipped."
        : ""
    );

    uploadingRef.current = true;
    setUploading(true);
    setUploadResults([]);

    const results: UploadResult[] = [];
    let successfulUploads = 0;

    for (const file of pdfs) {
      try {
        await uploadProjectPaper(projectId, file);

        successfulUploads += 1;

        results.push({
          filename: file.name,
          success: true,
          message: "Uploaded and indexed",
        });
      } catch (caught) {
        results.push({
          filename: file.name,
          success: false,
          message:
            caught instanceof Error
              ? caught.message
              : "Upload failed",
        });
      }

      if (selectedProjectRef.current?.id === projectId) {
        setUploadResults([...results]);
      }
    }

    try {
      await refreshProjects();

      if (selectedProjectRef.current?.id === projectId) {
        setPapersVersion((value) => value + 1);

        if (successfulUploads > 0) {
          setMatrixVersion((value) => value + 1);
        }
      }
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not refresh the project."
      );
    } finally {
      uploadingRef.current = false;
      setUploading(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      void handleFiles(event.target.files);
    }

    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);

    if (event.dataTransfer.files.length > 0) {
      void handleFiles(event.dataTransfer.files);
    }
  }

  function handleProjectChange(projectId: string) {
    const project =
      projects.find((item) => item.id === projectId) ??
      null;

    setPapers([]);
    setPapersError("");
    setUploadResults([]);
    setError("");
    setLibraryTab("papers");

    onSelectProject(project);
  }

  return (
    <section className="project-workspace">
      <div className="project-workspace-heading">
        <div>
          <span className="section-caption">
            RESEARCH WORKSPACE
          </span>

          <h2>My research projects</h2>

          <p>
            Each project has its own papers,
            research context and evidence matrix.
          </p>
        </div>

        <button
          type="button"
          className="project-create-button"
          onClick={() => setShowCreate((value) => !value)}
        >
          <Plus size={17} />
          New project
        </button>
      </div>

      {error && (
        <p className="project-error" role="alert">
          <AlertCircle size={16} />
          {error}
        </p>
      )}

      {showCreate && (
        <form
          className="project-create-form"
          onSubmit={handleCreate}
        >
          <input
            value={projectName}
            onChange={(event) =>
              setProjectName(event.target.value)
            }
            placeholder="Project name"
            aria-label="Project name"
            maxLength={100}
            required
          />

          <select
            value={creatingMode}
            onChange={(event) =>
              setCreatingMode(
                event.target.value as "document" | "research"
              )
            }
            aria-label="Project mode"
          >
            <option value="research">Research Lens</option>
            <option value="document">Document Lens</option>
          </select>

          <button type="submit" disabled={creating}>
            {creating ? "Creating..." : "Create project"}
          </button>
        </form>
      )}

      {loading ? (
        <p className="project-loading">
          <LoaderCircle size={17} className="spinning" />
          Loading projects...
        </p>
      ) : projects.length === 0 ? (
        <div className="project-empty">
          <FolderOpen size={28} />
          <p>Create your first project to begin.</p>
        </div>
      ) : (
        <>
          <label
            className="project-select-label"
            htmlFor="project-select"
          >
            Active project
          </label>

          <select
            id="project-select"
            className="project-select"
            value={selectedProject?.id ?? ""}
            onChange={(event) =>
              handleProjectChange(event.target.value)
            }
          >
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name} ({project.document_count} papers)
              </option>
            ))}
          </select>

          {selectedProject && (
            <div className="project-details">
              <div className="project-stat">
                <FileText size={19} />

                <span>
                  <strong>
                    {selectedProject.document_count}
                  </strong>{" "}
                  indexed document(s)
                </span>
              </div>

/div>

              {selectedProject.mode === "research" && selectedProject.document_count < 2 && (
                <p className="project-hint">
                  Research Lens requires at least
                  two indexed papers. Document Lens can work with one or more.
                </p>
              )}

              <div
                className={`project-dropzone ${
                  dragging
                    ? "project-dropzone-active"
                    : ""
                }`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setDragging(false);
                }}
                onDrop={handleDrop}
              >
                <UploadCloud size={31} />

                <strong>
                  Drag and drop {selectedProject.mode === "document" ? "documents" : "research papers"} here
                </strong>

                <span>
                  Or choose multiple files from your computer
                </span>

                <input
                  ref={inputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  multiple
                  hidden
                  onChange={handleFileChange}
                />

                <button
                  type="button"
                  disabled={uploading}
                  onClick={() => inputRef.current?.click()}
                >
                  {uploading
                    ? "Indexing papers..."
                    : "Browse PDFs"}
                </button>
              </div>

              {uploading && (
                <p className="project-loading" role="status">
                  <LoaderCircle
                    size={17}
                    className="spinning"
                  />
                  Processing your PDFs. Please wait...
                </p>
              )}

              {uploadResults.length > 0 && (
                <div className="project-upload-results">
                  {uploadResults.map((item, index) => (
                    <div key={`${item.filename}-${index}`}>
                      {item.success ? (
                        <CheckCircle2 size={17} />
                      ) : (
                        <AlertCircle size={17} />
                      )}

                      <span>
                        <strong>{item.filename}</strong>
                        {" — "}
                        {item.message}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <div className="project-paper-library">
                <div className="project-paper-heading">
                  <div>
                    <span className="section-caption">
                      PROJECT LIBRARY
                    </span>

                    <h3>{selectedProject.mode === "research" ? "Research sources" : "Documents"}</h3>

                    <p>
                      {selectedProject.mode === "research"
                        ? "View indexed papers or inspect the evidence matrix."
                        : "View the documents available for grounded conversation."}
                    </p>
                  </div>
                </div>

                <div
                  role="tablist"
                  aria-label="Project library views"
                  className="project-library-tabs"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={libraryTab === "papers"}
                    onClick={() => setLibraryTab("papers")}
                    className={libraryTab === "papers" ? "active" : ""}
                  >
                    <FileText size={17} />
                    Indexed Papers
                  </button>

                  {selectedProject.mode === "research" && (
                    <button
                    type="button"
                    role="tab"
                    aria-selected={libraryTab === "matrix"}
                    onClick={() => setLibraryTab("matrix")}
                    className={libraryTab === "matrix" ? "active" : ""}
                  >
                    <FileSpreadsheet size={17} />
                    Evidence Matrix
                  </button>
                  )}
                </div>

                {libraryTab === "papers" ? (
                  <>
                    <button
                      type="button"
                      className="project-refresh-button"
                      onClick={() =>
                        setPapersVersion((value) => value + 1)
                      }
                      disabled={papersLoading}
                      aria-label="Refresh indexed papers"
                    >
                      <RefreshCw size={17} />
                      Refresh papers
                    </button>

                    {papersLoading ? (
                      <p className="project-loading" role="status">
                        <LoaderCircle
                          size={17}
                          className="spinning"
                        />
                        Loading indexed papers...
                      </p>
                    ) : papersError ? (
                      <p className="project-error" role="alert">
                        <AlertCircle size={16} />
                        {papersError}
                      </p>
                    ) : papers.length === 0 ? (
                      <div className="project-paper-empty">
                        <FolderOpen size={24} />
                        <p>
                          No papers uploaded to this project yet.
                        </p>
                      </div>
                    ) : (
                      <div className="project-paper-list">
                        {papers.map((paper) => (
                          <div
                            className="project-paper-item"
                            key={paper.id}
                          >
                            <div className="project-paper-icon">
                              <FileText size={21} />
                            </div>

                            <div className="project-paper-info">
                              <strong title={paper.filename}>
                                {paper.filename}
                              </strong>

                              <span>
                                {paper.indexed_chunks} indexed chunks
                              </span>
                            </div>

                            <span
                              className={
                                paper.indexed
                                  ? "project-paper-status indexed"
                                  : "project-paper-status pending"
                              }
                            >
                              {paper.indexed
                                ? "Indexed"
                                : "Not indexed"}
                            </span>

                            <a
                              className="project-paper-open"
                              href={getProjectPaperUrl(
                                selectedProject.id,
                                paper.id
                              )}
                              target="_blank"
                              rel="noopener noreferrer"
                              aria-label={`Open ${paper.filename}`}
                            >
                              <ExternalLink size={16} />
                              Open PDF
                            </a>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <EvidenceMatrix
                    key={selectedProject.id}
                    projectId={selectedProject.id}
                    refreshVersion={matrixVersion}
                  />
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}