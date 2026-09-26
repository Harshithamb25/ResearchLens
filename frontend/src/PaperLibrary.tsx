
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  FileText,
  LoaderCircle,
  RefreshCw,
  UploadCloud,
} from "lucide-react";

import {
  getPapers,
  uploadPaper,
  type LibraryPaper,
} from "./api/research";

import "./PaperLibrary.css";

interface PaperLibraryProps {
  onStartResearch: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function PaperLibrary({
  onStartResearch,
}: PaperLibraryProps) {
  const fileInput = useRef<HTMLInputElement>(null);

  const [papers, setPapers] = useState<LibraryPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function refreshPapers() {
    setLoading(true);

    try {
      const result = await getPapers();
      setPapers(result.papers);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not load the paper library."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refreshPapers();
  }, []);

  async function handleUpload() {
    if (!selectedFile || uploading) return;

    setError(null);
    setSuccess(null);
    setUploading(true);

    try {
      const result = await uploadPaper(selectedFile);

      setSuccess(
        `${result.paper.document} uploaded successfully. ` +
          `${result.paper.pages} pages and ` +
          `${result.paper.chunks} chunks indexed.`
      );

      setSelectedFile(null);

      if (fileInput.current) {
        fileInput.current.value = "";
      }

      await refreshPapers();
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The PDF could not be uploaded."
      );
    } finally {
      setUploading(false);
    }
  }

  function selectFile(file: File | undefined) {
    setError(null);
    setSuccess(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      setSelectedFile(null);
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError("Maximum PDF size is 20 MB.");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  }

  return (
    <main className="paper-library-page">
      <div className="paper-library-heading">
        <div>
          <div className="paper-library-eyebrow">
            RESEARCH WORKSPACE
          </div>
          <h1>Paper library</h1>
          <p>
            Upload research PDFs, index their contents, and
            analyze evidence across your collection.
          </p>
        </div>

        <button
          type="button"
          className="paper-library-refresh"
          onClick={() => {
            setError(null);
            void refreshPapers();
          }}
          disabled={loading || uploading}
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>

      <section className="paper-upload-card">
        <div className="paper-upload-icon">
          <UploadCloud size={26} />
        </div>

        <h2>Upload a research paper</h2>

        <p>
          Select a text-based PDF. ResearchLens will extract,
          chunk, embed, and index it automatically.
        </p>

        <input
          ref={fileInput}
          id="paper-upload-input"
          type="file"
          accept=".pdf,application/pdf"
          disabled={uploading}
          onChange={(event) => {
            selectFile(event.target.files?.[0]);
          }}
        />

        {selectedFile && (
          <div className="paper-selected-file">
            <FileText size={19} />
            <span>
              {selectedFile.name}
              <small>{formatSize(selectedFile.size)}</small>
            </span>
          </div>
        )}

        <button
          type="button"
          className="paper-upload-button"
          disabled={!selectedFile || uploading}
          onClick={() => void handleUpload()}
        >
          {uploading ? (
            <>
              <LoaderCircle
                size={17}
                className="paper-spinner"
              />
              Uploading and indexing...
            </>
          ) : (
            <>
              <UploadCloud size={17} />
              Upload and index
            </>
          )}
        </button>

        <span className="paper-upload-hint">
          PDF only · Maximum 20 MB · Scanned PDFs not yet supported
        </span>
      </section>

      {error && (
        <div className="paper-message paper-message-error" role="alert">
          {error}
        </div>
      )}

      {success && (
        <div
          className="paper-message paper-message-success"
          role="status"
        >
          <CheckCircle2 size={18} />
          {success}
        </div>
      )}

      <section className="paper-collection">
        <div className="paper-collection-heading">
          <div>
            <h2>Indexed papers</h2>
            <p>
              {loading
                ? "Loading your collection..."
                : `${papers.length} document${
                    papers.length === 1 ? "" : "s"
                  } in your library`}
            </p>
          </div>
        </div>

        {loading ? (
          <div className="paper-empty">
            <LoaderCircle
              className="paper-spinner"
              size={24}
            />
            Loading papers...
          </div>
        ) : papers.length === 0 ? (
          <div className="paper-empty">
            No PDFs uploaded yet.
          </div>
        ) : (
          <div className="paper-list">
            {papers.map((paper) => (
              <article
                className="paper-list-item"
                key={paper.filename}
              >
                <div className="paper-file-icon">
                  <FileText size={21} />
                </div>

                <div className="paper-file-details">
                  <strong>{paper.filename}</strong>
                  <span>
                    {formatSize(paper.size_bytes)} ·{" "}
                    {paper.indexed_chunks} indexed chunks
                  </span>
                </div>

                <span
                  className={
                    paper.indexed
                      ? "paper-index-status is-indexed"
                      : "paper-index-status"
                  }
                >
                  {paper.indexed
                    ? "Indexed"
                    : "Not indexed"}
                </span>
              </article>
            ))}
          </div>
        )}
      </section>

      <button
        type="button"
        className="paper-start-research"
        onClick={onStartResearch}
      >
        Start research
        <ArrowRight size={17} />
      </button>
    </main>
  );
}