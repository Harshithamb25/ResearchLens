
import { useEffect, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Download,
  ExternalLink,
  FileSpreadsheet,
  LoaderCircle,
  RefreshCw,
  X,
} from "lucide-react";

import {
  getPaperMatrix,
  getPaperMatrixCsvUrl,
  type MatrixEntry,
  type MatrixFieldKey,
  type MatrixRow,
  type PaperMatrix,
} from "./api/matrix";

import { getProjectPaperUrl } from "./api/projects";
import "./EvidenceMatrix.css";

interface Props {
  projectId: string;
  refreshVersion?: number;
}

const FIELD_COLUMNS: {
  key: MatrixFieldKey;
  label: string;
}[] = [
  {
    key: "methodology",
    label: "Methodology / Algorithm used",
  },
  {
    key: "datasets",
    label: "Dataset(s)",
  },
  {
    key: "evaluation_metrics",
    label: "Eval Metrics",
  },
  {
    key: "accuracy_results",
    label: "Accuracy / Results",
  },
  {
    key: "advantages",
    label: "Advantages",
  },
  {
    key: "limitations",
    label: "Disadvantages / Limitations",
  },
  {
    key: "applications",
    label: "Applications",
  },
];

interface Selection {
  row: MatrixRow;
  field: MatrixFieldKey;
  label: string;
}

export default function EvidenceMatrix({
  projectId,
  refreshVersion = 0,
}: Props) {
  const [matrix, setMatrix] = useState<PaperMatrix | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");
    setMatrix(null);
    setSelection(null);

    getPaperMatrix(projectId, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setMatrix(data);
        }
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Could not load the evidence matrix."
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [projectId, refreshVersion, version]);

  function openPdf(
    documentId: string,
    page?: number
  ): string {
    const base = getProjectPaperUrl(projectId, documentId);

    return page && page >= 1
      ? `${base}#page=${page}`
      : base;
  }

  function exportCsv() {
    window.open(
      getPaperMatrixCsvUrl(projectId),
      "_blank",
      "noopener,noreferrer"
    );
  }

  function renderCell(
    row: MatrixRow,
    field: MatrixFieldKey,
    label: string
  ) {
    const entries = row.fields?.[field]?.entries ?? [];

    if (!entries.length) {
      return (
        <span className="matrix-unavailable">
          Not identified
        </span>
      );
    }

    return (
      <button
        type="button"
        className="matrix-cell-button"
        onClick={() => setSelection({ row, field, label })}
        title={`Inspect evidence for ${label}`}
      >
        <span className="matrix-cell-value">
          {row[field]}
        </span>
        <span className="matrix-cell-source">
          <BookOpen size={13} />
          {entries.length} source finding
          {entries.length === 1 ? "" : "s"}
        </span>
      </button>
    );
  }

  function renderSource(entry: MatrixEntry, index: number) {
    const page = Number(entry.source.page);
    const hasPage = Number.isInteger(page) && page >= 1;

    return (
      <article className="matrix-source-card" key={index}>
        <div className="matrix-source-heading">
          <strong>{entry.value}</strong>
          {hasPage && <span>Page {page}</span>}
        </div>

        <p>{entry.evidence_text}</p>

        <a
          href={openPdf(
            entry.source.document_id,
            hasPage ? page : undefined
          )}
          target="_blank"
          rel="noopener noreferrer"
          className="matrix-pdf-link"
        >
          <ExternalLink size={15} />
          Open original PDF
          {hasPage ? ` — Page ${page}` : ""}
        </a>
      </article>
    );
  }

  return (
    <section className="matrix-workspace">
      <header className="matrix-heading">
        <div>
          <span className="section-caption">
            PAPER-WISE COMPARISON
          </span>
          <h3>Cross-Paper Evidence Matrix</h3>
          <p>
            Compare extracted findings across the papers
            in your selected project.
          </p>
        </div>

        <div className="matrix-actions">
          <button
            type="button"
            onClick={() => setVersion((value) => value + 1)}
            disabled={loading}
            className="matrix-secondary-button"
          >
            <RefreshCw size={16} />
            Refresh
          </button>

          <button
            type="button"
            onClick={exportCsv}
            disabled={loading || !matrix?.rows.length}
            className="matrix-export-button"
          >
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </header>

      {loading ? (
        <div className="matrix-state" role="status">
          <LoaderCircle size={23} className="spinning" />
          <strong>Extracting evidence from your papers</strong>
          <p>
            Reading the original PDFs and organizing
            page-linked findings.
          </p>
        </div>
      ) : error ? (
        <div className="matrix-state matrix-error" role="alert">
          <AlertCircle size={22} />
          <strong>Matrix generation unsuccessful</strong>
          <p>{error}</p>
          <button
            type="button"
            onClick={() => setVersion((value) => value + 1)}
          >
            Try again
          </button>
        </div>
      ) : matrix ? (
        <>
          <div className="matrix-summary">
            <FileSpreadsheet size={19} />
            <strong>{matrix.paper_count} papers</strong>
            <span>·</span>
            <span>9 comparison columns</span>
            <span>·</span>
            <span>Original PDF references</span>
          </div>

          <p className="matrix-scroll-hint">
            Scroll horizontally to explore all nine columns.
            Select a populated cell to inspect its source.
          </p>

          <div
            className="matrix-table-scroll"
            role="region"
            aria-label="Cross-paper evidence matrix"
            tabIndex={0}
          >
            <table className="matrix-table">
              <thead>
                <tr>
                  <th scope="col">S. No.</th>
                  <th scope="col">Paper Title</th>
                  {FIELD_COLUMNS.map((column) => (
                    <th scope="col" key={column.key}>
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {matrix.rows.map((row) => (
                  <tr key={row.document_id}>
                    <td className="matrix-number">
                      {row.serial_number}
                    </td>

                    <td className="matrix-paper">
                      <strong>{row.paper_title}</strong>
                      <a
                        href={openPdf(row.document_id)}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <ExternalLink size={13} />
                        View PDF
                      </a>
                    </td>

                    {FIELD_COLUMNS.map((column) => (
                      <td key={column.key}>
                        {renderCell(
                          row,
                          column.key,
                          column.label
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="matrix-disclaimer">
            {matrix.note}
          </p>
        </>
      ) : null}

      {selection && (
        <div
          className="matrix-dialog-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSelection(null);
            }
          }}
        >
          <section
            className="matrix-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="matrix-dialog-title"
          >
            <header className="matrix-dialog-header">
              <div>
                <span className="section-caption">
                  SOURCE EVIDENCE
                </span>
                <h3 id="matrix-dialog-title">
                  {selection.label}
                </h3>
                <p>{selection.row.paper_title}</p>
              </div>

              <button
                type="button"
                aria-label="Close source evidence"
                onClick={() => setSelection(null)}
              >
                <X size={20} />
              </button>
            </header>

            <div className="matrix-dialog-body">
              {selection.row.fields[
                selection.field
              ].entries.map(renderSource)}

              <p className="matrix-disclaimer">
                These are automatically extracted candidate
                findings. Check the original passages before
                citing them.
              </p>
            </div>
          </section>
        </div>
      )}
    </section>
  );
}