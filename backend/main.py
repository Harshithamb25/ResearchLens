
"""
ResearchLens FastAPI application.

Provides research analysis, health checks, and PDF upload
with automatic indexing for the research paper library.
"""

import logging
from backend.workspace.history import router as history_router
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any
from backend.workspace.database import initialize_database
from backend.workspace.projects import router as projects_router

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ingestion.ingest import ingest_paper
from backend.pipeline import process_query
from backend.retrieval.vector_store import collection


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Prevent two uploads from checking and writing the same
# filename simultaneously within this server process.
upload_lock = Lock()


app = FastAPI(
    title="ResearchLens API",
    description=(
        "Evidence-grounded research intelligence API "
        "for multi-document research analysis."
    ),
    version="1.0.0",
)

initialize_database()
app.include_router(projects_router)
app.include_router(history_router)
# ---------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Research request model
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body accepted by the /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="Research question submitted by the user.",
    )

    retrieval_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "Number of retrieval candidates requested "
            "per indexed paper."
        ),
    )

    rerank_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Maximum number of evidence candidates "
            "retained after reranking."
        ),
    )

    claim_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic similarity threshold for "
            "grouping extracted research claims."
        ),
    )


# ---------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    """Return basic API information."""

    return {
        "name": "ResearchLens API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    """Check whether the API is running."""

    return {
        "status": "healthy",
    }


# ---------------------------------------------------------
# Research query endpoint
# ---------------------------------------------------------

@app.post("/query")
def query_research(
    request: QueryRequest,
) -> dict[str, Any]:
    """
    Execute the complete ResearchLens research pipeline.

    HTTP success and evidence-audit completion are
    reported independently.
    """

    try:
        result = process_query(
            question=request.question,
            retrieval_k=request.retrieval_k,
            rerank_k=request.rerank_k,
            claim_threshold=request.claim_threshold,
        )

        return {
            "success": True,
            "analysis_status": result["analysis_status"],
            "audit_completed": result["audit_completed"],
            "data": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "ResearchLens query execution failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "ResearchLens could not process the "
                "request. Please try again."
            ),
        ) from error


# ---------------------------------------------------------
# PDF upload helpers
# ---------------------------------------------------------

def get_existing_document_names() -> set[str]:
    """
    Return case-insensitive filenames already present
    on disk or indexed in ChromaDB.
    """

    disk_names = {
        path.name.casefold()
        for path in PAPERS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    }

    indexed = collection.get(include=["metadatas"])

    indexed_names = {
        metadata["document"].casefold()
        for metadata in (indexed["metadatas"] or [])
        if metadata and metadata.get("document")
    }

    return disk_names | indexed_names


def remove_indexed_document(document_name: str) -> None:
    """
    Remove chunks indexed during a failed upload.

    Only call this for a filename that was verified
    to be new before ingestion.
    """

    existing = collection.get(
        where={"document": document_name}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])


# ---------------------------------------------------------
# PDF upload endpoint
# ---------------------------------------------------------

@app.post("/papers/upload")
def upload_paper(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload and index one text-based research PDF.

    Maximum size: 20 MB.
    Existing filenames cannot be overwritten.
    """

    original_name = file.filename or ""
    filename = Path(original_name).name

    if (
        not filename
        or filename in {".", ".."}
        or not filename.lower().endswith(".pdf")
        or "/" in original_name
        or "\\" in original_name
        or "\x00" in original_name
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF with a valid filename.",
        )

    PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        # Save the incoming file temporarily so invalid
        # uploads never enter the permanent paper library.
        with TemporaryDirectory() as temporary_directory:
            temporary_path = (
                Path(temporary_directory) / filename
            )

            size = 0

            with temporary_path.open("wb") as destination:
                while True:
                    chunk = file.file.read(1024 * 1024)

                    if not chunk:
                        break

                    size += len(chunk)

                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Maximum PDF size is 20 MB.",
                        )

                    destination.write(chunk)

            if size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded PDF is empty.",
                )

            # Check the file itself, not just its extension.
            try:
                with pymupdf.open(temporary_path) as document:
                    if not document.is_pdf:
                        raise ValueError("Not a PDF")

                    if document.needs_pass:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "Password-protected PDFs "
                                "are not yet supported."
                            ),
                        )

                    if document.page_count == 0:
                        raise HTTPException(
                            status_code=422,
                            detail="The PDF contains no pages.",
                        )

            except HTTPException:
                raise

            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is not a valid PDF.",
                ) from error

            # For the local development server, serialize
            # uploads to avoid duplicate-name races.
            with upload_lock:
                if filename.casefold() in get_existing_document_names():
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A paper with this filename "
                            "already exists."
                        ),
                    )

                permanent_path = PAPERS_DIR / filename
                saved = False

                try:
                    # Exclusive creation prevents accidental
                    # replacement of an existing file.
                    with (
                        temporary_path.open("rb") as source,
                        permanent_path.open("xb") as destination,
                    ):
                        saved = True

                        while True:
                            chunk = source.read(1024 * 1024)

                            if not chunk:
                                break

                            destination.write(chunk)

                    # Reuse the existing ingestion pipeline.
                    result = ingest_paper(permanent_path)

                except FileExistsError as error:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A paper with this filename "
                            "already exists."
                        ),
                    ) from error

                except Exception as error:
                    # Undo a partial upload and its newly
                    # indexed chunks.
                    if saved:
                        try:
                            remove_indexed_document(filename)
                        except Exception:
                            logger.exception(
                                "Could not clean up indexed chunks"
                            )

                        permanent_path.unlink(missing_ok=True)

                    if isinstance(error, ValueError):
                        raise HTTPException(
                            status_code=422,
                            detail=str(error),
                        ) from error

                    raise

                return {
                    "success": True,
                    "message": "Paper uploaded and indexed.",
                    "paper": result,
                }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("PDF upload failed")

        raise HTTPException(
            status_code=500,
            detail="The paper could not be uploaded.",
        ) from error

    finally:
        file.file.close()

        
@app.get("/papers")
def list_papers() -> dict[str, Any]:
    """List locally stored PDFs and their indexed chunk counts."""

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    indexed = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}

    for metadata in indexed["metadatas"] or []:
        if not metadata:
            continue

        name = metadata.get("document")

        if name:
            counts[name] = counts.get(name, 0) + 1

    papers = []

    for path in sorted(
        PAPERS_DIR.iterdir(),
        key=lambda item: item.name.casefold(),
    ):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        papers.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "indexed_chunks": counts.get(path.name, 0),
            "indexed": counts.get(path.name, 0) > 0,
        })

    return {
        "papers": papers,
        "total": len(papers),
    }
