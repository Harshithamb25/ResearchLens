
"""ResearchLens project management, scoped PDF uploads, and research queries."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import pymupdf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.vector_store import (
    add_project_chunks,
    remove_project_document,
)
from backend.pipeline import process_query
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm_service import generate_document_answer
from backend.workspace.history import get_conversation_context
from backend.workspace.database import (
    get_connection,
    initialize_database,
)
from backend.workspace.history import save_research_session
from backend.workspace.paper_matrix import build_paper_matrix, matrix_to_csv


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects",
    tags=["Research Projects"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILES_DIR = PROJECT_ROOT / "data" / "projects"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

upload_lock = Lock()


# --------------------------------------------------
# Request models
# --------------------------------------------------

class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    mode: str = Field(default="research")


class UpdateProjectRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    mode: str | None = Field(default=None)
    is_pinned: bool | None = None


class ProjectQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    retrieval_k: int = Field(default=10, ge=1, le=50)
    rerank_k: int = Field(default=5, ge=1, le=20)
    claim_threshold: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )


# --------------------------------------------------
# Shared helpers
# --------------------------------------------------

def _project_row(project_id: str):
    """Return a project row or None."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()


def _require_project(project_id: str) -> None:
    """Raise HTTP 404 when a project does not exist."""
    if _project_row(project_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Research project not found.",
        )


# --------------------------------------------------
# Project creation
# --------------------------------------------------

@router.post("", status_code=201)
def create_project(request: CreateProjectRequest):
    initialize_database()

    name = request.name.strip()

    if not name:
        raise HTTPException(
            status_code=422,
            detail="Project name cannot be empty.",
        )

    project_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    description = request.description.strip()
    mode = request.mode.strip().lower()
    if mode not in {"document", "research"}:
        raise HTTPException(status_code=422, detail="Project mode must be 'document' or 'research'.")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO projects
                (
                    id,
                    name,
                    description,
                    status,
                    mode,
                    is_pinned,
                    created_at,
                    updated_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                name,
                description,
                "active",
                mode,
                0,
                timestamp,
                timestamp,
            ),
        )

        connection.commit()

    return {
        "id": project_id,
        "name": name,
        "description": description,
        "status": "active",
        "mode": mode,
        "is_pinned": False,
        "created_at": timestamp,
        "updated_at": timestamp,
        "document_count": 0,
    }


# --------------------------------------------------
# List projects
# --------------------------------------------------

@router.get("")
def list_projects():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.description,
                p.status,
                p.mode,
                p.is_pinned,
                p.created_at,
                p.updated_at,
                COUNT(d.id) AS document_count
            FROM projects AS p
            LEFT JOIN documents AS d
                ON d.project_id = p.id
            GROUP BY p.id
            ORDER BY p.is_pinned DESC, p.updated_at DESC, p.id DESC
            """
        ).fetchall()

    projects = [{**dict(row), "is_pinned": bool(row["is_pinned"])} for row in rows]

    return {
        "projects": projects,
        "total": len(projects),
    }


# --------------------------------------------------
# Get one project
# --------------------------------------------------

@router.get("/{project_id}")
def get_project(project_id: str):
    initialize_database()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.description,
                p.status,
                p.mode,
                p.is_pinned,
                p.created_at,
                p.updated_at,
                COUNT(d.id) AS document_count
            FROM projects AS p
            LEFT JOIN documents AS d
                ON d.project_id = p.id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (project_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Research project not found.",
        )

    return {**dict(row), "is_pinned": bool(row["is_pinned"])}


# --------------------------------------------------
# Project workspace metadata
# --------------------------------------------------

@router.patch("/{project_id}")
def update_project(project_id: str, request: UpdateProjectRequest):
    initialize_database()
    _require_project(project_id)

    updates = []
    values = []

    if request.name is not None:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Project name cannot be empty.")
        updates.append("name = ?")
        values.append(name)

    if request.description is not None:
        updates.append("description = ?")
        values.append(request.description.strip())

    if request.mode is not None:
        mode = request.mode.strip().lower()
        if mode not in {"document", "research"}:
            raise HTTPException(status_code=422, detail="Project mode must be 'document' or 'research'.")
        updates.append("mode = ?")
        values.append(mode)

    if request.is_pinned is not None:
        updates.append("is_pinned = ?")
        values.append(int(request.is_pinned))

    if not updates:
        raise HTTPException(status_code=422, detail="No project changes were supplied.")

    timestamp = datetime.now(timezone.utc).isoformat()
    updates.append("updated_at = ?")
    values.extend([timestamp, project_id])

    with get_connection() as connection:
        connection.execute(
            "UPDATE projects SET " + ", ".join(updates) + " WHERE id = ?",
            values,
        )
        connection.commit()

    return get_project(project_id)


@router.delete("/{project_id}")
def delete_project(project_id: str):
    initialize_database()
    _require_project(project_id)

    with get_connection() as connection:
        connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        connection.commit()

    import shutil
    shutil.rmtree(PROJECT_FILES_DIR / project_id, ignore_errors=True)

    return {"success": True, "project_id": project_id}

# --------------------------------------------------
# List indexed papers in a project
# --------------------------------------------------

@router.get("/{project_id}/papers")
def list_project_papers(project_id: str):
    initialize_database()
    _require_project(project_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                project_id,
                filename,
                indexed_chunks,
                created_at
            FROM documents
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()

    papers = [
        {
            "id": row["id"],
            "project_id": row["project_id"],
            "filename": row["filename"],
            "indexed_chunks": row["indexed_chunks"],
            "indexed": row["indexed_chunks"] > 0,
            "created_at": row["created_at"],
        }
        for row in rows
    ]

    return {
        "success": True,
        "project_id": project_id,
        "papers": papers,
        "total": len(papers),
    }


# --------------------------------------------------
# Upload and index one PDF
# --------------------------------------------------

@router.post(
    "/{project_id}/papers/upload",
    status_code=201,
)
def upload_project_paper(
    project_id: str,
    file: UploadFile = File(...),
):
    """
    Validate, save, index and register one PDF
    within the selected research project.
    """
    initialize_database()

    try:
        _require_project(project_id)

        filename = file.filename or ""

        if (
            not filename
            or filename in (".", "..")
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or not filename.lower().endswith(".pdf")
        ):
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid PDF filename.",
            )

        contents = file.file.read(
            MAX_UPLOAD_BYTES + 1
        )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 20 MB upload limit.",
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        try:
            with pymupdf.open(
                stream=contents,
                filetype="pdf",
            ) as pdf:
                if pdf.needs_pass:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Password-protected PDFs "
                            "are not supported."
                        ),
                    )

                if pdf.page_count == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF has no pages.",
                    )

                page_count = pdf.page_count

        except HTTPException:
            raise

        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail="The file is not a readable PDF.",
            ) from error

        document_id = str(uuid4())

        project_folder = (
            PROJECT_FILES_DIR / project_id
        )

        destination = (
            project_folder / f"{document_id}.pdf"
        )

        with upload_lock:
            with get_connection() as connection:
                duplicate = connection.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE project_id = ?
                      AND filename = ? COLLATE NOCASE
                    """,
                    (project_id, filename),
                ).fetchone()

                if duplicate:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "This project already contains "
                            "a PDF with the same filename."
                        ),
                    )

                project_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                destination.write_bytes(contents)

                try:
                    pages = extract_text_from_pdf(
                        destination
                    )

                    chunks = create_chunks(
                        pages,
                        filename,
                    )

                    if not chunks:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "No extractable text was found. "
                                "Scanned PDFs are not yet supported."
                            ),
                        )

                    embeddings = generate_embeddings(
                        chunks
                    )

                    add_project_chunks(
                        chunks=chunks,
                        embeddings=embeddings,
                        project_id=project_id,
                        document_id=document_id,
                    )

                    timestamp = datetime.now(
                        timezone.utc
                    ).isoformat()

                    connection.execute(
                        """
                        INSERT INTO documents
                            (
                                id,
                                project_id,
                                filename,
                                storage_path,
                                indexed_chunks,
                                created_at
                            )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            project_id,
                            filename,
                            str(destination),
                            len(chunks),
                            timestamp,
                        ),
                    )

                    connection.execute(
                        """
                        UPDATE projects
                        SET updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            timestamp,
                            project_id,
                        ),
                    )

                    connection.commit()

                except Exception:
                    connection.rollback()

                    try:
                        remove_project_document(
                            document_id
                        )

                    except Exception:
                        logger.exception(
                            "Failed to clean up indexed "
                            "document %s",
                            document_id,
                        )

                    destination.unlink(
                        missing_ok=True
                    )

                    raise

        return {
            "success": True,
            "message": "Paper uploaded and indexed.",
            "paper": {
                "id": document_id,
                "project_id": project_id,
                "filename": filename,
                "pages": page_count,
                "indexed_chunks": len(chunks),
                "indexed": True,
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Project PDF upload failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Could not upload and index this PDF.",
        ) from error

    finally:
        file.file.close()


# --------------------------------------------------
# Project-scoped research query
# WITH AUTOMATIC SESSION SAVING
# --------------------------------------------------

@router.post("/{project_id}/document-query")
def query_project_document(
    project_id: str,
    request: ProjectQueryRequest,
) -> dict[str, Any]:
    """Run ordinary grounded document understanding without research analysis."""
    initialize_database()
    _require_project(project_id)

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Document question cannot be empty.",
        )

    with get_connection() as connection:
        project = connection.execute(
            "SELECT mode FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        document_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE project_id = ?
              AND indexed_chunks > 0
            """,
            (project_id,),
        ).fetchone()[0]

    if project["mode"] != "document":
        raise HTTPException(
            status_code=409,
            detail="This project is configured for Research Lens.",
        )

    if document_count < 1:
        raise HTTPException(
            status_code=422,
            detail="Upload at least one indexed PDF before asking a document question.",
        )

    try:
        conversation_context = get_conversation_context(project_id)
        results = search(
            question,
            top_k=request.retrieval_k,
            project_id=project_id,
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        candidates = []
        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            if not text or not metadata:
                continue
            candidates.append(
                {
                    "text": text,
                    "document": metadata.get("document"),
                    "document_id": metadata.get("document_id"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "retrieval_score": (
                        1.0 - float(distance)
                        if distance is not None
                        else None
                    ),
                }
            )

        ranked = rerank(
            question,
            candidates,
            top_k=request.rerank_k,
        )

        evidence = [
            {
                **item,
                "evidence_scope": "UNCERTAIN",
                "contribution_type": "UNCERTAIN",
            }
            for item in ranked
        ]

        answer = generate_document_answer(
            question,
            evidence,
            conversation_context=conversation_context,
        )

        response_data = {
            "question": question,
            "query_type": "document",
            "route": "document_rag",
            "answer": answer,
            "answer_generated": answer is not None,
            "citation_validation": None,
            "analysis_status": "not_applicable",
            "analysis_message": (
                "Document Lens used project-scoped semantic "
                "retrieval and reranking without research analysis."
            ),
            "audit_completed": False,
            "failed_group_count": 0,
            "result": {
                "status": "not_applicable",
                "evidence": evidence,
                "cross_paper_evidence": False,
                "themes": [],
                "theme_comparisons": [],
                "analysis_failures": [],
            },
        }

        response = {
            "success": True,
            "project_id": project_id,
            "analysis_status": "not_applicable",
            "audit_completed": False,
            "data": response_data,
        }

        session_id = save_research_session(
            project_id=project_id,
            question=question,
            response=response,
        )

        return {
            **response,
            "session_id": session_id,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Project document query failed")
        raise HTTPException(
            status_code=500,
            detail="ResearchLens could not process the document query.",
        ) from error


@router.post("/{project_id}/query")
def query_project(
    project_id: str,
    request: ProjectQueryRequest,
) -> dict[str, Any]:
    """
    Run cross-paper analysis using only
    the selected project's documents.

    Save the complete successful response
    to project-scoped research history.
    """
    initialize_database()
    _require_project(project_id)

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Research question cannot be empty.",
        )

    with get_connection() as connection:
        paper_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE project_id = ?
              AND indexed_chunks > 0
            """,
            (project_id,),
        ).fetchone()[0]

    if paper_count < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "Research Mode requires at least "
                "two indexed papers. Upload another "
                "paper or use Document Mode."
            ),
        )

    try:
        conversation_context = get_conversation_context(project_id)
        result = process_query(
            question=question,
            retrieval_k=request.retrieval_k,
            rerank_k=request.rerank_k,
            claim_threshold=request.claim_threshold,
            project_id=project_id,
            conversation_context=conversation_context,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Project research query failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "ResearchLens could not process "
                "the project query."
            ),
        ) from error

    # Preserve the existing frontend response structure.
    response = {
        "success": True,
        "project_id": project_id,
        "analysis_status": result["analysis_status"],
        "audit_completed": result["audit_completed"],
        "data": result,
    }

    # Save the completed analysis, including its answer,
    # evidence, comparisons and citation validation.
    try:
        session_id = save_research_session(
            project_id=project_id,
            question=question,
            response=response,
        )

    except Exception as error:
        logger.exception(
            "Failed to save research history "
            "for project %s",
            project_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Research analysis completed, but "
                "its session could not be saved."
            ),
        ) from error

    # The additional session_id does not break
    # the existing React frontend.
    return {
        **response,
        "session_id": session_id,
    }


# --------------------------------------------------
# Serve the original PDF
# --------------------------------------------------

@router.get(
    "/{project_id}/papers/{document_id}/file"
)
def get_project_paper_file(
    project_id: str,
    document_id: str,
):
    """
    Open the original PDF for a document
    belonging to the selected project.
    """
    initialize_database()

    with get_connection() as connection:
        document = connection.execute(
            """
            SELECT
                d.storage_path,
                d.filename
            FROM documents AS d
            JOIN projects AS p
                ON p.id = d.project_id
            WHERE d.id = ?
              AND d.project_id = ?
            """,
            (
                document_id,
                project_id,
            ),
        ).fetchone()

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Paper not found in this project.",
        )

    project_directory = (
        PROJECT_FILES_DIR / project_id
    ).resolve()

    file_path = Path(
        document["storage_path"]
    ).resolve()

    if not file_path.is_relative_to(
        project_directory
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid paper storage location.",
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The original PDF is missing.",
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=document["filename"],
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
        },
    )