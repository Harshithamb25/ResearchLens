
"""Persistent research history and project-scoped evidence matrices."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from backend.workspace.database import (
    get_connection,
    initialize_database,
)
from backend.workspace.evidence_matrix import (
    build_evidence_matrix,
    matrix_to_csv,
)
from backend.workspace.paper_matrix import (
    build_paper_matrix,
    paper_matrix_to_csv,
)


router = APIRouter(
    prefix="/projects",
    tags=["Research History"],
)


def require_project(project_id: str) -> None:
    with get_connection() as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Research project not found.",
        )


def _require_session(
    project_id: str,
    session_id: str,
):
    initialize_database()
    require_project(project_id)

    with get_connection() as connection:
        session = connection.execute(
            """
            SELECT *
            FROM research_sessions
            WHERE id = ?
              AND project_id = ?
            """,
            (session_id, project_id),
        ).fetchone()

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Research session not found "
                "in this project."
            ),
        )

    return session


def save_research_session(
    project_id: str,
    question: str,
    response: dict,
) -> str:
    initialize_database()
    require_project(project_id)

    session_id = str(uuid4())
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    encoded_response = jsonable_encoder(
        response
    )

    result = encoded_response.get(
        "data",
        {},
    )

    analysis_status = result.get(
        "analysis_status",
        "partial",
    )

    audit_completed = bool(
        result.get("audit_completed", False)
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO research_sessions (
                id,
                project_id,
                question,
                response_json,
                analysis_status,
                audit_completed,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                project_id,
                question,
                json.dumps(
                    encoded_response,
                    ensure_ascii=False,
                ),
                analysis_status,
                int(audit_completed),
                created_at,
            ),
        )
        connection.commit()

    return session_id


@router.get("/{project_id}/sessions")
def list_research_sessions(
    project_id: str,
):
    initialize_database()
    require_project(project_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                project_id,
                question,
                analysis_status,
                audit_completed,
                created_at
            FROM research_sessions
            WHERE project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (project_id,),
        ).fetchall()

    sessions = [
        {
            **dict(row),
            "audit_completed": bool(
                row["audit_completed"]
            ),
        }
        for row in rows
    ]

    return {
        "success": True,
        "project_id": project_id,
        "sessions": sessions,
        "total": len(sessions),
    }


@router.get(
    "/{project_id}/sessions/{session_id}"
)
def get_research_session(
    project_id: str,
    session_id: str,
):
    row = _require_session(
        project_id,
        session_id,
    )

    return {
        "success": True,
        "id": row["id"],
        "project_id": row["project_id"],
        "question": row["question"],
        "analysis_status": row["analysis_status"],
        "audit_completed": bool(
            row["audit_completed"]
        ),
        "created_at": row["created_at"],
        "response": json.loads(
            row["response_json"]
        ),
    }


@router.delete(
    "/{project_id}/sessions/{session_id}"
)
def delete_research_session(
    project_id: str,
    session_id: str,
):
    initialize_database()
    require_project(project_id)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM research_sessions
            WHERE id = ?
              AND project_id = ?
            """,
            (session_id, project_id),
        )
        connection.commit()

    if cursor.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "Research session not found "
                "in this project."
            ),
        )

    return {
        "success": True,
        "message": "Research session deleted.",
        "session_id": session_id,
    }


def _session_matrix(
    project_id: str,
    session_id: str,
) -> dict:
    session = _require_session(
        project_id,
        session_id,
    )

    saved_response = json.loads(
        session["response_json"]
    )

    research_result = saved_response.get(
        "data",
        {},
    )

    if not isinstance(research_result, dict):
        raise HTTPException(
            status_code=422,
            detail=(
                "This session does not contain "
                "a usable research response."
            ),
        )

    with get_connection() as connection:
        papers = connection.execute(
            """
            SELECT
                id,
                filename,
                indexed_chunks,
                created_at
            FROM documents
            WHERE project_id = ?
              AND indexed_chunks > 0
            ORDER BY created_at ASC, id ASC
            """,
            (project_id,),
        ).fetchall()

    return build_evidence_matrix(
        project_id=project_id,
        session_id=session_id,
        question=session["question"],
        research_result=research_result,
        papers=[
            dict(paper)
            for paper in papers
        ],
    )


@router.get(
    "/{project_id}/sessions/{session_id}/matrix"
)
def get_research_matrix(
    project_id: str,
    session_id: str,
):
    return _session_matrix(
        project_id,
        session_id,
    )


@router.get(
    "/{project_id}/sessions/{session_id}/matrix.csv"
)
def export_research_matrix_csv(
    project_id: str,
    session_id: str,
):
    matrix = _session_matrix(
        project_id,
        session_id,
    )

    return Response(
        content="\ufeff" + matrix_to_csv(matrix),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                'attachment; filename="'
                f'researchlens-matrix-{session_id}.csv"'
            ),
            "Cache-Control": "private, no-store",
        },
    )


def _dedicated_project_matrix(
    project_id: str,
) -> dict:
    """Read indexed papers belonging only to this project."""
    initialize_database()
    require_project(project_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                storage_path
            FROM documents
            WHERE project_id = ?
              AND indexed_chunks > 0
            ORDER BY created_at ASC, id ASC
            """,
            (project_id,),
        ).fetchall()

    papers = [
        dict(row)
        for row in rows
    ]

    if not papers:
        raise HTTPException(
            status_code=422,
            detail=(
                "Upload and index at least one "
                "PDF before generating a matrix."
            ),
        )

    # The project upload endpoint stores PDFs here.
    project_directory = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "projects"
        / project_id
    ).resolve()

    for paper in papers:
        path = Path(
            paper["storage_path"]
        ).resolve()

        if not path.is_relative_to(
            project_directory
        ):
            raise HTTPException(
                status_code=403,
                detail="Invalid paper storage location.",
            )

        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Original PDF is missing: "
                    f"{paper['filename']}"
                ),
            )

    try:
        return build_paper_matrix(
            project_id=project_id,
            papers=papers,
        )

    except Exception as error:
        import logging

        logging.getLogger(__name__).exception(
            "Dedicated matrix extraction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not extract the project "
                "evidence matrix."
            ),
        ) from error


@router.get(
    "/{project_id}/paper-matrix"
)
def get_dedicated_paper_matrix(
    project_id: str,
):
    """Scan each indexed original PDF independently."""
    return _dedicated_project_matrix(
        project_id
    )


@router.get(
    "/{project_id}/paper-matrix.csv"
)
def export_dedicated_paper_matrix(
    project_id: str,
):
    """Export the dedicated nine-column matrix."""
    matrix = _dedicated_project_matrix(
        project_id
    )

    return Response(
        content=(
            "\ufeff"
            + paper_matrix_to_csv(matrix)
        ),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                'attachment; filename="'
                'researchlens-paper-matrix.csv"'
            ),
            "Cache-Control": "private, no-store",
        },
    )