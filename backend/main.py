
"""
ResearchLens FastAPI application.

Exposes research analysis results and explicitly
reports whether the evidence audit was completed.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import process_query


logger = logging.getLogger(__name__)

app = FastAPI(
    title="ResearchLens API",
    description=(
        "Evidence-grounded research intelligence API "
        "for multi-document research analysis."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body accepted by the /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="Research question submitted by the user."
    )

    retrieval_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of candidates retrieved from ChromaDB."
    )

    rerank_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of evidence candidates retained "
            "after reranking."
        )
    )

    claim_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic similarity threshold for grouping claims."
        )
    )


# ---------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "ResearchLens API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------

@app.post("/query")
def query_research(
    request: QueryRequest
) -> dict[str, Any]:
    """
    Execute the research pipeline.

    Request success and audit completion are separate:
    a successfully processed request may have a partial
    or failed cross-paper audit.
    """

    try:
        result = process_query(
            question=request.question,
            retrieval_k=request.retrieval_k,
            rerank_k=request.rerank_k,
            claim_threshold=request.claim_threshold
        )

        return {
            "success": True,
            "analysis_status": result["analysis_status"],
            "audit_completed": result["audit_completed"],
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "ResearchLens query execution failed"
        )

        # Avoid exposing provider errors, credentials
        # or internal implementation details.
        raise HTTPException(
            status_code=500,
            detail=(
                "ResearchLens could not process the "
                "request. Please try again."
            )
        ) from error