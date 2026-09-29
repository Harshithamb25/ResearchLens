
"""Safely rebuild the ChromaDB index for an existing project.

Usage:
    python -m backend.reindex_project PROJECT_ID
    python -m backend.reindex_project PROJECT_ID --apply

The default run is a preview. --apply backs up existing
vectors, replaces only the selected project's documents,
and updates their indexed chunk counts in SQLite.
"""

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.vector_store import (
    collection,
    add_project_chunks,
    remove_project_document,
)
from backend.workspace.database import (
    get_connection,
    initialize_database,
)


ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "data" / "reindex_backups"


def get_documents(project_id):
    """Read the existing project's registered documents."""
    initialize_database()

    with get_connection() as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

        if project is None:
            raise ValueError("Project does not exist.")

        rows = connection.execute(
            """
            SELECT id, filename, storage_path, indexed_chunks
            FROM documents
            WHERE project_id = ?
            ORDER BY filename
            """,
            (project_id,),
        ).fetchall()

    if not rows:
        raise ValueError("Project has no registered documents.")

    return [dict(row) for row in rows]


def prepare_documents(project_id, documents):
    """Extract and chunk everything before modifying ChromaDB."""
    project_folder = (
        ROOT / "data" / "projects" / project_id
    ).resolve()

    prepared = []

    for document in documents:
        path = Path(document["storage_path"]).resolve()

        if not path.is_relative_to(project_folder):
            raise ValueError(
                f"Unexpected storage location: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(path)

        pages = extract_text_from_pdf(path)
        chunks = create_chunks(
            pages,
            document["filename"],
        )

        if not chunks:
            raise ValueError(
                f"No extractable chunks: {document['filename']}"
            )

        print(
            f"{document['filename']}: "
            f"{len(pages)} pages, "
            f"{len(chunks)} new chunks"
        )

        prepared.append({
            "document": document,
            "pages": pages,
            "chunks": chunks,
        })

    return prepared


def snapshot_document(document_id, project_id):
    """Capture the existing vectors and all associated metadata."""
    result = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas", "embeddings"],
    )

    metadata = result.get("metadatas") or []

    if any(
        item.get("project_id") != project_id
        for item in metadata
    ):
        raise ValueError(
            f"Project mismatch for document {document_id}"
        )

    embeddings = result.get("embeddings")

    if embeddings is None:
        raise ValueError(
            f"Could not back up embeddings for {document_id}"
        )

    return {
        "ids": list(result["ids"]),
        "documents": list(result["documents"]),
        "metadatas": list(metadata),
        "embeddings": (
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else list(embeddings)
        ),
    }


def restore_document(document_id, snapshot):
    """Restore exactly the original records after a failed rebuild."""
    remove_project_document(document_id)

    if snapshot["ids"]:
        collection.add(
            ids=snapshot["ids"],
            documents=snapshot["documents"],
            metadatas=snapshot["metadatas"],
            embeddings=snapshot["embeddings"],
        )


def save_backup(project_id, snapshots):
    """Write a persistent backup before deleting any old records."""
    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = BACKUP_DIR / (
        f"{project_id}_{timestamp}.pkl"
    )

    if path.exists():
        raise FileExistsError(path)

    with path.open("xb") as file:
        pickle.dump(
            {
                "project_id": project_id,
                "snapshots": snapshots,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"\nBackup saved: {path}")
    return path


def verify_document(project_id, document_id, expected):
    """Check that the replacement contains only expected records."""
    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if len(result["ids"]) != expected:
        raise RuntimeError(
            f"Expected {expected} chunks for {document_id}; "
            f"found {len(result['ids'])}."
        )

    if any(
        metadata.get("project_id") != project_id
        or metadata.get("document_id") != document_id
        for metadata in result["metadatas"]
    ):
        raise RuntimeError(
            f"Incorrect metadata for {document_id}"
        )


def reindex(project_id, apply=False):
    documents = get_documents(project_id)
    prepared = prepare_documents(
        project_id,
        documents,
    )

    print(f"\nDocuments prepared: {len(prepared)}")

    if not apply:
        print(
            "PREVIEW ONLY: no embeddings generated, "
            "no database records changed."
        )
        return

    # Generate all embeddings before touching the existing index.
    for item in prepared:
        item["embeddings"] = generate_embeddings(
            item["chunks"]
        )

        if len(item["embeddings"]) != len(item["chunks"]):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

    # Back up all existing documents before replacing any of them.
    snapshots = {}

    for item in prepared:
        document = item["document"]
        document_id = document["id"]

        snapshot = snapshot_document(
            document_id,
            project_id,
        )

        if (
            len(snapshot["ids"])
            != document["indexed_chunks"]
        ):
            raise RuntimeError(
                "SQLite and ChromaDB counts differ for "
                f"{document['filename']}. "
                "Reindex aborted without changes."
            )

        snapshots[document_id] = snapshot

    save_backup(project_id, snapshots)

    try:
        for item in prepared:
            document = item["document"]
            document_id = document["id"]

            print(
                f"\nReplacing: {document['filename']}"
            )

            remove_project_document(document_id)

            add_project_chunks(
                chunks=item["chunks"],
                embeddings=item["embeddings"],
                project_id=project_id,
                document_id=document_id,
            )

            verify_document(
                project_id,
                document_id,
                len(item["chunks"]),
            )

        # Update counts only after all vector replacements succeed.
        with get_connection() as connection:
            for item in prepared:
                connection.execute(
                    """
                    UPDATE documents
                    SET indexed_chunks = ?
                    WHERE id = ? AND project_id = ?
                    """,
                    (
                        len(item["chunks"]),
                        item["document"]["id"],
                        project_id,
                    ),
                )

            connection.commit()

    except Exception:
        print(
            "\nReindex failed. Restoring original ChromaDB records..."
        )

        restoration_errors = []

        for document_id, snapshot in snapshots.items():
            try:
                restore_document(
                    document_id,
                    snapshot,
                )
            except Exception as error:
                restoration_errors.append(
                    (document_id, str(error))
                )

        if restoration_errors:
            print(
                "Automatic restoration was incomplete. "
                "Preserve the backup and do not run "
                "another reindex."
            )
            for error in restoration_errors:
                print(error)
        else:
            print(
                "Original ChromaDB records restored."
            )

        raise

    print(
        "\nSUCCESS: all selected documents were "
        "reindexed and verified."
    )

    for item in prepared:
        print(
            f"{item['document']['filename']}: "
            f"{len(item['chunks'])} chunks"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Safely reindex an existing ResearchLens project."
    )

    parser.add_argument(
        "project_id",
        help="Existing ResearchLens project UUID",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and replace the existing vectors",
    )

    args = parser.parse_args()
    project_id = str(UUID(args.project_id))

    reindex(
        project_id,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()