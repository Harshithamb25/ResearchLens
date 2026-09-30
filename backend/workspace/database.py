
"""SQLite database for ResearchLens workspaces and research history."""

import sqlite3
from pathlib import Path
from uuid import uuid4

DATABASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "researchlens.db"
)


def get_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return connection


def initialize_database():
    """Create missing tables without deleting existing data."""

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                mode TEXT NOT NULL DEFAULT 'research',
                is_pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL UNIQUE,
                indexed_chunks INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id)
                    REFERENCES projects(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS
                idx_documents_project
                ON documents(project_id);

            CREATE TABLE IF NOT EXISTS research_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                question TEXT NOT NULL,
                response_json TEXT NOT NULL,
                analysis_status TEXT NOT NULL,
                audit_completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id)
                    REFERENCES projects(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS
                idx_research_sessions_project_created
                ON research_sessions(
                    project_id,
                    created_at DESC
                );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (project_id)
                    REFERENCES projects(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                response_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS
                idx_conversation_messages_conversation_created
                ON conversation_messages(
                    conversation_id,
                    created_at ASC,
                    id ASC
                );
            """
        )

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(projects)"
            ).fetchall()
        }

        if "mode" not in columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN mode "
                "TEXT NOT NULL DEFAULT 'research'"
            )

        if "is_pinned" not in columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN is_pinned "
                "INTEGER NOT NULL DEFAULT 0"
            )

        projects = connection.execute(
            "SELECT id, created_at, updated_at FROM projects"
        ).fetchall()

        for project in projects:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    id, project_id, title, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    project["id"],
                    "New conversation",
                    project["created_at"],
                    project["updated_at"],
                ),
            )

        connection.commit()