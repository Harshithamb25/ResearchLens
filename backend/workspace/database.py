
"""SQLite database for ResearchLens workspaces and research history."""

import sqlite3
from pathlib import Path

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
            """
        )