"""
database.py
───────────
SQLite database setup for HealthGuard-XAI user authentication.
Uses Python's built-in sqlite3 for zero-config setup.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "healthguard.db")


def init_db() -> None:
    """Create the users table if it doesn't already exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                clerk_id        TEXT    NOT NULL UNIQUE,
                email           TEXT    NOT NULL UNIQUE,
                first_name      TEXT    DEFAULT '',
                last_name       TEXT    DEFAULT '',
                role            TEXT    NOT NULL DEFAULT 'patient',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    """Yield a SQLite connection; auto-closes on exit."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
