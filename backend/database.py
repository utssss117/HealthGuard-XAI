"""
database.py
───────────
SQLite database setup for HealthGuard-XAI user authentication.
Uses Python's built-in sqlite3 for zero-config setup.
"""

from __future__ import annotations

import os
import sqlite3
import bcrypt
from contextlib import contextmanager

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "healthguard.db")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash using bcrypt."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def init_db() -> None:
    """Create the users, predictions, and chat_history tables if they don't exist."""
    with get_db() as conn:
        # Create users table
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

        # Add password_hash column to users table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'password_hash' not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

        # Create predictions table to store patient biomarker data and model predictions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                patient_email       TEXT NOT NULL,
                timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                label               TEXT,
                risk_probability    REAL NOT NULL,
                risk_level          TEXT NOT NULL,
                pregnancies         REAL,
                glucose             REAL,
                blood_pressure      REAL,
                skin_thickness      REAL,
                insulin             REAL,
                bmi                 REAL,
                diabetes_pedigree   REAL,
                age                 REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create chat_history table to store user conversation history with health assistant
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                patient_email       TEXT NOT NULL,
                role                TEXT NOT NULL,
                content             TEXT NOT NULL,
                timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
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


# ── Authentication Helpers ───────────────────────────────────────────────────

def register_user(email: str, password: str, first_name: str, last_name: str, role: str) -> dict | None:
    """Register a new user in the local database. Password is hashed using bcrypt."""
    email_clean = email.lower().strip()
    hashed = hash_password(password)
    clerk_id = f"local_{email_clean}"  # compatibility for NOT NULL UNIQUE clerk_id

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO users (clerk_id, email, password_hash, first_name, last_name, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clerk_id, email_clean, hashed, first_name.strip(), last_name.strip(), role)
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email_clean,)).fetchone()
            return dict(row) if row else None
    except sqlite3.IntegrityError:
        return None


def authenticate_user(email: str, password: str) -> dict | None:
    """Authenticate user with email and password. Returns user dict or None."""
    email_clean = email.lower().strip()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email_clean,)).fetchone()
        if row and row['password_hash']:
            if verify_password(password, row['password_hash']):
                return dict(row)
    return None


# ── Data Persistence Helpers ─────────────────────────────────────────────────

def save_prediction(
    user_id: int,
    patient_email: str,
    biomarkers: dict,
    risk_probability: float,
    risk_level: str,
    label: str = None
) -> int:
    """Save risk prediction and biomarkers record to the database."""
    patient_clean = patient_email.lower().strip()
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                user_id, patient_email, label, risk_probability, risk_level,
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_clean,
                label,
                risk_probability,
                risk_level,
                biomarkers.get("Pregnancies"),
                biomarkers.get("Glucose"),
                biomarkers.get("BloodPressure"),
                biomarkers.get("SkinThickness"),
                biomarkers.get("Insulin"),
                biomarkers.get("BMI"),
                biomarkers.get("DiabetesPedigreeFunction") or biomarkers.get("DiabetesPedigree"),
                biomarkers.get("Age")
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_prediction_history(user_id: int, patient_email: str = None) -> list:
    """Retrieve saved prediction history for a user, optionally filtered by patient."""
    with get_db() as conn:
        if patient_email:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? AND patient_email = ? ORDER BY timestamp ASC",
                (user_id, patient_email.lower().strip())
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp ASC",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_unique_patients(user_id: int) -> list:
    """Return all unique patient emails assessed by this clinician/user."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT patient_email FROM predictions WHERE user_id = ? ORDER BY patient_email ASC",
            (user_id,)
        ).fetchall()
        return [r['patient_email'] for r in rows]


def save_chat_message(user_id: int, patient_email: str, role: str, content: str) -> None:
    """Save an AI assistant conversation message to the database."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, patient_email, role, content) VALUES (?, ?, ?, ?)",
            (user_id, patient_email.lower().strip(), role, content)
        )
        conn.commit()


def get_chat_history(user_id: int, patient_email: str) -> list:
    """Retrieve chat conversation history for a user/patient pair."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE user_id = ? AND patient_email = ? ORDER BY timestamp ASC",
            (user_id, patient_email.lower().strip())
        ).fetchall()
        return [dict(r) for r in rows]


def clear_chat_history(user_id: int, patient_email: str) -> None:
    """Delete all chat logs for a specific user/patient pair."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND patient_email = ?",
            (user_id, patient_email.lower().strip())
        )
        conn.commit()

