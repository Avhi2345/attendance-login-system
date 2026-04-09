"""
Database Layer — Google Sheets
- Embeddings sheet: stores face embeddings per user
- Attendance sheet: logs every login attempt with all scores
- Self-learning: updates embeddings via exponential moving average
- SECURITY: uses json.loads instead of eval()
"""

import json
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ── Auth ──────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
client = gspread.authorize(creds)


# ── Sheet Initialization ─────────────────────────────────────
def _get_or_create_spreadsheet():
    try:
        return client.open("FaceAuthDB")
    except gspread.SpreadsheetNotFound:
        return client.create("FaceAuthDB")


def _init_embeddings_sheet():
    ss = _get_or_create_spreadsheet()
    try:
        return ss.worksheet("Embeddings")
    except gspread.WorksheetNotFound:
        sheet = ss.add_worksheet(title="Embeddings", rows=1000, cols=3)
        sheet.append_row(["Name", "Embedding", "UpdatedAt"])
        return sheet


def _init_attendance_sheet():
    ss = _get_or_create_spreadsheet()
    try:
        return ss.worksheet("Attendance")
    except gspread.WorksheetNotFound:
        sheet = ss.add_worksheet(title="Attendance", rows=5000, cols=6)
        sheet.append_row([
            "Name", "Timestamp", "FaceScore",
            "AntispoofScore", "LivenessScore", "Status",
        ])
        return sheet


embeddings_sheet = _init_embeddings_sheet()
attendance_sheet = _init_attendance_sheet()


# ── Embedding Operations ─────────────────────────────────────
def save_embedding(name: str, embedding: list):
    """Save a new embedding for a user (JSON-serialized, not eval!)."""
    embeddings_sheet.append_row([
        name,
        json.dumps(embedding),
        datetime.now().isoformat(),
    ])


def load_all_users() -> dict:
    """Load all users and their embeddings.
    Returns: {name: [[emb1], [emb2], ...]}"""
    data = embeddings_sheet.get_all_records()
    users = {}
    for row in data:
        try:
            emb = json.loads(row["Embedding"])
            users.setdefault(row["Name"], []).append(emb)
        except (json.JSONDecodeError, KeyError):
            continue
    return users


def update_embedding_self_learn(name: str, new_embedding: list, alpha: float = 0.3):
    """
    Self-Learning: blend new observation into stored embeddings.

    Formula:
        updated = (1 - alpha) * avg_existing + alpha * new_embedding
        updated = normalize(updated)

    This lets the system gradually adapt to appearance changes
    (aging, glasses, lighting, etc.) while staying stable.
    """
    new_emb = np.array(new_embedding, dtype=np.float64)

    # Find all rows for this user
    data = embeddings_sheet.get_all_records()
    user_rows = []
    for i, row in enumerate(data):
        if row["Name"] == name:
            try:
                emb = np.array(json.loads(row["Embedding"]), dtype=np.float64)
                user_rows.append({"row_num": i + 2, "embedding": emb})  # +2: header + 0-index
            except (json.JSONDecodeError, KeyError):
                continue

    if not user_rows:
        save_embedding(name, new_embedding)
        return

    # Compute average of all existing embeddings
    existing = np.array([r["embedding"] for r in user_rows])
    avg_old = np.mean(existing, axis=0)

    # Blend old average with new observation
    updated = (1.0 - alpha) * avg_old + alpha * new_emb

    # Normalize to unit vector
    norm = np.linalg.norm(updated)
    if norm > 0:
        updated = updated / norm

    # Overwrite the most recent embedding row with the blended result
    latest = user_rows[-1]
    embeddings_sheet.update_cell(latest["row_num"], 2, json.dumps(updated.tolist()))
    embeddings_sheet.update_cell(latest["row_num"], 3, datetime.now().isoformat())


# ── Attendance Logging ────────────────────────────────────────
def log_attendance(
    name: str,
    face_score: float,
    antispoof_score: float,
    liveness_score: float,
    status: str,
):
    """Log an attendance entry with all verification scores."""
    attendance_sheet.append_row([
        name,
        datetime.now().isoformat(),
        round(face_score, 4),
        round(antispoof_score, 4),
        round(liveness_score, 4),
        status,
    ])


def get_attendance_log(name: str = None, limit: int = 50) -> list:
    """Retrieve attendance log entries, optionally filtered by name."""
    data = attendance_sheet.get_all_records()
    if name:
        data = [r for r in data if r.get("Name") == name]
    return data[-limit:]
