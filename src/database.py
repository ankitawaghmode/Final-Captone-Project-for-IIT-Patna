"""
SQLite-based ticket and user management for the IT Help Desk system.
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "tickets.db"


def init_db() -> None:
    """Create schema and seed sample users/tickets."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                global_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL,
                department  TEXT NOT NULL,
                manager     TEXT
            );

            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id   TEXT PRIMARY KEY,
                global_id   TEXT NOT NULL,
                title       TEXT NOT NULL,
                description TEXT NOT NULL,
                category    TEXT NOT NULL,
                priority    TEXT NOT NULL DEFAULT 'Medium',
                status      TEXT NOT NULL DEFAULT 'Open',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                resolution  TEXT,
                FOREIGN KEY (global_id) REFERENCES users(global_id)
            );
        """)

        # ── Sample users ──────────────────────────────────────────────
        conn.executemany(
            "INSERT OR IGNORE INTO users VALUES (?,?,?,?,?)",
            [
                ("GID001", "Alice Johnson", "alice.johnson@company.com", "Engineering", "Bob Smith"),
                ("GID002", "Bob Smith",    "bob.smith@company.com",    "Engineering", "Carol Davis"),
                ("GID003", "Carol Davis",  "carol.davis@company.com",  "HR",          "Eve Wilson"),
                ("GID004", "David Lee",    "david.lee@company.com",    "Finance",     "Frank Brown"),
                ("GID005", "Eve Wilson",   "eve.wilson@company.com",   "IT",          None),
            ],
        )

        # ── Sample tickets ────────────────────────────────────────────
        conn.executemany(
            "INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "TKT-001", "GID001",
                    "Cannot access VPN",
                    "VPN connection fails after recent system update. Error: 'Authentication failed'.",
                    "Network", "High", "In Progress",
                    "2024-01-15 09:00:00", "2024-01-15 14:00:00", None,
                ),
                (
                    "TKT-002", "GID001",
                    "Outlook not syncing emails",
                    "Outlook stuck on Sending/Receiving. Inbox not loading new emails.",
                    "Email", "Medium", "Resolved",
                    "2024-01-10 10:00:00", "2024-01-11 11:00:00",
                    "Cleared Outlook cache and ran the Office repair tool.",
                ),
                (
                    "TKT-003", "GID002",
                    "SSO portal login failure",
                    "Cannot login to company SSO portal. Password appears correct.",
                    "Access", "High", "Open",
                    "2024-01-16 08:30:00", "2024-01-16 08:30:00", None,
                ),
                (
                    "TKT-004", "GID003",
                    "Laptop extremely slow after update",
                    "Laptop performance degraded significantly after latest Windows update.",
                    "Hardware", "Medium", "Open",
                    "2024-01-16 09:00:00", "2024-01-16 09:00:00", None,
                ),
                (
                    "TKT-005", "GID004",
                    "Cannot print to network printer",
                    "Network printer HP-FLOOR3 not printing. Job queues but nothing comes out.",
                    "Printer", "Low", "Open",
                    "2024-01-16 10:00:00", "2024-01-16 10:00:00", None,
                ),
            ],
        )


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def _query(sql: str, params: tuple = ()) -> list[dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = _query(sql, params)
    return rows[0] if rows else None


def get_user(global_id: str) -> dict | None:
    return _query_one(
        "SELECT * FROM users WHERE global_id = ?",
        (global_id.strip().upper(),),
    )


def get_user_tickets(global_id: str, status_filter: str | None = None) -> list[dict]:
    if status_filter:
        return _query(
            "SELECT * FROM tickets WHERE global_id = ? AND status = ? ORDER BY created_at DESC",
            (global_id.strip().upper(), status_filter),
        )
    return _query(
        "SELECT * FROM tickets WHERE global_id = ? ORDER BY created_at DESC",
        (global_id.strip().upper(),),
    )


def get_ticket(ticket_id: str) -> dict | None:
    return _query_one(
        "SELECT * FROM tickets WHERE ticket_id = ?",
        (ticket_id.strip().upper(),),
    )


def search_similar_tickets(global_id: str, keywords: list[str]) -> list[dict]:
    """Return open/in-progress tickets whose title or description match any keyword."""
    seen, results = set(), []
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        for kw in keywords:
            pattern = f"%{kw.lower()}%"
            rows = conn.execute(
                """
                SELECT * FROM tickets
                WHERE  global_id = ?
                  AND  status IN ('Open', 'In Progress')
                  AND  (LOWER(title) LIKE ? OR LOWER(description) LIKE ?)
                ORDER BY created_at DESC
                """,
                (global_id.strip().upper(), pattern, pattern),
            ).fetchall()
            for r in rows:
                t = dict(r)
                if t["ticket_id"] not in seen:
                    seen.add(t["ticket_id"])
                    results.append(t)
    return results


def create_ticket(
    global_id: str,
    title: str,
    description: str,
    category: str,
    priority: str = "Medium",
) -> dict:
    ticket_id = f"TKT-{str(uuid.uuid4())[:6].upper()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ticket_id, global_id.strip().upper(), title, description,
             category, priority, "Open", now, now, None),
        )
    return {
        "ticket_id": ticket_id,
        "global_id": global_id.strip().upper(),
        "title": title,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "Open",
        "created_at": now,
    }
