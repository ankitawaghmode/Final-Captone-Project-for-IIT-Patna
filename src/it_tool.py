"""
LangChain tool definitions that wrap the database and knowledge base.
These same functions are exposed as MCP tools in mcp_server.py.
"""
import json
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.tools import tool

from src import database as db
from src import knowledge_base as kb

# Injected at runtime via set_vectorstore()
_vectorstore: Optional[Chroma] = None


def set_vectorstore(vs: Chroma) -> None:
    global _vectorstore
    _vectorstore = vs


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_user_info(global_id: str) -> str:
    """Look up a company employee by their Global ID (e.g. GID001).
    Returns their name, email, department, and manager."""
    user = db.get_user(global_id)
    if not user:
        return (
            f"No employee found with Global ID '{global_id}'. "
            "Please verify the ID (format: GIDxxx) and try again."
        )
    return json.dumps(user, indent=2)


@tool
def get_user_tickets(global_id: str, status: str = "") -> str:
    """Retrieve all IT support tickets for an employee.
    Optionally filter by status: 'Open', 'In Progress', or 'Resolved'.
    Leave status blank to see all tickets."""
    user = db.get_user(global_id)
    if not user:
        return f"No employee found with Global ID '{global_id}'."

    allowed = {"Open", "In Progress", "Resolved"}
    status_filter = status if status in allowed else None
    tickets = db.get_user_tickets(global_id, status_filter)

    if not tickets:
        label = f" with status '{status_filter}'" if status_filter else ""
        return f"No tickets found for {user['name']} ({global_id}){label}."

    lines = [f"Tickets for {user['name']} ({global_id}):\n"]
    for t in tickets:
        lines.append(
            f"  [{t['ticket_id']}] {t['title']}\n"
            f"    Status: {t['status']} | Priority: {t['priority']} | Category: {t['category']}\n"
            f"    Created: {t['created_at']}"
            + (f"\n    Resolution: {t['resolution']}" if t.get("resolution") else "")
        )
    return "\n".join(lines)


@tool
def get_ticket_details(ticket_id: str) -> str:
    """Get full details of a specific IT ticket by its ID (e.g. TKT-001)."""
    ticket = db.get_ticket(ticket_id)
    if not ticket:
        return f"No ticket found with ID '{ticket_id}'."
    return json.dumps(ticket, indent=2)


@tool
def search_similar_tickets(global_id: str, issue_description: str) -> str:
    """Check whether the employee already has an open or in-progress ticket
    for the same issue before creating a new one.
    Pass the employee's Global ID and a short description of the issue."""
    user = db.get_user(global_id)
    if not user:
        return f"No employee found with Global ID '{global_id}'."

    # Simple keyword extraction — meaningful words only
    stop = {"the", "and", "for", "with", "that", "this", "have", "from", "not", "are"}
    keywords = [
        w.lower() for w in issue_description.split()
        if len(w) > 3 and w.lower() not in stop
    ][:6]

    similar = db.search_similar_tickets(global_id, keywords)
    if not similar:
        return (
            f"No similar open/in-progress tickets found for {user['name']} ({global_id}). "
            "You may proceed to create a new ticket."
        )

    lines = [f"Found {len(similar)} existing open/in-progress ticket(s) for {user['name']}:\n"]
    for t in similar:
        lines.append(
            f"  [{t['ticket_id']}] {t['title']}\n"
            f"    Status: {t['status']} | Priority: {t['priority']}\n"
            f"    Opened: {t['created_at']}\n"
            f"    Description: {t['description'][:200]}"
        )
    return "\n".join(lines)


@tool
def create_new_ticket(
    global_id: str,
    title: str,
    description: str,
    category: str,
    priority: str = "Medium",
) -> str:
    """Create a new IT support ticket ONLY after explicit user confirmation.

    category must be one of: Network, Email, Access, Hardware, Software, Printer, Security, Other
    priority must be one of: Low, Medium, High
    """
    user = db.get_user(global_id)
    if not user:
        return f"Cannot create ticket — no employee found with Global ID '{global_id}'."

    valid_categories = {"Network", "Email", "Access", "Hardware", "Software", "Printer", "Security", "Other"}
    valid_priorities = {"Low", "Medium", "High"}

    if category not in valid_categories:
        category = "Other"
    if priority not in valid_priorities:
        priority = "Medium"

    ticket = db.create_ticket(global_id, title, description, category, priority)

    return (
        f"✅ Ticket created successfully!\n\n"
        f"  Ticket ID : {ticket['ticket_id']}\n"
        f"  Title     : {ticket['title']}\n"
        f"  Category  : {ticket['category']}\n"
        f"  Priority  : {ticket['priority']}\n"
        f"  Status    : {ticket['status']}\n"
        f"  Created   : {ticket['created_at']}\n\n"
        f"The IT support team has been notified. "
        f"Updates will be sent to {user['email']}."
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search the IT knowledge base for troubleshooting steps and self-service solutions.
    Always call this FIRST before creating a ticket to see if the issue can be resolved
    without IT intervention."""
    if _vectorstore is None:
        return "Knowledge base is not initialized."

    results = kb.search_kb(_vectorstore, query, k=2)
    if not results:
        return "No relevant articles found in the knowledge base for this query."

    lines = [f"Found {len(results)} relevant KB article(s):\n"]
    for i, art in enumerate(results, 1):
        body = art["content"][:900] + ("..." if len(art["content"]) > 900 else "")
        lines.append(
            f"{'─'*60}\n"
            f"Article {i}: {art['title']}  [{art['category']}]\n"
            f"{'─'*60}\n"
            f"{body}\n"
        )
    return "\n".join(lines)


def get_all_tools() -> list:
    """Return the complete list of IT helpdesk LangChain tools."""
    return [
        get_user_info,
        get_user_tickets,
        get_ticket_details,
        search_similar_tickets,
        create_new_ticket,
        search_knowledge_base,
    ]
