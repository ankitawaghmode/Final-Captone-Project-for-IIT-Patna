"""
IT Help Desk MCP Server - FastMCP implementation

Run:
    python -m src.mcp_server
"""

import json
import sys
from pathlib import Path
from typing import Annotated

# Ensure repo root is importable when invoked as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer
from src import database as db


# Initialize database
db.init_db()

# Create MCP server
mcp = MCPServer("it-helpdesk")


@mcp.tool()
def get_user_info(
    global_id: Annotated[str, "Company Global ID, e.g. GID001"]
) -> str:
    """
    Look up an employee by their company Global ID.
    """
    user = db.get_user(global_id)

    if not user:
        return f"User not found: {global_id}"

    return json.dumps(user, indent=2)


@mcp.tool()
def get_user_tickets(
    global_id: Annotated[str, "Company Global ID"],
    status: Annotated[
        str,
        "Optional filter: Open, In Progress, Resolved"
    ] = ""
) -> str:
    """
    Get all IT tickets for a user.
    """
    tickets = db.get_user_tickets(
        global_id,
        status if status else None
    )

    return json.dumps(tickets, indent=2)


@mcp.tool()
def get_ticket_details(
    ticket_id: Annotated[str, "Ticket ID, e.g. TKT-001"]
) -> str:
    """
    Get details of a specific ticket.
    """
    ticket = db.get_ticket(ticket_id)

    if not ticket:
        return f"Ticket not found: {ticket_id}"

    return json.dumps(ticket, indent=2)


@mcp.tool()
def search_similar_tickets(
    global_id: Annotated[str, "Company Global ID"],
    issue_description: Annotated[str, "Issue description"]
) -> str:
    """
    Search for similar existing tickets.
    """

    stop_words = {
        "the", "and", "for", "with", "that",
        "this", "have", "from", "not", "are"
    }

    keywords = [
        word.lower()
        for word in issue_description.split()
        if len(word) > 3 and word.lower() not in stop_words
    ][:6]

    results = db.search_similar_tickets(
        global_id,
        keywords
    )

    return json.dumps(results, indent=2)


@mcp.tool()
def create_ticket(
    global_id: Annotated[str, "User Global ID"],
    title: Annotated[str, "Ticket title"],
    description: Annotated[str, "Issue description"],
    category: Annotated[
        str,
        "Network, Email, Access, Hardware, Software, Printer, Security, Other"
    ],
    priority: Annotated[
        str,
        "Low, Medium, High"
    ] = "Medium"
) -> str:
    """
    Create a new IT support ticket.
    """

    valid_categories = {
        "Network",
        "Email",
        "Access",
        "Hardware",
        "Software",
        "Printer",
        "Security",
        "Other",
    }

    valid_priorities = {
        "Low",
        "Medium",
        "High",
    }

    category = category if category in valid_categories else "Other"
    priority = priority if priority in valid_priorities else "Medium"

    ticket = db.create_ticket(
        global_id,
        title,
        description,
        category,
        priority,
    )

    return json.dumps(ticket, indent=2)


if __name__ == "__main__":
    print(
        "IT Help Desk MCP server starting...",
        file=sys.stderr
    )

    mcp.run()