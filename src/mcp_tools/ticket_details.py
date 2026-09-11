import json
from typing import Annotated

from src import database as db


def register(mcp) -> None:
    @mcp.tool()
    def get_ticket_details(ticket_id: Annotated[str, "Ticket ID, e.g. TKT-001"]) -> str:
        """Get details of a specific ticket."""
        ticket = db.get_ticket(ticket_id)
        if not ticket:
            return f"Ticket not found: {ticket_id}"
        return json.dumps(ticket, indent=2)
