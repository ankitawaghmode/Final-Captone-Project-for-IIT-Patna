import json
from typing import Annotated

from src import database as db


def register(mcp) -> None:
    @mcp.tool()
    def get_user_tickets(
        global_id: Annotated[str, "Company Global ID"],
        status: Annotated[str, "Optional filter: Open, In Progress, Resolved"] = "",
    ) -> str:
        """Get all IT tickets for a user."""
        tickets = db.get_user_tickets(global_id, status if status else None)
        return json.dumps(tickets, indent=2)
