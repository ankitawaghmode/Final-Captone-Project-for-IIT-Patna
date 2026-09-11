import json
from typing import Annotated

from src import database as db


def register(mcp) -> None:
    @mcp.tool()
    def create_ticket(
        global_id: Annotated[str, "User Global ID"],
        title: Annotated[str, "Ticket title"],
        description: Annotated[str, "Issue description"],
        category: Annotated[str, "Network, Email, Access, Hardware, Software, Printer, Security, Other"],
        priority: Annotated[str, "Low, Medium, High"] = "Medium",
    ) -> str:
        """Create a new IT support ticket."""
        valid_categories = {"Network", "Email", "Access", "Hardware", "Software", "Printer", "Security", "Other"}
        valid_priorities = {"Low", "Medium", "High"}
        category = category if category in valid_categories else "Other"
        priority = priority if priority in valid_priorities else "Medium"
        ticket = db.create_ticket(global_id, title, description, category, priority)
        return json.dumps(ticket, indent=2)
