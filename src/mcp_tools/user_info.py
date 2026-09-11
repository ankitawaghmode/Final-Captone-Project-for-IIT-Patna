import json
from typing import Annotated

from src import database as db


def register(mcp) -> None:
    @mcp.tool()
    def get_user_info(global_id: Annotated[str, "Company Global ID, e.g. GID001"]) -> str:
        """Look up an employee by their company Global ID."""
        user = db.get_user(global_id)
        if not user:
            return f"User not found: {global_id}"
        return json.dumps(user, indent=2)
