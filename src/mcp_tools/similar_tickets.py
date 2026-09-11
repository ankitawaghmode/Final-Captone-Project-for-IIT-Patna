import json
from typing import Annotated

from src import database as db


def register(mcp) -> None:
    @mcp.tool()
    def search_similar_tickets(
        global_id: Annotated[str, "Company Global ID"],
        issue_description: Annotated[str, "Issue description"],
    ) -> str:
        """Search for similar existing tickets."""
        stop_words = {"the", "and", "for", "with", "that", "this", "have", "from", "not", "are"}
        keywords = [
            word.lower()
            for word in issue_description.split()
            if len(word) > 3 and word.lower() not in stop_words
        ][:6]
        results = db.search_similar_tickets(global_id, keywords)
        return json.dumps(results, indent=2)
