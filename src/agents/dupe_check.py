from .models import AgentState


def dupe_check_node(state: AgentState, tool_map: dict) -> dict:
    """Check for existing similar open/in-progress tickets."""
    result = tool_map["search_similar_tickets"].invoke({
        "global_id": state.get("global_id", ""),
        "issue_description": state.get("query", ""),
    })
    return {"tool_result": result, "awaiting_confirm": True}
