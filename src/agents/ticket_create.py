from .models import AgentState


def ticket_create_node(state: AgentState, tool_map: dict) -> dict:
    """Create a new IT support ticket after user confirmation."""
    query = state.get("query") or "IT support request"
    global_id = state.get("global_id", "")
    result = tool_map["create_new_ticket"].invoke({
        "global_id": global_id,
        "title": query[:80],
        "description": query,
        "category": "Other",
        "priority": "Medium",
    })
    print(f"[ticket_create_node] gid={global_id!r}  result={result[:80]}")
    return {"tool_result": result, "awaiting_confirm": False}
