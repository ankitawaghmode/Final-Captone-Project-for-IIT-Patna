from .models import AgentState


def ticket_lookup_node(state: AgentState, tool_map: dict) -> dict:
    """Fetch a specific ticket or list all tickets for the user."""
    ticket_id = state.get("ticket_id")
    if ticket_id:
        result = tool_map["get_ticket_details"].invoke({"ticket_id": ticket_id})
    else:
        result = tool_map["get_user_tickets"].invoke(
            {"global_id": state.get("global_id", ""), "status": ""}
        )
    return {"tool_result": result}
