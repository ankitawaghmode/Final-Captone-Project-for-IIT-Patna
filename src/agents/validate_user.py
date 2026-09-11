import json

from .models import AgentState


def validate_user_node(state: AgentState, tool_map: dict) -> dict:
    """Validate Global ID and store user record in state."""
    raw = tool_map["get_user_info"].invoke({"global_id": state.get("global_id", "")})
    try:
        user_info = json.loads(raw)
        if isinstance(user_info, dict) and "global_id" in user_info:
            return {"user_info": user_info, "tool_result": f"Validated: {raw}"}
    except Exception:
        pass
    return {"user_info": None, "tool_result": raw}
