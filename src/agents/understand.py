from langchain_core.messages import SystemMessage

from .models import AgentState, UserIntent
from .prompts import CLASSIFY_PROMPT


def understand_node(state: AgentState, classifier) -> dict:
    """Classify intent and extract fields using structured LLM output."""
    try:
        result: UserIntent = classifier.invoke(
            [SystemMessage(content=CLASSIFY_PROMPT)] + list(state["messages"])[-4:]
        )
    except Exception:
        result = UserIntent(intent="unknown", query="")

    updates: dict = {"intent": result.intent, "tool_result": ""}
    if result.query:
        updates["query"] = result.query
    if result.global_id and not state.get("user_info"):
        updates["global_id"] = result.global_id.strip().upper()
    if result.ticket_id:
        updates["ticket_id"] = result.ticket_id.strip().upper()

    if result.intent == "confirm_yes":
        updates["awaiting_confirm"] = state.get("awaiting_confirm", False)
    elif result.intent == "confirm_no":
        updates["awaiting_confirm"] = False
    else:
        updates["awaiting_confirm"] = False

    return updates
