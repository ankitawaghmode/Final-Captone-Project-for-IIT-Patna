"""LangGraph workflow assembly for the IT Help Desk agent."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents.dupe_check import dupe_check_node
from .agents.kb_search import kb_search_node
from .agents.models import AgentState, UserIntent
from .agents.respond import respond_node
from .agents.ticket_create import ticket_create_node
from .agents.ticket_lookup import ticket_lookup_node
from .agents.understand import understand_node
from .agents.validate_user import validate_user_node


def _route_after_understand(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    global_id = state.get("global_id")
    user_info = state.get("user_info")

    if intent == "greeting":
        return "respond"
    if not global_id:
        return "respond"
    if not user_info:
        return "validate_user"
    if intent == "confirm_yes" and state.get("awaiting_confirm"):
        return "ticket_create"
    if intent in ("confirm_no", "unknown"):
        return "respond"
    if intent == "kb_question":
        return "kb_search"
    if intent == "ticket_status":
        return "ticket_lookup"
    if intent == "ticket_create":
        return "dupe_check"
    return "respond"


def _route_after_validate(state: AgentState) -> str:
    if not state.get("user_info"):
        return "respond"
    intent = state.get("intent", "unknown")
    if intent == "kb_question":
        return "kb_search"
    if intent == "ticket_status":
        return "ticket_lookup"
    if intent == "ticket_create":
        return "dupe_check"
    return "respond"


def build_graph(llm, tools: list):
    """Compile the deterministic LangGraph help desk workflow."""
    tool_map = {tool.name: tool for tool in tools}
    classifier = llm.with_structured_output(UserIntent)

    builder = StateGraph(AgentState)
    builder.add_node("understand", lambda state: understand_node(state, classifier))
    builder.add_node("validate_user", lambda state: validate_user_node(state, tool_map))
    builder.add_node("kb_search", lambda state: kb_search_node(state, tool_map))
    builder.add_node("ticket_lookup", lambda state: ticket_lookup_node(state, tool_map))
    builder.add_node("dupe_check", lambda state: dupe_check_node(state, tool_map))
    builder.add_node("ticket_create", lambda state: ticket_create_node(state, tool_map))
    builder.add_node("respond", lambda state: respond_node(state, llm))

    builder.add_edge(START, "understand")
    builder.add_conditional_edges("understand", _route_after_understand, {
        "respond": "respond",
        "validate_user": "validate_user",
        "kb_search": "kb_search",
        "ticket_lookup": "ticket_lookup",
        "dupe_check": "dupe_check",
        "ticket_create": "ticket_create",
    })
    builder.add_conditional_edges("validate_user", _route_after_validate, {
        "respond": "respond",
        "kb_search": "kb_search",
        "ticket_lookup": "ticket_lookup",
        "dupe_check": "dupe_check",
    })
    for node in ("kb_search", "ticket_lookup", "dupe_check", "ticket_create"):
        builder.add_edge(node, "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())
