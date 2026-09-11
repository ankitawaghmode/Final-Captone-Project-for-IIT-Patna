from .models import AgentState


def kb_search_node(state: AgentState, tool_map: dict) -> dict:
    """Search the knowledge base for the user's issue."""
    query = state.get("query") or list(state["messages"])[-1].content
    result = tool_map["search_knowledge_base"].invoke({"query": query})
    return {"tool_result": result}
