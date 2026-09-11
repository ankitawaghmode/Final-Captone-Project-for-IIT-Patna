from datetime import date

from langchain_core.messages import AIMessage, SystemMessage

from .models import AgentState
from .prompts import RESPOND_PROMPT


def respond_node(state: AgentState, llm) -> dict:
    """Generate the user-facing reply using trimmed state context."""
    user_info = state.get("user_info")
    tool_result = state.get("tool_result", "")

    context = [RESPOND_PROMPT.format(today=date.today().isoformat())]
    if user_info:
        context.append(
            f"User: {user_info.get('name')} ({user_info.get('global_id')}) "
            f"— {user_info.get('department')}"
        )
    if tool_result:
        context.append(f"Tool output:\n{tool_result}")
    if not state.get("global_id"):
        context.append("No global_id yet — ask for it first.")
    if state.get("awaiting_confirm"):
        context.append("Ask user to confirm ticket creation.")
    context.append(
        "RULE: Never invent ticket IDs or claim a ticket was created unless "
        "the Tool output above explicitly confirms it."
    )

    recent_messages = list(state["messages"])[-6:]
    try:
        response = llm.invoke([SystemMessage(content="\n".join(context))] + recent_messages)
    except Exception as exc:
        response = AIMessage(content=f"Sorry, I encountered an error: {exc}. Please try again.")
    return {"messages": [response]}
