import uuid

from langchain_core.messages import HumanMessage


class ITHelpdeskChatbot:
    """Stateful chatbot backed by a compiled LangGraph."""

    def __init__(self, graph) -> None:
        self._graph = graph
        self._thread_id = str(uuid.uuid4())

    @property
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self._thread_id}}

    def chat(self, user_input: str) -> str:
        """Send a message and return the agent's reply."""
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self._config,
        )
        return result["messages"][-1].content

    def reset(self) -> None:
        """Start a fresh session by rotating the thread ID."""
        self._thread_id = str(uuid.uuid4())

    @property
    def history(self) -> list:
        """Return full message history for the current session."""
        state = self._graph.get_state(self._config)
        return list(state.values.get("messages", [])) if state else []
