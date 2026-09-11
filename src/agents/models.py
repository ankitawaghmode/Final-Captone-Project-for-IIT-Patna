from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class UserIntent(BaseModel):
    intent: Literal[
        "greeting", "kb_question", "ticket_status",
        "ticket_create", "confirm_yes", "confirm_no", "unknown"
    ]
    global_id: str | None = Field(None, description="GIDxxx if found in conversation")
    query: str = Field("", description="IT issue description or search keywords")
    ticket_id: str | None = Field(None, description="TKT-xxx if user mentioned one")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    query: str
    ticket_id: str | None
    global_id: str | None
    user_info: dict | None
    awaiting_confirm: bool
    tool_result: str
