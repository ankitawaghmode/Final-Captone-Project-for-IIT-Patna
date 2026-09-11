"""
IT Help Desk agent — LangGraph orchestrated workflow (one explicit node per stage).
Supports OpenAI and Google Gemini via LLM_PROVIDER env var.

Graph topology
──────────────
                                    ┌── kb_search ─────────────┐
                                    ├── ticket_lookup ──────────┤
  START → [understand] ──► [validate_user] ──► [dupe_check] ───┤──► [respond] → END
               │                │                              ├── ticket_create ─┤
               └── (no tool) ──►└── (not found) ──────────────┘
                                                               └── (respond) ─────┘

LLM is used in exactly two nodes:
  1. understand  — structured-output classification (intent + field extraction)
  2. respond     — natural language response generation

All other nodes call tools/DB directly — routing is fully deterministic.

Decision flow (per turn)
────────────────────────
  understand: classify intent, extract global_id / query / ticket_id
      │
      ├── greeting / unknown / no global_id → respond (ask for ID or chitchat)
      ├── has global_id, not yet validated  → validate_user
      │       └── validated → route by intent
      ├── kb_question   → kb_search   → respond
      ├── ticket_status → ticket_lookup → respond
      ├── ticket_create → dupe_check  → respond (ask to confirm)
      └── confirm_yes (awaiting_confirm=True) → ticket_create → respond
"""
import json
import os
import uuid
from datetime import date
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Prompts ───────────────────────────────────────────────────────────────────

# Kept short deliberately — classifier only needs the last message + minimal context.
CLASSIFY_PROMPT = """Classify the latest IT Help Desk user message.

Intents: greeting | kb_question | ticket_status | ticket_create | confirm_yes | confirm_no | unknown

Rules:
- If the user only provides a Global ID (GIDxxx), keep the prior intent.
- global_id: GIDxxx format only. ticket_id: TKT-xxx format only.
- confirm_yes/no refers to ticket-creation confirmation only."""

# Kept short deliberately — respond node gets tool output inline, no need for verbose rules.
RESPOND_PROMPT = """You are a concise IT Help Desk Assistant. Reply in 1-4 sentences unless steps are needed.

Context rules (apply whichever matches):
- No global_id      → ask for Company Global ID (format GIDxxx)
- User not found    → apologise, ask to verify the ID
- KB results        → numbered steps, then ask “Did this resolve your issue?”
- Ticket info       → summarise clearly, offer next steps
- Similar tickets   → list them, ask if user wants to track one instead
- No similar ticket → ask user to confirm new ticket creation
- Ticket created    → confirm ticket ID, say IT team will follow up
- Greeting          → greet briefly, ask how you can help
Today: {today}"""


# ── Structured intent model ───────────────────────────────────────────────────

class UserIntent(BaseModel):
    intent: Literal[
        "greeting", "kb_question", "ticket_status",
        "ticket_create", "confirm_yes", "confirm_no", "unknown"
    ]
    global_id: str | None = Field(None, description="GIDxxx if found in conversation")
    query: str = Field("", description="IT issue description or search keywords")
    ticket_id: str | None = Field(None, description="TKT-xxx if user mentioned one")


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Conversation (append-only via add_messages) ───────────────────────────
    messages:         Annotated[list, add_messages]
    # ── Extracted per-turn (reset each turn by understand node) ───────────────
    intent:           str
    query:            str
    ticket_id:        str | None
    # ── Persisted across turns ────────────────────────────────────────────────
    global_id:        str | None   # user's confirmed Global ID
    user_info:        dict | None  # validated user record
    awaiting_confirm: bool         # True when we're waiting for ticket-create confirm
    # ── Tool output consumed by respond node ──────────────────────────────────
    tool_result:      str




# ── LLM / embeddings factories ────────────────────────────────────────────────

def create_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Return a chat LLM for the given provider (openai | gemini)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def create_embeddings(provider: str | None = None, model: str | None = None):
    """Return embeddings for the given provider (openai | gemini)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(llm: BaseChatModel, tools: list):
    """
    Compile the orchestrated LangGraph workflow.

    Node responsibilities
    ─────────────────────
    understand    — LLM (structured output): classify intent, extract IDs/query
    validate_user — call get_user_info tool; populate user_info in state
    kb_search     — call search_knowledge_base; populate tool_result
    ticket_lookup — call get_user_tickets or get_ticket_details; populate tool_result
    dupe_check    — call search_similar_tickets; populate tool_result; set awaiting_confirm
    ticket_create — call create_new_ticket; populate tool_result; clear awaiting_confirm
    respond       — LLM: read full state + tool_result → generate final reply

    Routing is fully deterministic (no LLM in the router).
    """
    tool_map = {t.name: t for t in tools}
    classifier = llm.with_structured_output(UserIntent)

    # ── Node 1: understand ────────────────────────────────────────────────────
    def understand_node(state: AgentState) -> dict:
        """Classify intent and extract fields using structured LLM output."""
        # Only the last 4 messages (2 turns) are needed for intent classification.
        try:
            result: UserIntent = classifier.invoke(
                [SystemMessage(content=CLASSIFY_PROMPT)] + list(state["messages"])[-4:]
            )
        except Exception:
            result = UserIntent(intent="unknown", query="")

        updates: dict = {
            "intent":      result.intent,
            "tool_result": "",   # clear previous turn's tool output
        }
        if result.query:
            updates["query"] = result.query
        # Only update global_id if newly extracted and user not yet validated
        if result.global_id and not state.get("user_info"):
            updates["global_id"] = result.global_id.strip().upper()
        if result.ticket_id:
            updates["ticket_id"] = result.ticket_id.strip().upper()

        # Explicitly set awaiting_confirm every turn so it never silently resets.
        if result.intent == "confirm_yes":
            updates["awaiting_confirm"] = state.get("awaiting_confirm", False)
        elif result.intent == "confirm_no":
            updates["awaiting_confirm"] = False
        else:
            updates["awaiting_confirm"] = False

        return updates

    # ── Node 2: validate_user ─────────────────────────────────────────────────
    def validate_user_node(state: AgentState) -> dict:
        """Validate Global ID and store user record in state."""
        raw = tool_map["get_user_info"].invoke({"global_id": state.get("global_id", "")})
        try:
            user_info = json.loads(raw)
            if isinstance(user_info, dict) and "global_id" in user_info:
                return {"user_info": user_info, "tool_result": f"Validated: {raw}"}
        except Exception:
            pass
        return {"user_info": None, "tool_result": raw}

    # ── Node 3: kb_search ─────────────────────────────────────────────────────
    def kb_search_node(state: AgentState) -> dict:
        """Search the knowledge base for the user's issue."""
        query = state.get("query") or list(state["messages"])[-1].content
        result = tool_map["search_knowledge_base"].invoke({"query": query})
        return {"tool_result": result}

    # ── Node 4: ticket_lookup ─────────────────────────────────────────────────
    def ticket_lookup_node(state: AgentState) -> dict:
        """Fetch a specific ticket or list all tickets for the user."""
        tid = state.get("ticket_id")
        if tid:
            result = tool_map["get_ticket_details"].invoke({"ticket_id": tid})
        else:
            result = tool_map["get_user_tickets"].invoke(
                {"global_id": state.get("global_id", ""), "status": ""}
            )
        return {"tool_result": result}

    # ── Node 5: dupe_check ────────────────────────────────────────────────────
    def dupe_check_node(state: AgentState) -> dict:
        """Check for existing similar open/in-progress tickets."""
        result = tool_map["search_similar_tickets"].invoke({
            "global_id":         state.get("global_id", ""),
            "issue_description": state.get("query", ""),
        })
        # Pause here — respond node will ask user to confirm
        return {"tool_result": result, "awaiting_confirm": True}

    # ── Node 6: ticket_create ─────────────────────────────────────────────────
    def ticket_create_node(state: AgentState) -> dict:
        """Create a new IT support ticket after user confirmation."""
        query  = state.get("query") or "IT support request"
        gid    = state.get("global_id", "")
        result = tool_map["create_new_ticket"].invoke({
            "global_id":   gid,
            "title":       query[:80],
            "description": query,
            "category":    "Other",
            "priority":    "Medium",
        })
        print(f"[ticket_create_node] gid={gid!r}  result={result[:80]}")  # debug
        return {"tool_result": result, "awaiting_confirm": False}

    # ── Node 7: respond ───────────────────────────────────────────────────────
    def respond_node(state: AgentState) -> dict:
        """Generate the user-facing reply using trimmed state context."""
        today      = date.today().isoformat()
        tool_result = state.get("tool_result", "")
        user_info   = state.get("user_info")

        # Build a compact system message — only include lines that are relevant.
        ctx_parts = [RESPOND_PROMPT.format(today=today)]
        if user_info:
            ctx_parts.append(
                f"User: {user_info.get('name')} ({user_info.get('global_id')}) "
                f"— {user_info.get('department')}"
            )
        if tool_result:
            ctx_parts.append(f"Tool output:\n{tool_result}")
        if not state.get("global_id"):
            ctx_parts.append("No global_id yet — ask for it first.")
        if state.get("awaiting_confirm"):
            ctx_parts.append("Ask user to confirm ticket creation.")
        # Anti-hallucination: the LLM must not invent ticket IDs or claim actions it did not perform.
        ctx_parts.append("RULE: Never invent ticket IDs or claim a ticket was created unless the Tool output above explicitly confirms it.")

        # Cap history to last 6 messages to avoid unbounded token growth.
        recent_msgs = list(state["messages"])[-6:]
        try:
            response = llm.invoke([SystemMessage(content="\n".join(ctx_parts))] + recent_msgs)
        except Exception as exc:
            response = AIMessage(content=f"Sorry, I encountered an error: {exc}. Please try again.")
        return {"messages": [response]}

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_understand(state: AgentState) -> str:
        intent    = state.get("intent", "unknown")
        global_id = state.get("global_id")
        user_info = state.get("user_info")

        if intent == "greeting":
            return "respond"
        if not global_id:
            return "respond"          # respond will ask for Global ID
        if not user_info:
            return "validate_user"    # validate before any tool action

        # User is validated — route by intent
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

    def route_after_validate(state: AgentState) -> str:
        """After user validation, route based on original intent."""
        if not state.get("user_info"):
            return "respond"          # validation failed
        intent = state.get("intent", "unknown")
        if intent == "kb_question":
            return "kb_search"
        if intent == "ticket_status":
            return "ticket_lookup"
        if intent == "ticket_create":
            return "dupe_check"
        return "respond"

    # ── Assemble and compile ──────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    builder.add_node("understand",    understand_node)
    builder.add_node("validate_user", validate_user_node)
    builder.add_node("kb_search",     kb_search_node)
    builder.add_node("ticket_lookup", ticket_lookup_node)
    builder.add_node("dupe_check",    dupe_check_node)
    builder.add_node("ticket_create", ticket_create_node)
    builder.add_node("respond",       respond_node)

    builder.add_edge(START, "understand")

    builder.add_conditional_edges("understand", route_after_understand, {
        "respond":       "respond",
        "validate_user": "validate_user",
        "kb_search":     "kb_search",
        "ticket_lookup": "ticket_lookup",
        "dupe_check":    "dupe_check",
        "ticket_create": "ticket_create",
    })
    builder.add_conditional_edges("validate_user", route_after_validate, {
        "respond":       "respond",
        "kb_search":     "kb_search",
        "ticket_lookup": "ticket_lookup",
        "dupe_check":    "dupe_check",
    })

    for node in ("kb_search", "ticket_lookup", "dupe_check", "ticket_create"):
        builder.add_edge(node, "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())

# ── Chatbot session wrapper ───────────────────────────────────────────────────

class ITHelpdeskChatbot:
    """
    Stateful chatbot backed by a compiled LangGraph.

    Every session is isolated by a unique thread_id in the MemorySaver
    checkpointer.  reset() rotates the thread_id, starting a fresh session
    while the graph and its weights remain intact.

    Usage:
        graph   = build_graph(llm, tools)
        chatbot = ITHelpdeskChatbot(graph)
        reply   = chatbot.chat("Hello, my VPN is broken")
        chatbot.reset()          # new session
        msgs    = chatbot.history # inspect full message history
    """

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
