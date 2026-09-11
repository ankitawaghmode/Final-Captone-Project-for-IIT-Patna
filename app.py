"""
IT Help Desk Assistant — Production Streamlit UI
Run: streamlit run app.py

Features:
  • Chat assistant (LangGraph agent) with 👤 / 🤖 message bubbles
  • Session history — persisted across browser reloads
  • Knowledge Base viewer, search, and add-new article
  • My Tickets — lookup by Global ID with status/priority badges
  • Manual ticket creation form
  • Quick Help sidebar links
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="IT Help Desk Assistant",
    page_icon="Images/technical-support.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv(ROOT / ".env")

from src.agent import ITHelpdeskChatbot, build_graph, create_embeddings, create_llm
from src.database import create_ticket as db_create_ticket
from src.database import get_user, get_user_tickets as db_get_tickets, init_db
from src.it_tool import get_all_tools, set_vectorstore
from src.knowledge_base import init_knowledge_base
import base64

# ── Constants ─────────────────────────────────────────────────────────────────
KB_JSON       = ROOT / "data" / "kb_documents.json"
SESSIONS_FILE = ROOT / "data" / "chat_sessions.json"
# Embed bot avatar as base64 so it works inside st.markdown HTML
_bot_img_path = ROOT / "Images" / "virtual.png"
if _bot_img_path.exists():
    _bot_b64 = base64.b64encode(_bot_img_path.read_bytes()).decode()
    BOT_AVATAR_HTML = f'<img src="data:image/png;base64,{_bot_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />'
else:
    BOT_AVATAR_HTML = "IT"  # fallback text if image not found

_user_image_path = ROOT / "Images" / "user.png"
if _user_image_path.exists():
    _user_b64 = base64.b64encode(_user_image_path.read_bytes()).decode()
    USER_AVATAR_HTML = f'<img src="data:image/png;base64,{_user_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />'
else:
    USER_AVATAR_HTML = "You"  # fallback text if image not found

_quick_help_img_path = ROOT / "Images" / "book.png"
if _quick_help_img_path.exists():
    _quick_help_b64 = base64.b64encode(_quick_help_img_path.read_bytes()).decode()
    QUICK_HELP_AVATAR_HTML = f'<img src="data:image/png;base64,{_quick_help_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />'
else :
    QUICK_HELP_AVATAR_HTML = "QH"  # fallback text if image not found

_chat_history_img_path = ROOT / "Images" / "talking.png"
if _chat_history_img_path.exists():
    _chat_history_b64 = base64.b64encode(_chat_history_img_path.read_bytes()).decode()
    CHAT_HISTORY_AVATAR_HTML = f'<img src="data:image/png;base64,{_chat_history_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />'
else:
    CHAT_HISTORY_AVATAR_HTML = "CH"  # fallback text if image not found

CATEGORIES    = ["Network", "Email", "Access", "Hardware", "Software",
                 "Printer", "Security", "Other"]
PRIORITIES    = ["Low", "Medium", "High"]
KB_CATEGORIES = ["Access Management", "Network", "Email", "Collaboration",
                 "Software", "Hardware", "Printer", "Security",
                 "Remote Access", "Cloud Storage"]

# ══════════════════════════════════════════════════════════════════════════════
# CSS — monochromatic modern enterprise design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>

/* ── Reset & base ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] > .main { background: #f4f5f7; }

/* ── Global padding ────────────────────────────────────────────── */
.block-container { padding: 1.5rem 2rem 2rem !important; }

/* ══ SIDEBAR ══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #1c1c1e !important;
    border-right: 1px solid #2c2c2e;
}
/* Remove Streamlit's default inner padding — our HTML blocks own the spacing */
section[data-testid="stSidebar"] > div:first-child { padding: 0 0 0 0 !important; }
section[data-testid="stSidebar"] * { color: #e5e5ea !important; }
section[data-testid="stSidebar"] strong { color: #ffffff !important; }

/* ── Pure-HTML sidebar blocks ────────────────────────────────── */
.sb-header {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 16px 14px 12px;
    border-bottom: 1px solid #2c2c2e;
}
.sb-logo {
    width: 34px; height: 34px;
    background: #fff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden;
}
.sb-logo img { width: 100%; height: 100%; object-fit: cover; }
.sb-brand-name  { font-size: 13px; font-weight: 700; color: #fff !important; line-height: 1.2; }
.sb-brand-sub   { font-size: 10px; color: #6e6e73 !important; margin-top: 1px; }

.sb-status {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 14px;
    background: #141416;
    border-bottom: 1px solid #2c2c2e;
    font-size: 11px; color: #8e8e93 !important;
}
.sb-status .dot { width: 6px; height: 6px; background: #34c759; border-radius: 50%; flex-shrink: 0; }
.sb-status .prov { margin-left: auto; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px; color: #6e6e73 !important; }

.sb-session-info {
    margin: 0 12px 0;
    background: #141416;
    border: 1px solid #2c2c2e;
    border-radius: 8px;
    padding: 8px 11px;
    font-size: 10.5px; color: #6e6e73 !important; line-height: 1.55;
}
.sb-session-info code { font-family: monospace; color: #8e8e93 !important;
    background: transparent !important; padding: 0; }

.sb-section-label {
    padding: 12px 14px 5px;
    font-size: 9.5px; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase;
    color: #48484a !important;
}

.sb-footer {
    padding: 10px 14px;
    border-top: 1px solid #2c2c2e;
    font-size: 10px; color: #3a3a3c !important;
    text-align: center; letter-spacing: 0.3px;
}

/* ── New Chat — primary button ────────────────────────────────── */
/* Target by kind attribute that Streamlit sets on primary buttons */
section[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background: #ffffff !important;
    color: #1c1c1e !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 12.5px !important;
    padding: 9px 14px !important;
    width: 100% !important;
    text-align: center !important;
    transition: background .15s;
}
section[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    background: #e5e5ea !important;
}

/* ── Nav buttons (history + quick links) — secondary default ──── */
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    color: #aeaeb2 !important;
    text-align: left !important;
    font-size: 11.5px !important;
    font-weight: 400 !important;
    padding: 7px 10px !important;
    border-radius: 6px !important;
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: background .12s, color .12s;
}
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: #2c2c2e !important;
    color: #ffffff !important;
}

/* Strip the scrollable container border & default padding */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
    padding: 0 4px !important;
}
section[data-testid="stSidebar"] [data-testid="element-container"] {
    margin: 0 !important; padding: 0 !important;
}

.stButton > button {
    background-color : transparent;
}
.st-emotion-cache-1kre5gu{
    background-color :  rgb(255, 75, 75) !important;
    border : 1px solid rgb(255, 75, 75) !important;
}

section[data-testid="stSidebar"] .stButton button {
    white-space: nowrap !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] .stButton button p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100%;
}

.st-emotion-cache-1lads1q{
    justify-content: left;
}

/* ══ HEADER BANNER ════════════════════════════════════════════════ */

.it-banner {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.it-banner-icon {
    width: 36px; height: 36px;
    background: #1c1c1e;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: -.5px;
    flex-shrink: 0;
}
.it-banner h2 { color: #1c1c1e; margin: 0; font-size: 1.05rem; font-weight: 700; }
.it-banner p  { color: #6e6e73; margin: 1px 0 0; font-size: 0.77rem; }

/* ══ TABS ════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.83rem;
    font-weight: 500;
    color: #6e6e73;
    border-bottom: 2px solid transparent;
    padding: 6px 14px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1c1c1e !important;
    border-bottom-color: #1c1c1e !important;
    font-weight: 700;
}

.st-emotion-cache-qksclw[data-selected] .react-aria-SelectionIndicator {
    background-color: transparent !important;
}


/* ══ CHAT SCROLL AREA ═════════════════════════════════════════════ */
.chat-scroll-area {
    height: 500px;
    overflow-y: auto;
    background: #f4f5f7;
    border: 1px solid #e5e5ea;
    border-radius: 12px 12px 0 0;
    padding: 16px 14px 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    scroll-behavior: smooth;
    margin:30px;
}
.chat-scroll-area::-webkit-scrollbar { width: 5px; }
.chat-scroll-area::-webkit-scrollbar-thumb { background: #d1d1d6; border-radius: 4px; }

/* ══ MESSAGE ROWS ═════════════════════════════════════════════════ */
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 8px;
}
.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-end;
    gap: 8px;
}

/* ══ AVATARS ══════════════════════════════════════════════════════ */
.avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: -.3px;
    flex-shrink: 0;
    text-transform: uppercase;
}
.avatar-user { background: #1c1c1e; color: #ffffff; }
.avatar-bot  { background: #e5e5ea; color: #1c1c1e; border: 1px solid #d1d1d6; }

/* ══ BUBBLES ══════════════════════════════════════════════════════ */
.bubble-user {
    background: #1c1c1e;
    color: #f5f5f7;
    border-radius: 16px 4px 16px 16px;
    padding: 10px 14px;
    word-wrap: break-word;
    font-size: 0.875rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,.18);
}
.bubble-bot {
    background: #ffffff;
    color: #1c1c1e;
    border: 1px solid #e5e5ea;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px;
    word-wrap: break-word;
    font-size: 0.875rem;
    line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.bubble-meta-user { text-align: right; color: #8e8e93; font-size: 0.65rem; margin-top: 4px; }
.bubble-meta-bot  { color: #8e8e93; font-size: 0.65rem; margin-top: 4px; }

/* ══ CHAT INPUT BAR ═══════════════════════════════════════════════ */
[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid #e5e5ea !important;
    padding: 10px 14px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,.06) !important;
    margin: 0px 30px;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #8e8e93 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.10) !important;
}

.st-emotion-cache-184dg47:hover{
    background-color : gray !important;
}
.st-emotion-cache-184dg47 {
    background-color : lightgray !important;
}

/* ══ TYPING INDICATOR ═════════════════════════════════════════════ */
.typing-indicator { display: flex; gap: 4px; margin-top: 6px; }
.typing-indicator span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #8e8e93;
    animation: typing-bounce 1.3s infinite ease-in-out;
}
.typing-indicator span:nth-child(2) { animation-delay: .18s; }
.typing-indicator span:nth-child(3) { animation-delay: .36s; }
@keyframes typing-bounce {
    0%,80%,100% { transform: translateY(0); opacity: .4; }
    40% { transform: translateY(-5px); opacity: 1; }
}

/* ══ BADGE SYSTEM ════════════════════════════════════════════════ */
.badge {
    display: inline-flex; align-items: center;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.71rem; font-weight: 600;
    letter-spacing: .01em;
}
.b-open       { background: #f4f5f7; color: #3a3a3c; border: 1px solid #d1d1d6; }
.b-inprogress { background: #2c2c2e; color: #f5f5f7; }
.b-resolved   { background: #1c1c1e; color: #ffffff; }
.b-high       { background: #1c1c1e; color: #ffffff; }
.b-medium     { background: #48484a; color: #ffffff; }
.b-low        { background: #f4f5f7; color: #3a3a3c; border: 1px solid #d1d1d6; }

/* ══ ONLINE DOT ═══════════════════════════════════════════════════ */
.online-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #34c759;
    border-radius: 50%;
    margin-right: 5px;
}

/* ══ CARDS & EXPANDERS ════════════════════════════════════════════ */
.ticket-card, .section-card {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.ticket-card:hover { border-color: #8e8e93; }

/* ══ DATAFRAME ════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e5ea;
    border-radius: 10px;
    overflow: hidden;
}

/* ══ FORM / INPUT FIELDS ═════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    border-color: #d1d1d6 !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #8e8e93 !important;
    box-shadow: none !important;
}

/* ══ PRIMARY BUTTON ══════════════════════════════════════════════ */
[data-testid="baseButton-primary"] {
    background: #1c1c1e !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: .01em !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #3a3a3c !important;
}
.stButton > button {
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    transition: .15s ease;
}

button[data-testid="stBaseButton-primary"] > div {
    justify-content: center !important;
}

/* ══ METRICS ════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #f4f5f7;
    border: 1px solid #e5e5ea;
    border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #6e6e73 !important; }
[data-testid="stMetricValue"] { font-size: 1rem !important; font-weight: 700 !important; color: #1c1c1e !important; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Backend — cached once per process
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔧 Starting IT Help Desk engine…")
def _init_backend():
    init_db()
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    emb   = create_embeddings(provider)
    vs    = init_knowledge_base(emb, collection_name=f"it_knowledge_base_{provider}")
    set_vectorstore(vs)
    tools = get_all_tools()
    llm   = create_llm(provider)
    graph = build_graph(llm, tools)
    return graph, vs, provider


graph, vectorstore, PROVIDER = _init_backend()


# ══════════════════════════════════════════════════════════════════════════════
# Session persistence
# ══════════════════════════════════════════════════════════════════════════════
def _load_sessions() -> list:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_sessions(sessions: list) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(
        json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _persist_session() -> None:
    if not st.session_state.get("messages"):
        return
    sessions = _load_sessions()
    sid      = st.session_state.session_id
    first    = st.session_state.messages[0]["content"]
    record   = {
        "id":        sid,
        "started":   st.session_state.session_started,
        "title":     first[:58] + ("…" if len(first) > 58 else ""),
        "global_id": st.session_state.get("chat_gid", ""),
        "msg_count": len(st.session_state.messages),
        "messages":  st.session_state.messages,
    }
    idx = next((i for i, s in enumerate(sessions) if s["id"] == sid), None)
    if idx is not None:
        sessions[idx] = record
    else:
        sessions.insert(0, record)
    _save_sessions(sessions[:50])


def _start_new_session() -> None:
    _persist_session()
    st.session_state.chatbot.reset()
    st.session_state.messages        = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am an IT Help Desk Assistant. How can I help you today?",
            "time": datetime.now().strftime("%H:%M")
        }
    ]
    st.session_state.session_id      = str(uuid.uuid4())[:8]
    st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_gid        = ""
    st.session_state.pending_input   = None


def _restore_session(record: dict) -> None:
    _persist_session()
    st.session_state.chatbot.reset()
    st.session_state.messages        = record["messages"]
    st.session_state.session_id      = record["id"]
    st.session_state.session_started = record["started"]
    st.session_state.chat_gid        = record.get("global_id", "")
    st.session_state.pending_input   = None


# ══════════════════════════════════════════════════════════════════════════════
# Session state bootstrap
# ══════════════════════════════════════════════════════════════════════════════
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ITHelpdeskChatbot(graph)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
        "role": "assistant",
        "content": "👋 Hello! I am an IT Help Desk Assistant. How can I help you today?",
        "time": datetime.now().strftime("%H:%M")
        }
    ]
    st.session_state.session_id      = str(uuid.uuid4())[:8]
    st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_gid        = ""
    st.session_state.pending_input   = None

    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message = None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _send_chat(user_input: str) -> str:
    """Send a message, update state, persist session, return reply."""
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "time": ts})
    try:
        reply = st.session_state.chatbot.chat(user_input)
        g_state = st.session_state.chatbot._graph.get_state(
            st.session_state.chatbot._config
        )
        if g_state and g_state.values.get("global_id"):
            st.session_state.chat_gid = g_state.values["global_id"]
    except Exception as exc:
        reply = f"⚠️ I encountered an error: {exc}"
    ts2 = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "assistant", "content": reply, "time": ts2})
    _persist_session()
    return reply


def _get_kb_articles() -> list:
    try:
        return json.loads(KB_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF file given its raw bytes."""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")


def _add_kb_article(title: str, category: str, content: str, keywords: list) -> bool:
    try:
        articles  = _get_kb_articles()
        new_id    = f"kb{len(articles) + 1:03d}"
        articles.append({"id": new_id, "title": title, "category": category,
                          "content": content, "keywords": keywords})
        KB_JSON.write_text(json.dumps(articles, indent=2, ensure_ascii=False), encoding="utf-8")
        from langchain_core.documents import Document
        vectorstore.add_documents([
            Document(
                page_content=f"Title: {title}\nCategory: {category}\n\n{content}",
                metadata={"id": new_id, "title": title, "category": category,
                          "keywords": ", ".join(keywords)},
            )
        ])
        return True
    except Exception as e:
        st.error(f"Failed to add article: {e}")
        return False


def _status_badge(status: str) -> str:
    cls = {"Open": "b-open", "In Progress": "b-inprogress", "Resolved": "b-resolved"}.get(status, "b-open")
    return f'<span class="badge {cls}">{status}</span>'


def _priority_badge(priority: str) -> str:
    cls = {"High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(priority, "b-medium")
    return f'<span class="badge {cls}">{priority}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand header — pure HTML, no widgets
    _logo_path = ROOT / "Images" / "technical-support.png"
    if _logo_path.exists():
        _lb64 = base64.b64encode(_logo_path.read_bytes()).decode()
        _logo_html = f'<img src="data:image/png;base64,{_lb64}" style="width:100%;height:100%;object-fit:cover;"/>'
    else:
        _logo_html = "IT"
    st.markdown(f"""
    <div class="sb-header">
        <div class="sb-logo">{_logo_html}</div>
        <div>
            <div class="sb-brand-name">IT Help Desk</div>
            <div class="sb-brand-sub">AI Support Assistant</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Status strip — pure HTML
    _gid_span = f'&thinsp;&middot;&thinsp;<span style="color:#d1d1d6;font-weight:600">{st.session_state.chat_gid}</span>' \
                if st.session_state.chat_gid else ""
    st.markdown(f"""
    <div class="sb-status">
        <span class="dot"></span>
        <span>Online{_gid_span}</span>
        <span class="prov">{PROVIDER}</span>
    </div>""", unsafe_allow_html=True)

    # New Chat — native PRIMARY button (styled via data-testid="baseButton-primary")
    st.markdown('<div style="padding:12px 12px 4px;">', unsafe_allow_html=True)
    if st.button("+ New Chat", use_container_width=True, key="btn_new_sess", type="primary"):
        _start_new_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Session info pill — pure HTML
    st.markdown(f"""
    <div class="sb-session-info">
        <code>{st.session_state.session_id}</code>
        &nbsp;&middot;&nbsp;{st.session_state.session_started}
    </div>""", unsafe_allow_html=True)

    # Chat History
    st.markdown('<div class="sb-section-label">Chat History</div>', unsafe_allow_html=True)
    _sessions = _load_sessions()
    if not _sessions:
        st.markdown('<p style="padding:4px 14px 8px;font-size:11px;color:#48484a;margin:0">No sessions yet</p>',
                    unsafe_allow_html=True)
    else:
        # Use native scrollable container — buttons styled via baseButton-secondary
        with st.container(height=200, border=False):
            for s in _sessions[:50]:
                if st.button(s.get("title", "Session")[:44],
                             key=f"h_{s['id']}", use_container_width=True):
                    _restore_session(s)
                    st.rerun()

    # Quick Help
    st.markdown('<div class="sb-section-label" style="margin-top:4px">Quick Help</div>',
                unsafe_allow_html=True)
    with st.container(height=195, border=False):
        for art in _get_kb_articles()[:8]:
            if st.button(art['title'], key=f"qk_{art['id']}", use_container_width=True):
                st.session_state.pending_input = f"How do I: {art['title']}?"
                st.rerun()

    # Footer — pure HTML, pushed down with flex spacer
    st.markdown('<div class="sb-footer">IT Help Desk v2.0 &thinsp;&middot;&thinsp; LangGraph + FastMCP</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="it-banner">
    <div class="it-banner-icon">IT</div>
    <div>
        <h2>IT Help Desk Assistant</h2>
        <p>AI-powered support &nbsp;&middot;&nbsp; Ticket Management &nbsp;&middot;&nbsp; Knowledge Base</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_tickets, tab_kb, tab_create = st.tabs([
    "Chat Assistant",
    "My Tickets",
    "Knowledge Base",
    "Create Ticket",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Process Quick Help pending input (fires before rendering)
    if st.session_state.pending_input:
        pi = st.session_state.pending_input
        st.session_state.pending_input = None
        with st.spinner("🤔 Thinking…"):
            _send_chat(pi)
        st.rerun()

    # ── Build HTML message history ────────────────────────────────────────────
    bubbles_html = []
    for msg in st.session_state.messages:
        content = msg["content"].replace("\n", "<br>")
        time    = msg.get("time", "")
        if msg["role"] == "user":
            bubbles_html.append(f"""
            <div class="msg-row-user">
            <div>
                <div class="bubble-user">{content}</div>
                <div class="bubble-meta-user">🕐 {time}</div>
            </div>
           <div class="avatar avatar-user" style="overflow:hidden;padding:0;">{USER_AVATAR_HTML}</div>
            </div>""")
        else:
            bubbles_html.append(f"""
            <div class="msg-row-bot">
            <div class="avatar avatar-bot" style="overflow:hidden;padding:0;">{BOT_AVATAR_HTML}</div>
            <div>
                <div class="bubble-bot">{content}</div>
                <div class="bubble-meta-bot">🕐 {time}</div>
            </div>
            </div>""")

    if st.session_state.processing:
        bubbles_html.append("""
        <div class="msg-row-bot">
            <div class="avatar avatar-bot">🤖</div>
            <div>
                <div class="bubble-bot">
                    Thinking...
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        </div>
    """)
    # Render scrollable chat window — JS scrolls to bottom on every render
    st.markdown(
        f'<div class="chat-scroll-area" id="chat-box">'
        + "".join(bubbles_html)
        + """</div>
        <script>
          var box = document.getElementById('chat-box');
          if(box) box.scrollTop = box.scrollHeight;
        </script>""",
        unsafe_allow_html=True,
    )

    if (
    st.session_state.processing
    and st.session_state.pending_user_message
    ):
        try:
            reply = st.session_state.chatbot.chat(
                st.session_state.pending_user_message
            )

            g_state = st.session_state.chatbot._graph.get_state(
                st.session_state.chatbot._config
            )

            if g_state and g_state.values.get("global_id"):
                st.session_state.chat_gid = g_state.values["global_id"]

        except Exception as exc:
            reply = f"⚠️ Error: {exc}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "time": datetime.now().strftime("%H:%M")
            }
        )

        st.session_state.processing = False
        st.session_state.pending_user_message = None

        _persist_session()

        st.rerun()

    # ── Fixed-bottom chat input (st.chat_input always sticks to page bottom) ──
    if user_input := st.chat_input(
    "💬 Type your message… e.g. 'My VPN is not working'"
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "time": datetime.now().strftime("%H:%M"),
            }
        )

        st.session_state.pending_user_message = user_input
        st.session_state.processing = True

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY TICKETS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tickets:
    st.subheader("My Tickets")

    col_id, col_btn = st.columns([3, 1])
    with col_id:
        gid_input = st.text_input(
            "Company Global ID:",
            value=st.session_state.chat_gid,
            placeholder="e.g. GID001",
            key="ticket_gid_input",
        )
    with col_btn:
        st.write(""); st.write("")
        do_lookup = st.button("🔍 Look Up Tickets", use_container_width=True)

    if do_lookup and gid_input:
        gid  = gid_input.strip().upper()
        user = get_user(gid)
        if not user:
            st.error(f"❌ No employee found with Global ID **{gid}**. Please verify the ID.")
        else:
            st.success(
                f" **{user['name']}** &nbsp;·&nbsp; {user['department']} "
                f"&nbsp;·&nbsp; {user['email']}"
            )

            status_opts = ["All", "Open", "In Progress", "Resolved"]
            col_f1, col_f2 = st.columns([2, 4])
            with col_f1:
                sf = st.selectbox("Filter by status:", status_opts, key="sf")
            tickets = db_get_tickets(gid, sf if sf != "All" else None)

            if not tickets:
                st.info("No tickets found.")
            else:
                st.markdown(f"**{len(tickets)} ticket(s)**")

                # Summary table
                rows = []
                for t in tickets:
                    rows.append({
                        "Ticket ID":  t["ticket_id"],
                        "Title":      t["title"][:50],
                        "Status":     t["status"],
                        "Priority":   t["priority"],
                        "Category":   t["category"],
                        "Created":    t["created_at"][:10],
                    })
                df = pd.DataFrame(rows)

                def color_status(val):
                    color_map = {"Open": "#f4f5f7", "In Progress": "#e5e5ea", "Resolved": "#d1d1d6"}
                    return f"background-color: {color_map.get(val, 'white')}; color: #1c1c1e"

                def color_priority(val):
                    color_map = {"High": "#2c2c2e", "Medium": "#48484a", "Low": "#f4f5f7"}
                    text_map  = {"High": "#ffffff",  "Medium": "#ffffff",  "Low": "#3a3a3c"}
                    return f"background-color: {color_map.get(val, 'white')}; color: {text_map.get(val, '#1c1c1e')}"

                styled = df.style.map(color_status, subset=["Status"]) \
                                 .map(color_priority, subset=["Priority"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Detailed expandable view
                st.markdown("#### Ticket Details")
                for t in tickets:
                    s_icon = {"Open": "○", "In Progress": "◑", "Resolved": "●"}.get(t["status"], "○")
                    p_icon = {"High": "▲", "Medium": "–", "Low": "▼"}.get(t["priority"], "–")
                    with st.expander(f"{s_icon} **[{t['ticket_id']}]** {t['title']}"):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Status",   t["status"])
                        m2.metric("Priority", t["priority"])
                        m3.metric("Category", t["category"])
                        m4.metric("Created",  t["created_at"][:10])
                        st.markdown("**Description:**")
                        st.info(t["description"])
                        if t.get("resolution"):
                            st.markdown("**Resolution:**")
                            st.success(t["resolution"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.subheader("Knowledge Base")

    browse_col, add_col = st.columns([3, 2], gap="large")

    # ── Browse ────────────────────────────────────────────────────────────────
    with browse_col:
        st.markdown("#### Browse & Search Articles")
        all_arts = _get_kb_articles()

        s_col, c_col = st.columns([3, 2])
        with s_col:
            kw = st.text_input("🔍 Search:", placeholder="VPN, password, printer…", key="kb_search")
        with c_col:
            cats_avail = ["All"] + sorted({a["category"] for a in all_arts})
            cat_f = st.selectbox("Category:", cats_avail, key="kb_cat")

        filtered = all_arts
        if kw:
            q = kw.lower()
            filtered = [a for a in filtered
                        if q in a["title"].lower() or q in a["content"].lower()
                        or any(q in k for k in a.get("keywords", []))]
        if cat_f != "All":
            filtered = [a for a in filtered if a["category"] == cat_f]

        st.caption(f"{len(filtered)} article(s) found")
        for art in filtered:
            with st.expander(f"📄 **{art['title']}**  —  *{art['category']}*"):
                st.markdown(art["content"])
                if art.get("keywords"):
                    st.caption("🔖 Keywords: " + "  ·  ".join(art["keywords"]))
                if st.button("Ask about this in Chat",
                             key=f"chat_art_{art['id']}",
                             use_container_width=True):
                    st.session_state.pending_input = (
                        f"I need help with: {art['title']}. Can you walk me through the steps?"
                    )
                    st.toast("✅ Message sent — switch to Chat Assistant tab", icon="💬")
                    st.rerun()

    # ── Add new article ───────────────────────────────────────────────────────
    with add_col:
        st.markdown("#### Add New Article")

        # ── PDF Upload ────────────────────────────────────────────────────────
        with st.expander("Upload PDF document", expanded=False):
            uploaded_pdf = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                key="pdf_uploader",
                label_visibility="collapsed",
            )
            if uploaded_pdf is not None:
                try:
                    raw_text = _extract_pdf_text(uploaded_pdf.read())
                    if not raw_text:
                        st.warning("No readable text found in this PDF. Try a text-based PDF.")
                    else:
                        default_title = Path(uploaded_pdf.name).stem.replace("_", " ").replace("-", " ").title()
                        st.success(f"{len(raw_text):,} characters extracted from {uploaded_pdf.name}")

                        pdf_title = st.text_input("Title *", value=default_title, key="pdf_title")
                        pdf_cat   = st.selectbox("Category *", KB_CATEGORIES, key="pdf_cat")
                        pdf_kw    = st.text_input(
                            "Keywords (comma-separated)",
                            placeholder="vpn, network, troubleshoot",
                            key="pdf_kw",
                        )
                        with st.expander("Preview extracted text"):
                            st.text_area(
                                "First 1 500 characters",
                                value=raw_text[:1500],
                                height=160,
                                disabled=True,
                                label_visibility="collapsed",
                            )

                        if st.button("Publish PDF to Knowledge Base", type="primary",
                                     use_container_width=True, key="pub_pdf"):
                            if not pdf_title.strip():
                                st.error("Title is required.")
                            else:
                                kw_list = [k.strip() for k in pdf_kw.split(",") if k.strip()]
                                if _add_kb_article(pdf_title.strip(), pdf_cat, raw_text, kw_list):
                                    st.success(f"✅ '{pdf_title}' added from PDF!")
                                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")

        st.divider()
        st.markdown("Or add manually")
        with st.form("add_kb_form", clear_on_submit=True):
            new_title    = st.text_input("Title *", placeholder="e.g. How to configure 2FA")
            new_category = st.selectbox("Category *", KB_CATEGORIES, key="new_kb_cat")
            new_content  = st.text_area(
                "Content * (step-by-step guide)",
                height=220,
                placeholder="1. Open Settings…\n2. Navigate to Security…\n3. Click Enable MFA…",
            )
            new_keywords = st.text_input(
                "Keywords (comma-separated)",
                placeholder="2fa, mfa, authentication, login",
            )
            submitted = st.form_submit_button(
                "Publish to Knowledge Base",
                use_container_width=True,
                type="primary",
            )
            if submitted:
                if not new_title.strip() or not new_content.strip():
                    st.error("Title and Content are required.")
                else:
                    kw_list = [k.strip() for k in new_keywords.split(",") if k.strip()]
                    if _add_kb_article(new_title.strip(), new_category, new_content.strip(), kw_list):
                        st.success(f"✅ **'{new_title}'** added to the Knowledge Base!")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CREATE TICKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_create:
    st.subheader("Create IT Support Ticket")
    st.markdown("Manually log a new support request. The system will assign a ticket ID automatically.")

    form_col, tips_col = st.columns([3, 2], gap="large")

    with form_col:
        with st.form("create_ticket_form", clear_on_submit=True):
            ct_gid  = st.text_input(
                "Company Global ID *",
                value=st.session_state.chat_gid,
                placeholder="e.g. GID001",
            )
            ct_title = st.text_input("Issue Title *", placeholder="Brief summary of the problem")
            ct_desc  = st.text_area(
                "Detailed Description *",
                height=130,
                placeholder="Describe the issue, any error messages, when it started…",
            )
            cc, pc = st.columns(2)
            with cc:
                ct_cat = st.selectbox("Category *", CATEGORIES)
            with pc:
                ct_pri = st.selectbox("Priority *", PRIORITIES, index=1)

            submit_btn = st.form_submit_button(
                "Submit Ticket", use_container_width=True, type="primary"
            )

            if submit_btn:
                errors = []
                if not ct_gid.strip():    errors.append("Global ID is required.")
                if not ct_title.strip():  errors.append("Issue Title is required.")
                if not ct_desc.strip():   errors.append("Description is required.")
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    user = get_user(ct_gid.strip().upper())
                    if not user:
                        st.error(f"❌ No employee found with Global ID **{ct_gid}**.")
                    else:
                        ticket = db_create_ticket(
                            ct_gid.strip().upper(),
                            ct_title.strip(),
                            ct_desc.strip(),
                            ct_cat,
                            ct_pri,
                        )
                        st.success(f"✅ Ticket **{ticket['ticket_id']}** created successfully!")
                        st.markdown(f"""
| Field | Value |
|-------|-------|
| **Ticket ID** | `{ticket['ticket_id']}` |
| **Raised by** | {user['name']} ({user['global_id']}) |
| **Title** | {ticket['title']} |
| **Category** | {ticket['category']} |
| **Priority** | {ticket['priority']} |
| **Status** | {ticket['status']} |
| **Created** | {ticket['created_at']} |
""")
                        st.info(
                            f"📧 Email updates will be sent to **{user['email']}**. "
                            "The IT team will contact you within 1 business day."
                        )

    with tips_col:
        st.markdown("#### 💡 Priority Guide")
        st.markdown("""
| Priority | When to use |
|----------|-------------|
| 🔴 **High** | Work completely blocked, no workaround exists |
| 🟡 **Medium** | Impacting productivity, partial workaround |
| 🟢 **Low** | Minor inconvenience, full workaround available |

#### 📋 Category Guide
| Category | Examples |
|----------|---------|
| Network | VPN, WiFi, internet, DNS |
| Email | Outlook, Exchange, calendar |
| Access | Passwords, SSO, MFA, permissions |
| Hardware | Laptop, monitor, keyboard |
| Software | Office, Teams, installation |
| Printer | Printing, scanning |
| Security | Phishing, malware, data loss |
""")
        st.info(
            "💡 **Tip:** Use the **Chat Assistant** tab for guided troubleshooting — "
            "the AI checks for duplicate tickets automatically before creating a new one."
        )
