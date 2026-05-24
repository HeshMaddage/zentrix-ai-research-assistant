from __future__ import annotations

import datetime
import gc
import json
import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

st.set_page_config(
    page_title="Zentrix Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading AI models and database…")
def get_graph():
    from agent.graph import build_graph
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_memory():
    from memory.chroma_manager import ChromaMemoryManager
    manager = ChromaMemoryManager()
    manager.initialise()
    return manager


# ── Session state initialisation ─────────────────────────────────────────────

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.datetime.now()
    if "messages" not in st.session_state:
        # Displayed messages — list of {"role": "user"|"assistant", "content": str}
        st.session_state.messages = []
    if "status_text" not in st.session_state:
        st.session_state.status_text = ""
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""
    if "last_memory_hit" not in st.session_state:
        st.session_state.last_memory_hit = False
    if "refresh_requested" not in st.session_state:
        st.session_state.refresh_requested = None   # topic string to refresh


_init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intent_badge(intent: str, memory_hit: bool) -> str:
    if memory_hit:
        return "🧠 Answered from memory"
    elif intent == "research_and_answer":
        return "🌐 Researched from web"
    elif intent == "answer_from_context":
        return "💬 Answered from conversation"
    else:
        return ""


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.80:
        return "#22c55e"   # green
    elif confidence >= 0.65:
        return "#f59e0b"   # amber
    else:
        return "#ef4444"   # red


def _stream_text(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.012)   # ~80 words/sec — fast but visibly animated



def run_graph_with_status(user_query: str, status_placeholder) -> dict:
    from agent.graph import get_initial_state

    graph = get_graph()
    session_id = st.session_state.session_id
    config = {"configurable": {"thread_id": session_id}}

    # Determine if this is the first message or a follow-up
    if len(st.session_state.messages) <= 1:
        # First message — use full initial state
        initial_input = get_initial_state(session_id, user_query)
    else:
        # Subsequent messages — checkpointer restores prior state
        initial_input = {"messages": [HumanMessage(content=user_query)]}

    # Node-name → human-readable status message
    NODE_STATUS = {
        "classify_intent":      "🔍 Searching memory…",
        "retrieve_from_memory": "📚 Reading stored notes…",
        "search_web":           "🌐 Searching the web…",
        "save_to_memory":       "✍️  Synthesising and saving…",
        "generate_answer":      "💬 Generating answer…",
    }

    final_state = {}

    # stream() yields (node_name, state_update) tuples after each node
    for node_name, state_update in graph.stream(
        initial_input,
        config=config,
        stream_mode="updates",
    ):
        status_msg = NODE_STATUS.get(node_name, f"⚙️ Running {node_name}…")
        status_placeholder.markdown(
            f'<div style="color:#6b7280;font-size:0.85rem">{status_msg}</div>',
            unsafe_allow_html=True,
        )
        final_state.update(state_update)

    status_placeholder.empty()
    return final_state



def render_sidebar() -> None:
    """
    Right sidebar showing the agent's growing knowledge base.

    Displays each stored topic with:
      - Topic name
      - Stored date and age
      - Confidence badge (colour-coded)
      - Refresh button (deletes + re-research on next query)
    """
    with st.sidebar:
        st.markdown("## 🧠 Memory Explorer")
        st.caption("Everything this agent has learned and stored.")

        memory = get_memory()

        note_count = memory.note_count()
        chunk_count = memory.chunk_count()

        col1, col2 = st.columns(2)
        col1.metric("Research Notes", note_count)
        col2.metric("Source Chunks", chunk_count)

        st.divider()

        if note_count == 0:
            st.info(
                "No notes stored yet.\n\n"
                "Ask a question and the agent will research it and "
                "save the findings here automatically."
            )
            return

        # Get full note details from ChromaDB
        try:
            raw = memory._notes_collection.get(
                include=["metadatas", "documents"]
            )
        except Exception:
            st.warning("Could not load memory notes.")
            return

        metadatas  = raw.get("metadatas", [])
        documents  = raw.get("documents", [])

        # Sort by timestamp descending (most recent first)
        note_data = sorted(
            zip(metadatas, documents),
            key=lambda x: float(x[0].get("timestamp", 0)),
            reverse=True,
        )

        for meta, doc in note_data:
            topic      = meta.get("topic", "unknown")
            timestamp  = float(meta.get("timestamp", 0))
            confidence = float(meta.get("confidence", 0))
            key_facts_raw = meta.get("key_facts", "[]")

            try:
                key_facts = json.loads(key_facts_raw)
            except Exception:
                key_facts = []

            age_days = (time.time() - timestamp) / 86400
            if age_days < 1:
                age_str = "Today"
            elif age_days < 2:
                age_str = "Yesterday"
            else:
                age_str = f"{int(age_days)}d ago"

            stored_date = datetime.datetime.fromtimestamp(timestamp).strftime("%b %d")
            conf_color = _confidence_color(confidence)
            conf_pct = int(confidence * 100)

            with st.expander(f"**{topic.title()}**", expanded=False):
                # Badge row
                st.markdown(
                    f'<span style="background:{conf_color};color:white;'
                    f'padding:2px 8px;border-radius:12px;font-size:0.75rem">'
                    f'{conf_pct}% confidence</span>&nbsp;&nbsp;'
                    f'<span style="color:#6b7280;font-size:0.75rem">'
                    f'{age_str} · {stored_date}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Summary preview
                st.caption(doc[:200] + "…" if len(doc) > 200 else doc)

                # Key facts
                if key_facts:
                    st.markdown("**Key facts:**")
                    for fact in key_facts[:3]:
                        st.markdown(f"• {fact}")

                # Refresh button
                if st.button(
                    "🔄 Refresh this topic",
                    key=f"refresh_{topic}",
                    help="Deletes this note and re-researches the topic on your next query",
                ):
                    deleted = memory.delete_note(topic)
                    if deleted:
                        st.success(f"Deleted '{topic}'. Ask about it again to refresh.")
                        st.rerun()
                    else:
                        st.error("Could not delete note. Try again.")

        st.divider()
        st.caption(
            f"Session: `{st.session_state.session_id[:8]}…`\n\n"
            f"Started: {st.session_state.session_start.strftime('%H:%M:%S')}"
        )


def render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 Zentrix Research Assistant")
        st.caption(
            "Ask me anything. I search the web, synthesise findings into memory, "
            "and recall them instantly in future sessions."
        )
    with col2:
        st.markdown("")
        st.markdown(
            f"**Session**  \n`{st.session_state.session_id[:12]}…`  \n"
            f"**Started**  \n{st.session_state.session_start.strftime('%H:%M:%S')}"
        )


def render_chat() -> None:
    """Render message history and handle new input."""

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    # Status indicator placeholder (lives below last message, above input)
    status_placeholder = st.empty()

    # Chat input
    user_input = st.chat_input("Ask me to research anything…")
    if not user_input:
        return

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the graph
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer_placeholder.markdown("…")

        try:
            final_state = run_graph_with_status(user_input, status_placeholder)
        except Exception as exc:
            error_msg = (
                f"⚠️ An error occurred: {exc}\n\n"
                "Please check your API keys in `.env` and try again."
            )
            answer_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            return

        answer = final_state.get("final_answer", "I was unable to generate a response.")
        intent = final_state.get("intent", "")
        memory_hit = final_state.get("memory_hit", False)

        # Stream the answer into the placeholder
        streamed = ""
        for chunk in _stream_text(answer):
            streamed += chunk
            answer_placeholder.markdown(streamed + "▌")
        answer_placeholder.markdown(answer)

        # Intent badge
        badge = _intent_badge(intent, memory_hit)
        if badge:
            st.caption(badge)

        # Save to display history
        st.session_state.last_intent = intent
        st.session_state.last_memory_hit = memory_hit
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "badge": badge,
        })


def main() -> None:
    render_sidebar()
    render_header()
    st.divider()
    render_chat()


if __name__ == "__main__":
    main()