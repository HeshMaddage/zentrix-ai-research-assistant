
from __future__ import annotations

import datetime
import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from models.research_note import ResearchNote

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY_MESSAGES: int = 6  


def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in .env.\n"
            "Sign up free at https://console.groq.com"
        )
    return ChatGroq(model=GROQ_MODEL, temperature=0.3)


def _format_note_for_memory_prompt(note: ResearchNote) -> str:
    """
    Format a ResearchNote as a readable attributed block.

    Example output:
      TOPIC: quantum computing 2024
      STORED: 2 days ago (March 20, 2025) · Confidence: 87%
      SUMMARY: ...
      KEY FACTS:
        1. ...
      SOURCES:
        [1] https://...
    """
    age = note.age_days()
    if age < 1:
        age_str = "today"
    elif age < 2:
        age_str = "yesterday"
    else:
        age_str = f"{int(age)} days ago"

    stored_date = datetime.datetime.fromtimestamp(note.timestamp).strftime("%B %d, %Y")
    conf_pct = int(note.confidence * 100)
    facts_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(note.key_facts))
    sources_str = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(note.sources))

    return (
        f"TOPIC: {note.topic}\n"
        f"STORED: {age_str} ({stored_date}) · Confidence: {conf_pct}%\n"
        f"SUMMARY:\n{note.summary}\n"
        f"KEY FACTS:\n{facts_str}\n"
        f"SOURCES:\n{sources_str}"
    )


def _format_note_for_web_prompt(note: ResearchNote) -> str:
    """Format a freshly synthesised note — emphasise recency, cite sources by number."""
    conf_pct = int(note.confidence * 100)
    facts_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(note.key_facts))
    sources_str = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(note.sources))

    return (
        f"TOPIC: {note.topic}\n"
        f"CONFIDENCE: {conf_pct}%\n"
        f"SUMMARY:\n{note.summary}\n"
        f"KEY FACTS:\n{facts_str}\n"
        f"SOURCES (cite by number in your answer):\n{sources_str}"
    )


def _format_history(messages: List[BaseMessage]) -> str:
    """
    Format the N messages before the most recent one as a readable transcript.
    The current query (messages[-1]) is passed separately in the user prompt.
    """
    history = messages[-(MAX_HISTORY_MESSAGES + 1):-1]
    if not history:
        return "(No prior conversation in this session.)"

    lines = []
    for msg in history:
        role = type(msg).__name__.replace("Message", "").upper()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)



_MEMORY_SYSTEM = """\
You are a research assistant with a long-term memory system. You previously \
researched topics and stored detailed notes. You are now drawing on those \
stored notes to answer the user's question.

TONE: Cautious and attributive. Always signal you are drawing from stored memory. \
Use phrases like:
  - "According to my notes from [time]…"
  - "My research notes indicate…"
  - "Based on what I stored [X] days ago…"

RULES:
  - Do NOT invent information beyond what the notes contain.
  - If notes are older than 7 days, add a caveat that information may be outdated \
    and offer to refresh the research.
  - If the notes don't fully answer the question, say so explicitly and suggest \
    the user ask you to search for more information.
  - Write in fluent prose, 3-5 paragraphs. No JSON, no raw bullet lists.
"""

_MEMORY_USER = """\
CONVERSATION HISTORY:
{history}

STORED MEMORY NOTES:
{notes_context}

USER'S QUESTION: {query}

Answer using the stored notes. Attribute your answer to your memory.
"""


def generate_memory_answer(
    query: str,
    notes: List[ResearchNote],
    messages: List[BaseMessage],
) -> str:
    
    llm = _build_llm()
    chain = ChatPromptTemplate.from_messages([
        ("system", _MEMORY_SYSTEM),
        ("user", _MEMORY_USER),
    ]) | llm

    notes_context = "\n\n---\n\n".join(
        _format_note_for_memory_prompt(n) for n in notes
    )

    logger.info(
        f"generate_memory_answer: '{query[:60]}' "
        f"notes={len(notes)} history={len(messages)} msgs"
    )
    response = chain.invoke({
        "query": query,
        "notes_context": notes_context,
        "history": _format_history(messages),
    })
    return response.content


_WEB_SYSTEM = """\
You are a research assistant. You just performed a live web search and \
synthesised the findings into structured notes. Use those notes to answer \
the user's question thoroughly and confidently.

TONE: Confident and direct. You have fresh, current information. Use phrases like:
  - "According to recent sources…"
  - "Research shows that…"
  - "Based on current information…"
  - Cite sources by number: "…as reported by [1]."

STRUCTURE:
  1. Direct answer to the question (1-2 sentences)
  2. Detailed explanation from the notes (2-3 paragraphs)
  3. Confidence level and sources summary (1 sentence)

Write in natural prose. Cite sources by number where specific facts are mentioned. \
Do NOT output JSON.
"""

_WEB_USER = """\
CONVERSATION HISTORY:
{history}

FRESHLY RESEARCHED NOTES:
{note_context}

USER'S QUESTION: {query}

Answer the question using the fresh research notes. Be direct and thorough.
"""


def generate_web_answer(
    query: str,
    note: ResearchNote,
    messages: List[BaseMessage],
) -> str:

    llm = _build_llm()
    chain = ChatPromptTemplate.from_messages([
        ("system", _WEB_SYSTEM),
        ("user", _WEB_USER),
    ]) | llm

    logger.info(
        f"generate_web_answer: '{query[:60]}' "
        f"topic='{note.topic}' confidence={note.confidence:.2f}"
    )
    response = chain.invoke({
        "query": query,
        "note_context": _format_note_for_web_prompt(note),
        "history": _format_history(messages),
    })
    return response.content


_CONTEXT_SYSTEM = """\
You are a research assistant in the middle of a conversation. The user is asking \
a follow-up that references something already discussed — they are NOT requesting \
new research.

Read the conversation history and answer the follow-up based solely on what has \
already been discussed. Do NOT invent information not present in the history.

If the follow-up references something you genuinely did not discuss, say so clearly \
and offer to research it.

TONE: Natural and conversational — this is a follow-up exchange, not a formal \
research report. Be concise: 1-3 paragraphs.
"""

_CONTEXT_USER = """\
CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {query}

Answer based on the conversation above.
"""


def generate_context_answer(
    query: str,
    messages: List[BaseMessage],
) -> str:

    llm = _build_llm()
    chain = ChatPromptTemplate.from_messages([
        ("system", _CONTEXT_SYSTEM),
        ("user", _CONTEXT_USER),
    ]) | llm

    logger.info(f"generate_context_answer: follow-up='{query[:60]}'")
    response = chain.invoke({
        "query": query,
        "history": _format_history(messages),
    })
    return response.content