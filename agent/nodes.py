"""
Node execution order (determined by graph edges in graph.py):

  classify_intent          ← Node 1: routing brain
    │
    ├── "answer_from_memory"  ──► retrieve_from_memory  ──► generate_answer
    ├── "answer_from_context" ──────────────────────────► generate_answer
    └── "research_and_answer" ──► search_web ──► save_to_memory ──► generate_answer
                                                                          │
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agent.state import AgentState
from memory.chroma_manager import ChromaMemoryManager
from models.research_note import ResearchNote
from prompts.synthesis_promt import synthesise_note
from tools.web_search import WebSearchResult, search_and_chunk

load_dotenv()
logger = logging.getLogger(__name__)

# Config constants 

SIMILARITY_THRESHOLD: float = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.75"))
MEMORY_FRESHNESS_DAYS: float = float(os.getenv("MEMORY_FRESHNESS_DAYS", "7"))
MEMORY_MIN_CONFIDENCE: float = float(os.getenv("MEMORY_MIN_CONFIDENCE", "0.6"))
ROUTER_TOP_K: int = 3   # how many candidate notes to retrieve for routing

# Phrases that signal a conversational follow-up (no ChromaDB needed)
FOLLOWUP_PATTERNS: List[str] = [
    r"you mentioned",
    r"tell me more",
    r"what did you mean",
    r"elaborate on",
    r"can you expand",
    r"what about the",
    r"go back to",
    r"earlier you said",
    r"the (\w+ )?point you",
    r"that last",
    r"your previous",
]

# Shared memory manager instance
# Instantiated once per process — embedding model is expensive to reload.
# graph.py passes this in via closure (see get_nodes() factory below).

def _is_followup(query: str) -> bool:
    """
    Return True if the query looks like a conversational follow-up 
    """
    q_lower = query.lower().strip()
    return any(re.search(pattern, q_lower) for pattern in FOLLOWUP_PATTERNS)


def _passes_routing_threshold(
    note: ResearchNote,
    similarity: float,
) -> bool:
    """
    Three-condition gate from the build plan:
      1. Similarity score >= SIMILARITY_THRESHOLD
      2. Note age < MEMORY_FRESHNESS_DAYS
      3. Confidence score >= MEMORY_MIN_CONFIDENCE

    All three must pass for memory to be used.
    """
    conditions = {
        "similarity": similarity >= SIMILARITY_THRESHOLD,
        "freshness": note.is_fresh(MEMORY_FRESHNESS_DAYS),
        "confidence": note.is_confident(MEMORY_MIN_CONFIDENCE),
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        logger.debug(
            f"Note '{note.topic}' failed routing conditions: {failed} | "
            f"sim={similarity:.3f}, age={note.age_days():.1f}d, conf={note.confidence:.2f}"
        )
    return all(conditions.values())


# Node factory 

def get_nodes(memory: ChromaMemoryManager) -> Dict:
    """
    Returns a dict of node functions closed over the shared memory manager.

    """

    def classify_intent(state: AgentState) -> dict:
        """
        Node 1: Routing brain of the agent.

        Reads the latest user message, decides whether to answer from:
          - memory (ChromaDB hit, all conditions pass)
          - web search (no good memory hit)
          - current conversation context (conversational follow-up)

        Returns partial state update:
          query, intent, retrieved_notes, memory_hit
        """
        # Extract query from latest message 
        messages = state.get("messages", [])
        if not messages:
            logger.warning("classify_intent: no messages in state, defaulting to clarify")
            return {
                "query": "",
                "intent": "clarify",
                "retrieved_notes": [],
                "memory_hit": False,
            }

        last_message = messages[-1]
        query: str = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        ).strip()

        logger.info(f"classify_intent: query='{query[:80]}{'...' if len(query) > 80 else ''}'")

        # Conversational follow-up check 
        # Check this BEFORE hitting ChromaDB — no embedding needed
        if _is_followup(query):
            logger.info("classify_intent: follow-up detected → 'answer_from_context'")
            return {
                "query": query,
                "intent": "answer_from_context",
                "retrieved_notes": state.get("retrieved_notes", []),  # keep existing
                "memory_hit": False,
            }

        #  Semantic memory search 

        candidate_notes: List[ResearchNote] = memory.search_memory(
            query=query,
            top_k=ROUTER_TOP_K,
            threshold=0.0,          # get all candidates; gate applied below
            min_confidence=0.0,     # gate applied below
            max_age_days=None,      # gate applied below
        )

        passing_notes: List[ResearchNote] = []

        if candidate_notes:
            passing_notes = memory.search_memory(
                query=query,
                top_k=ROUTER_TOP_K,
                threshold=SIMILARITY_THRESHOLD,
                min_confidence=MEMORY_MIN_CONFIDENCE,
                max_age_days=MEMORY_FRESHNESS_DAYS,
            )

        # Route based on memory hit 
        if passing_notes:
            logger.info(
                f"classify_intent: memory hit — {len(passing_notes)} note(s) pass all "
                f"conditions → 'answer_from_memory'"
            )
            for note in passing_notes:
                logger.debug(f"  · '{note.topic}' conf={note.confidence:.2f} age={note.age_days():.1f}d")

            return {
                "query": query,
                "intent": "answer_from_memory",
                "retrieved_notes": passing_notes,
                "memory_hit": True,
            }
        else:
            logger.info("classify_intent: no memory hit → 'research_and_answer'")
            return {
                "query": query,
                "intent": "research_and_answer",
                "retrieved_notes": [],
                "memory_hit": False,
            }


    def retrieve_from_memory(state: AgentState) -> dict:
        """
        Node 2a: Refine and finalise the memory retrieval.
        """
        notes = state.get("retrieved_notes", [])
        logger.info(
            f"[STUB] retrieve_from_memory: {len(notes)} note(s) available. "
            f"Topics: {[n.topic for n in notes]}"
        )
        # Pass-through — no state change needed (classify_intent did the work)
        return {}


    def search_web(state: AgentState) -> dict:
        """
        Search Tavily, chunk results, and immediately persist raw evidence to
        ChromaDB source_chunks — BEFORE synthesis, so evidence is never lost.
 
        Raw chunks are saved per-URL so that each chunk carries an accurate
        'url' metadata key for source attribution in future retrieval.
 
        State reads:  query, session_id
        State writes: web_results (serialised WebSearchResult dict)
        """
        query: str = state.get("query", "")
        session_id: str = state.get("session_id", "unknown-session")
 
        if not query:
            logger.warning("search_web: empty query — returning empty web_results")
            return {"web_results": {"query": "", "results": [], "chunks": [], "total_chunks": 0}}
 
        logger.info(f"search_web: calling Tavily for '{query}'")
 
        # search and chunk
        web_result: WebSearchResult = search_and_chunk(query)
 
        logger.info(
            f"search_web: {len(web_result.results)} results, "
            f"{web_result.total_chunks} chunks"
        )
 
        # Persist raw chunks to source_chunks BEFORE synthesis 
        # Save per-URL so metadata correctly attributes each chunk.
        chunks_saved_total = 0
        for result in web_result.results:
            url = result.get("url", "unknown")
            title = result.get("title", "")
 
            url_chunk_texts = [
                c.text for c in web_result.chunks
                if c.source_url == url
            ]
 
            if not url_chunk_texts:
                continue
 
            memory.save_source_chunks(
                chunks=url_chunk_texts,
                metadata={
                    "url": url,
                    "title": title,
                    "topic": query,
                    "session_id": session_id,
                },
            )
            chunks_saved_total += len(url_chunk_texts)
 
        logger.info(
            f"search_web: persisted {chunks_saved_total} chunks across "
            f"{len(web_result.results)} URLs to source_chunks"
        )
 
        # Store serialised result in state 
        return {"web_results": web_result.to_state_dict()}

    def save_to_memory(state: AgentState) -> dict:
        """
        Synthesise web results into a ResearchNote via GPT-4o-mini, then save
        to ChromaDB research_notes so the router finds it on future queries.
 
        If synthesis fails for any reason, new_note is set to None and
        generate_answer degrades gracefully rather than crashing.
 
        State reads:  web_results, session_id, query
        State writes: new_note
        """
        raw_web = state.get("web_results", {})
        session_id: str = state.get("session_id", "unknown-session")
        query: str = state.get("query", "")
 
        if not raw_web or not raw_web.get("results"):
            logger.warning("save_to_memory: no web_results — skipping synthesis")
            return {"new_note": None}
 
        # Reconstruct WebSearchResult from state dict
        web_result = WebSearchResult.from_state_dict(raw_web)
 
        logger.info(
            f"save_to_memory: synthesising note for '{query}' "
            f"({web_result.total_chunks} chunks, {len(web_result.results)} sources)"
        )
 
        # Synthesise via LLM 
        try:
            note: ResearchNote = synthesise_note(web_result, session_id=session_id)
        except Exception as exc:
            logger.error(
                f"save_to_memory: synthesis FAILED for '{query}': {exc}\n"
                "Setting new_note=None — generate_answer will degrade gracefully."
            )
            return {"new_note": None}
 
        # Save to ChromaDB research_notes
        try:
            memory.save_research_note(note)
            logger.info(
                f"save_to_memory: saved → topic='{note.topic}' "
                f"confidence={note.confidence:.2f} "
                f"facts={len(note.key_facts)} sources={len(note.sources)}"
            )
        except Exception as exc:
            # Note was synthesised but couldn't be persisted — still answer this turn
            logger.error(
                f"save_to_memory: ChromaDB persist FAILED for '{note.topic}': {exc}. "
                "This turn can still answer but note won't be in memory next session."
            )
 
        return {"new_note": note}

    def generate_answer(state: AgentState) -> dict:
        """
        Node 4: Generate the final answer and append it to messages.

        """
        intent = state.get("intent", "unknown")
        query = state.get("query", "")
        retrieved_notes = state.get("retrieved_notes", [])
        memory_hit = state.get("memory_hit", False)
        web_results = state.get("web_results", {})

        if intent == "answer_from_memory":
            topics = [n.topic for n in retrieved_notes]
            stub_text = (
                f"[STUB] Intent: answer_from_memory ✓\n"
                f"Query: '{query}'\n"
                f"Memory hit: {memory_hit}\n"
                f"Retrieved notes ({len(retrieved_notes)}): {topics}\n\n"
                f"→ This node will call GPT-4o-mini with the retrieved "
                f"note summaries and produce a real answer."
            )
        elif intent == "answer_from_context":
            stub_text = (
                f"[STUB] Intent: answer_from_context\n"
                f"Query: '{query}'\n"
                f"→ Follow-up detected. This will reference prior "
                f"messages in the conversation to answer without hitting ChromaDB."
            )
        elif intent == "research_and_answer":
            result_count = len(web_results.get("results", []))
            stub_text = (
                f"[STUB] Intent: research_and_answer\n"
                f"Query: '{query}'\n"
                f"Web results: {result_count} (stub — web search not yet implemented)\n"
                f"→ This will synthesise web results into a note, "
                f"save it, and generate a full answer."
            )
        else:
            stub_text = (
                f"[STUB] Intent: {intent}\n"
                f"Query: '{query}'\n"
                f"→ Unrecognised intent. Clarification would be requested here."
            )

        logger.info(f"generate_answer: intent={intent}, memory_hit={memory_hit}")

        answer_message = AIMessage(content=stub_text)

        return {
            "final_answer": stub_text,
            "messages": [answer_message],   # add_messages reducer appends this
        }

    # Return all nodes as a dict 
    return {
        "classify_intent": classify_intent,
        "retrieve_from_memory": retrieve_from_memory,
        "search_web": search_web,
        "save_to_memory": save_to_memory,
        "generate_answer": generate_answer,
    }