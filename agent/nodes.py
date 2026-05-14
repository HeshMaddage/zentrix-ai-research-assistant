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
        Node 2b: Search the web using Tavily.

        """
        query = state.get("query", "")
        logger.info(f"[STUB] search_web: would search for '{query}'")

        # Placeholder so downstream nodes don't crash on missing key
        return {
            "web_results": {
                "query": query,
                "results": [],
                "status": "stub — web search not yet implemented",
            }
        }

    def save_to_memory(state: AgentState) -> dict:
        """
        Node 3: Synthesise web results into a ResearchNote and save to ChromaDB.

        """
        query = state.get("query", "")
        web_results = state.get("web_results", {})
        logger.info(
            f"[STUB] save_to_memory: would synthesise and save note for '{query}'. "
            f"Got {len(web_results.get('results', []))} web result(s)."
        )
        return {"new_note": None}

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