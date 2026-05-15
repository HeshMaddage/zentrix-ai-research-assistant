"""
LangGraph StateGraph assembly for the AI Research Assistant.

This file does three things:
  1. Instantiates the shared ChromaMemoryManager (loaded once per process)
  2. Wires all nodes and edges into a StateGraph
  3. Compiles the graph with a SqliteSaver checkpointer for session persistence

"""

from __future__ import annotations

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.nodes import get_nodes
from agent.state import AgentState
from memory.chroma_manager import ChromaMemoryManager

load_dotenv()
logger = logging.getLogger(__name__)

#  Config 

SESSIONS_DB_PATH: str = os.getenv("SESSIONS_DB_PATH", "./sessions.db")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_data")


# Router function 

def route_after_classify(
    state: AgentState,
) -> Literal["retrieve_from_memory", "search_web", "generate_answer"]:
    """
    Conditional edge function called after classify_intent.

    Reads `state["intent"]` and returns the name of the next node.
    LangGraph uses the returned string as the edge destination.

    Intent → next node mapping:
      "answer_from_memory"  → "retrieve_from_memory"
      "research_and_answer" → "search_web"
      "answer_from_context" → "generate_answer"  (skip retrieval entirely)
      anything else         → "generate_answer"  (safe fallback)
    """
    intent = state.get("intent", "")
    logger.info(f"Router: intent='{intent}'")

    if intent == "answer_from_memory":
        return "retrieve_from_memory"
    elif intent == "research_and_answer":
        return "search_web"
    else:
        # "answer_from_context" and "clarify" both go straight to answer
        return "generate_answer"


# Graph builder 

def build_graph(
    chroma_path: str = CHROMA_PATH,
    sessions_db: str = SESSIONS_DB_PATH,
) -> StateGraph:

    #  Initialise shared memory manager 
    logger.info("build_graph: initialising ChromaMemoryManager …")
    memory = ChromaMemoryManager(chroma_path=chroma_path)
    memory.initialise()

    # Get node functions 
    nodes = get_nodes(memory)

    # Build the graph 
    builder = StateGraph(AgentState)

    # Add all five nodes
    builder.add_node("classify_intent",      nodes["classify_intent"])
    builder.add_node("retrieve_from_memory", nodes["retrieve_from_memory"])
    builder.add_node("search_web",           nodes["search_web"])
    builder.add_node("save_to_memory",       nodes["save_to_memory"])
    builder.add_node("generate_answer",      nodes["generate_answer"])

    # Wire edges 
    # Entry point
    builder.add_edge(START, "classify_intent")

    # Conditional routing after classify_intent
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_from_memory": "retrieve_from_memory",
            "search_web":           "search_web",
            "generate_answer":      "generate_answer",
        },
    )

    # Memory path: retrieve → answer
    builder.add_edge("retrieve_from_memory", "generate_answer")

    # Web path: search → save → answer
    builder.add_edge("search_web",     "save_to_memory")
    builder.add_edge("save_to_memory", "generate_answer")

    # All paths end here
    builder.add_edge("generate_answer", END)

    # Attach SqliteSaver checkpointer 
    # SqliteSaver persists the full AgentState to SQLite after every node.
    # Each conversation is stored under its thread_id (session_id).
    # The sessions.db file is created automatically on first run.
    checkpointer = SqliteSaver.from_conn_string(sessions_db)
    logger.info(f"build_graph: checkpointer → '{sessions_db}'")

    # Compile and return
    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("build_graph: graph compiled successfully ✓")
    return compiled


def get_initial_state(session_id: str, user_message: str) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "query": "",               # set by classify_intent
        "intent": "",              # set by classify_intent
        "retrieved_notes": [],     # set by classify_intent or retrieve_from_memory
        "web_results": {},         # set by search_web
        "new_note": None,          # set by save_to_memory
        "final_answer": "",        # set by generate_answer
        "memory_hit": False,       # set by classify_intent
    }