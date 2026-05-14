"""
Node ownership map:

                        
   classify_intent - query, intent, retrieved_notes,memory_hit                           
   retrieve_from_memory - retrieved_notes (final selection)    
   search_web - web_results                          
   save_to_memory - new_note                             
   generate_answer - final_answer, messages               
 
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from models.research_note import ResearchNote


class AgentState(TypedDict):
    """
    Shared state object passed between every node in the research agent graph.

    LangGraph calls each node with the current state and merges the returned
    dict back into the state. Fields not returned by a node remain unchanged.
    """

    # Conversation history 
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    intent: str

    # Memory retrieval 
    retrieved_notes: List[ResearchNote]
    memory_hit: bool

    # Web research 
    web_results: Dict

    # Synthesis 
    new_note: Optional[ResearchNote]

    # Output 
    final_answer: str

    # Session identity 
    session_id: str
