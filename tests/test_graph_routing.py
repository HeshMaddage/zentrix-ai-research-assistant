"""
Day 2 smoke-test suite.

"""

from __future__ import annotations

import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage

from agent.graph import build_graph, get_initial_state
from memory.chroma_manager import ChromaMemoryManager
from models.research_note import ResearchNote

#  Test paths (isolated from production data) 

TEST_CHROMA_PATH = "./chroma_data_test_d2"
TEST_SESSIONS_DB = "./sessions_test_d2.db"

# Seed data (same topics as Day 1) 

SEED_NOTES = [
    ResearchNote.create(
        topic="quantum computing 2024",
        summary=(
            "In 2024, quantum computing saw major advances in error correction. "
            "Google's Willow chip demonstrated below-threshold error rates. "
            "IBM reached 1,000+ qubit systems with improved coherence times."
        ),
        key_facts=["Google Willow below-threshold error correction", "IBM 1000+ qubits"],
        sources=["https://nature.com/quantum"],
        confidence=0.88,
    ),
    ResearchNote.create(
        topic="climate policy 2024",
        summary=(
            "Global climate policy in 2024 was shaped by COP29 outcomes and "
            "record solar deployment of 450 GW. The EU Carbon Border Adjustment "
            "Mechanism entered its transitional phase."
        ),
        key_facts=["450 GW solar added globally", "EU CBAM transitional phase"],
        sources=["https://iea.org/renewables-2024"],
        confidence=0.82,
    ),
    ResearchNote.create(
        topic="llm fine-tuning techniques",
        summary=(
            "LoRA and QLoRA became the standard for fine-tuning large models on "
            "consumer hardware. DPO displaced RLHF for alignment."
        ),
        key_facts=["LoRA/QLoRA standard for consumer GPU fine-tuning", "DPO > RLHF"],
        sources=["https://arxiv.org/lora"],
        confidence=0.91,
    ),
    ResearchNote.create(
        topic="rust programming language",
        summary=(
            "Rust continued its rise in systems programming, with adoption in the "
            "Linux kernel and major browser engines. Memory safety guarantees remain "
            "its primary selling point over C and C++."
        ),
        key_facts=["Rust in Linux kernel", "Memory safe alternative to C"],
        sources=["https://rustlang.org"],
        confidence=0.85,
    ),
    ResearchNote.create(
        topic="transformer architecture advances",
        summary=(
            "Transformer architectures in 2024 saw innovations including extended "
            "context windows up to 1M tokens, Mixture-of-Experts scaling, and "
            "state-space model hybrids like Mamba."
        ),
        key_facts=["1M token context windows achieved", "MoE and Mamba architectures"],
        sources=["https://arxiv.org/transformers2024"],
        confidence=0.87,
    ),
]

# 5 queries that SHOULD hit memory (semantically close to seeded topics)
MEMORY_QUERIES = [
    "what are the latest quantum computing breakthroughs?",
    "tell me about solar energy and climate change policy",
    "how does LoRA work for fine-tuning language models?",
    "is Rust replacing C in systems programming?",
    "what is the Mixture of Experts architecture in LLMs?",
]

# 5 queries that should NOT hit memory (unknown topics)
WEB_QUERIES = [
    "what is the current state of nuclear fusion reactors?",
    "how does CRISPR gene editing work in 2024?",
    "tell me about the latest developments in autonomous vehicles",
    "what are the best practices for kubernetes security?",
    "explain the recent advances in protein folding prediction",
]

# Queries that should trigger follow-up detection
FOLLOWUP_QUERIES = [
    "you mentioned something about error correction earlier",
    "tell me more about that last point",
    "can you elaborate on what you said?",
    "go back to the first thing you mentioned",
    "what did you mean by that exactly?",
]

# Helpers

PASS = "✓"
FAIL = "✗"

def check(condition: bool, message: str) -> None:
    status = PASS if condition else FAIL
    print(f"  {status} {message}")
    if not condition:
        raise AssertionError(f"FAILED: {message}")


def seed_memory(chroma_path: str) -> ChromaMemoryManager:
    """Initialise ChromaDB and seed with test notes."""
    manager = ChromaMemoryManager(chroma_path=chroma_path)
    manager.initialise()
    manager.clear_all_notes()
    for note in SEED_NOTES:
        manager.save_research_note(note)
    print(f"  Seeded {manager.note_count()} notes into ChromaDB")
    return manager


# Test functions 

def test_graph_compiles() -> object:
    """Graph must compile without raising exceptions."""
    print("\n[Test 1] Graph compilation …")
    graph = build_graph(
        chroma_path=TEST_CHROMA_PATH,
        sessions_db=TEST_SESSIONS_DB,
    )
    check(graph is not None, "build_graph() returned a compiled graph")
    print(f"  {PASS} Graph compiled successfully")
    return graph


def test_sessions_db_created(graph) -> None:
    """sessions.db must exist after first graph invocation."""
    print("\n[Test 2] sessions.db creation (checkpointer) …")

    # Clean up any existing DB
    if os.path.exists(TEST_SESSIONS_DB):
        os.remove(TEST_SESSIONS_DB)

    session_id = str(uuid.uuid4())
    initial_state = get_initial_state(session_id, "hello, test query")

    graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )

    check(os.path.exists(TEST_SESSIONS_DB), f"sessions.db created at '{TEST_SESSIONS_DB}'")
    print(f"  {PASS} sessions.db exists — checkpointing is active")


def test_memory_routing(graph) -> None:
    """Known topic queries should route to answer_from_memory."""
    print(f"\n[Test 3] Memory routing — {len(MEMORY_QUERIES)} known-topic queries …")
    hits = 0
    for query in MEMORY_QUERIES:
        session_id = str(uuid.uuid4())
        result = graph.invoke(
            get_initial_state(session_id, query),
            config={"configurable": {"thread_id": session_id}},
        )
        intent = result.get("intent", "unknown")
        hit = intent == "answer_from_memory"
        if hit:
            hits += 1
        status = PASS if hit else "~"
        print(f"    {status} '{query[:55]}…' → {intent}")

    # Allow up to 1 miss — embeddings are probabilistic and some queries
    # may fall just under the threshold on the first run
    check(hits >= 4, f"At least 4/5 known queries should hit memory, got {hits}/5")
    print(f"  {PASS} {hits}/5 known queries routed to answer_from_memory")


def test_web_routing(graph) -> None:
    """Unknown topic queries should route to research_and_answer."""
    print(f"\n[Test 4] Web routing — {len(WEB_QUERIES)} unknown-topic queries …")
    hits = 0
    for query in WEB_QUERIES:
        session_id = str(uuid.uuid4())
        result = graph.invoke(
            get_initial_state(session_id, query),
            config={"configurable": {"thread_id": session_id}},
        )
        intent = result.get("intent", "unknown")
        hit = intent == "research_and_answer"
        if hit:
            hits += 1
        status = PASS if hit else FAIL
        print(f"    {status} '{query[:55]}…' → {intent}")

    check(hits == 5, f"All 5 unknown queries should route to research_and_answer, got {hits}/5")
    print(f"  {PASS} {hits}/5 unknown queries routed to research_and_answer")


def test_followup_routing(graph) -> None:
    """Follow-up phrases should route to answer_from_context."""
    print(f"\n[Test 5] Follow-up routing — {len(FOLLOWUP_QUERIES)} follow-up queries …")
    hits = 0
    for query in FOLLOWUP_QUERIES:
        session_id = str(uuid.uuid4())
        result = graph.invoke(
            get_initial_state(session_id, query),
            config={"configurable": {"thread_id": session_id}},
        )
        intent = result.get("intent", "unknown")
        hit = intent == "answer_from_context"
        if hit:
            hits += 1
        status = PASS if hit else FAIL
        print(f"    {status} '{query[:55]}' → {intent}")

    check(hits == len(FOLLOWUP_QUERIES),
          f"All follow-up queries should route to answer_from_context, got {hits}/{len(FOLLOWUP_QUERIES)}")
    print(f"  {PASS} {hits}/{len(FOLLOWUP_QUERIES)} follow-up queries correctly detected")


def test_message_accumulation(graph) -> None:
    """
    Messages must accumulate across turns in the same session
    (proves add_messages reducer is working).
    """
    print("\n[Test 6] Message accumulation across turns …")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # Turn 1
    result1 = graph.invoke(get_initial_state(session_id, "what is quantum computing?"), config=config)
    msg_count_1 = len(result1["messages"])

    # Turn 2 — same session, new message only
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="tell me more about qubits")]},
        config=config,
    )
    msg_count_2 = len(result2["messages"])

    check(
        msg_count_2 > msg_count_1,
        f"Messages should grow across turns: turn1={msg_count_1}, turn2={msg_count_2}"
    )
    print(
        f"  {PASS} Messages accumulated: "
        f"turn 1 = {msg_count_1} msg(s), turn 2 = {msg_count_2} msg(s)"
    )


def test_final_answer_populated(graph) -> None:
    """final_answer must be a non-empty string after graph runs."""
    print("\n[Test 7] final_answer is populated …")
    session_id = str(uuid.uuid4())
    result = graph.invoke(
        get_initial_state(session_id, "what is machine learning?"),
        config={"configurable": {"thread_id": session_id}},
    )
    answer = result.get("final_answer", "")
    check(isinstance(answer, str) and len(answer) > 0, "final_answer is a non-empty string")
    print(f"  {PASS} final_answer length: {len(answer)} chars")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Day 2 — AgentState + Graph Routing Smoke Test")
    print("=" * 65)

    # Seed ChromaDB with known topics
    print("\n[Setup] Seeding ChromaDB with test notes …")
    seed_memory(TEST_CHROMA_PATH)

    try:
        graph = test_graph_compiles()
        test_sessions_db_created(graph)
        test_memory_routing(graph)
        test_web_routing(graph)
        test_followup_routing(graph)
        test_message_accumulation(graph)
        test_final_answer_populated(graph)

        print("\n" + "=" * 65)
        print("  ✅ All Day 2 tests passed!")
        print("=" * 65)

    except AssertionError as exc:
        print(f"\n  ❌ {exc}")
        sys.exit(1)

    finally:
        # Clean up test artefacts
        for path in [TEST_CHROMA_PATH, TEST_SESSIONS_DB]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        print("\n  (cleaned up test artefacts)")


if __name__ == "__main__":
    main()