from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

# Enable INFO logging so node transitions are visible
logging.basicConfig(
    level=logging.WARNING,   # suppress ChromaDB noise
    format="%(name)s | %(message)s",
)
logging.getLogger("agent").setLevel(logging.INFO)
logging.getLogger("prompts").setLevel(logging.INFO)

from langchain_core.messages import HumanMessage

from agent.graph import build_graph, get_initial_state


TEST_CHROMA = "./chroma_data_test_demo"
TEST_DB     = "./sessions_test_demo.db"


PASS = "✓"
FAIL = "✗"

def check(condition: bool, message: str) -> None:
    status = PASS if condition else FAIL
    print(f"    {status} {message}")
    if not condition:
        raise AssertionError(f"FAILED: {message}")

def preview(text: str, length: int = 120) -> str:
    text = text.strip()
    return text[:length] + "…" if len(text) > length else text

def separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def main() -> None:
    print("=" * 60)
    print("  Day 4 — Memory Demonstration Test")
    print("  Core proof: cross-session memory retrieval")
    print("=" * 60)

    # Check API keys
    missing = [k for k in ("TAVILY_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"\n  Missing API keys: {missing}")
        print("  Add them to .env and retry.")
        sys.exit(1)

    # Build graph once — shared between both sessions
    print("\n[Setup] Building graph (loads ChromaDB + embedding model)…")
    graph = build_graph(chroma_path=TEST_CHROMA, sessions_db=TEST_DB)
    print("  Graph ready")

    session_1_id = str(uuid.uuid4())
    session_2_id = str(uuid.uuid4())

    saved_topic = None

    try:
        # ══════════════════════════════════════════════════════════════════════
        # SESSION 1 — Query 1: new topic → web search
        # ══════════════════════════════════════════════════════════════════════
        separator("Session 1 · Query 1 — New topic (expect: research_and_answer)")

        query_1 = "What are the main applications of transformer models in NLP?"
        print(f"  Query: \"{query_1}\"")

        result_1 = graph.invoke(
            get_initial_state(session_1_id, query_1),
            config={"configurable": {"thread_id": session_1_id}},
        )

        intent_1 = result_1.get("intent", "")
        new_note  = result_1.get("new_note")
        answer_1  = result_1.get("final_answer", "")

        check(intent_1 == "research_and_answer", f"Intent: {intent_1}")
        check(new_note is not None, "A ResearchNote was synthesised and saved")
        check(len(answer_1) > 50, "Answer is non-empty")

        if new_note:
            saved_topic = new_note.get("topic") if isinstance(new_note, dict) else new_note.topic
            confidence = new_note.get("confidence", 0.0) if isinstance(new_note, dict) else new_note.confidence
            print(f"    ℹ Note saved: '{saved_topic}' (confidence={confidence:.2f})")

        print(f"    ℹ Answer preview: {preview(answer_1)}")

        # ══════════════════════════════════════════════════════════════════════
        # SESSION 1 — Query 2: follow-up → context
        # ══════════════════════════════════════════════════════════════════════
        separator("Session 1 · Query 2 — Follow-up (expect: answer_from_context)")

        query_2 = "Tell me more about the fine-tuning application you mentioned"
        print(f"  Query: \"{query_2}\"")

        # Same session_id — checkpointer restores state automatically
        result_2 = graph.invoke(
            {"messages": [{"type": "human", "content": query_2}]},
            config={"configurable": {"thread_id": session_1_id}},
        )
        
        #changed using G
        intent_2 = result_2.get("intent", "")
        answer_2 = result_2.get("final_answer", "")

        check(intent_2 == "answer_from_context", f"Intent: {intent_2}")
        check(len(answer_2) > 50, "Answer is non-empty")

        msg_count = len(result_2.get("messages", []))
        check(msg_count >= 3, f"Messages accumulated: {msg_count} (expect ≥ 3 after 2 turns)")

        print(f"    ℹ Answer preview: {preview(answer_2)}")
        print(f"    ℹ Total messages in session: {msg_count}")

        # ══════════════════════════════════════════════════════════════════════
        # SESSION 2 — New session, same topic → memory hit
        # This is the CORE PROOF of the project
        # ══════════════════════════════════════════════════════════════════════
        separator("Session 2 · Query 1 — NEW session, related topic (expect: answer_from_memory)")
        print("  *** This is the CORE PROOF — cross-session memory retrieval ***")

        query_3 = "How are transformers used in language tasks?"
        print(f"  Query: \"{query_3}\"")
        print(f"  Session 1 saved topic: '{saved_topic}'")
        print(f"  Session 2 ID (new):    {session_2_id[:16]}…")

        result_3 = graph.invoke(
            get_initial_state(session_2_id, query_3),
            config={"configurable": {"thread_id": session_2_id}},
        )

        intent_3      = result_3.get("intent", "")
        retrieved      = result_3.get("retrieved_notes", [])
        memory_hit     = result_3.get("memory_hit", False)
        answer_3       = result_3.get("final_answer", "")

        check(intent_3 == "answer_from_memory",
              f"Intent: {intent_3}  ← CORE PROOF (must be answer_from_memory)")
        check(memory_hit is True, "memory_hit=True")
        check(len(retrieved) > 0, f"{len(retrieved)} note(s) retrieved from ChromaDB")
        check(len(answer_3) > 50, "Answer is non-empty")

        if retrieved:
            # print(f"    ℹ Retrieved note: '{retrieved[0].topic}'")
            # print(f"    ℹ Note age: {retrieved[0].age_days():.2f} days")
            # print(f"    ℹ Confidence: {retrieved[0].confidence:.2f}")
            

            first_note = retrieved[0]
            # Handle dictionary lookups safely to satisfy the test printing
            topic_name = first_note.get('topic') if isinstance(first_note, dict) else getattr(first_note, 'topic', 'Unknown')
            conf_score = first_note.get('confidence', 0.0) if isinstance(first_note, dict) else getattr(first_note, 'confidence', 0.0)
            
            print(f"    ℹ Retrieved note: '{topic_name}'")
            print(f"    ℹ Confidence: {conf_score:.2f}")
        print(f"    ℹ Answer preview: {preview(answer_3)}")

        # ══════════════════════════════════════════════════════════════════════
        # Summary
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Session 1 Q1 intent: {intent_1}")
        print(f"  Session 1 Q2 intent: {intent_2}")
        print(f"  Session 2 Q1 intent: {intent_3}  ← CORE PROOF")
        print()
        print("  Memory demonstration COMPLETE")
        print("  Cross-session persistence CONFIRMED")
        print()
        print("  → Screenshot this output and add to your README.")
        print("  → Session 2 returning 'answer_from_memory' proves the")
        print("    pipeline works end-to-end across separate sessions.")
        print("=" * 60)

    except AssertionError as exc:
        print(f"\n  error{exc}")
        sys.exit(1)

    finally:
        # Release ChromaDB handles before cleanup (Windows file lock fix)
        try:
            # Access the memory manager via the graph's node closure
            # Simplest approach: just let GC handle it after setting to None
            del graph
        except Exception:
            pass
        gc.collect()

        for path in [TEST_CHROMA, TEST_DB]:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except PermissionError:
                    print(
                        f"\n  Could not delete '{path}' — "
                        "delete it manually. Test results are unaffected."
                    )

        print("\n  (test artefacts cleaned up)")


if __name__ == "__main__":
    main()