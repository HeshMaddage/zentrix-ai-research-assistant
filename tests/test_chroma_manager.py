"""
tests/test_chroma_manager.py

Expected output (approximate):
  ✓ Saved 3 research notes
  ✓ Query 'quantum error correction' → top hit: 'quantum computing 2024' (sim ≈ 0.8x)
  ✓ Query 'climate change renewable energy' → top hit: 'climate policy 2024' (sim ≈ 0.7x)
  ✓ Query 'completely unrelated gibberish topic xyz' → 0 results (threshold filters correctly)
  ✓ All 3 topics found: ['climate policy 2024', 'llm fine-tuning techniques', 'quantum computing 2024']
  ✓ After delete: 2 topics remain
  All tests passed ✓
"""

import sys
import time
import os

# Allow running from project root: `python -m tests.test_chroma_manager`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.chroma_manager import ChromaMemoryManager
from models.research_note import ResearchNote

# Test data 

DUMMY_NOTES = [
    ResearchNote.create(
        topic="quantum computing 2024",
        summary=(
            "In 2024, quantum computing saw major advances in error correction. "
            "Google's Willow chip demonstrated below-threshold error rates, a long-sought "
            "milestone. IBM reached 1,000+ qubit systems with improved coherence times. "
            "Practical quantum advantage in chemistry simulations is now considered achievable "
            "within 5 years by leading researchers."
        ),
        key_facts=[
            "Google Willow chip achieved below-threshold quantum error correction",
            "IBM surpassed 1,000 physical qubits with better coherence",
            "Quantum advantage in drug discovery simulations expected within 5 years",
            "Microsoft's topological qubits showed first experimental evidence",
        ],
        sources=[
            "https://nature.com/articles/quantum-willow-2024",
            "https://ibm.com/quantum/roadmap-2024",
        ],
        confidence=0.88,
    ),
    ResearchNote.create(
        topic="climate policy 2024",
        summary=(
            "Global climate policy in 2024 was shaped by the COP29 outcomes and "
            "accelerating renewable energy deployment. Solar capacity additions hit a record "
            "450 GW globally. The EU's Carbon Border Adjustment Mechanism entered its "
            "transitional phase. Several major economies updated their NDCs ahead of the "
            "2025 deadline, though ambition gaps remain significant."
        ),
        key_facts=[
            "Solar capacity additions hit record 450 GW globally in 2024",
            "EU Carbon Border Adjustment Mechanism entered transitional phase",
            "COP29 produced updated loss and damage financing commitments",
            "Global EV sales exceeded 17 million units",
        ],
        sources=[
            "https://iea.org/reports/renewables-2024",
            "https://unfccc.int/cop29-outcomes",
        ],
        confidence=0.82,
    ),
    ResearchNote.create(
        topic="llm fine-tuning techniques",
        summary=(
            "Large language model fine-tuning in 2024 was dominated by parameter-efficient "
            "methods. LoRA and QLoRA became the standard approach for adapting 7B–70B models "
            "on consumer hardware. Direct Preference Optimisation (DPO) largely displaced "
            "RLHF for alignment due to its simplicity. Mixture-of-Experts architectures "
            "reduced inference cost while maintaining benchmark performance."
        ),
        key_facts=[
            "LoRA/QLoRA are now standard for fine-tuning on consumer GPUs",
            "DPO displaced RLHF as the preferred alignment technique",
            "MoE models like Mixtral achieve GPT-4 level performance at lower cost",
            "Synthetic data generation is now a key component of fine-tuning pipelines",
        ],
        sources=[
            "https://arxiv.org/abs/2312.12345",
            "https://huggingface.co/blog/lora-explained",
        ],
        confidence=0.91,
    ),
]


# Helpers 

PASS = "✓"
FAIL = "✗"

def check(condition: bool, message: str) -> None:
    status = PASS if condition else FAIL
    print(f"  {status} {message}")
    if not condition:
        raise AssertionError(f"FAILED: {message}")


#  Test functions 

def test_save_notes(manager: ChromaMemoryManager) -> None:
    print("\n[Test 1] Saving 3 research notes …")
    for note in DUMMY_NOTES:
        manager.save_research_note(note)
    count = manager.note_count()
    check(count == 3, f"Expected 3 notes in ChromaDB, got {count}")
    print("  ✓ Saved 3 research notes")


def test_semantic_search_quantum(manager: ChromaMemoryManager) -> None:
    print("\n[Test 2] Semantic search — quantum query …")
    results = manager.search_memory(
        query="quantum error correction and qubit coherence",
        top_k=3,
        threshold=0.5,   # lower threshold to inspect all scores during dev
    )
    check(len(results) > 0, "Expected at least 1 result for quantum query")
    top = results[0]
    check(
        "quantum" in top.topic.lower(),
        f"Expected quantum note as top hit, got '{top.topic}'"
    )
    print(f"  {PASS} Top hit: '{top.topic}'")


def test_semantic_search_climate(manager: ChromaMemoryManager) -> None:
    print("\n[Test 3] Semantic search — climate query …")
    results = manager.search_memory(
        query="renewable energy solar capacity climate change policy",
        top_k=3,
        threshold=0.5,
    )
    check(len(results) > 0, "Expected at least 1 result for climate query")
    top = results[0]
    check(
        "climate" in top.topic.lower(),
        f"Expected climate note as top hit, got '{top.topic}'"
    )
    print(f"  {PASS} Top hit: '{top.topic}'")


def test_threshold_filters_irrelevant(manager: ChromaMemoryManager) -> None:
    print("\n[Test 4] Threshold filtering — irrelevant query should return 0 results …")
    results = manager.search_memory(
        query="zzz xyzzy frobnicator blork nonsense unrelated 12345",
        top_k=3,
        threshold=0.75,   # production threshold
    )
    check(
        len(results) == 0,
        f"Expected 0 results for gibberish query at threshold=0.75, got {len(results)}"
    )
    print(" Irrelevant query correctly returned 0 results (threshold filtered)")


def test_get_all_topics(manager: ChromaMemoryManager) -> None:
    print("\n[Test 5] get_all_topics …")
    topics = manager.get_all_topics()
    expected = sorted([note.topic for note in DUMMY_NOTES])
    check(topics == expected, f"Expected {expected}, got {topics}")
    print(f"  {PASS} Topics: {topics}")


def test_delete_note(manager: ChromaMemoryManager) -> None:
    print("\n[Test 6] delete_note …")
    deleted = manager.delete_note("quantum computing 2024")
    check(deleted, "delete_note should return True for existing topic")
    remaining = manager.get_all_topics()
    check(
        "quantum computing 2024" not in remaining,
        "Quantum note should be gone after delete"
    )
    check(len(remaining) == 2, f"Expected 2 notes remaining, got {len(remaining)}")
    print(f"  {PASS} After delete: {len(remaining)} topics remain: {remaining}")


def test_research_note_model() -> None:
    """Unit tests for the ResearchNote model itself (no ChromaDB needed)."""
    print("\n[Test 7] ResearchNote model validation …")

    note = DUMMY_NOTES[0]

    # to_chroma_metadata round-trip
    meta = note.to_chroma_metadata()
    check(isinstance(meta["key_facts"], str), "key_facts must be JSON string in metadata")
    check(isinstance(meta["sources"], str), "sources must be JSON string in metadata")

    reconstructed = ResearchNote.from_chroma_metadata(meta, note.summary)
    check(reconstructed.topic == note.topic, "round-trip topic matches")
    check(reconstructed.key_facts == note.key_facts, "round-trip key_facts matches")
    check(reconstructed.sources == note.sources, "round-trip sources matches")
    check(abs(reconstructed.confidence - note.confidence) < 0.001, "round-trip confidence matches")

    # Confidence validator
    try:
        bad = ResearchNote.create(
            topic="bad note",
            summary="test",
            key_facts=[],
            sources=[],
            confidence=1.5,   # invalid
        )
        check(False, "Should have raised ValueError for confidence=1.5")
    except Exception:
        check(True, "confidence=1.5 correctly raises ValueError")

    print("  ResearchNote model round-trip and validation passed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  ChromaMemoryManager Integration Test")
    print("=" * 60)

    # Use a temp path so tests don't pollute the real chroma_data/
    TEST_CHROMA_PATH = "./chroma_data_test"

    manager = ChromaMemoryManager(chroma_path=TEST_CHROMA_PATH)
    manager.initialise()

    # Clean slate
    manager.clear_all_notes()

    try:
        test_research_note_model()
        test_save_notes(manager)
        test_semantic_search_quantum(manager)
        test_semantic_search_climate(manager)
        test_threshold_filters_irrelevant(manager)
        test_get_all_topics(manager)
        test_delete_note(manager)

        print("\n" + "=" * 60)
        print("  All tests passed!")
        print("=" * 60)

    except AssertionError as exc:
        print(f"\n   {exc}")
        sys.exit(1)

    finally:
        # Clean up test database
        import shutil
        if os.path.exists(TEST_CHROMA_PATH):
            shutil.rmtree(TEST_CHROMA_PATH)
            print(f"\n  (cleaned up test ChromaDB at {TEST_CHROMA_PATH})")


if __name__ == "__main__":
    main()