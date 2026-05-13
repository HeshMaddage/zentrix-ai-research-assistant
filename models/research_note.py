"""
models/research_note.py
────────────────────────────────────────────────────────────────────────────────
Pydantic data model for a single research note stored in ChromaDB.

Design rules enforced here:
  • confidence is validated to be in [0.0, 1.0]
  • to_chroma_metadata() always returns a *flat* dict — ChromaDB only accepts
    str / int / float values, so List fields are JSON-serialised.
  • from_chroma_metadata() is the inverse, deserialising back to a full model.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
import uuid
from typing import List

from pydantic import BaseModel, field_validator, model_validator


class ResearchNote(BaseModel):
    """
    A structured summary of what the agent learned about a topic.

    Stored in ChromaDB's `research_notes` collection:
      - The `summary` field is embedded as the document text.
      - All other fields go into ChromaDB metadata (flat dict).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    topic: str
    """Short, descriptive topic label e.g. 'quantum computing 2024'."""

    session_id: str
    """UUID identifying the research session that created this note."""

    # ── Content ───────────────────────────────────────────────────────────────
    summary: str
    """2–5 sentence synthesis of the research findings for this topic."""

    key_facts: List[str]
    """3–7 bullet-point facts extracted from the web sources."""

    sources: List[str]
    """URLs of the web pages consulted."""

    # ── Quality metadata ──────────────────────────────────────────────────────
    timestamp: float
    """Unix timestamp (seconds since epoch) when this note was created."""

    confidence: float
    """
    Agent's self-assessed confidence in this note (0.0 – 1.0).
    Below 0.6 → agent flags the note as low-quality on retrieval.
    """

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {v}"
            )
        return round(v, 4)

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic cannot be an empty string")
        return v

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("summary cannot be an empty string")
        return v

    # ── Convenience constructors ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        topic: str,
        summary: str,
        key_facts: List[str],
        sources: List[str],
        confidence: float,
        session_id: str | None = None,
        timestamp: float | None = None,
    ) -> "ResearchNote":
        """
        Factory that fills in defaults for session_id and timestamp.
        Use this instead of calling ResearchNote(...) directly.

        Example
        -------
        note = ResearchNote.create(
            topic="quantum computing 2024",
            summary="Several breakthroughs in error correction...",
            key_facts=["Google achieved 99.9% gate fidelity", ...],
            sources=["https://nature.com/..."],
            confidence=0.85,
        )
        """
        return cls(
            topic=topic,
            summary=summary,
            key_facts=key_facts,
            sources=sources,
            confidence=confidence,
            session_id=session_id or str(uuid.uuid4()),
            timestamp=timestamp or time.time(),
        )

    # ── ChromaDB serialisation ────────────────────────────────────────────────

    def to_chroma_metadata(self) -> dict:
        """
        Returns a *flat* dict safe to pass as ChromaDB metadata.

        ChromaDB only accepts str / int / float values at the top level —
        no nested dicts, no lists. Lists are JSON-serialised to strings here
        and deserialised in `from_chroma_metadata`.

        Returns
        -------
        dict
            Flat key-value pairs, all values are str / int / float.
        """
        return {
            "topic": self.topic,
            "session_id": self.session_id,
            "timestamp": self.timestamp,          # float — ChromaDB accepts this
            "confidence": self.confidence,        # float — ChromaDB accepts this
            "key_facts": json.dumps(self.key_facts),   # list → JSON string
            "sources": json.dumps(self.sources),        # list → JSON string
            # summary is stored as the ChromaDB document text, not metadata,
            # but we include a truncated version here for quick inspection.
            "summary_preview": self.summary[:200],
        }

    @classmethod
    def from_chroma_metadata(cls, metadata: dict, summary: str) -> "ResearchNote":
        """
        Reconstruct a ResearchNote from ChromaDB metadata + document text.

        Parameters
        ----------
        metadata : dict
            The flat metadata dict returned by ChromaDB.
        summary : str
            The document text stored alongside the metadata embedding.
        """
        return cls(
            topic=metadata["topic"],
            session_id=metadata["session_id"],
            summary=summary,
            key_facts=json.loads(metadata["key_facts"]),
            sources=json.loads(metadata["sources"]),
            timestamp=float(metadata["timestamp"]),
            confidence=float(metadata["confidence"]),
        )

    # ── Display helpers ───────────────────────────────────────────────────────

    def age_days(self) -> float:
        """Returns how many days old this note is (float)."""
        return (time.time() - self.timestamp) / 86_400

    def is_fresh(self, max_age_days: float = 7.0) -> bool:
        """True if the note is younger than `max_age_days`."""
        return self.age_days() < max_age_days

    def is_confident(self, min_confidence: float = 0.6) -> bool:
        """True if confidence meets the minimum threshold."""
        return self.confidence >= min_confidence

    def __repr__(self) -> str:
        return (
            f"ResearchNote(topic={self.topic!r}, "
            f"confidence={self.confidence:.2f}, "
            f"age={self.age_days():.1f}d, "
            f"facts={len(self.key_facts)})"
        )