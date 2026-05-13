"""
Collections managed:
  • research_notes   — one document per topic; document = summary text
  • source_chunks    — raw web content chunks for retrieval augmentation

Embedding model:
  BAAI/bge-small-en-v1.5 via sentence-transformers.
  Instantiated ONCE at __init__ time and reused — loading from disk takes 2–4s,
  subsequent calls are <100ms.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from models.research_note import ResearchNote

load_dotenv()

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_PATH: str = "./chroma_data"
RESEARCH_NOTES_COLLECTION = "research_notes"
SOURCE_CHUNKS_COLLECTION = "source_chunks"

DEFAULT_SIMILARITY_THRESHOLD: float = float(
    os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.75")
)
DEFAULT_TOP_K: int = 5


# Manager 


class ChromaMemoryManager:
    """
    Manages all interactions with ChromaDB for the AI Research Assistant.
    """

    def __init__(self, chroma_path: str = CHROMA_PATH) -> None:
        self.chroma_path = chroma_path
        self._client: Optional[chromadb.PersistentClient] = None
        self._notes_collection = None
        self._chunks_collection = None

        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' — first load takes ~3s …")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready.")

    # Lifecycle

    def initialise(self) -> None:
        """
        Creates (or loads from disk) both ChromaDB collections.
        """
        self._client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection is idempotent
        self._notes_collection = self._client.get_or_create_collection(
            name=RESEARCH_NOTES_COLLECTION,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for BGE
        )
        self._chunks_collection = self._client.get_or_create_collection(
            name=SOURCE_CHUNKS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB initialised at '{self.chroma_path}'. "
            f"Notes: {self._notes_collection.count()}, "
            f"Chunks: {self._chunks_collection.count()}"
        )

    # Internal helpers 

    def _assert_initialised(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "ChromaMemoryManager.initialise() must be called before use."
            )

    def _embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats."""
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        return self._embedder.encode(
            texts, normalize_embeddings=True, batch_size=32
        ).tolist()

    # research_notes collection 

    def save_research_note(self, note: ResearchNote) -> None:
        """
        Upsert a ResearchNote into the `research_notes` collection.

        Uses upsert (not add) so re-researching the same topic refreshes the
        existing note rather than creating a duplicate.

        The document text (what gets embedded) is the note's summary.
        All other fields live in metadata as a flat dict.

        Parameters
        ----------
        note : ResearchNote
            A fully constructed ResearchNote instance.
        """
        self._assert_initialised()

        embedding = self._embed(note.summary)
        metadata = note.to_chroma_metadata()

        # ChromaDB document ID — use topic as the natural key so upsert works
        doc_id = f"note::{note.topic.lower().replace(' ', '_')}"

        self._notes_collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[note.summary],
            metadatas=[metadata],
        )
        logger.info(f"Saved research note: '{note.topic}' (id={doc_id})")

    def search_memory(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_confidence: float = float(os.getenv("MEMORY_MIN_CONFIDENCE", "0.6")),
        max_age_days: Optional[float] = None,
    ) -> List[ResearchNote]:
        """
        Semantic search over stored research notes.

        1. Embeds the query with the same model used for storage.
        2. Queries ChromaDB for the `top_k` nearest neighbours.
        3. Filters by cosine similarity >= threshold.
        4. Optionally filters by confidence and note age.

        Parameters
        ----------
        query : str
            The user's question or topic string.
        top_k : int
            Maximum number of candidates to retrieve from ChromaDB.
        threshold : float
            Minimum cosine similarity (0–1) to include a note. Default 0.75.
        min_confidence : float
            Minimum note confidence score to include. Default 0.6.
        max_age_days : float | None
            If set, exclude notes older than this many days.

        Returns
        -------
        List[ResearchNote]
            Matching notes, sorted by similarity (highest first).
        """
        self._assert_initialised()

        if self._notes_collection.count() == 0:
            logger.debug("search_memory: collection is empty, returning []")
            return []

        query_embedding = self._embed(query)

        results = self._notes_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._notes_collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        notes: List[ResearchNote] = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]   # cosine distance = 1 - similarity

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - dist   # convert distance → similarity

            if similarity < threshold:
                logger.debug(
                    f"Skipping '{meta.get('topic')}': "
                    f"similarity={similarity:.3f} < threshold={threshold}"
                )
                continue

            note = ResearchNote.from_chroma_metadata(meta, doc)

            if not note.is_confident(min_confidence):
                logger.debug(
                    f"Skipping '{note.topic}': "
                    f"confidence={note.confidence:.2f} < {min_confidence}"
                )
                continue

            if max_age_days is not None and not note.is_fresh(max_age_days):
                logger.debug(
                    f"Skipping '{note.topic}': age={note.age_days():.1f}d > {max_age_days}d"
                )
                continue

            logger.debug(f"Hit: '{note.topic}' sim={similarity:.3f}")
            notes.append(note)

        return notes

    # ── source_chunks collection ──────────────────────────────────────────────

    def save_source_chunks(
        self,
        chunks: List[str],
        metadata: dict,
    ) -> None:
        """
        Save raw web-content chunks to the `source_chunks` collection.

        Each chunk gets a unique ID derived from the URL + chunk index.
        Metadata must be flat (no nested dicts/lists).

        Parameters
        ----------
        chunks : List[str]
            List of text chunks from a scraped web page.
        metadata : dict
            Flat dict with at least 'url' and 'topic' keys.
            Lists in metadata will be auto-serialised to JSON strings.
        """
        self._assert_initialised()

        if not chunks:
            logger.warning("save_source_chunks called with empty chunks list — skipping.")
            return

        # Ensure metadata is flat (serialise any list/dict values)
        flat_meta = {
            k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in metadata.items()
        }
        flat_meta["saved_at"] = time.time()

        embeddings = self._embed_batch(chunks)
        url_slug = flat_meta.get("url", "unknown").replace("/", "_")[:60]
        ids = [f"chunk::{url_slug}::{i}" for i in range(len(chunks))]
        metadatas = [flat_meta] * len(chunks)

        self._chunks_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(
            f"Saved {len(chunks)} source chunks for URL: {flat_meta.get('url')}"
        )

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_all_topics(self) -> List[str]:
        """
        Return a sorted list of all unique topic strings stored in memory.
        Used by the Streamlit UI to display the knowledge base contents.

        Returns
        -------
        List[str]
            Unique topic labels, sorted alphabetically.
        """
        self._assert_initialised()

        if self._notes_collection.count() == 0:
            return []

        results = self._notes_collection.get(include=["metadatas"])
        topics = [meta["topic"] for meta in results["metadatas"]]
        return sorted(set(topics))

    def delete_note(self, topic: str) -> bool:
        """
        Delete a research note by its topic string.

        Used by the "refresh" workflow — delete the stale note, then let the
        agent re-research and save a fresh one.

        Parameters
        ----------
        topic : str
            The topic string to match (case-insensitive, spaces normalised).

        Returns
        -------
        bool
            True if a note was deleted, False if no matching note was found.
        """
        self._assert_initialised()

        doc_id = f"note::{topic.lower().replace(' ', '_')}"

        try:
            self._notes_collection.delete(ids=[doc_id])
            logger.info(f"Deleted research note: '{topic}' (id={doc_id})")
            return True
        except Exception as exc:
            logger.warning(f"delete_note: could not delete '{doc_id}': {exc}")
            return False

    def note_count(self) -> int:
        """Return total number of stored research notes."""
        self._assert_initialised()
        return self._notes_collection.count()

    def chunk_count(self) -> int:
        """Return total number of stored source chunks."""
        self._assert_initialised()
        return self._chunks_collection.count()

    def clear_all_notes(self) -> None:
        """
        Delete ALL research notes. Destructive — use only in tests.
        Source chunks are not affected.
        """
        self._assert_initialised()
        self._client.delete_collection(RESEARCH_NOTES_COLLECTION)
        self._notes_collection = self._client.get_or_create_collection(
            name=RESEARCH_NOTES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("All research notes deleted.")