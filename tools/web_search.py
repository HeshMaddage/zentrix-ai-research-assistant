"""
Tavily web search wrapper with LangChain RecursiveCharacterTextSplitter chunking.

"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

# Config 

TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# Data structures


@dataclass
class SourceChunk:
    """
    A single text chunk from one web result, with its origin URL attached.

    Stored in ChromaDB's source_chunks collection via
    ChromaMemoryManager.save_source_chunks(). The url field becomes the
    'url' key in the flat metadata dict.
    """
    text: str
    source_url: str
    result_index: int
    title: str = ""

@dataclass
class WebSearchResult:
    """
    The complete output of search_and_chunk().

    Passed into AgentState["web_results"] and consumed by:
      - save_to_memory node  → synthesis prompt reads `results` + `chunks`
      - search_web node      → saves `chunks` to ChromaDB source_chunks
    """

    query: str
    results: List[Dict[str, Any]]
    chunks: List[SourceChunk]
    total_chunks: int

    def as_formatted_context(self, max_chunks: int = 20) -> str:
        """
        Format the top `max_chunks` chunks as a readable string for injection
        into the synthesis prompt. Groups by source URL.

        Used by prompts/synthesis_prompt.py.
        """
        seen_urls: dict[str, List[str]] = {}
        for chunk in self.chunks[:max_chunks]:
            seen_urls.setdefault(chunk.source_url, []).append(chunk.text)

        sections: List[str] = []
        for url, texts in seen_urls.items():
            combined = "\n".join(texts)
            sections.append(f"SOURCE: {url}\n{combined}")

        return "\n\n---\n\n".join(sections)

    def get_source_urls(self) -> List[str]:
        """Return deduplicated list of source URLs, preserving order."""
        seen = set()
        urls = []
        for chunk in self.chunks:
            if chunk.source_url not in seen:
                seen.add(chunk.source_url)
                urls.append(chunk.source_url)
        return urls

    def to_state_dict(self) -> dict:
        """
        Serialise to a plain dict safe to store in AgentState["web_results"].

        AgentState uses TypedDict[web_results: Dict] — we store a dict here,
        not the dataclass, so the graph can checkpoint it cleanly via SqliteSaver.
        """
        return {
            "query": self.query,
            "results": self.results,
            "chunks": [
                {
                    "text": c.text,
                    "source_url": c.source_url,
                    "result_index": c.result_index,
                    "title": c.title,
                }
                for c in self.chunks
            ],
            "total_chunks": self.total_chunks,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "WebSearchResult":
        """Reconstruct from AgentState["web_results"] dict."""
        chunks = [
            SourceChunk(
                text=c["text"],
                source_url=c["source_url"],
                result_index=c["result_index"],
                title=c.get("title", ""),
            )
            for c in d.get("chunks", [])
        ]
        return cls(
            query=d.get("query", ""),
            results=d.get("results", []),
            chunks=chunks,
            total_chunks=d.get("total_chunks", len(chunks)),
        )


#  Splitter 

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph → sentence → word
)


# Core functions 
def _build_tavily_tool() -> TavilySearchResults:
    """
    Instantiate TavilySearchResults.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY not found. Add it to your .env file.\n"
            "Sign up free at https://tavily.com — 1,000 searches/month."
        )

    return TavilySearchResults(
        max_results=TAVILY_MAX_RESULTS,
        include_raw_content=True,
        include_answer=False,   # synthesis own
    )


def _chunk_result(result: Dict[str, Any], result_index: int) -> List[SourceChunk]:
    """
    Split one Tavily result into SourceChunks.

    Content priority:
      1. raw_content  — full page body (richest, preferred)
      2. content      — Tavily's cleaned snippet (fallback)
      3. title        — last resort, prevents empty chunk list

    Each chunk gets the source URL and result index attached so
    ChromaDB metadata attribution is always complete.
    """
    url: str = result.get("url", "unknown")
    title: str = result.get("title", "")

    # Pick the richest available text
    body: str = (
        result.get("raw_content")
        or result.get("content")
        or title
        or ""
    ).strip()

    if not body:
        logger.warning(f"_chunk_result: no content for URL '{url}', skipping")
        return []

    raw_chunks: List[str] = _splitter.split_text(body)

    chunks = [
        SourceChunk(
            text=chunk_text,
            source_url=url,
            result_index=result_index,
            title=title,
        )
        for chunk_text in raw_chunks
        if chunk_text.strip()   # discard whitespace-only chunks
    ]

    logger.debug(f"  → {len(chunks)} chunks from '{url[:60]}'")
    return chunks


def search_and_chunk(query: str) -> WebSearchResult:
    """
    Search Tavily for `query` and return chunked results.

    Steps:
      1. Call TavilySearchResults.invoke(query) → list of result dicts
      2. For each result, split raw_content with RecursiveCharacterTextSplitter
         (chunk_size=500, overlap=50) → list of SourceChunk objects
      3. Collect all chunks across all results into a flat list
      4. Return a WebSearchResult containing both raw results and all chunks

    Parameters
    ----------
    query : str
        The research query string (comes from AgentState["query"]).

    Returns
    -------
    WebSearchResult
        Contains raw Tavily results and all text chunks with source URLs.

    Raises
    ------
    EnvironmentError
        If TAVILY_API_KEY is not set.
    Exception
        If Tavily API call fails (network, quota, bad key).
    """
    logger.info(f"search_and_chunk: searching for '{query}'")

    tool = _build_tavily_tool()

    # TavilySearchResults.invoke() returns a list of dicts
    try:
        raw_results: List[Dict[str, Any]] = tool.invoke(query)
    except Exception as exc:
        logger.error(f"Tavily search failed for '{query}': {exc}")
        raise

    logger.info(f"search_and_chunk: got {len(raw_results)} results from Tavily")

    # Chunk each result
    all_chunks: List[SourceChunk] = []
    for i, result in enumerate(raw_results):
        result_chunks = _chunk_result(result, result_index=i)
        all_chunks.extend(result_chunks)

    total = len(all_chunks)
    logger.info(
        f"search_and_chunk: {total} total chunks from {len(raw_results)} results "
        f"(avg {total // max(len(raw_results), 1)} chunks/result)"
    )

    return WebSearchResult(
        query=query,
        results=raw_results,
        chunks=all_chunks,
        total_chunks=total,
    )


def chunks_for_chroma(web_result: WebSearchResult) -> tuple[List[str], List[dict]]:
    """
    Extract parallel lists of (chunk_texts, metadatas) for ChromaDB ingestion.

    Called by the search_web node before handing off to ChromaMemoryManager.
    Returns two aligned lists: texts[i] has metadata[i].

    Each metadata dict is flat (str/int/float values only) — ChromaDB safe.

    Parameters
    ----------
    web_result : WebSearchResult

    Returns
    -------
    tuple[List[str], List[dict]]
        (chunk_texts, metadatas) — same length, aligned by index.
    """
    texts: List[str] = []
    metadatas: List[dict] = []

    for chunk in web_result.chunks:
        texts.append(chunk.text)
        metadatas.append({
            "url": chunk.source_url,
            "title": chunk.title,
            "result_index": chunk.result_index,
            "query": web_result.query,
        })

    return texts, metadatas