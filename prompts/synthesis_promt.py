"""

Converts raw Tavily web results into a structured ResearchNote via GPT-4o-mini.

Public API:
    synthesise_note(web_result: WebSearchResult, session_id: str) → ResearchNote

Pipeline:
    web_result
        │
        ▼
    format_web_context()          ← build the user-turn string
        │
        ▼
    ChatPromptTemplate            ← system + user messages
        │
        ▼
    ChatOpenAI (gpt-4o-mini)      ← JSON mode enforced
        │
        ▼
    _parse_llm_output()           ← PydanticOutputParser → ResearchNote
        │ (if parse fails)
        ▼
    OutputFixingParser             ← sends malformed JSON back for correction
        │
        ▼
    ResearchNote                  ← fully validated Pydantic object

Key design decisions:
  • response_format={"type": "json_object"} — forces the LLM to output pure
    JSON. This is more reliable than prompting "respond only in JSON" because
    the model enforces the format at the token-sampling level.
  • PydanticOutputParser is the primary parser. It instantiates ResearchNote
    directly from the JSON, so all field validators (confidence range, topic
    not-empty) run automatically.
  • OutputFixingParser is the fallback — it sends the broken output back to
    the LLM with an auto-generated correction prompt. This handles edge cases
    like nested JSON, extra preamble, or truncated output.
  • summary instructed to be "approximately 200 words" — consistent length
    gives consistent embedding vector magnitude, which improves similarity
    scoring stability in ChromaDB.
  • Confidence scoring by source type is embedded in the system prompt so the
    LLM self-assesses based on evidence quality, not just topic familiarity.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models.research_note import ResearchNote
from tools.web_search import WebSearchResult

load_dotenv()
logger = logging.getLogger(__name__)

# Config 

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHUNKS: int = int(os.getenv("SYNTHESIS_MAX_CHUNKS", "20"))

# ── System prompt ─────────────────────────────────────────────────────────────
#
# This is the most important prompt in the project. It controls:
#   1. Output format (pure JSON, no preamble)
#   2. Field quality (specific key_facts, 200-word summary)
#   3. Confidence scoring by source type
#   4. Topic label format (short, lowercase, year-qualified)
#
# DO NOT change the field names in the JSON schema — they must match
# ResearchNote exactly (topic, summary, key_facts, sources, confidence).
# The session_id and timestamp are injected by synthesise_note(), not the LLM.

SYNTHESIS_SYSTEM_PROMPT = """\
You are a precise research analyst. Your job is to synthesise web search \
results into a structured JSON research note.

CRITICAL: Respond with ONLY valid JSON. No preamble, no explanation, no \
markdown code fences. The JSON must be parseable by Python's json.loads().

Output this exact JSON structure:
{{
  "topic": "<short descriptive label, lowercase, include year if time-sensitive>",
  "summary": "<exactly ~200 words synthesising the key findings across all sources>",
  "key_facts": [
    "<specific, falsifiable fact with concrete details>",
    "<another specific fact — numbers, names, dates preferred>",
    "<3 to 7 facts total>"
  ],
  "sources": ["<url1>", "<url2>", "..."],
  "confidence": <float between 0.0 and 1.0>
}}

FIELD RULES:

topic:
  - 2–6 words, all lowercase
  - Include year for time-sensitive topics: "llm fine-tuning 2024"
  - Use the query as a starting point but make it more precise

summary:
  - Approximately 200 words (this is important for embedding consistency)
  - Synthesise across ALL sources — do not summarise just one
  - Use past tense for completed events, present for ongoing situations
  - Do not repeat the topic verbatim in the first sentence

key_facts:
  - 3 to 7 facts, each on its own line in the list
  - Each fact must be SPECIFIC and FALSIFIABLE
    ✓ "GPT-4 was released on March 14, 2023 with a 32K context window"
    ✗ "GPT-4 is a powerful AI model"
  - Include numbers, dates, names, percentages where available
  - One fact per list item — no compound sentences with "and also"

sources:
  - Include the URL of every source that contributed information
  - If a source was irrelevant, omit it
  - Use the exact URLs from the provided search results

confidence:
  Assess evidence quality and set confidence accordingly:
  - 0.85–0.95: Peer-reviewed papers, official documentation, government sources
  - 0.70–0.84: Major news outlets (Reuters, BBC, NYT, WSJ, AP), established tech blogs
  - 0.55–0.69: General news sites, aggregator articles, Wikipedia
  - 0.40–0.54: Personal blogs, opinion pieces, forums (Reddit, HN)
  - 0.20–0.39: Single unverified source, speculative content
  When sources conflict, use the lower bound of the range.
  When multiple high-quality sources agree, use the upper bound.
"""

SYNTHESIS_USER_PROMPT = """\
Research query: {query}

Web search results:
{context}

Synthesise the above results into the JSON research note format. \
Remember: respond with ONLY the JSON object, nothing else.
"""

# LLM + parser setup 


def _build_llm() -> ChatOpenAI:
    """
    Build the ChatOpenAI instance with JSON mode enforced.

    model_kwargs={"response_format": {"type": "json_object"}} tells the OpenAI
    API to only emit valid JSON tokens. This is more reliable than prompt-only
    JSON enforcement because it operates at the sampling level.

    Temperature 0 for deterministic synthesis — research notes should be
    consistent for the same input, not creative.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file.\n"
            "Get yours at https://platform.openai.com/api-keys"
        )

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _build_chain(llm: ChatOpenAI):
    """
    Build the prompt | llm chain.

    Returns the chain object — does NOT include the parser because we want
    to handle parse failures gracefully with OutputFixingParser separately.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        ("user", SYNTHESIS_USER_PROMPT),
    ])
    return prompt | llm


def _parse_llm_output(
    raw_output: str,
    session_id: str,
    llm: ChatOpenAI,
) -> ResearchNote:
    """
    Parse the LLM's JSON output into a ResearchNote.

    Primary path: PydanticOutputParser → ResearchNote directly.
    Fallback path: OutputFixingParser sends malformed output back to the LLM
    with an auto-generated correction prompt.

    The session_id and timestamp are NOT in the LLM output (the LLM doesn't
    know the session). We inject them here after parsing the LLM fields.

    Parameters
    ----------
    raw_output : str
        The LLM's raw string output (should be valid JSON).
    session_id : str
        Session UUID to inject into the ResearchNote.
    llm : ChatOpenAI
        The LLM instance, used by OutputFixingParser for correction calls.

    Returns
    -------
    ResearchNote
        Fully validated Pydantic object with all fields populated.

    Raises
    ------
    ValueError
        If parsing fails even after OutputFixingParser correction.
    """
    # Step 1: Parse raw JSON → dict
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.warning(f"LLM output is not valid JSON: {exc}. Attempting fix …")
        # Try to extract JSON from output that has preamble/postamble text
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                logger.info("JSON extracted from surrounding text via regex")
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse LLM output as JSON even after extraction.\n"
                    f"Raw output: {raw_output[:500]}"
                )
        else:
            raise ValueError(
                f"No JSON object found in LLM output.\nRaw: {raw_output[:500]}"
            )

    # Step 2: Inject fields the LLM doesn't produce
    data["session_id"] = session_id
    data["timestamp"] = time.time()

    # Step 3: Construct ResearchNote via Pydantic (runs all validators)
    try:
        note = ResearchNote(**data)
        logger.info(
            f"Synthesis successful: topic='{note.topic}' "
            f"confidence={note.confidence:.2f} "
            f"facts={len(note.key_facts)} sources={len(note.sources)}"
        )
        return note

    except Exception as exc:
        logger.warning(
            f"PydanticOutputParser failed: {exc}\n"
            f"Attempting OutputFixingParser correction …"
        )

        # Step 4: OutputFixingParser fallback
        base_parser = PydanticOutputParser(pydantic_object=ResearchNote)
        fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

        try:
            # OutputFixingParser sends the bad output back with a correction prompt
            fixed_note = fixing_parser.parse(raw_output)
            # Still need to inject session_id / timestamp
            fixed_note_dict = fixed_note.model_dump()
            fixed_note_dict["session_id"] = session_id
            fixed_note_dict["timestamp"] = time.time()
            return ResearchNote(**fixed_note_dict)

        except Exception as fix_exc:
            raise ValueError(
                f"OutputFixingParser also failed: {fix_exc}\n"
                f"Original error: {exc}\n"
                f"Raw output: {raw_output[:500]}"
            ) from fix_exc


# Public API 


def synthesise_note(
    web_result: WebSearchResult,
    session_id: str,
) -> ResearchNote:
    """
    Convert a WebSearchResult into a ResearchNote using GPT-4o-mini.

    This is the only function nodes.py needs to import from this module.

    Parameters
    ----------
    web_result : WebSearchResult
        The output of tools.web_search.search_and_chunk().
    session_id : str
        The current session UUID (from AgentState["session_id"]).

    Returns
    -------
    ResearchNote
        A fully validated ResearchNote ready to save to ChromaDB.

    Raises
    ------
    EnvironmentError
        If OPENAI_API_KEY is not configured.
    ValueError
        If LLM synthesis and JSON parsing both fail after correction attempts.

    Example
    -------
    web_result = search_and_chunk("large language model fine-tuning 2024")
    note = synthesise_note(web_result, session_id=str(uuid.uuid4()))
    print(note.topic, note.confidence)
    """
    logger.info(
        f"synthesise_note: query='{web_result.query}' "
        f"chunks={web_result.total_chunks}"
    )

    llm = _build_llm()
    chain = _build_chain(llm)

    # Format web content for the prompt (top MAX_CONTEXT_CHUNKS chunks)
    context = web_result.as_formatted_context(max_chunks=MAX_CONTEXT_CHUNKS)

    if not context.strip():
        raise ValueError(
            f"No content available to synthesise for query: '{web_result.query}'. "
            "Check that Tavily returned results with raw_content."
        )

    logger.debug(f"Context length: {len(context)} chars, feeding to {OPENAI_MODEL}")

    # Invoke the LLM
    response = chain.invoke({
        "query": web_result.query,
        "context": context,
    })

    raw_output: str = response.content

    # Parse and return the ResearchNote
    return _parse_llm_output(raw_output, session_id=session_id, llm=llm)