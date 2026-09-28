# Zentrix AI Research Assistant

A Python-based AI research assistant for web research, memory retrieval, and answer synthesis.

## Overview

Zentrix is an AI-powered research workflow that combines agent orchestration, semantic memory, and web search to answer user questions in a structured and reusable way. The system is designed to classify intent, recall prior knowledge from stored notes, search the web when needed, and save synthesized findings back into memory for future sessions.

## AI Stack

This project uses a modern AI and retrieval stack:

- LangGraph for orchestration of the research workflow
- LangChain components for model integration and structured prompting
- Groq as the primary LLM provider for fast inference
- ChromaDB for persistent vector storage and semantic retrieval
- SentenceTransformers for embeddings (BAAI/bge-small-en-v1.5)
- Tavily for web search and source retrieval
- Streamlit for the user interface
- Python dotenv for environment configuration

## How the AI Works

The assistant follows a simple but powerful research loop:

1. The user asks a question.
2. The agent classifies the intent.
3. It checks memory for relevant prior notes.
4. If needed, it performs a web search for fresh information.
5. It synthesizes the best answer from memory and/or web results.
6. It stores the final research summary back into the vector database for future recall.

This gives the system both short-term contextual reasoning and long-term memory across sessions.

## Features

- Intent-based AI routing
- Semantic memory using vector embeddings
- Persistent knowledge storage with ChromaDB
- Web research and source retrieval
- Answer generation with LLMs
- Session-aware conversational flow
- Memory explorer in the Streamlit UI

## Project Structure

- `app.py` — Streamlit application entry point
- `agent/` — LangGraph workflow and node logic
- `memory/` — ChromaDB memory management
- `models/` — data models for research notes
- `tools/` — research and utility tools
- `prompts/` — prompt templates
- `tests/` — project tests

## Installation

```bash
git clone https://github.com/HeshMaddage/zentrix-ai-research-assistant.git
cd zentrix-ai-research-assistant
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- A configured `.env` file with required API keys

## License

MIT
