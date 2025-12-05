# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smithers is a RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, and OpenAI. It provides a conversational interface to query a knowledge base of markdown documents stored in `src/data/` (500+ files with 31k+ lines of technical notes).

## Key Commands

### Development
```bash
# Run the API server
uvicorn src.smithers.api:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run integration tests only
pytest -m integration

# Run specific test
pytest tests/test_chain_integration.py::test_chain_answers_using_in_memory_store

# Linting
ruff check .
ruff format .
```

### RAG Indexing
```bash
# Index all markdown files from src/data/ to OpenAI Vector Store
python -m src.smithers.rag.indexer

# Recreate vector store and reindex everything
python -m src.smithers.rag.indexer --recreate-store

# Index limited files for testing
python -m src.smithers.rag.indexer --max-files 10
```

## Architecture

### Agent Architecture

Smithers uses a LangChain agent with multiple tools:

**Tools:**
- **Knowledge Base Search** (`search_knowledge_base`): Searches markdown documents in the RAG system
- **Surf Forecast** (`get_surf_forecast`): Retrieves wave/wind conditions from Open-Meteo Marine API

The agent (`src/smithers/chain.py`) uses OpenAI's function calling to automatically select and invoke the appropriate tool based on user queries.

### RAG System Flow

1. **Indexing (src/smithers/rag/indexer.py)**
   - Scans `src/data/` for markdown files
   - Uploads files to OpenAI Vector Store via `VectorStore.upload_file()`
   - OpenAI automatically handles chunking and embedding generation
   - Batch uploads optimize costs (10 files per batch by default)

2. **Vector Store (src/smithers/rag/vector_store.py)**
   - Wraps OpenAI Vector Store API
   - Manages file uploads, deletion, and search operations
   - Uses `text-embedding-3-small` model (1536 dimensions)
   - Implements `VectorStoreLike` protocol for test injection

3. **Retrieval (src/smithers/rag/retriever.py)**
   - Implements LangChain `BaseRetriever` interface
   - Delegates to `VectorStore.search()` for similarity search
   - Converts search results to LangChain `Document` objects
   - Default k=5 documents retrieved per query

4. **Agent (src/smithers/chain.py)**
   - LangChain agent created with `create_agent()` API
   - Uses `gpt-4o-mini` with temperature=0 for deterministic responses
   - Agent has access to multiple tools (knowledge search, surf forecast)
   - Supports both streaming and non-streaming invocation
   - Input format: `{"messages": [HumanMessage(...), ...]}`

5. **Tools (src/smithers/tools/)**
   - **Knowledge Search** (`knowledge_search.py`): Wraps existing Retriever for RAG queries
   - **Surf Forecast** (`surf_forecast.py`): Fetches marine data from Open-Meteo API (free, no key required)
   - Tools use LangChain's `@tool` decorator for automatic schema generation

6. **API (src/smithers/api.py)**
   - FastAPI endpoints: `/chat`, `/chat/stream`, `/health`
   - In-memory session storage with conversation history (use Redis in production)
   - Server-Sent Events (SSE) for streaming responses
   - CORS enabled for development (configure for production)

### Key Design Patterns

- **Agent with tools**: Uses LangChain's `create_agent()` to enable multi-tool access (RAG + external APIs)
- **Protocol-based testing**: `VectorStoreLike` protocol allows injecting `InMemoryVectorStore` in tests without mocking OpenAI API
- **Tool modularity**: Each tool is a separate module with `@tool` decorator for easy extension
- **Streaming support**: Both `/chat` (sync) and `/chat/stream` (async SSE) endpoints for different UX needs
- **Configuration management**: `pydantic-settings` loads config from `.env` with sensible defaults

## Configuration

Environment variables in `.env`:
- `OPENAI_API_KEY`: Required for OpenAI API access (embeddings, chat, vector store)
- `RAG_INDEX_NAME`: Vector store name (default: "smithers-knowledge-index")
- `RAG_EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
- `RAG_SEARCH_TOP_K`: Number of documents to retrieve (default: 5)

**Note:** Surf forecast tool requires no API key (uses free Open-Meteo Marine API)

## Testing Strategy

- Integration tests use `InMemoryVectorStore` to avoid OpenAI API calls
- Mark integration tests with `@pytest.mark.integration`
- Tests verify agent behavior: correct answers when docs exist, admits uncertainty when they don't
- Tool tests (`test_surf_forecast.py`) verify external API integration
- Test files mirror production agent structure for realistic validation

## Common Development Tasks

**Adding new documents to knowledge base:**
1. Add markdown files to `src/data/`
2. Run `python -m src.smithers.rag.indexer` to upload to vector store
3. Restart API server to use updated index

**Modifying agent behavior:**
- Adjust retrieval count: Change `k` parameter in `tools/knowledge_search.py`
- Modify system prompt: Edit system message in `chain.py:59-68`
- Change LLM settings: Update model/temperature in `chain.py:72-76`
- Add new tools: Create tool in `tools/` directory, add to `chain.py` tools list

**Adding a new tool:**
1. Create tool file in `src/smithers/tools/` with `@tool` decorator
2. Add tool to `tools/__init__.py` exports
3. Import and add to tools list in `chain.py:58-61`
4. Create tests in `tests/test_<tool_name>.py`

**Testing agent changes:**
- Run `pytest -m integration` to validate end-to-end behavior
- Run `pytest tests/test_surf_forecast.py` to test surf forecast tool
- Use `InMemoryVectorStore` pattern for fast iteration without API costs

**Example queries:**
- Knowledge base: "What's in the knowledge base?", "Tell me about RAG"
- Surf forecast: "What's the surf forecast for Pipeline?", "How are the waves at Mavericks?"
- If you get an error like ModuleNotFoundError: No module named 'smithers' when running the application, it's most likely because the smithers package needs to be installed. You can install it by running pip install -e .