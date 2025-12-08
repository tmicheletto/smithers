# Smithers

<p align="center">
  <img src="static/smithers.avif" alt="Smithers and Mr. Burns" width="600">
</p>

A RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, and OpenAI. Smithers provides a conversational interface to query a knowledge base of markdown documents and access additional tools like surf forecasting.

## Features

- **Knowledge Base Search**: Query 500+ markdown files (31k+ lines) using RAG
- **Surf Forecast**: Get wave and wind conditions for surf spots worldwide
- **Agent Architecture**: LangChain agent with multi-tool support
- **Streaming Support**: Real-time responses via Server-Sent Events
- **Session Management**: In-memory conversation history

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Install dependencies
uv sync

# Create .env.local file with your OpenAI API key
echo "OPENAI_API_KEY=your-api-key" > .env.local
```

### Index Knowledge Base

```bash
# Index all markdown files from src/data/
uv run python -m src.smithers.rag.indexer

# For testing, index limited files
uv run python -m src.smithers.rag.indexer --max-files 10

# Recreate vector store from scratch
uv run python -m src.smithers.rag.indexer --recreate-store
```

### Run Server

```bash
uv run uvicorn src.smithers.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## Configuration

Environment variables (`.env.local`):

```bash
OPENAI_API_KEY=your-api-key          # Required
RAG_INDEX_NAME=smithers-knowledge-index  # Vector store name
RAG_EMBEDDING_MODEL=text-embedding-3-small  # Embedding model
RAG_SEARCH_TOP_K=5                   # Documents per query
```

## Usage

### API Endpoints

**Chat (Synchronous)**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?", "session_id": "user123"}'
```

**Chat (Streaming)**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?", "session_id": "user123"}'
```

**Health Check**
```bash
curl http://localhost:8000/health
```

### Example Queries

- Knowledge base: "What's in the knowledge base?", "Tell me about RAG"
- Surf forecast: "What's the surf forecast for Pipeline?", "How are the waves at Mavericks?"

## Architecture

### Agent Flow

```
User Query � Agent � Tool Selection � Execution � Response
                �
           [Knowledge Search, Surf Forecast]
```

### Components

- **Agent** (`chain.py`): LangChain agent with GPT-4o-mini
- **Tools** (`tools/`): Knowledge search, surf forecast
- **Vector Store** (`rag/vector_store.py`): OpenAI Vector Store wrapper
- **Retriever** (`rag/retriever.py`): LangChain retriever interface
- **Indexer** (`rag/indexer.py`): Document processing and upload
- **API** (`api.py`): FastAPI endpoints with streaming support

### Key Design Patterns

- **Agent with tools**: Multi-tool access via LangChain's `create_agent()`
- **Protocol-based testing**: `VectorStoreLike` protocol for test injection
- **Streaming support**: Both sync and async SSE endpoints
- **Tool modularity**: Each tool as separate module with `@tool` decorator

## Development

### Testing

```bash
# Run all tests
uv run pytest

# Run integration tests only
uv run pytest -m integration

# Run specific test
uv run pytest tests/test_chain_integration.py::test_chain_answers_using_in_memory_store
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Type checking
uv run pyright
```

### Adding a New Tool

1. Create tool file in `src/smithers/tools/` with `@tool` decorator
2. Add tool to `tools/__init__.py` exports
3. Import and add to tools list in `chain.py`
4. Create tests in `tests/test_<tool_name>.py`

### Modifying Agent Behavior

- **Retrieval count**: Change `k` parameter in `tools/knowledge_search.py`
- **System prompt**: Edit system message in `chain.py:59-68`
- **LLM settings**: Update model/temperature in `chain.py:72-76`

## Contributing

See CLAUDE.md for detailed development guidelines including:
- Code style requirements
- Testing strategy
- Package management rules
- Commit conventions

## License

[Add license information]
