"""FastAPI web application for the chatbot.

Adds basic logging for startup, requests, streaming, and errors.
"""

import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse
from .chain import build_chain

# Configure module-level logger
logger = logging.getLogger("smithers.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Smithers Chatbot",
    description="AI-powered chatbot using PydanticAI",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# In-memory session storage
# In production, use Redis or a database
sessions: dict[str, list[ChatMessage]] = defaultdict(list)


chain = None


@app.on_event("startup")
async def startup_event():
    """Initialize LangChain RAG chain on startup."""
    logger.info("Initializing LangChain RAG chain...")
    try:
        global chain
        chain = build_chain()
        logger.info("LangChain RAG chain ready")
    except Exception as e:
        logger.exception("Failed to initialize LangChain chain: %s", e)


def get_or_create_session(session_id: str | None) -> str:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id
    return str(uuid.uuid4())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=settings.model,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response.

    Args:
        request: The chat request containing the user's message and optional session_id.

    Returns:
        ChatResponse with the assistant's message and session_id.
    """
    # Get or create session
    session_id = get_or_create_session(request.session_id)

    # Add user message to history
    user_msg = ChatMessage(role="user", content=request.message)
    sessions[session_id].append(user_msg)

    # Build conversation history for future use (not yet threaded into chain)
    conversation_history = [
        {"role": msg.role, "content": msg.content} for msg in sessions[session_id]
    ]

    try:
        logger.info("Chat request", extra={
            "session_id": session_id,
            "message_len": len(request.message),
        })
        # Invoke LangChain RAG chain (non-streaming)
        assistant_message = chain.invoke({
            "question": request.message,
            "chat_history": conversation_history,
        })

        # Add assistant message to history
        assistant_msg = ChatMessage(role="assistant", content=assistant_message)
        sessions[session_id].append(assistant_msg)

        response = ChatResponse(
            message=assistant_message,
            session_id=session_id,
            timestamp=datetime.now(),
        )
        logger.info("Chat response", extra={
            "session_id": session_id,
            "response_len": len(assistant_message),
        })
        return response

    except Exception as e:
        logger.exception("Chat error: %s", e, extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and stream the response.

    Args:
        request: The chat request containing the user's message and optional session_id.

    Returns:
        Streaming response with Server-Sent Events.
    """
    # Get or create session
    session_id = get_or_create_session(request.session_id)

    # Add user message to history
    user_msg = ChatMessage(role="user", content=request.message)
    sessions[session_id].append(user_msg)

    # Build conversation history
    conversation_history = [
        {"role": msg.role, "content": msg.content} for msg in sessions[session_id]
    ]

    async def event_stream():
        """Generate Server-Sent Events for streaming response."""
        try:
            accumulated = ""
            # Stream tokens from the chain
            for chunk in chain.stream({
                "question": request.message,
                "chat_history": conversation_history,
            }):
                text = str(chunk)
                accumulated += text
                yield f"data: {text}\n\n"

            # After streaming, add final message to history and emit session id
            assistant_msg = ChatMessage(role="assistant", content=accumulated)
            sessions[session_id].append(assistant_msg)
            logger.info("Stream completed", extra={
                "session_id": session_id,
                "bytes": len(accumulated),
            })
            yield f"event: session\ndata: {session_id}\n\n"

        except Exception as e:
            logger.exception("Stream error: %s", e, extra={"session_id": session_id})
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session.

    Args:
        session_id: The session ID.

    Returns:
        List of messages in the conversation.
    """
    if session_id not in sessions:
        logger.warning("History requested for missing session", extra={"session_id": session_id})
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session.

    Args:
        session_id: The session ID to delete.

    Returns:
        Success message.
    """
    if session_id in sessions:
        del sessions[session_id]
        logger.info("Session deleted", extra={"session_id": session_id})
        return {"message": "Session deleted"}

    logger.warning("Delete requested for missing session", extra={"session_id": session_id})
    raise HTTPException(status_code=404, detail="Session not found")
