"""FastAPI web application for the chatbot."""

import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import agent
from .config import settings
from .models import AgentDependencies
from .schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse

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


def get_or_create_session(session_id: str | None) -> str:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id
    return str(uuid.uuid4())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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
    
    # Build conversation history for the agent
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in sessions[session_id]
    ]
    
    # Create dependencies with conversation history
    deps = AgentDependencies(conversation_history=conversation_history)
    
    try:
        # Run the agent with conversation history in the message context
        # Format history as context for the agent
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[:-1]  # Exclude the current message
        ]) if len(conversation_history) > 1 else ""
        
        # Prepend history to the message if there is any
        message_with_context = request.message
        if history_text:
            message_with_context = f"Previous conversation:\n{history_text}\n\nCurrent message: {request.message}"
        
        result = await agent.run(message_with_context, deps=deps)
        
        # Extract the response
        assistant_message = result.output.result if hasattr(result.output, "result") else str(result.output)
        
        # Add assistant message to history
        assistant_msg = ChatMessage(role="assistant", content=assistant_message)
        sessions[session_id].append(assistant_msg)
        
        return ChatResponse(
            message=assistant_message,
            session_id=session_id,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
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
        {"role": msg.role, "content": msg.content}
        for msg in sessions[session_id]
    ]
    
    async def event_stream():
        """Generate Server-Sent Events for streaming response."""
        deps = AgentDependencies(conversation_history=conversation_history)
        
        try:
            # Stream the agent's response
            async with agent.run_stream(request.message, deps=deps) as response:
                async for chunk in response.stream_text():
                    # Send chunk as SSE
                    yield f"data: {chunk}\n\n"
                
                # Get final result
                final_result = await response.get_output()
                assistant_message = final_result.result if hasattr(final_result, "result") else str(final_result)
                
                # Add to history
                assistant_msg = ChatMessage(role="assistant", content=assistant_message)
                sessions[session_id].append(assistant_msg)
                
                # Send session ID at the end
                yield f"event: session\ndata: {session_id}\n\n"
                
        except Exception as e:
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
        return {"message": "Session deleted"}
    
    raise HTTPException(status_code=404, detail="Session not found")
