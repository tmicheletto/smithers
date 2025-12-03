"""API schemas for the chatbot."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request to send a message to the chatbot."""

    message: str = Field(..., min_length=1, description="The user's message")
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    """Response from the chatbot."""

    message: str = Field(..., description="The assistant's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model: str
