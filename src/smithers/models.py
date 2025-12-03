"""Data models for the agentic application."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class AgentDependencies:
    """Dependencies injected into agent tools.

    This dataclass holds shared resources and context that tools need access to.
    Examples: API clients, database connections, configuration, etc.
    """

    context: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)


class TaskRequest(BaseModel):
    """User request for the agent to process."""

    task: str = Field(..., description="The task or question for the agent")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata about the request"
    )


class TaskResponse(BaseModel):
    """Agent response to a task request."""

    result: str = Field(..., description="The result of processing the task")
    success: bool = Field(
        default=True, description="Whether the task was completed successfully"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the execution"
    )
