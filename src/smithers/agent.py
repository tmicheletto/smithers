"""Main agentic application using PydanticAI."""

from __future__ import annotations

import logfire
from pydantic_ai import Agent, RunContext

from .config import settings
from .models import AgentDependencies, TaskRequest, TaskResponse

# Configure Logfire for observability
# 'if-token-present' means nothing will be sent if logfire is not configured
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


# Create the agent at module level
agent = Agent(
    settings.model,
    deps_type=AgentDependencies,
    output_type=TaskResponse,
    system_prompt=(
        "You are a helpful assistant. Process user requests efficiently "
        "and provide clear, structured responses. Use available tools when needed."
    ),
)


@agent.tool
async def process_text(ctx: RunContext[AgentDependencies], text: str) -> str:
    """Process and analyze text content.

    Args:
        ctx: The agent run context with dependencies.
        text: The text to process.

    Returns:
        Processed text analysis.
    """
    logfire.info("Processing text", text_length=len(text))
    # Placeholder: Add your text processing logic here
    analysis = {
        "length": len(text),
        "word_count": len(text.split()),
    }
    ctx.deps.context["last_analysis"] = analysis
    return f"Processed {analysis['word_count']} words"


@agent.tool
async def get_context(ctx: RunContext[AgentDependencies]) -> str:
    """Retrieve current context information.

    Args:
        ctx: The agent run context with dependencies.

    Returns:
        Current context as a string.
    """
    context_str = str(ctx.deps.context)
    logfire.info("Retrieved context", context_length=len(context_str))
    return context_str


async def process_task(task_request: TaskRequest) -> TaskResponse:
    """Process a task request using the agent.

    Args:
        task_request: The task to process.

    Returns:
        The agent's response.
    """
    deps = AgentDependencies(context=task_request.metadata)
    result = await agent.run(task_request.task, deps=deps)
    return result.output
