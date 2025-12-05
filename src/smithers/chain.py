"""LangChain agent wiring for Smithers using OpenAI with tools.

The agent has access to:
- Knowledge base search tool (RAG over markdown docs)
- Surf forecast tool (Open-Meteo Marine API)

Uses OpenAI function calling for reliable tool use.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from smithers.tools import get_knowledge_search_tool, get_surf_forecast_tool
from smithers.config import settings


def _format_chat_history(history: list[dict]) -> list:
    """Convert chat history dicts to LangChain message objects.

    Excludes the current question (last user message) since it's
    passed separately in the prompt.

    Args:
        history: List of message dicts with 'role' and 'content' keys.

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage).
    """
    messages = []
    # Skip last message (current question)
    for msg in history[:-1]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def build_chain() -> Any:
    """Construct the LangChain agent with tools.

    Returns a CompiledStateGraph that can be invoked with messages.
    The agent has access to:
    - search_knowledge_base: Search markdown documents
    - get_surf_forecast: Get wave/wind conditions for surf spots
    """
    # Initialize tools
    tools = [
        get_knowledge_search_tool(),
        get_surf_forecast_tool(),
    ]

    # System prompt
    system_prompt = (
        "You are Smithers, a helpful assistant with access to tools.\n\n"
        "You can:\n"
        "- Search the knowledge base for information (use search_knowledge_base)\n"
        "- Get surf forecasts for locations (use get_surf_forecast)\n\n"
        "For surf forecasts:\n"
        "- Common spots: Pipeline (21.6644, -158.0533), Mavericks (37.4936, -122.4969), "
        "Huntington Beach (33.6584, -118.0056), Trestles (33.3720, -117.5901)\n"
        "- If user gives a location name, use your knowledge to estimate coordinates or ask for them\n\n"
        "Be concise and helpful. Use tools when needed."
    )

    # Model: OpenAI gpt-4o-mini with function calling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    # Create agent using new API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent
