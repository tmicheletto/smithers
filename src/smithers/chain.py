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
        "- Get surf forecasts for Australian locations (use get_surf_forecast)\n\n"
        "For surf forecasts:\n"
        "- Extract the location name and time reference from user queries\n"
        "- Call get_surf_forecast with location_name and when parameters\n"
        "- Time references: 'today', 'tomorrow', day names (Monday-Sunday)\n"
        "- If no time specified, default to 'today'\n"
        "- Forecasts are provided in 3 sessions: Morning (6-10 AM), Midday (10 AM-2 PM), Afternoon (2-6 PM)\n"
        "- Each session includes a 1-10 rating based on wave height, period, and wind conditions\n"
        "- Focus on Australian surf spots as the tool is optimized for Australia\n\n"
        "Examples:\n"
        "- 'What's the surf like at Bells Beach tomorrow?' → get_surf_forecast('Bells Beach', 'tomorrow')\n"
        "- 'How's Torquay on Sunday?' → get_surf_forecast('Torquay', 'Sunday')\n"
        "- 'Surf at Barwon Heads' → get_surf_forecast('Barwon Heads', 'today')\n\n"
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
