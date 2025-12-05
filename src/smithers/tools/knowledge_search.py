"""Knowledge base search tool using the existing RAG retriever.

Searches through the markdown documents in the knowledge base to answer questions.
"""

from __future__ import annotations

from langchain_core.tools import tool
from smithers.rag.retriever import Retriever


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a readable string.

    Args:
        docs: List of Document objects from the retriever

    Returns:
        Formatted string with source and content
    """
    parts = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source") or meta.get("path") or meta.get("title")
        header = f"Source: {source}" if source else ""
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information related to a query.

    This tool searches through markdown documents in the knowledge base
    and returns relevant information. Use this when you need to answer
    questions about topics that might be in the documentation or notes.

    Args:
        query: The search query or question

    Returns:
        Relevant information from the knowledge base, or a message if nothing found.
    """
    try:
        retriever = Retriever(k=5)
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the knowledge base."

        return _format_docs(docs)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


def get_knowledge_search_tool():
    """Return the knowledge search tool for LangChain agent."""
    return search_knowledge_base
