"""LangChain chain wiring for Smithers RAG chat using OpenAI.

Option A: Build an LC chain that uses `LCVectorStoreRetriever` to
reuse the existing OpenAI Vector Store without reindexing. The chain
combines a prompt, retriever, and `ChatOpenAI` model `gpt-40-mini`.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from smithers.rag.retriever import Retriever
from smithers.config import settings


def _format_docs(docs: list) -> str:
    """Concatenate documents as context for the prompt."""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source") or meta.get("path") or meta.get("title")
        header = f"Source: {source}" if source else ""
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain() -> Any:
    """Construct the LangChain `Runnable` for RAG chat.

    Returns a chain that accepts an input dict with keys:
    - `question`: user query string
    - `chat_history` (optional): list of prior messages (unused in this
      minimal implementation but can be threaded later)
    """
    retriever = Retriever(k=5)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Smithers, a concise, helpful assistant."
                " Use the provided context to answer accurately."
                " If the answer is not in the context, say you don't know."
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )

    # Model: OpenAI gpt-4o-mini with deterministic behavior.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    # Map input -> retrieve -> format -> prompt -> llm -> parse
    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x["question"]))
            | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
