import pytest

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from smithers.config import settings
from smithers.rag.retriever import Retriever
from smithers.rag.vector_store import VectorStoreLike


class InMemoryVectorStore(VectorStoreLike):
    """Simple in-memory vector store used only for tests.

    Stores small text snippets and returns them using a naive
    word-overlap scoring, just enough to drive integration tests.
    """

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def search(self, query: str, k: int = 5):  # matches VectorStore.search signature
        q_words = set(query.lower().split())
        scored: list[dict] = []

        for doc in self._docs:
            text = doc["content"]
            d_words = set(text.lower().split())
            score = len(q_words & d_words)
            if score <= 0:
                continue

            scored.append(
                {
                    "id": doc.get("id", ""),
                    "content": text,
                    "metadata": doc.get("metadata", {}),
                    "score": float(score),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


def build_test_chain(vector_store: InMemoryVectorStore):
    """Build a RAG chain using the test in-memory vector store.

    This mirrors the production chain wiring but swaps the retriever
    backend for an in-memory implementation.
    """

    retriever = Retriever(k=5, vector_store=vector_store)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Smithers, a concise, helpful assistant. "
                "Use the provided context to answer accurately. "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    llm = ChatOpenAI(
        model=settings.model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    def _format_docs(docs):
        parts: list[str] = []
        for d in docs:
            meta = d.metadata or {}
            source = meta.get("source") or meta.get("path") or meta.get("title")
            header = f"Source: {source}" if source else ""
            parts.append(f"{header}\n{d.page_content}")
        return "\n\n".join(parts)

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


@pytest.mark.integration
def test_chain_answers_using_in_memory_store():
    docs = [
        {
            "id": "doc1",
            "content": "yq is a command-line YAML processor, similar to jq.",
            "metadata": {"source": "test-doc"},
        },
        {
            "id": "doc2",
            "content": "jq is used to process JSON on the command line.",
            "metadata": {"source": "test-doc"},
        },
    ]

    store = InMemoryVectorStore(docs)
    chain = build_test_chain(store)

    question = "What is yq used for?"
    answer = chain.invoke({"question": question})

    lower = answer.lower()
    assert "yaml" in lower
    assert "command" in lower


@pytest.mark.integration
def test_chain_says_dont_know_without_relevant_docs():
    docs = [
        {
            "id": "doc1",
            "content": "This document talks only about Git commands.",
            "metadata": {"source": "git-doc"},
        }
    ]

    store = InMemoryVectorStore(docs)
    chain = build_test_chain(store)

    question = "How do I use kubectl port-forward?"
    answer = chain.invoke({"question": question})

    # We don't assert exact wording, only that it admits uncertainty.
    lower = answer.lower()
    assert "don't know" in lower or "do not know" in lower or "not sure" in lower
