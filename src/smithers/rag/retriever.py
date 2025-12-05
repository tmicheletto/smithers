"""LangChain retriever wrapper delegating to existing VectorStore.

This implements Option A: reuse the current OpenAI Vector Store backend
via `VectorStore.search()` and expose results as LangChain `Document`s
through the `BaseRetriever` interface. This avoids reindexing while
enabling LangChain chains to consume retrieval outputs.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from smithers.rag.vector_store import VectorStore


class Retriever(BaseRetriever):
    """LangChain `BaseRetriever` that delegates to `VectorStore`.

    The retriever converts vector store search results into LangChain
    `Document` objects with `page_content` and `metadata` fields.
    """

    def __init__(self, *, k: int = 4) -> None:
        """Initialize the retriever.

        Parameters:
            k: Default number of documents to retrieve.
        """
        super().__init__()
        self._k = k
        self._vector_store = VectorStore()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve top-k relevant documents for the given query.

        Delegates to `VectorStore.search()` and maps results to `Document`s.
        """
        results = self._vector_store.search(query=query, k=self._k)

        docs: List[Document] = []
        for item in results:
            # Expecting each item to contain text and metadata fields.
            text = item.get("text") or item.get("content") or ""
            metadata = item.get("metadata") or {}
            # Normalize common metadata keys from OpenAI vector store responses.
            # Keep source, file_id, chunk_index if present.
            normalized_meta = {
                key: metadata.get(key)
                for key in [
                    "source",
                    "file_id",
                    "chunk_index",
                    "score",
                    "vector_store_id",
                    "document_id",
                    "path",
                    "title",
                ]
                if key in metadata
            }
            docs.append(Document(page_content=text, metadata=normalized_meta))

        return docs
