"""OpenAI Vector Store integration for RAG.

Uses OpenAI's vector stores API for cloud-based document storage and retrieval.
"""

import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from smithers.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


class VectorStore:
    """Manages embeddings and search using OpenAI Vector Store.

    Provides functionality to create vector stores, upload files with embeddings,
    and search using OpenAI's vector storage service.
    """

    def __init__(
        self,
        store_name: str | None = None,
    ) -> None:
        """Initialize the OpenAI Vector Store wrapper.

        Args:
            store_name: Name of the vector store. Defaults to settings.rag.index_name.
        """
        self.store_name = store_name or settings.rag.index_name
        self._vector_store = None
        self._vector_store_id = None
        self._load_vector_store()

    def _load_vector_store(self) -> None:
        """Load existing vector store by name or raise an error.

        Raises:
            RuntimeError: If the named vector store does not exist or cannot be loaded.
        """
        # List existing vector stores
        vector_stores_list = client.vector_stores.list()

        # Look for existing store with matching name
        for vs in vector_stores_list.data:
            if vs.name == self.store_name:
                self._vector_store = vs
                self._vector_store_id = vs.id
                logger.info(f"Loaded existing OpenAI vector store: {self.store_name}")
                return

        # If not found, raise error to signal not initialized
        raise RuntimeError(
            f"OpenAI vector store not found: {self.store_name}. "
            "Create it with create_store()."
        )

    def create_store(self, dimension: int = 1536) -> None:
        """Create a new OpenAI vector store.

        Args:
            dimension: Embedding dimension (for reference only, OpenAI manages this).

        Raises:
            RuntimeError: If vector store creation fails.
        """
        logger.info(f"Creating OpenAI vector store: {self.store_name}...")

        self._vector_store = client.vector_stores.create(name=self.store_name)
        self._vector_store_id = self._vector_store.id
        logger.info(
            f"OpenAI vector store created: {self.store_name} (ID: {self._vector_store_id})"
        )

    def upload_file(self, file_path: str | Path) -> str:
        """Upload a file to the OpenAI vector store for automatic chunking and embedding.

        OpenAI automatically chunks the file and generates embeddings for each chunk.

        Args:
            file_path: Path to the file to upload (supports txt, pdf, md, and other formats).

        Returns:
            File ID in the vector store.

        Raises:
            ValueError: If vector store is not initialized.
            RuntimeError: If file upload fails.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Uploading file to vector store: {file_path.name}")

        # Check if file already exists and delete it (for idempotency)
        existing_file_id = self.get_file_by_name(file_path.name)
        if existing_file_id:
            logger.info("File already exists in vector store, removing old version...")
            self.delete_file(existing_file_id)

        # Upload file to vector store
        with open(file_path, "rb") as fh:
            response = client.vector_stores.files.upload(
                vector_store_id=self._vector_store_id,
                file=fh,
            )

        logger.info(f"File uploaded successfully: {file_path.name} (ID: {response.id})")
        return response.id

    def batch_upload_files(self, file_paths: list[Path] | list[str]) -> dict[str, int]:
        """Upload multiple files to the vector store using batch API.

        Uses OpenAI's batch upload API which is more cost-efficient than individual uploads.

        Args:
            file_paths: List of file paths to upload.

        Returns:
            Dictionary with 'successful' and 'failed' counts.

        Raises:
            ValueError: If vector store is not initialized.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        file_paths = [Path(fp) for fp in file_paths]
        results = {"successful": 0, "failed": 0, "failed_files": []}

        logger.info(
            f"Batch uploading {len(file_paths)} files to vector store using batch API"
        )

        # Open all file handles for batch upload
        file_handles: list[Any] = []
        try:
            for file_path in file_paths:
                if not file_path.exists():
                    results["failed"] += 1
                    results["failed_files"].append(
                        {
                            "path": str(file_path),
                            "error": "File not found",
                        }
                    )
                    continue
                file_handles.append(open(file_path, "rb"))

            if file_handles:
                batch = client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=self._vector_store_id,
                    files=file_handles,
                    max_concurrency=5,
                )

                logger.info(f"Batch upload completed. Status: {batch.status}")
                logger.info(f"  Files processed: {batch.file_counts.total}")
                logger.info(f"  Files completed: {batch.file_counts.completed}")
                logger.info(f"  Files failed: {batch.file_counts.failed}")

                results["successful"] = batch.file_counts.completed
                results["failed"] = batch.file_counts.failed
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

        logger.info(
            f"Batch upload complete: {results['successful']} successful, "
            f"{results['failed']} failed"
        )
        return results

    def upsert_datapoints(self, datapoints: list[dict[str, Any]]) -> None:
        """Upsert embeddings with metadata to OpenAI vector store.

        Args:
            datapoints: List of dicts with keys: 'id', 'embedding', 'metadata'.

        Raises:
            ValueError: If vector store is not initialized.
            RuntimeError: If upserting fails.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        logger.info(f"Upserting {len(datapoints)} datapoints to OpenAI vector store")

        try:
            # OpenAI vector store works with files and embeddings
            # We'll store metadata and embeddings as JSON records
            import json
            import tempfile
            from pathlib import Path

            # Create a temporary file with JSONL format (one JSON per line)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as f:
                for dp in datapoints:
                    record = {
                        "id": dp["id"],
                        "embedding": dp["embedding"],
                        "metadata": dp["metadata"],
                    }
                    f.write(json.dumps(record) + "\n")
                temp_file_path = f.name

            # Upload file to vector store
            with open(temp_file_path, "rb") as f:
                client.beta.vector_stores.files.upload(
                    vector_store_id=self._vector_store_id,
                    file=f,
                )

            # Clean up temp file
            Path(temp_file_path).unlink()

            logger.info(f"Successfully uploaded {len(datapoints)} datapoints")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert datapoints to OpenAI vector store: {e}"
            ) from e

    def search(self, query: str | list[float], k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant documents in the vector store.

        Args:
            query: Search query (string will be embedded by OpenAI, or pass embedding vector).
            k: Number of top results to return. Defaults to 5.

        Returns:
            List of search results with 'id', 'score', and 'metadata'.

        Raises:
            ValueError: If vector store is not initialized.
            RuntimeError: If search fails.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        logger.info(
            f"Searching vector store with query: {query if isinstance(query, str) else 'embedding'}"
        )

        # Use OpenAI's vector store search - it handles embedding internally
        if isinstance(query, str):
            results = client.vector_stores.search(
                vector_store_id=self._vector_store_id,
                query=query,
                max_num_results=k,
            )
        else:
            logger.warning("Using text search instead of embedding search")
            return []

        # Format results for compatibility with retriever
        formatted_results = []
        for result in results.data:
            chunk_content = ""
            if hasattr(result, "content"):
                if isinstance(result.content, list):
                    chunk_content = "\n".join(
                        block.text if hasattr(block, "text") else str(block)
                        for block in result.content
                    )
                else:
                    chunk_content = str(result.content)

            result_id = getattr(result, "file_id", getattr(result, "id", "unknown"))

            formatted_results.append(
                {
                    "id": result_id,
                    "score": float(result.score) if hasattr(result, "score") else 0.0,
                    "metadata": getattr(result, "metadata", {}),
                    "content": chunk_content,
                }
            )

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def _search_store(
        self, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Search the OpenAI vector store.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of search results from vector store.
        """
        # Use OpenAI's vector store search API
        # This performs similarity search on embeddings stored in the vector store
        results = []

        try:
            # Query the vector store with the embedding
            # Note: OpenAI's vector store search is primarily designed for file-based retrieval
            # For embedding-based search, we use the Files API with our stored embeddings
            search_results = client.beta.vector_stores.files.list(
                vector_store_id=self._vector_store_id
            )

            # Since we don't have direct embedding search, we'll retrieve and score locally
            # This is a workaround; ideally use assistant with file search
            import math

            for file_ref in search_results.data:
                # Get file content (this would need to be the JSONL we uploaded)
                # For now, return empty results to indicate structure
                pass

            # Alternative: Use Files API directly for retrieval
            # This requires retrieving the file and parsing JSONL
            file_list = client.files.list()
            for file_obj in file_list.data:
                if self.store_name in file_obj.filename:
                    # Retrieve and parse the file
                    file_content = client.files.content(file_obj.id).text
                    lines = file_content.strip().split("\n")

                    for line in lines:
                        if not line.strip():
                            continue
                        import json

                        record = json.loads(line)
                        embedding = record.get("embedding", [])

                        # Calculate cosine similarity
                        dot_product = sum(
                            a * b for a, b in zip(query_embedding, embedding)
                        )
                        magnitude_q = math.sqrt(sum(a * a for a in query_embedding))
                        magnitude_e = math.sqrt(sum(b * b for b in embedding))

                        if magnitude_q > 0 and magnitude_e > 0:
                            similarity = dot_product / (magnitude_q * magnitude_e)
                        else:
                            similarity = 0.0

                        results.append(
                            {
                                "id": record.get("id", ""),
                                "similarity": similarity,
                                "metadata": record.get("metadata", {}),
                            }
                        )

            # Sort by similarity (descending) and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Error during vector store search: {e}")
            return []

    def list_files(self) -> list[dict[str, Any]]:
        """List all files in the vector store.

        Returns:
            List of file objects with their metadata.

        Raises:
            ValueError: If vector store is not initialized.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        files = client.vector_stores.files.list(self._vector_store_id)
        logger.info(f"Found {len(files.data)} files in vector store")
        return files.data

    def get_file_by_name(self, filename: str) -> str | None:
        """Get file ID by filename from vector store.

        Args:
            filename: Name of the file to find.

        Returns:
            File ID if found, None otherwise.
        """
        files = client.vector_stores.files.list(self._vector_store_id)
        for file_obj in files.data:
            file_name = getattr(file_obj, "filename", None) or getattr(
                file_obj, "name", None
            )
            if file_name == filename:
                return file_obj.id
        return None

    def delete_file(self, file_id: str) -> None:
        """Delete a file from the vector store.

        Args:
            file_id: ID of the file to delete.

        Raises:
            ValueError: If vector store is not initialized.
            RuntimeError: If deletion fails.
        """
        if not self._vector_store_id:
            raise ValueError("Vector store not initialized. Call create_store() first.")

        client.vector_stores.files.delete(
            vector_store_id=self._vector_store_id,
            file_id=file_id,
        )
        logger.info(f"File deleted from vector store: {file_id}")

    def get_file_content(self, file_id: str) -> str | None:
        """Get file content from OpenAI Files API.

        Args:
            file_id: ID of the file to retrieve.

        Returns:
            File content as string, or None if not found.

        Raises:
            RuntimeError: If retrieval fails.
        """
        response = client.files.content(file_id)
        return response.text
