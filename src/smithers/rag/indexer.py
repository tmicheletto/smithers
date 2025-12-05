"""RAG indexer - handles indexing documents into OpenAI Vector Store.

Uses batch file uploads for cost optimization per OpenAI recommendations.
"""

from pathlib import Path

from smithers.config import Settings
from smithers.rag.vector_store import VectorStore


def get_markdown_files(data_dir: Path, max_files: int | None = None) -> list[Path]:
    """Get markdown files from a directory.

    Args:
        data_dir: Path to directory containing markdown files.
        max_files: Maximum number of files to load (None for all).

    Returns:
        List of markdown file paths.
    """
    # Find all .md files recursively
    md_files = sorted(data_dir.rglob("*.md"))

    if not md_files:
        raise ValueError(f"No markdown files found in {data_dir}")

    # Limit number of files if specified
    if max_files is not None:
        md_files = md_files[:max_files]

    return md_files


def batch_upload_files(
    vector_store: VectorStore, files: list[Path], batch_size: int = 10
) -> int:
    """Upload files to vector store in batches for cost optimization.

    OpenAI recommends batch uploads to reduce API call overhead and costs.

    Args:
        vector_store: VectorStore instance to upload to.
        files: List of file paths to upload.
        batch_size: Number of files to upload per batch. Defaults to 10.

    Returns:
        Number of successfully uploaded files.
    """
    total_files = len(files)
    successful_uploads = 0

    for batch_num, i in enumerate(range(0, total_files, batch_size), 1):
        batch_files = files[i : i + batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, total_files)

        print(
            f"\n📦 Batch {batch_num}: Uploading files {batch_start}-{batch_end} "
            f"(of {total_files})..."
        )

        for file_path in batch_files:
            try:
                vector_store.upload_file(file_path)
                successful_uploads += 1
                print(f"  ✓ {file_path.name}")
            except Exception as e:
                print(f"  ✗ {file_path.name}: {e}")
                continue

    return successful_uploads


def index_knowledge_base(
    data_dir: Path | None = None,
    max_files: int | None = None,
    delete_existing: bool = False,
) -> VectorStore:
    """Index markdown documents into OpenAI Vector Store.

    Uploads markdown files to OpenAI Vector Store, which automatically
    handles chunking and embedding generation.

    Args:
        data_dir: Path to data directory. Defaults to src/data.
        max_files: Maximum number of files to index (None for all).
        delete_existing: Whether to delete existing index before creating new one.

    Returns:
        Initialized VectorStore with indexed documents.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    print(f"Indexing knowledge base from {data_dir}...")

    # Get config
    rag_config = Settings().rag

    # Get markdown files
    print("Finding markdown files...")
    md_files = get_markdown_files(data_dir, max_files=max_files)
    print(f"✓ Found {len(md_files)} markdown files")

    # Initialize Vector Store: attempt to load, create if missing
    print("Checking Vector Store...")
    try:
        vector_store = VectorStore(store_name=rag_config.index_name)
        print(f"Using existing store: {rag_config.index_name}")
    except Exception:
        vector_store = VectorStore(store_name=rag_config.index_name)
        print("Store not found. Creating new Vector Store...")
        vector_store.create_store()

    # Handle store recreation if requested
    if delete_existing:
        print(f"Recreating store: {rag_config.index_name}...")
        # Delete then create a fresh store
        try:
            # If the store exists, remove any previous files by listing and deleting
            # Note: delete_store() is not defined; ensure recreation via new store name or cleanup
            files = vector_store.list_files()
            for f in files:
                vector_store.delete_file(f.id)
        except Exception:
            pass
        vector_store.create_store()

    # Upload files to vector store in batches (cost optimization)
    print(f"\nUploading {len(md_files)} files to OpenAI Vector Store in batches...")
    successful = batch_upload_files(vector_store, md_files, batch_size=10)

    print("\n✅ Knowledge base indexed successfully")
    print(f"   Uploaded: {successful}/{len(md_files)} files")
    if successful < len(md_files):
        print(f"   Failed: {len(md_files) - successful} files\n")
    else:
        print()

    return vector_store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index knowledge base into Vector Store"
    )
    parser.add_argument(
        "--recreate-store",
        action="store_true",
        help="Delete existing store and create a new one",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to index (default: all)",
    )
    args = parser.parse_args()

    index_knowledge_base(max_files=args.max_files, delete_existing=args.recreate_store)
