#!/usr/bin/env python3
"""Test script for the RAG retriever."""

import logging

from src.smithers.config import settings
from src.smithers.rag.retriever import RAGRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test the retriever."""
    query = "unique features of 1 password"
    
    logger.info("Initializing retriever...")
    retriever = RAGRetriever()
    
    logger.info(f"Searching for: {query}")
    
    # Use search_knowledge_base which returns formatted context
    result = retriever.search_knowledge_base(query, k=3)
    
    print("\n" + "=" * 80)
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    main()
