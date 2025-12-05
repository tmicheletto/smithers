#!/usr/bin/env python
"""List the contents of the vector store."""

from smithers.config import Settings
from smithers.rag.vector_search import VectorStore

rag_config = Settings().rag
store = VectorStore(store_name=rag_config.index_name)

if store.is_initialized():
    print(f'Vector Store: {rag_config.index_name}')
    print(f'Store ID: {store._vector_store_id}')
    print()
    
    files = store.list_files()
    print(f'Files in store: {len(files)}')
    print()
    
    for i, file in enumerate(files, 1):
        print(f'{i}. File ID: {file.id}')
        print(f'   Status: {file.status}')
        print(f'   Created at: {file.created_at}')
        print()
else:
    print('Vector store not found')
