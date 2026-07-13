import os
import asyncio
from typing import List, Dict, Any
from memory.db import MemoryDB, MemoryChunk

class MDFacilitator:
    def __init__(self, db: MemoryDB, embedding_provider):
        self.db = db
        self.embedding_provider = embedding_provider

    async def ingest_files(self, file_paths: List[str]):
        """Batch ingest markdown files into the vector store."""
        for path in file_paths:
            if not os.path.exists(path):
                continue
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self._split_content(content)
            
            # Batch embed and store
            for chunk in chunks:
                embedding = await self.embedding_provider.get_embedding(chunk)
                self.db.add_memory(MemoryChunk(
                    content=chunk,
                    embedding=embedding,
                    metadata={"source": path, "chunk_size": len(chunk)},
                    mem_type="DOC"
                ))

    def _split_content(self, content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple recursive character splitting for markdown."""
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunks.append(content[start:end])
            start += chunk_size - overlap
        return chunks
