import asyncio
from unittest.mock import AsyncMock, MagicMock
from memory.db import MemoryDB, MemoryChunk
from memory.facilitator import MDFacilitator
from memory.retriever import HybridRetriever

class MockEmbeddingProvider:
    async def get_embedding(self, text: str):
        # Return a simple deterministic vector based on length for testing
        return [float(len(text)) / 100.0] * 128

async def test_memory_pipeline():
    # Setup
    db = MemoryDB(":memory:") # Use in-memory DB for tests
    # Force DB init just in case
    db._init_db()
    emb_provider = MockEmbeddingProvider()
    facilitator = MDFacilitator(db, emb_provider)
    retriever = HybridRetriever(db, emb_provider)

    # 1. Test Ingestion (Facilitator)
    # Create a dummy md file
    with open("test_doc.md", "w") as f:
        f.write("The Motion Harness uses a hybrid retrieval system with FTS5 and Vector DB.")

    await facilitator.ingest_files(["test_doc.md"])
    
    # 2. Test Keyword Search (Sparse)
    # 'FTS5' is a very specific keyword
    results = await retriever.retrieve("FTS5")
    assert len(results) > 0
    assert "FTS5" in results[0]["content"]

    # 3. Test Semantic Search (Dense)
    # 'retrieval system' should match conceptually
    results = await retriever.retrieve("how does it find information?")
    assert len(results) > 0
    
    print("✅ Memory pipeline tests passed!")

if __name__ == "__main__":
    asyncio.run(test_memory_pipeline())
