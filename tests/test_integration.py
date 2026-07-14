import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.providers import ModelConfig, ProviderFactory, LocalProvider, CloudProvider
from core.caveman import CavemanProtocol, CavemanCompressor
from memory.db import MemoryDB, MemoryChunk, EMBEDDING_DIM
from memory.retriever import HybridRetriever
from main import MotionAgent
import hashlib


class MockProvider:
    async def complete(self, prompt, system_prompt="", **kwargs):
        return "Certainly! I have analyzed the files and found the bug is in line 42. I'm sorry for the inconvenience."

    async def close(self):
        pass


class MockEmbeddingProvider:
    async def get_embedding(self, text):
        # Produce a deterministic EMBEDDING_DIM-length vector from the text hash.
        # We cycle the SHA-256 hash bytes to fill all 128 dimensions.
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h]  # 32 floats from 32 bytes
        vec = (raw * ((EMBEDDING_DIM // len(raw)) + 1))[:EMBEDDING_DIM]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


async def test_golden_path():
    """
    Tests the full 'Golden Path':
    User Prompt -> Hybrid Recall -> Model completion -> Caveman Compression -> Skill Crystallization
    """
    model_config = ModelConfig(name="test-model", endpoint="http://localhost", provider_type="local")
    agent = MotionAgent(model_config)
    agent.provider = MockProvider()

    # Add a memory to test recall
    agent.memory.add_memory(MemoryChunk(
        content="The bug in line 42 is caused by a null pointer in the handler.",
        embedding=[0.1] * EMBEDDING_DIM,
        metadata={"source": "test.md"},
        mem_type="DOC"
    ))

    # 2. Execution
    prompt = "Where is the bug?"
    response = await agent.run(prompt, target="agent")

    # 3. Assertions
    # Caveman should strip fluff from the MockProvider's response
    assert "Certainly!" not in response
    assert "bug is in line 42" in response

    print("✅ Golden Path integration test passed!")


async def test_caveman_bidirectional():
    """
    Tests that Caveman compression is reversible via expand().
    """
    original = "Certainly! I have analyzed the files and found that the bug is in line 42. I'm sorry for the inconvenience."
    compressed, fragments = CavemanCompressor.compress(original)
    expanded = CavemanCompressor.expand(compressed, fragments)

    # Compressed should be shorter
    assert len(compressed) < len(original)
    # Expanded should contain the original core content
    assert "bug is in line 42" in expanded
    # Fluff should be restored
    assert "Certainly!" in expanded

    print("✅ Caveman bidirectional test passed!")


async def test_hybrid_retrieval():
    """
    Tests that keyword search finds exact matches in the memory DB.
    """
    db = MemoryDB(":memory:")
    emb_provider = MockEmbeddingProvider()
    retriever = HybridRetriever(db, emb_provider)

    db.add_memory(MemoryChunk(
        content="The FTS5 module enables fast full-text search.",
        embedding=[0.1] * EMBEDDING_DIM,
        metadata={"source": "docs"},
        mem_type="DOC"
    ))

    results = await retriever.retrieve("FTS5")
    assert len(results) > 0
    assert "FTS5" in results[0]["content"]

    print("✅ Hybrid retrieval test passed!")


if __name__ == "__main__":
    asyncio.run(test_golden_path())
    asyncio.run(test_caveman_bidirectional())
    asyncio.run(test_hybrid_retrieval())
