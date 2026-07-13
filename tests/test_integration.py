import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.providers import ModelConfig, ProviderFactory
from core.caveman import CavemanProtocol
from memory.db import MemoryDB, MemoryChunk
from memory.retriever import HybridRetriever
from main import MotionAgent

class MockProvider:
    async def complete(self, prompt, system_prompt="", **kwargs):
        return "Certainly! I have analyzed the files and found the bug is in line 42. I'm sorry for the inconvenience."

class MockEmbeddingProvider:
    async def get_embedding(self, text):
        return [0.1] * 128

async def test_golden_path():
    """
    Tests the full 'Golden Path':
    User Prompt -> Hybrid Recall -> Model Completion -> Caveman Compression -> Final Output
    """
    # 1. Setup
    model_config = ModelConfig(name="test-model", endpoint="http://localhost", provider_type="local")
    # We monkeypatch the provider to use our mock
    agent = MotionAgent(model_config)
    agent.provider = MockProvider()
    agent.embedding_provider = MockEmbeddingProvider() # For the retriever
    
    # Add a dummy memory to test recall
    agent.memory.add_memory(MemoryChunk(
        content="The bug in line 42 is caused by a null pointer in the handler.",
        embedding=[0.1]*128,
        metadata={"source": "test.md"},
        mem_type="DOC"
    ))

    # 2. Execution
    prompt = "Where is the bug?"
    response = await agent.run(prompt, target="agent") # target="agent" to trigger Caveman

    # 3. Assertions
    # Check if Caveman stripped the fluff from the MockProvider's response
    assert "Certainly!" not in response
    assert "I'm sorry" not in response
    assert "bug is in line 42" in response
    
    print("✅ Golden Path integration test passed!")

if __name__ == "__main__":
    # Simple runner for the integration test
    asyncio.run(test_golden_path())
