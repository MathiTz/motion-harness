import asyncio
from pathlib import Path
import tempfile
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
    # Create a dummy md file in an isolated temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / "test_doc.md"
        doc_path.write_text(
            "The Motion Harness uses a hybrid retrieval system with FTS5 and Vector DB.",
            encoding="utf-8",
        )
        await facilitator.ingest_files([str(doc_path)])
    
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

async def test_semantic_search_skips_zero_norm_embeddings():
    """Regression: cosine distance is undefined (NULL) for a zero-norm
    embedding. semantic_search must skip such rows instead of crashing with
    `unsupported operand type(s) for -: 'float' and 'NoneType'` - this
    happened in practice because SkillSynthesizer used to hardcode
    embedding=[0.0]*128 for every synthesized skill.
    """
    from memory.db import MemoryDB, MemoryChunk, EMBEDDING_DIM

    class _FixedEmbeddingProvider:
        async def get_embedding(self, text: str):
            return [1.0] + [0.0] * (EMBEDDING_DIM - 1)

    db = MemoryDB(":memory:")
    db.add_memory(MemoryChunk(
        content="Zero vector placeholder memory",
        embedding=[0.0] * EMBEDDING_DIM,
        metadata={},
        mem_type="DOC",
    ))
    db.add_memory(MemoryChunk(
        content="Real memory with a real embedding",
        embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
        metadata={},
        mem_type="DOC",
    ))

    retriever = HybridRetriever(db, _FixedEmbeddingProvider())
    # Must not raise.
    results = await retriever.retrieve("anything", top_k=5)
    contents = [r["content"] for r in results]
    assert "Zero vector placeholder memory" not in contents
    assert "Real memory with a real embedding" in contents

    print("✅ Zero-norm embedding regression test passed!")


async def test_skill_synthesizer_never_stores_zero_embedding():
    """Regression: SkillSynthesizer used to hardcode embedding=[0.0]*128."""
    from unittest.mock import AsyncMock
    from core.learning import SkillSynthesizer, Trajectory
    from core.providers import ModelConfig
    from memory.db import MemoryDB, EMBEDDING_DIM
    import shutil
    import struct
    import tempfile

    model_config = ModelConfig(name="test", endpoint="http://localhost", provider_type="local")
    db = MemoryDB(":memory:")
    tmp_skills_dir = tempfile.mkdtemp()
    try:
        synthesizer = SkillSynthesizer(model_config, db, skills_dir=tmp_skills_dir, embedding_provider=None)
        synthesizer.provider = AsyncMock()
        synthesizer.provider.complete.return_value = "# Skill\n## Description\nTest\n## Procedure\n1. Do it."

        trajectory = Trajectory(
            task_id="t1", prompt="test skill", steps=[], final_result="done", success=True,
        )
        await synthesizer.synthesize(trajectory)

        row = db.conn.execute("SELECT embedding FROM memories ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        vals = struct.unpack(f"<{EMBEDDING_DIM}f", row[0])
        norm = sum(v * v for v in vals) ** 0.5
        assert norm > 0, "synthesized skill must not be stored with a zero-norm embedding"
    finally:
        shutil.rmtree(tmp_skills_dir, ignore_errors=True)
        db.close()

    print("✅ Skill synthesizer non-zero embedding regression test passed!")


async def test_retriever_drops_unrelated_semantic_matches_and_dedupes():
    """Regression: a low-similarity semantic hit must not be blindly
    injected as context, and duplicate content (e.g. from duplicate DB rows)
    must be collapsed to one entry. This is what let an unrelated, heavily
    duplicated memory hijack unrelated conversations.
    """
    from memory.db import MemoryDB, MemoryChunk, EMBEDDING_DIM

    class _FixedEmbeddingProvider:
        def __init__(self, vector):
            self.vector = vector

        async def get_embedding(self, text: str):
            return self.vector

    db = MemoryDB(":memory:")
    query_vec = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    # "Relevant" memory: identical direction to the query -> similarity ~1.0.
    db.add_memory(MemoryChunk(
        content="Relevant memory about the actual topic.",
        embedding=query_vec,
        metadata={},
        mem_type="DOC",
    ))
    # "Unrelated" memory: orthogonal-ish vector -> low similarity, should be
    # filtered out by the relevance threshold instead of always appearing.
    unrelated_vec = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    for _ in range(3):
        db.add_memory(MemoryChunk(
            content="The bug in line 42 is caused by a null pointer in the handler.",
            embedding=unrelated_vec,
            metadata={},
            mem_type="DOC",
        ))

    retriever = HybridRetriever(db, _FixedEmbeddingProvider(query_vec))
    results = await retriever.retrieve("actual topic", top_k=5)

    contents = [r["content"] for r in results]
    assert "Relevant memory about the actual topic." in contents
    assert "The bug in line 42 is caused by a null pointer in the handler." not in contents

    print("✅ Retriever relevance/dedup regression test passed!")


if __name__ == "__main__":
    asyncio.run(test_memory_pipeline())
    asyncio.run(test_retriever_drops_unrelated_semantic_matches_and_dedupes())
