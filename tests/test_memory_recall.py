"""Recall quality: natural-language keyword search, rank fusion, notes, embeddings."""
from memory.db import EMBEDDING_DIM, MemoryChunk, MemoryDB, NoteStore, fts_query
from memory.retriever import HybridRetriever
from core.providers import ModelConfig
from main import MotionAgent


def add(db, content, vec=None, mem_type="DOC"):
    db.add_memory(MemoryChunk(content, vec or [0.0] * EMBEDDING_DIM, {}, mem_type))


def test_fts_query_builds_or_of_significant_terms():
    assert fts_query("how do we deploy to staging?") == '"deploy" OR "staging"'
    assert fts_query("Find the AUTH_TOKEN_SESS variable") == '"Find" OR "AUTH_TOKEN_SESS" OR "variable"'
    assert fts_query("the of and") == "" and fts_query("") == ""
    assert fts_query('say "hi" AND (x)').count("OR") >= 1  # punctuation/operators are neutralised
    assert len(fts_query(" ".join(f"term{i}" for i in range(50))).split(" OR ")) == 16


def test_natural_language_question_now_matches_stored_fact():
    db = MemoryDB(":memory:")
    add(db, "The deployment uses blue-green rollout on the staging cluster")
    assert db.keyword_search("how do we deploy to staging?")  # was [] with whole-prompt phrase matching
    assert db.keyword_search("blue-green rollout")
    assert db.keyword_search("unrelated gibberish query") == []


def test_keyword_search_survives_fts_syntax_in_user_text():
    db = MemoryDB(":memory:")
    add(db, "plain memory about parsing")
    for nasty in ('"unbalanced', "a AND OR NOT", "col:val*", "NEAR(", "'; DROP TABLE memories; --"):
        db.keyword_search(nasty)  # must not raise


class FakeEmbedder:
    def __init__(self, vec, semantic=True):
        self.vec, self.semantic_available, self.calls = vec, semantic, 0

    async def get_embedding(self, text):
        self.calls += 1
        return self.vec


async def test_retriever_skips_embedding_for_empty_db_and_blank_query():
    db, emb = MemoryDB(":memory:"), FakeEmbedder([1.0] + [0.0] * (EMBEDDING_DIM - 1))
    r = HybridRetriever(db, emb)
    assert await r.retrieve("anything at all") == [] and emb.calls == 0
    add(db, "something")
    assert await r.retrieve("   ") == [] and emb.calls == 0


async def test_retriever_ignores_semantic_search_when_embeddings_are_fake():
    """Hash 'embeddings' are all-positive, so every memory looks similar to every query."""
    db = MemoryDB(":memory:")
    fake = [1.0 / EMBEDDING_DIM ** 0.5] * EMBEDDING_DIM
    add(db, "Completely unrelated memory about bananas", fake)
    r = HybridRetriever(db, FakeEmbedder(fake, semantic=False))
    assert await r.retrieve("kubernetes ingress") == []
    r2 = HybridRetriever(db, FakeEmbedder(fake, semantic=True))
    assert await r2.retrieve("kubernetes ingress")  # (semantic on: proves the gate is what filtered it)


async def test_rank_fusion_prefers_items_found_by_both_retrievers():
    db = MemoryDB(":memory:")
    vec = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    add(db, "alpha keyword only memory about tokens", [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2))
    add(db, "vector only memory zzz", vec)
    add(db, "tokens plus vector memory", vec)
    r = HybridRetriever(db, FakeEmbedder(vec))
    out = await r.retrieve("tokens", top_k=3)
    assert out[0]["content"] == "tokens plus vector memory" and out[0]["type"] == "hybrid"
    assert {o["content"] for o in out} >= {"tokens plus vector memory", "vector only memory zzz"}


def test_notes_are_persistent_keyed_and_replaceable():
    db = MemoryDB(":memory:")
    notes = NoteStore(db)
    notes.save("api", "use v2")
    notes.save("api", "use v3")
    notes.save("db", "sqlite")
    assert notes.get("api") == "use v3" and notes.get("db") == "sqlite" and notes.get("nope") is None
    assert db.count() == 2  # replaced, not duplicated
    assert db.keyword_search("sqlite")  # findable by later recall


def test_delete_memory_removes_from_all_indexes():
    db = MemoryDB(":memory:")
    add(db, "temporary secret fact")
    mid = db.conn.execute("SELECT id FROM memories").fetchone()[0]
    db.delete_memory(mid)
    assert db.count() == 0 and db.keyword_search("secret") == []


async def test_agent_embeddings_are_cached_resized_and_flagged():
    agent = MotionAgent(ModelConfig(name="x", endpoint="http://localhost:11434", provider_type="local"), memory_path=":memory:")
    calls = []

    async def embed(text):
        calls.append(text)
        return [1.0] * 768  # e.g. nomic-embed-text; the DB is 128-wide

    agent.provider.embed = embed
    v1 = await agent.get_embedding("hello")
    v2 = await agent.get_embedding("hello")
    assert len(v1) == EMBEDDING_DIM and v1 == v2 and calls == ["hello"]
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6 and agent.semantic_available
    agent.memory.add_memory(MemoryChunk("stored", v1, {}, "DOC"))  # would raise on a 768-dim vector


async def test_agent_without_embedder_falls_back_and_says_so():
    agent = MotionAgent(ModelConfig(name="c", endpoint="https://ollama.com/v1", provider_type="cloud"), memory_path=":memory:")
    vec = await agent.get_embedding("hello")
    assert len(vec) == EMBEDDING_DIM and not agent.semantic_available


async def test_agent_embedding_failure_degrades_to_keyword_only():
    agent = MotionAgent(ModelConfig(name="x", endpoint="http://localhost:1", provider_type="local"), memory_path=":memory:")

    async def boom(text):
        raise RuntimeError("ollama down")

    agent.provider.embed = boom
    await agent.get_embedding("q")
    assert not agent.semantic_available
