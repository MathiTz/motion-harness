from typing import List, Dict, Any
from memory.db import MemoryDB


class HybridRetriever:
    # Semantic matches below this cosine-similarity score are dropped. Both
    # backends behind MemoryDB.semantic_search report true cosine similarity
    # (the sqlite-vec path uses distance_metric=cosine, converted back via
    # 1 - distance; the brute-force fallback computes cosine similarity
    # directly), so this threshold is meaningful and consistent regardless
    # of which backend is active. Orthogonal (unrelated) vectors score 0.0.
    # Keyword (FTS5) hits are exempt since they already require a real
    # textual match.
    DEFAULT_MIN_SEMANTIC_SCORE = 0.3
    # Reciprocal-rank-fusion constant (standard value); dampens the influence
    # of any single list's top ranks so neither retriever dominates.
    RRF_K = 60

    def __init__(self, db: MemoryDB, embedding_provider, min_semantic_score: float = DEFAULT_MIN_SEMANTIC_SCORE):
        self.db = db
        self.embedding_provider = embedding_provider
        self.min_semantic_score = min_semantic_score

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Nothing stored (or nothing to search for): skip the embedding call.
        if not (query or "").strip() or self.db.count() == 0:
            return []

        # 1. Sparse: FTS5 keyword search (OR of significant terms, BM25-ranked).
        keyword_results = self.db.keyword_search(query, limit=top_k * 2)

        # 2. Dense: only when the embedder produces real semantic vectors. The
        # hash-based fallback has no meaning (all its vectors look alike), so
        # searching with it would inject unrelated memories.
        semantic_results = []
        query_emb = await self.embedding_provider.get_embedding(query)
        if getattr(self.embedding_provider, "semantic_available", True):
            semantic_results = [
                (score, content)
                for score, content in self.db.semantic_search(query_emb, limit=top_k * 2)
                if score >= self.min_semantic_score
            ]

        # 3. Reciprocal rank fusion. BM25 ranks and cosine scores are not
        # comparable, but ranks are.
        fused: Dict[str, Dict[str, Any]] = {}
        for kind, results in (("semantic", semantic_results), ("keyword", keyword_results)):
            for rank, (_score, content) in enumerate(results):
                entry = fused.setdefault(content, {"score": 0.0, "content": content, "type": kind})
                entry["score"] += 1.0 / (self.RRF_K + rank + 1)
                if entry["type"] != kind:
                    entry["type"] = "hybrid"

        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
