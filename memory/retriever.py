from typing import List, Tuple, Dict, Any
from memory.db import MemoryDB

class HybridRetriever:
    # Semantic matches below this cosine-similarity score are dropped. Both
    # backends behind MemoryDB.semantic_search report true cosine similarity
    # (the sqlite-vec path uses distance_metric=cosine, converted back via
    # 1 - distance; the brute-force fallback computes cosine similarity
    # directly), so this threshold is meaningful and consistent regardless
    # of which backend is active. Orthogonal (unrelated) vectors score 0.0.
    # This matters most for the hash-based embedding fallback (used by cloud
    # providers without a real embedding endpoint), which can otherwise rank
    # completely unrelated memories as "nearest neighbors" and inject them
    # into every unrelated conversation. Keyword (FTS5) hits are exempt since
    # they already require a real textual match.
    DEFAULT_MIN_SEMANTIC_SCORE = 0.3

    def __init__(self, db: MemoryDB, embedding_provider, min_semantic_score: float = DEFAULT_MIN_SEMANTIC_SCORE):
        self.db = db
        self.embedding_provider = embedding_provider
        self.min_semantic_score = min_semantic_score

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Generate query embedding
        query_emb = await self.embedding_provider.get_embedding(query)
        
        # 2. Semantic Search (Dense)
        semantic_results = self.db.semantic_search(query_emb, limit=top_k * 2)
        
        # 3. Keyword Search (Sparse)
        keyword_results = self.db.keyword_search(query, limit=top_k * 2)
        
        # 4. Hybrid Merge & Rerank
        # Simple reciprocal rank fusion or just merging for this foundation
        merged = []
        for score, content in semantic_results:
            merged.append({"score": score, "content": content, "type": "semantic"})
        for score, content in keyword_results:
            # FTS5 rank is lower = better, so we invert it for merging
            merged.append({"score": -score, "content": content, "type": "keyword"})

        # Drop weakly-related semantic matches (see DEFAULT_MIN_SEMANTIC_SCORE).
        merged = [
            item for item in merged
            if item["type"] == "keyword" or item["score"] >= self.min_semantic_score
        ]

        # De-duplicate identical content - the same memory can surface from
        # both search types, and duplicate rows can exist in the DB.
        seen = set()
        deduped = []
        for item in merged:
            if item["content"] in seen:
                continue
            seen.add(item["content"])
            deduped.append(item)

        # Sort by score descending
        deduped.sort(key=lambda x: x["score"], reverse=True)
        
        return deduped[:top_k]
