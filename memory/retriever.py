from typing import List, Tuple, Dict, Any
from memory.db import MemoryDB

class HybridRetriever:
    def __init__(self, db: MemoryDB, embedding_provider):
        self.db = db
        self.embedding_provider = embedding_provider

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
            
        # Sort by score descending
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged[:top_k]
