from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.db import MemoryDB


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """SQLite's CURRENT_TIMESTAMP is space-separated ("2026-09-30 02:46:09"); code that seeds a
    specific timestamp (tests, evals/memory_quality.py) tends to write ISO-8601 ("T"-separated).
    Accept either; None (or anything unparseable) means "age unknown", not "just now" - a memory
    with no readable timestamp must not silently look fresher than it is."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace(" ", "T", 1) if "T" not in value else value)
    except ValueError:
        return None


def format_age(timestamp: Optional[str], now: Optional[datetime] = None) -> Optional[str]:
    """A short, human age label ("today", "3d ago", "2y ago") for a memory's timestamp, or None if
    it can't be parsed. Surfacing this (issue #18's follow-up) is what lets a model prefer a recent
    fact over a stale one that merely scores similarly - text alone doesn't carry that signal."""
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return None
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    days = (now - parsed).days
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{days // 30}mo ago"
    return f"{days // 365}y ago"


class HybridRetriever:
    # Semantic matches below this cosine-similarity score are dropped. Both
    # backends behind MemoryDB.semantic_search report true cosine similarity
    # (the sqlite-vec path uses distance_metric=cosine, converted back via
    # 1 - distance; the brute-force fallback computes cosine similarity
    # directly), so this threshold is meaningful and consistent regardless
    # of which backend is active. Orthogonal (unrelated) vectors score 0.0.
    # Keyword (FTS5) hits are exempt since they already require a real
    # textual match.
    #
    # This floor alone does not mean "confident" - 0.3 is deliberately
    # permissive (semantic search should not go silent just because nothing
    # scored above some arbitrary "good" bar). A raw score just above this
    # floor (e.g. 0.4) is a real but weak match, and retrieve() now reports
    # that raw score on every semantic/hybrid result instead of only the
    # fused rank score, which has no absolute meaning at all (see below) -
    # so a caller (or the model reading the formatted memory context) can
    # tell a borderline match from a strong one instead of the two looking
    # identical once they're both just "recalled content."
    DEFAULT_MIN_SEMANTIC_SCORE = 0.3
    # Below this raw cosine score, a semantic match is real but weak enough that presenting it as
    # plain fact (core/agent_loop.py's _format_recalled_chunk) risks the exact failure a maintainer
    # flagged: a ~0.4-confidence guess aggregated into the prompt looking exactly as authoritative
    # as a strong match. Chosen as roughly the midpoint between the floor above (0.3, deliberately
    # permissive) and a score that's actually a good match; like RECENCY_TIE_MARGIN below, this is
    # an explicit, revisitable choice, not something derived from this harness's own embedding
    # model's real score distribution - it has none on hand to calibrate against.
    WEAK_SEMANTIC_SCORE = 0.5
    # Reciprocal-rank-fusion constant (standard value); dampens the influence
    # of any single list's top ranks so neither retriever dominates.
    RRF_K = 60
    # If the top-ranked result and a runner-up are within this fraction of each other's fused
    # score, prefer the more recent one instead of leaving the order to incidental fusion
    # arithmetic. Fused scores are rank-based (roughly 1/60ish per contributing list), not a
    # meaningful confidence number, so two topically-similar but factually conflicting memories
    # (an old decision and the one that superseded it) routinely land within a percent or two of
    # each other - exactly the real, measured case in docs/memory-quality.md (0.0164 vs 0.0161,
    # a ~2% gap) where the STALE one ranked first. 15% is a deliberately generous margin to catch
    # that kind of near-tie; it is a starting point, not a calibrated constant - revisit if it
    # turns out to reorder results that were never actually competing for the same fact.
    RECENCY_TIE_MARGIN = 0.15

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
                (score, content, ts)
                for score, content, ts in self.db.semantic_search(query_emb, limit=top_k * 2)
                if score >= self.min_semantic_score
            ]

        # 3. Reciprocal rank fusion. BM25 ranks and cosine scores are not
        # comparable, but ranks are.
        fused: Dict[str, Dict[str, Any]] = {}
        for kind, results in (("semantic", semantic_results), ("keyword", keyword_results)):
            for rank, (score, content, timestamp) in enumerate(results):
                entry = fused.setdefault(content, {
                    "score": 0.0, "content": content, "type": kind,
                    "raw_score": None, "timestamp": timestamp,
                })
                entry["score"] += 1.0 / (self.RRF_K + rank + 1)
                if entry["type"] != kind:
                    entry["type"] = "hybrid"
                if kind == "semantic":
                    entry["raw_score"] = max(entry["raw_score"] or 0.0, score)
                if not entry.get("timestamp"):
                    entry["timestamp"] = timestamp

        for entry in fused.values():
            entry["age"] = format_age(entry.get("timestamp"))

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return self._break_ties_by_recency(ranked)[:top_k]

    def _break_ties_by_recency(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Within a leading group of near-tied fused scores (see RECENCY_TIE_MARGIN), reorder by
        timestamp (most recent first; unparseable/missing timestamps sort last, never assumed
        recent). Only the leading tie group moves - a clear winner is never displaced by ties
        further down the list."""
        if len(ranked) < 2 or not ranked[0]["score"]:
            return ranked
        top_score = ranked[0]["score"]
        tie_count = 1
        for entry in ranked[1:]:
            if top_score - entry["score"] <= top_score * self.RECENCY_TIE_MARGIN:
                tie_count += 1
            else:
                break
        if tie_count < 2:
            return ranked
        tied = ranked[:tie_count]
        tied.sort(key=lambda e: _parse_timestamp(e.get("timestamp")) or datetime.min, reverse=True)
        return tied + ranked[tie_count:]
