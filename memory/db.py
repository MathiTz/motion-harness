import re
import sqlite3
import json
import struct
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

EMBEDDING_DIM = 128

# Words that carry no retrieval signal. Removing them keeps natural-language
# questions ("how do we deploy to staging?") from needing every word to match.
_STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in into is it its "
    "me my of on or our should so than that the their them then there these they this to up us was we were "
    "what when where which who why will with would you your please about just like want need make get use "
    "using help show tell give let lets can't don't isn't".split()
)


def fts_query(text: str, max_terms: int = 16) -> str:
    """Turn free text into an FTS5 OR-query of quoted terms ("" if nothing useful).

    The previous implementation searched the entire prompt as ONE exact phrase,
    which matches essentially never for a natural-language question.
    """
    seen = set()
    terms = []
    for tok in re.findall(r"[A-Za-z0-9_]{2,}", text or ""):
        low = tok.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(f'"{tok}"')
        if len(terms) >= max_terms:
            break
    return " OR ".join(terms)

@dataclass
class MemoryChunk:
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    mem_type: str

class MemoryDB:
    def __init__(self, db_path: str = "motion_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.conn.enable_load_extension(True)
        try:
            import sqlite_vec
            self.conn.load_extension(sqlite_vec.loadable_path())
            self._vec_available = True
        except Exception:
            self._vec_available = False
        self.conn.enable_load_extension(False)

        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, metadata)")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding BLOB,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mem_type TEXT
        )""")
        if self._vec_available:
            dim = EMBEDDING_DIM
            # Use cosine distance explicitly (vec0 defaults to L2, which is
            # magnitude-sensitive and was previously being misread as a
            # higher-is-better similarity score by HybridRetriever, silently
            # ranking the LEAST similar memories first). Cosine distance
            # matches the brute-force fallback's cosine similarity exactly.
            existing_sql = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'memories_vec'"
            ).fetchone()
            needs_migration = existing_sql is not None and "distance_metric=cosine" not in (existing_sql[0] or "")
            if needs_migration:
                self.conn.execute("DROP TABLE memories_vec")
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )
            if needs_migration:
                # Re-populate the rebuilt index from the durable `memories`
                # table so existing data survives the metric migration.
                for mem_id, emb_blob in self.conn.execute("SELECT id, embedding FROM memories").fetchall():
                    self.conn.execute(
                        "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                        (mem_id, emb_blob),
                    )
        self.conn.commit()

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        return struct.pack(f"<{len(embedding)}f", *embedding)

    def add_memory(self, chunk: MemoryChunk):
        embedding_blob = self._serialize_embedding(chunk.embedding)
        metadata_json = json.dumps(chunk.metadata)
        cursor = self.conn.execute(
            "INSERT INTO memories (content, embedding, metadata, mem_type) VALUES (?, ?, ?, ?)",
            (chunk.content, embedding_blob, metadata_json, chunk.mem_type),
        )
        mem_id = cursor.lastrowid
        self.conn.execute(
            "INSERT INTO memories_fts (rowid, content, metadata) VALUES (?, ?, ?)",
            (mem_id, chunk.content, metadata_json),
        )
        if self._vec_available:
            self.conn.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (mem_id, self._serialize_embedding(chunk.embedding)),
            )
        self.conn.commit()

    def keyword_search(self, query: str, limit: int = 5) -> List[Tuple[float, str]]:
        match = fts_query(query)
        if not match:
            return []
        sql = "SELECT rank, content FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?"
        try:
            return self.conn.execute(sql, (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def delete_memory(self, mem_id: int) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (mem_id,))
        if self._vec_available:
            self.conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (mem_id,))
        self.conn.commit()

    # ── notes (memory_save / memory_get tools) ───────────────────────────
    def save_note(self, key: str, text: str) -> None:
        """Persist a keyed note. Notes are findable by keyword recall; they get
        a zero vector (no semantic signal), which semantic search skips."""
        for (mem_id,) in self.conn.execute(
            "SELECT id FROM memories WHERE mem_type = 'NOTE' AND json_extract(metadata, '$.note_key') = ?", (key,)
        ).fetchall():
            self.delete_memory(mem_id)
        self.add_memory(MemoryChunk(
            content=f"[note:{key}] {text}",
            embedding=[0.0] * EMBEDDING_DIM,
            metadata={"note_key": key},
            mem_type="NOTE",
        ))

    def get_note(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content FROM memories WHERE mem_type = 'NOTE' AND json_extract(metadata, '$.note_key') = ? "
            "ORDER BY id DESC LIMIT 1", (key,)
        ).fetchone()
        if not row:
            return None
        prefix = f"[note:{key}] "
        return row[0][len(prefix):] if row[0].startswith(prefix) else row[0]

    def semantic_search(self, query_embedding: List[float], limit: int = 5) -> List[Tuple[float, str]]:
        query_blob = self._serialize_embedding(query_embedding)
        if self._vec_available:
            rows = self.conn.execute(
                """SELECT m.content, v.distance
                   FROM memories_vec v
                   JOIN memories m ON m.id = v.rowid
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (query_blob, limit),
            ).fetchall()
            # The vec0 table is created with distance_metric=cosine (see
            # _init_db), so distance == 1 - cosine_similarity. Convert back
            # to cosine similarity so callers (HybridRetriever) can treat
            # "score" as higher-is-better identically across this path and
            # the brute-force cosine-similarity fallback below. Without this,
            # callers that sort descending by score would rank the LEAST
            # similar memories first whenever this vector-index path is used.
            #
            # Cosine distance is undefined (NULL) for zero-norm embeddings
            # (e.g. a stored placeholder embedding). Skip those rather than
            # crashing on `1.0 - None` - they simply can't participate in
            # semantic search, but remain findable via keyword search.
            return [(1.0 - row[1], row[0]) for row in rows if row[1] is not None]

        # Fallback: brute-force cosine similarity using numpy
        cursor = self.conn.execute("SELECT content, embedding FROM memories")
        results = []
        query_arr = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return []
        for content, emb_blob in cursor:
            emb = np.frombuffer(emb_blob, dtype="<f")
            if len(emb) != len(query_arr):
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                # Cosine similarity is undefined for a zero-norm embedding
                # (e.g. a stored placeholder embedding); skip it rather than
                # dividing by zero and injecting a NaN-scored "match".
                continue
            score = float(np.dot(query_arr, emb) / (query_norm * emb_norm))
            results.append((score, content))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:limit]

    def close(self):
        self.conn.close()


class NoteStore:
    """Adapter giving WorkspaceTools a persistent memory_save/memory_get backend."""

    def __init__(self, db: MemoryDB) -> None:
        self.db = db

    def save(self, key: str, text: str) -> None:
        self.db.save_note(key, text)

    def get(self, key: str) -> Optional[str]:
        return self.db.get_note(key)
