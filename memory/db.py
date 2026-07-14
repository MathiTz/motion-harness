import sqlite3
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

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
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, metadata)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB, metadata TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, mem_type TEXT)")
        self.conn.commit()

    def add_memory(self, chunk: MemoryChunk):
        embedding_blob = json.dumps(chunk.embedding).encode("utf-8")
        metadata_json = json.dumps(chunk.metadata)
        cursor = self.conn.execute("INSERT INTO memories (content, embedding, metadata, mem_type) VALUES (?, ?, ?, ?)", (chunk.content, embedding_blob, metadata_json, chunk.mem_type))
        mem_id = cursor.lastrowid
        self.conn.execute("INSERT INTO memories_fts (rowid, content, metadata) VALUES (?, ?, ?)", (mem_id, chunk.content, metadata_json))
        self.conn.commit()

    def keyword_search(self, query: str, limit: int = 5) -> List[Tuple[float, str]]:
        if not query:
            return []

        # Treat the input as a phrase to avoid FTS parser issues with punctuation.
        escaped_query = query.replace('"', '""')
        phrase_query = f'"{escaped_query}"'

        sql = "SELECT rank, content FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?"
        cursor = self.conn.execute(sql, (phrase_query, limit))
        return cursor.fetchall()

    def semantic_search(self, query_embedding: List[float], limit: int = 5) -> List[Tuple[float, str]]:
        cursor = self.conn.execute("SELECT content, embedding FROM memories LIMIT 100")
        results = []
        for row in cursor:
            content, emb_blob = row
            emb = json.loads(emb_blob.decode("utf-8"))
            score = sum(a * b for a, b in zip(query_embedding, emb))
            results.append((score, content))
        return sorted(results, key=lambda x: x[0], reverse=True)[:limit]

    def close(self):
        self.conn.close()
