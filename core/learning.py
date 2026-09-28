import hashlib
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from core.providers import ModelConfig, ProviderFactory
from core.skill_state import ACTIVE, REJECTED, Provenance, load_meta, save_meta, set_status, write_candidate
from core.skill_state import rollback as _rollback_meta
from memory.db import MemoryDB, MemoryChunk, EMBEDDING_DIM


def _fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deterministic, always-non-zero embedding for when no embedding
    provider is available. Mirrors MotionAgent.get_embedding's fallback so
    behavior is consistent across the codebase."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [float(b) / 255.0 for b in h]
    vec = (raw * ((dim // len(raw)) + 1))[:dim]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]

@dataclass
class Trajectory:
    task_id: str
    prompt: str
    steps: List[Dict[str, Any]]  # List of {tool: str, input: str, output: str}
    final_result: str
    success: bool

# The installed location of the harness (parent of core/), not the caller's
# CWD - auto-synthesized skills accumulate here regardless of which project
# directory `motion` is currently pointed at.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SkillSynthesizer:
    """
    The 'Crystallization' engine. 
    Turns successful tool-call trajectories into reusable .md skills.
    """
    def __init__(self, model_config: ModelConfig, db: MemoryDB, skills_dir: Optional[str] = None, embedding_provider=None):
        self.provider = ProviderFactory.get_provider(model_config)
        self.db = db
        self.skills_dir = skills_dir or os.path.join(_REPO_DIR, "skills")
        # Used to compute a real embedding for synthesized skills so they can
        # actually participate in semantic recall. Expected to expose an
        # async get_embedding(text) -> list[float] (e.g. a MotionAgent
        # instance, which already has a safe hash-based fallback built in).
        self.embedding_provider = embedding_provider
        
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

    async def synthesize(self, trajectory: Trajectory) -> Optional[str]:
        """
        Analyzes a successful trajectory and creates a CANDIDATE skill document (issue #16):
        written to disk with status "candidate" and never indexed for recall here - a candidate is
        not surfaced to any live turn (core/skills.py's SkillLibrary only ever lists ACTIVE skills)
        until something explicitly calls promote() below. Real outcome verification (did the turn
        actually succeed, not just "a tool ran") is gated behind the eval-baseline's task-success
        signal (issue #13); until a caller supplies that, every candidate's provenance is stamped
        verified=False so it's traceable as lower-confidence rather than silently treated as good.
        """
        if not trajectory.success:
            return None

        # Construct a prompt for the LLM to summarize the trajectory into a skill
        steps_summary = "\n".join([
            f"Step {i+1}: {s['tool']}({s['input']}) -> {s['output'][:100]}..." 
            for i, s in enumerate(trajectory.steps)
        ])
        
        synthesis_prompt = (
            f"You are a skill synthesis engine. Analyze the following successful tool trajectory "
            f"and extract the core reusable procedure. \n\n"
            f"Goal: {trajectory.prompt}\n"
            f"Steps:\n{steps_summary}\n"
            f"Final Result: {trajectory.final_result}\n\n"
            f"Create a concise, high-density '.md' skill. Use 'Caveman' style: no fluff, just "
            f"precise steps and constraints. Format as: # Skill Name\\n## Description\\n## Procedure"
        )

        try:
            skill_content = await self.provider.complete(synthesis_prompt, system_prompt="You are an expert in procedural knowledge extraction.")
            
            # Generate a filename based on the goal
            # Slugified: prompts are free text and must never contain path
            # separators (a prompt like "../../x" used to escape skills_dir).
            slug = re.sub(r"[^a-z0-9_-]+", "", re.sub(r"\s+", "_", trajectory.prompt.strip().lower()))[:40] or "skill"
            file_path = os.path.join(self.skills_dir, f"{slug}.md")

            # Regenerating an already-promoted skill supersedes its indexed content with a fresh,
            # unevaluated candidate - the old memory entry would otherwise stay recallable (orphaned:
            # not reachable from the current meta at all) even though the live file now holds
            # different content. write_candidate() moves the old meta into history for provenance,
            # but has no MemoryDB reference to de-index it, so that happens here instead.
            old_mem_id = (load_meta(file_path) or {}).get("mem_id")
            options = getattr(getattr(self.provider, "config", None), "options", None)
            model = options.get("model", "") if isinstance(options, dict) else ""
            write_candidate(file_path, skill_content, Provenance(
                task_id=trajectory.task_id, prompt=trajectory.prompt, model=model, verified=False,
            ))
            if old_mem_id is not None:
                self.db.delete_memory(old_mem_id)
            return file_path
        except Exception as e:
            print(f"Skill synthesis failed: {e}")
            return None

    async def _embed(self, content: str) -> List[float]:
        """A zero-vector embedding is never valid: cosine similarity/distance is undefined for a
        zero-norm vector, which crashes semantic search rather than just being unhelpful."""
        embedding = None
        if self.embedding_provider is not None:
            try:
                embedding = await self.embedding_provider.get_embedding(content)
            except Exception:
                embedding = None
        return embedding or _fallback_embedding(content)

    async def promote(self, skill_path: str) -> bool:
        """Move a candidate/evaluated skill to ACTIVE and, only now, index its content in the
        MemoryDB for semantic recall - this is the one gate that makes a synthesized skill visible
        to a live turn at all (core/skills.py's SkillLibrary already filters file listing/use_skill
        by status; this handles the separate memory-recall surface). Records the resulting mem_id
        in the skill's own metadata so reject()/rollback() can remove it again. Returns False if
        the skill has no sidecar (nothing to promote - e.g. a hand-saved skill, already active)."""
        meta = load_meta(skill_path)
        if meta is None:
            return False
        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()
        embedding = await self._embed(content)
        mem_id = self.db.add_memory(MemoryChunk(
            content=content, embedding=embedding, metadata={"file": skill_path, "type": "SKILL"}, mem_type="DOC",
        ))
        meta = set_status(skill_path, ACTIVE)
        meta["mem_id"] = mem_id
        save_meta(skill_path, meta)
        return True

    def reject(self, skill_path: str) -> bool:
        """Mark a candidate as rejected (never promoted) and remove it from recall if it had
        already been indexed. The file stays on disk (in its history) for provenance/audit."""
        meta = load_meta(skill_path)
        if meta is None:
            return False
        mem_id = meta.get("mem_id")
        if mem_id is not None:
            self.db.delete_memory(mem_id)
        set_status(skill_path, REJECTED)
        return True

    def rollback(self, skill_path: str) -> bool:
        """Restore the previous version, de-indexing the current one from recall first if it was
        active. Returns False if there is no prior version to roll back to."""
        meta = load_meta(skill_path)
        mem_id = (meta or {}).get("mem_id")
        restored = _rollback_meta(skill_path)
        if restored is None:
            return False
        if mem_id is not None:
            self.db.delete_memory(mem_id)
        return True
