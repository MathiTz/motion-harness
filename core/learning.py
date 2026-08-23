import hashlib
import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from core.providers import ModelConfig, ProviderFactory
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
        Analyzes a successful trajectory and creates a concise skill document.
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
            skill_name = trajectory.prompt.replace(" ", "_").lower()[:30] + ".md"
            file_path = os.path.join(self.skills_dir, skill_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(skill_content)
            
            # Also index the skill in the MemoryDB for semantic recall. A
            # zero-vector embedding is never valid here: cosine
            # similarity/distance is undefined for a zero-norm vector, which
            # crashes semantic search rather than just being unhelpful.
            embedding = None
            if self.embedding_provider is not None:
                try:
                    embedding = await self.embedding_provider.get_embedding(skill_content)
                except Exception:
                    embedding = None
            if not embedding:
                embedding = _fallback_embedding(skill_content)
            self.db.add_memory(MemoryChunk(
                content=skill_content,
                embedding=embedding,
                metadata={"file": file_path, "type": "SKILL"},
                mem_type="DOC"
            ))
            
            return file_path
        except Exception as e:
            print(f"Skill synthesis failed: {e}")
            return None
