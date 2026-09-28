"""Per-project memory scoping (issue #17): a fact remembered while working in one workspace must
not be recalled while working in an unrelated one. Before this, every workspace shared one DB at
the harness's own install directory (main.py's REPO_DIR/motion_memory.db); now each workspace gets
its own <workspace>/.motion/memory.db, derived the same way as sessions/skills/trajectories
(core/session.py's state_dir).
"""
from pathlib import Path

from core.providers import ModelConfig
from core.session import state_dir
from main import REPO_DIR, MotionAgent, _default_memory_path
from tests.test_agent_loop import EmptyRetriever, Scripted, call, run, text


def agent_for(workspace: "str | Path | None" = None, memory_path=None):
    agent = MotionAgent(
        ModelConfig(name="t", endpoint="http://x", provider_type="local"),
        memory_path=memory_path, workspace=str(workspace) if workspace else None,
    )
    agent.retriever = EmptyRetriever()
    return agent


# ── path derivation ──────────────────────────────────────────────────────────

def test_default_memory_path_is_under_the_workspaces_own_state_dir(tmp_path: Path):
    path = _default_memory_path(str(tmp_path))
    assert path == str(state_dir(tmp_path, create=False) / "memory.db")
    assert Path(path).parent == tmp_path / ".motion"


def test_two_different_workspaces_get_two_different_default_paths(tmp_path: Path):
    a, b = tmp_path / "project-a", tmp_path / "project-b"
    a.mkdir()
    b.mkdir()
    assert _default_memory_path(str(a)) != _default_memory_path(str(b))


def test_no_workspace_falls_back_to_the_pre_fix_shared_path_unchanged():
    """Only for genuinely workspace-less callers (main.py's `--test` self-check, a library caller
    that never resolved a workspace) - every real entry point (TUI, headless) always has one."""
    assert _default_memory_path(None) == str(Path(REPO_DIR) / "motion_memory.db")


def test_an_explicit_memory_path_always_wins_over_the_workspace_default(tmp_path: Path):
    agent = agent_for(workspace=tmp_path, memory_path=":memory:")
    assert agent.memory.db_path == ":memory:"


# ── the acceptance criterion: recorded in A, absent from B ──────────────────

async def test_a_memory_recorded_in_one_workspace_is_absent_from_an_unrelated_one(tmp_path: Path):
    workspace_a = tmp_path / "project-a"
    workspace_b = tmp_path / "project-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    agent_a = agent_for(workspace_a)
    agent_a.provider = Scripted([call("1", "list_files"), text("There are no files here.")])
    agent_a.auto_remember = True
    await run(agent_a, "the deploy key rotates every Tuesday", workspace=str(workspace_a))
    import asyncio

    await asyncio.gather(*list(agent_a._bg_tasks))
    assert agent_a.memory.keyword_search("deploy key rotates")  # recorded, recallable in its own workspace

    # A second, independent agent bound to the OTHER workspace - simulating `motion` launched
    # against a different project, the real scenario this issue is about - must not see it.
    agent_b = agent_for(workspace_b)
    assert agent_b.memory.keyword_search("deploy key rotates") == []
    assert agent_b.memory.db_path != agent_a.memory.db_path

    # And a THIRD agent reopening workspace A's own DB (a later `motion` launch on the same
    # project) still finds it - scoping isn't accidentally per-process, it's per-workspace on disk.
    agent_a_again = agent_for(workspace_a)
    assert agent_a_again.memory.keyword_search("deploy key rotates")


def test_the_memory_db_file_actually_lands_under_dot_motion(tmp_path: Path):
    workspace = tmp_path / "myproject"
    workspace.mkdir()
    agent = agent_for(workspace)
    from memory.db import EMBEDDING_DIM, MemoryChunk

    agent.memory.add_memory(MemoryChunk("a fact", [0.0] * EMBEDDING_DIM, {}, "DOC"))
    assert (workspace / ".motion" / "memory.db").exists()


# ── the already-isolated sub-agent path (acceptance criterion: unaffected) ──

def test_subagent_memory_stays_fully_ephemeral_regardless_of_workspace(tmp_path: Path):
    """core/orchestrator.py always passes memory_path=":memory:" for sub-agents - already more
    isolated than per-workspace scoping (no cross-session leakage at all), so this issue leaves it
    unaffected. Guard that a workspace passed alongside an explicit memory_path never overrides it."""
    agent = agent_for(workspace=tmp_path, memory_path=":memory:")
    assert agent.memory.db_path == ":memory:"
    assert not (tmp_path / ".motion" / "memory.db").exists()


# ── the old shared DB: left alone, not silently discarded ───────────────────

def test_old_shared_db_file_is_never_touched_by_workspace_scoped_construction(tmp_path: Path, monkeypatch):
    """Regression guard for the migration decision (see README.md's "Where state lives"): building
    a workspace-scoped agent must not read from or write to REPO_DIR's old motion_memory.db."""
    fake_repo_dir = tmp_path / "fake-install"
    fake_repo_dir.mkdir()
    old_shared_db = fake_repo_dir / "motion_memory.db"
    monkeypatch.setattr("main.REPO_DIR", str(fake_repo_dir))

    workspace = tmp_path / "project"
    workspace.mkdir()
    agent_for(workspace)  # constructing a workspace-scoped agent...
    assert not old_shared_db.exists()  # ...must never create or touch the old shared path
