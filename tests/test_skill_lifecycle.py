"""Skill lifecycle: candidate -> evaluated -> active, provenance, rollback (issue #16).

Before this, SkillSynthesizer.synthesize() wrote a skill straight into the active skills directory
and indexed it for recall in the same call, with success hardcoded to True regardless of what the
turn actually did. There was no gate between "the synthesizer produced this" and "a live turn can
use it."
"""
from unittest.mock import AsyncMock

import pytest

from core.learning import SkillSynthesizer, Trajectory
from core.providers import ModelConfig
from core.skills import SkillLibrary
from core.skill_state import CANDIDATE, REJECTED, is_active, load_meta
from memory.db import MemoryDB


def synthesizer_for(tmp_path) -> SkillSynthesizer:
    s = SkillSynthesizer(
        ModelConfig(name="t", endpoint="http://x", provider_type="local"),
        MemoryDB(":memory:"), skills_dir=str(tmp_path), embedding_provider=None,
    )
    s.provider = AsyncMock()
    s.provider.complete.return_value = "# Skill\n## Description\nTest\n## Procedure\n1. Do it."
    return s


async def synthesize(s: SkillSynthesizer, prompt="a test skill", success=True) -> str:
    path = await s.synthesize(Trajectory(task_id="t1", prompt=prompt, steps=[], final_result="done", success=success))
    assert path
    return path


# ── a synthesized skill starts non-active and stays out of recall ──────────

async def test_a_freshly_synthesized_skill_is_a_candidate_not_active(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s)
    meta = load_meta(path)
    assert meta["status"] == CANDIDATE
    assert not is_active(path)


async def test_skilllibrary_never_surfaces_a_candidate(tmp_path):
    s = synthesizer_for(tmp_path)
    await synthesize(s, prompt="a candidate skill")
    lib = SkillLibrary([tmp_path])
    assert lib.index() == []
    assert lib.get("a_candidate_skill") is None
    # ...but it IS visible when explicitly asked for, e.g. for a review UI.
    assert lib.index(include_inactive=True) != []
    assert lib.get("a_candidate_skill", include_inactive=True) is not None


async def test_candidate_content_is_not_semantically_recallable_until_promoted(tmp_path):
    """The second surfacing path: synthesize() must not index the skill into the MemoryDB at all -
    that's what promote() does. A regression here would make a candidate recallable even though
    SkillLibrary correctly hides it from file-based listing."""
    s = synthesizer_for(tmp_path)
    await synthesize(s)
    assert s.db.count() == 0


async def test_hand_saved_skills_with_no_sidecar_are_unaffected_and_stay_active(tmp_path):
    """/skill save writes a plain .md with no metadata sidecar - this issue's scope is synthesized
    skills, not hand-authored ones, which were already trusted."""
    (tmp_path / "my_skill.md").write_text("# My Skill\ndo the thing\n")
    lib = SkillLibrary([tmp_path])
    assert lib.index() == [("my_skill", "My Skill")]
    assert is_active(tmp_path / "my_skill.md")


# ── success is a real signal, not hardcoded ─────────────────────────────────

async def test_synthesis_is_skipped_for_a_trajectory_marked_unsuccessful(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await s.synthesize(Trajectory(task_id="t1", prompt="x", steps=[], final_result="", success=False))
    assert path is None
    assert s.db.count() == 0 and list(tmp_path.glob("*.md")) == []


async def test_provenance_records_task_prompt_model_and_is_unverified_by_default(tmp_path):
    s = synthesizer_for(tmp_path)
    s.provider.config = ModelConfig(name="t", endpoint="http://x", provider_type="local", options={"model": "glm-5.2"})
    path = await synthesize(s, prompt="fix the parser bug")
    meta = load_meta(path)
    prov = meta["provenance"]
    assert prov["task_id"] == "t1" and prov["prompt"] == "fix the parser bug" and prov["model"] == "glm-5.2"
    assert prov["verified"] is False  # no eval-baseline outcome check ran (issue #13's task-success signal)


# ── promote / reject / rollback ──────────────────────────────────────────────

async def test_promote_makes_a_candidate_active_and_recallable(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s, prompt="promote me")
    ok = await s.promote(path)
    assert ok is True
    assert is_active(path) and load_meta(path)["status"] == "active"
    lib = SkillLibrary([tmp_path])
    assert lib.get("promote_me") is not None
    assert s.db.count() == 1  # now indexed for semantic recall


async def test_promoting_a_hand_saved_skill_with_no_sidecar_is_a_no_op(tmp_path):
    s = synthesizer_for(tmp_path)
    (tmp_path / "hand_saved.md").write_text("# Hand Saved\nx\n")
    assert await s.promote(str(tmp_path / "hand_saved.md")) is False


async def test_reject_removes_an_active_skill_from_recall_but_keeps_the_file(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s, prompt="bad skill")
    await s.promote(path)
    assert s.db.count() == 1

    assert s.reject(path) is True
    assert s.db.count() == 0  # de-indexed
    assert load_meta(path)["status"] == REJECTED
    assert SkillLibrary([tmp_path]).get("bad_skill") is None  # not active -> not surfaced
    from pathlib import Path

    assert Path(path).exists()  # kept on disk for provenance/audit


async def test_rollback_restores_the_previous_version_and_deindexes_the_current_one(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s, prompt="versioned skill")
    await s.promote(path)
    first_content = open(path).read()

    s.provider.complete.return_value = "# Skill v2\n## Description\nChanged\n## Procedure\n1. Different."
    path2 = await synthesize(s, prompt="versioned skill")  # same slug -> regenerates, pushes v1 to history
    assert path2 == path
    meta = load_meta(path)
    assert meta["status"] == CANDIDATE and meta["version"] == 2 and len(meta["history"]) == 1
    await s.promote(path)
    assert open(path).read() != first_content

    assert s.rollback(path) is True
    assert open(path).read() == first_content
    assert load_meta(path)["version"] == 1
    assert s.db.count() == 0  # v2's index entry removed; rollback doesn't re-promote v1 automatically


async def test_rollback_with_no_history_reports_failure_and_changes_nothing(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s, prompt="only version")
    content_before = open(path).read()
    assert s.rollback(path) is False
    assert open(path).read() == content_before


async def test_regenerating_a_skill_preserves_the_prior_versions_provenance(tmp_path):
    s = synthesizer_for(tmp_path)
    path = await synthesize(s, prompt="same skill twice")
    first_meta = load_meta(path)

    s.provider.complete.return_value = "# Skill v2\n## Description\nChanged\n## Procedure\n1. Different."
    await synthesize(s, prompt="same skill twice")
    meta = load_meta(path)
    assert meta["version"] == 2
    assert meta["history"][0]["provenance"]["prompt"] == first_meta["provenance"]["prompt"]
    assert meta["history"][0]["content"] != open(path).read()
