"""scripts/skill_ab_test.py: the A/B promotion check (issue #16), tested against fakes - never a
real provider call. Real, billed verification of this script itself is a maintainer/local exercise
like scripts/run_evals.py and scripts/live_check.py, not something run in CI.
"""
import argparse
from pathlib import Path

import pytest

import scripts.skill_ab_test as ab
from evals.lib import TaskResult, load_tasks


def fake_result(passed: bool) -> TaskResult:
    return TaskResult(
        task_id="t", category="bug_fix", held_out=False, ok=True, passed=passed, rubric=None,
        error=None, cost_usd=0.001, elapsed_s=1.0, steps=1, tool_calls=1, provider="p", model="m",
    )


def test_install_skill_places_the_file_under_dot_motion_skills(tmp_path):
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")
    ab.install_skill(skill)(tmp_path / "workdir")
    assert (tmp_path / "workdir" / ".motion" / "skills" / "candidate.md").read_text() == "# Candidate\n"


async def test_run_variant_installs_the_skill_only_when_given(monkeypatch, tmp_path):
    seen_before_run = []

    async def fake_run_task(task, *, provider_id, workdir, before_run=None, **kw):
        seen_before_run.append(before_run)
        if before_run:
            before_run(workdir)
        return fake_result(True)

    monkeypatch.setattr(ab, "run_task", fake_run_task)
    task = load_tasks(only=["bugfix-off-by-one"])[0]
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")

    assert await ab.run_variant(task, None, None) is True
    assert seen_before_run[-1] is None

    assert await ab.run_variant(task, None, skill) is True
    assert seen_before_run[-1] is not None


async def test_main_reports_no_regression_and_exits_zero(monkeypatch, tmp_path, capsys):
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")

    async def always_pass(task, provider_id, skill_path):
        return True

    monkeypatch.setattr(ab, "run_variant", always_pass)
    code = await ab.main_async(argparse.Namespace(skill=str(skill), tasks="bugfix-off-by-one", repeat=1, provider=None))
    out = capsys.readouterr().out
    assert code == 0 and "No regression" in out and "safe to consider promoting" in out
    assert "/skill promote candidate" in out


async def test_main_reports_a_regression_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")

    async def pass_without_fail_with(task, provider_id, skill_path):
        return skill_path is None  # passes without the skill, fails with it: a real regression

    monkeypatch.setattr(ab, "run_variant", pass_without_fail_with)
    code = await ab.main_async(argparse.Namespace(skill=str(skill), tasks="bugfix-off-by-one", repeat=1, provider=None))
    out = capsys.readouterr().out
    assert code == 1 and "DO NOT PROMOTE" in out and "bugfix-off-by-one" in out


async def test_main_uses_repeat_count_and_averages_across_attempts(monkeypatch, tmp_path, capsys):
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")
    calls = {"without": 0, "with": 0}

    async def counting(task, provider_id, skill_path):
        key = "with" if skill_path else "without"
        calls[key] += 1
        return True

    monkeypatch.setattr(ab, "run_variant", counting)
    await ab.main_async(argparse.Namespace(skill=str(skill), tasks="bugfix-off-by-one", repeat=3, provider=None))
    assert calls == {"without": 3, "with": 3}


async def test_main_rejects_a_missing_skill_file(tmp_path, capsys):
    code = await ab.main_async(argparse.Namespace(skill=str(tmp_path / "nope.md"), tasks=None, repeat=1, provider=None))
    assert code == 2 and "no such skill file" in capsys.readouterr().err


async def test_main_rejects_a_task_id_that_is_rubric_scored_only(tmp_path, capsys):
    """investigate-shared-state is rubric-scored - it can't feed a pass/fail A/B at all."""
    skill = tmp_path / "candidate.md"
    skill.write_text("# Candidate\n")
    code = await ab.main_async(argparse.Namespace(skill=str(skill), tasks="investigate-shared-state", repeat=1, provider=None))
    assert code == 2 and "no scriptable tasks" in capsys.readouterr().err
