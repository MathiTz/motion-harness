"""Tests for the eval-baseline runner (evals/lib.py, issue #13).

These test the runner's own mechanics with scripted (free, offline) providers - never a real
provider call. The suite existing and being runnable is what CI checks; actually running it against
a real model, for real cost, is a human decision (see docs/evals.md) and is deliberately not done
here or anywhere in CI.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evals.lib import (
    RESULTS_DIR,
    TASKS_DIR,
    build_report,
    load_task,
    load_tasks,
    run_check,
    run_resume_task,
    run_task,
    stage_workspace,
)
from tests.test_agent_loop import Scripted, call, make_agent, text

REPO = Path(__file__).resolve().parents[1]


def factory(steps):
    def make():
        agent = make_agent(Scripted(steps))
        agent.provider.config.provider_type = "cloud"
        agent.provider.config.options.update(input_mtok=1.0, output_mtok=2.0)
        return agent
    return make


# ── task loading ─────────────────────────────────────────────────────────────

def test_all_real_tasks_load_and_cover_every_named_category():
    tasks = load_tasks(include_held_out=True)
    assert len(tasks) >= 6
    categories = {t.category for t in tasks}
    assert categories == {"bug_fix", "feature_mod", "investigate", "resume"}
    for t in tasks:
        assert t.prompt.strip()
        assert (t.check is not None) != (t.rubric is not None), t.id  # exactly one, never both/neither


def test_held_out_tasks_are_excluded_by_default_and_included_on_request():
    default = load_tasks(include_held_out=False)
    everything = load_tasks(include_held_out=True)
    held_out_ids = {t.id for t in everything if t.held_out}
    assert held_out_ids  # the fixture set actually has some
    assert not ({t.id for t in default} & held_out_ids)
    assert {t.id for t in everything} - {t.id for t in default} == held_out_ids


def test_only_filters_to_the_requested_task_ids():
    tasks = load_tasks(only=["bugfix-off-by-one"])
    assert [t.id for t in tasks] == ["bugfix-off-by-one"]


def test_resume_tasks_carry_their_two_phase_fields():
    tasks = {t.id: t for t in load_tasks(include_held_out=True)}
    resume_tasks = [t for t in tasks.values() if t.category == "resume"]
    assert resume_tasks
    for t in resume_tasks:
        assert t.resume_prompt and t.phase1_max_steps


@pytest.mark.parametrize("extra", ["check: tests/t.py\nrubric: r.md\n", ""])  # both set, or neither
def test_load_task_rejects_a_task_that_does_not_declare_exactly_one_scoring_method(tmp_path, extra):
    d = tmp_path / "bad-task"
    d.mkdir()
    (d / "task.yaml").write_text(f"id: bad-task\ncategory: bug_fix\nprompt: x\n{extra}")
    with pytest.raises(ValueError, match="exactly one"):
        load_task(d)


# ── staging and scoring mechanics (no LLM involved) ─────────────────────────

def test_stage_workspace_copies_fixture_but_never_the_hidden_test(tmp_path):
    task = load_task(TASKS_DIR / "bugfix-off-by-one")
    stage_workspace(task, tmp_path)
    assert (tmp_path / "utils.py").exists()
    assert not (tmp_path / "test_utils.py").exists()


def test_run_check_fails_against_the_real_buggy_fixture_and_passes_once_fixed(tmp_path):
    task = load_task(TASKS_DIR / "bugfix-off-by-one")
    stage_workspace(task, tmp_path)
    passed, output = run_check(task, tmp_path)
    assert passed is False and "test_returns_exactly_n_items" in output

    (tmp_path / "utils.py").write_text(
        "def get_last_n(items, n):\n    return [] if n <= 0 else items[-n:]\n"
    )
    passed, output = run_check(task, tmp_path)
    assert passed is True and "4 passed" in output


def test_run_check_returns_none_for_a_rubric_task(tmp_path):
    task = load_task(TASKS_DIR / "investigate-shared-state")
    stage_workspace(task, tmp_path)
    passed, output = run_check(task, tmp_path)
    assert passed is None and output == ""


# ── run_task / run_resume_task with a scripted (free) provider ─────────────

async def test_run_task_reports_success_when_the_scripted_fix_is_correct(tmp_path):
    task = load_task(TASKS_DIR / "bugfix-off-by-one")
    fixed = "def get_last_n(items, n):\n    return [] if n <= 0 else items[-n:]\n"
    steps = [
        call("1", "read_file", path="utils.py"),
        call("2", "write_file", path="utils.py", content=fixed),
        text("fixed it"),
    ]
    result = await run_task(task, provider_id=None, workdir=tmp_path, agent_factory=factory(steps))
    assert result.ok and result.passed and result.success
    assert result.task_id == "bugfix-off-by-one" and result.category == "bug_fix"
    assert result.cost_usd is not None and result.cost_usd >= 0


async def test_run_task_reports_failure_when_the_agent_does_not_fix_the_bug(tmp_path):
    task = load_task(TASKS_DIR / "bugfix-off-by-one")
    result = await run_task(task, provider_id=None, workdir=tmp_path, agent_factory=factory([text("looks fine to me")]))
    assert result.ok and result.passed is False and not result.success


async def test_a_provider_error_is_not_silently_a_pass(tmp_path):
    task = load_task(TASKS_DIR / "bugfix-off-by-one")
    result = await run_task(task, provider_id=None, workdir=tmp_path, agent_factory=factory([RuntimeError("boom")]))
    assert result.ok is False and not result.success  # ok=False regardless of what passed ends up being


async def test_run_resume_task_scores_only_the_state_after_both_phases(tmp_path):
    task = load_task(TASKS_DIR / "resume-shape-areas")
    phase1_steps = [
        call("1", "read_file", path="shapes.py"),
        call("2", "write_file", path="shapes.py", content=(
            "import math\n\nclass Circle:\n    def __init__(self, radius):\n        self.radius = radius\n\n"
            "    def area(self):\n        return math.pi * self.radius ** 2\n\n"
            "class Square:\n    def __init__(self, side):\n        self.side = side\n\n"
            "class Triangle:\n    def __init__(self, base, height):\n        self.base = base\n        self.height = height\n"
        )),
        text("did the circle, ran out of steps"),
    ]
    phase2_steps = [
        call("3", "read_file", path="shapes.py"),
        call("4", "write_file", path="shapes.py", content=(
            "import math\n\nclass Circle:\n    def __init__(self, radius):\n        self.radius = radius\n\n"
            "    def area(self):\n        return math.pi * self.radius ** 2\n\n"
            "class Square:\n    def __init__(self, side):\n        self.side = side\n\n"
            "    def area(self):\n        return self.side ** 2\n\n"
            "class Triangle:\n    def __init__(self, base, height):\n        self.base = base\n        self.height = height\n\n"
            "    def area(self):\n        return 0.5 * self.base * self.height\n"
        )),
        text("finished the rest"),
    ]
    result = await run_resume_task(
        task, provider_id=None, workdir=tmp_path,
        agent_factory=factory(phase1_steps), resume_agent_factory=factory(phase2_steps),
    )
    assert result.passed and result.success
    assert result.tool_calls >= 2  # both phases actually ran tools, not just talked


async def test_run_resume_task_fails_if_phase_two_never_finishes_the_rest(tmp_path):
    task = load_task(TASKS_DIR / "resume-shape-areas")
    phase1_steps = [text("thinking about it")]
    phase2_steps = [text("I believe this is already complete")]
    result = await run_resume_task(
        task, provider_id=None, workdir=tmp_path,
        agent_factory=factory(phase1_steps), resume_agent_factory=factory(phase2_steps),
    )
    assert result.ok and not result.success


# ── reporting ────────────────────────────────────────────────────────────────

def test_build_report_separates_rubric_tasks_from_the_success_rate():
    from evals.lib import TaskResult

    results = [
        TaskResult("a", "bug_fix", False, True, True, None, None, 0.01, 1.0, 3, 2, "p", "m"),
        TaskResult("b", "bug_fix", False, True, False, None, None, 0.02, 1.0, 3, 2, "p", "m"),
        TaskResult("c", "investigate", False, True, None, "rubric.md", None, 0.01, 1.0, 2, 1, "p", "m"),
    ]
    report = build_report(results)
    s = report["summary"]
    assert s["total"] == 3 and s["scriptable"] == 2 and s["needs_human_review"] == 1
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["cost_per_successful_task"] == pytest.approx(0.01)  # only task "a" succeeded
    assert s["total_cost_usd"] == pytest.approx(0.04)


def test_build_report_handles_no_scriptable_tasks_without_dividing_by_zero():
    from evals.lib import TaskResult

    results = [TaskResult("c", "investigate", False, True, None, "rubric.md", None, 0.01, 1.0, 2, 1, "p", "m")]
    s = build_report(results)["summary"]
    assert s["success_rate"] is None and s["cost_per_successful_task"] is None


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_list_shows_every_task_without_running_or_spending_anything():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_evals.py"), "--list"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "bugfix-off-by-one" in proc.stdout
    assert "held-out" not in proc.stdout  # default listing excludes held-out tasks
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_evals.py"), "--list", "--include-held-out"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert "[held-out]" in proc.stdout


def test_cli_unknown_task_id_exits_cleanly_instead_of_a_traceback():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_evals.py"), "--task", "does-not-exist", "--list"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2 and "no matching tasks" in proc.stderr


@pytest.fixture(autouse=True)
def _clean_pycache_in_eval_tasks():
    """pytest (run by run_check against a task's hidden test) leaves __pycache__/.pytest_cache
    under evals/tasks/<id>/tests/ - clean up so the repo stays tidy across local test runs."""
    yield
    for junk in ("__pycache__", ".pytest_cache"):
        for p in TASKS_DIR.rglob(junk):
            shutil.rmtree(p, ignore_errors=True)
