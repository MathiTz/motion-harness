"""Task-evaluation baseline (issue #13): load fixed tasks from evals/tasks/, run each against
the harness in headless mode, score it, and report pass/fail, cost, time and steps - a repeatable
way to tell whether a change makes the agent better or worse at real tasks, instead of judging by
code review and unit tests alone.

Reuses core.headless.run_headless (the same engine `motion -p` uses) rather than reimplementing a
runner; this module only adds task loading, fixture staging, scoring and reporting around it.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Task:
    id: str
    category: str
    description: str
    prompt: str
    dir: Path
    held_out: bool = False
    check: Optional[str] = None  # path (relative to dir) to a hidden pytest file, or...
    rubric: Optional[str] = None  # ...a path to a human-scoring rubric - never both
    resume_prompt: Optional[str] = None  # set for category "resume": phase-2 prompt
    phase1_max_steps: Optional[int] = None

    @property
    def scriptable(self) -> bool:
        return self.check is not None


def load_task(task_dir: Path) -> Task:
    data = yaml.safe_load((task_dir / "task.yaml").read_text())
    check, rubric = data.get("check"), data.get("rubric")
    if bool(check) == bool(rubric):  # both set, or neither - either way the task can't be scored
        raise ValueError(
            f"{task_dir.name}: task.yaml must set exactly one of `check` or `rubric` "
            f"(got check={check!r}, rubric={rubric!r})"
        )
    return Task(
        id=data["id"], category=data["category"], description=data.get("description", ""),
        prompt=data["prompt"].strip(), dir=task_dir, held_out=bool(data.get("held_out", False)),
        check=check, rubric=rubric,
        resume_prompt=(data.get("resume_prompt") or "").strip() or None,
        phase1_max_steps=data.get("phase1_max_steps"),
    )


def load_tasks(include_held_out: bool = True, only: Optional[list[str]] = None) -> list[Task]:
    tasks = [load_task(d) for d in sorted(TASKS_DIR.iterdir()) if (d / "task.yaml").is_file()]
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if t.id in wanted]
    if not include_held_out:
        tasks = [t for t in tasks if not t.held_out]
    return tasks


def stage_workspace(task: Task, workdir: Path) -> None:
    """Copy the task's fixture (what the agent sees) into workdir. The hidden reference test
    under tests/ (if any) is deliberately NOT copied here - only stage_check copies it, and only
    after the agent has finished, so the agent cannot read or edit its own scoring test."""
    fixture = task.dir / "fixture"
    if fixture.is_dir():
        shutil.copytree(fixture, workdir, dirs_exist_ok=True)


def run_check(task: Task, workdir: Path) -> tuple[Optional[bool], str]:
    """Run the task's hidden pytest check against workdir. Returns (passed, output); passed is
    None for a rubric-scored task (nothing automated to run - see run_rubric_stub)."""
    if not task.check:
        return None, ""
    test_src = task.dir / task.check
    test_dst = workdir / Path(task.check).name
    shutil.copy(test_src, test_dst)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_dst.name, "-q"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]


@dataclass
class TaskResult:
    task_id: str
    category: str
    held_out: bool
    ok: bool  # the headless run completed without a provider/tool-loop error
    passed: Optional[bool]  # scripted check result, or None if rubric-scored (needs human review)
    rubric: Optional[str]
    error: Optional[str]
    cost_usd: Optional[float]
    elapsed_s: float
    steps: int
    tool_calls: int
    provider: str
    model: Optional[str]
    check_output: str = ""
    workdir: Optional[str] = None

    @property
    def success(self) -> bool:
        """A task counts as a success only when there's an automated PASS. A rubric task never
        self-reports success - see docs/evals.md on why "the agent says done" is not a check."""
        return bool(self.ok and self.passed is True)


async def _run_headless_capturing(prompt: str, *, provider_id: Optional[str], workspace: str,
                                   limits: Optional[dict] = None, agent_factory=None) -> dict[str, Any]:
    from core.headless import run_headless

    buf = io.StringIO()
    await run_headless(
        prompt, provider_id=provider_id, workspace=workspace, output_format="json",
        limits=limits or {}, out=buf, err=io.StringIO(), agent_factory=agent_factory,
    )
    return json.loads(buf.getvalue().splitlines()[-1])


async def run_task(task: Task, *, provider_id: Optional[str], workdir: Path,
                    keep_workdir: bool = False, agent_factory=None) -> TaskResult:
    stage_workspace(task, workdir)
    summary = await _run_headless_capturing(
        task.prompt, provider_id=provider_id, workspace=str(workdir), agent_factory=agent_factory,
    )
    cost = summary.get("cost_usd")
    elapsed = float(summary.get("elapsed_s") or 0)
    steps = int(summary.get("steps") or 0)
    tool_calls = int(summary.get("tool_calls") or 0)

    passed: Optional[bool] = None
    check_output = ""
    if task.scriptable:
        passed, check_output = run_check(task, workdir)
    return TaskResult(
        task_id=task.id, category=task.category, held_out=task.held_out,
        ok=bool(summary.get("ok")), passed=passed, rubric=task.rubric,
        error=summary.get("error"), cost_usd=cost, elapsed_s=elapsed, steps=steps,
        tool_calls=tool_calls, provider=summary.get("provider", provider_id or ""),
        model=summary.get("model"), check_output=check_output,
        workdir=str(workdir) if keep_workdir else None,
    )


async def run_resume_task(task: Task, *, provider_id: Optional[str], workdir: Path,
                           keep_workdir: bool = False, agent_factory=None,
                           resume_agent_factory=None) -> TaskResult:
    """Two headless calls sharing one workspace: phase 1 is capped to task.phase1_max_steps so it
    cannot finish a multi-part task; phase 2 is a fresh call (headless has no session/history - see
    docs/evals.md) with a prompt that tells the model to check current file state and finish the
    rest. Scores only the end state after both phases. Each phase gets its own agent_factory since
    they are two separate agent.run() calls, possibly against two different scripted providers."""
    stage_workspace(task, workdir)
    phase1 = await _run_headless_capturing(
        task.prompt, provider_id=provider_id, workspace=str(workdir),
        limits={"max_steps": task.phase1_max_steps} if task.phase1_max_steps else None,
        agent_factory=agent_factory,
    )
    phase2 = await _run_headless_capturing(
        task.resume_prompt, provider_id=provider_id, workspace=str(workdir),
        agent_factory=resume_agent_factory or agent_factory,
    )
    cost1, cost2 = phase1.get("cost_usd"), phase2.get("cost_usd")
    cost = (cost1 or 0) + (cost2 or 0) if (cost1 is not None or cost2 is not None) else None
    elapsed = float(phase1.get("elapsed_s") or 0) + float(phase2.get("elapsed_s") or 0)
    steps = int(phase1.get("steps") or 0) + int(phase2.get("steps") or 0)
    tool_calls = int(phase1.get("tool_calls") or 0) + int(phase2.get("tool_calls") or 0)
    passed, check_output = run_check(task, workdir) if task.scriptable else (None, "")
    return TaskResult(
        task_id=task.id, category=task.category, held_out=task.held_out,
        ok=bool(phase2.get("ok")), passed=passed, rubric=task.rubric,
        error=phase2.get("error"), cost_usd=cost, elapsed_s=elapsed, steps=steps,
        tool_calls=tool_calls, provider=phase2.get("provider", provider_id or ""),
        model=phase2.get("model"), check_output=check_output,
        workdir=str(workdir) if keep_workdir else None,
    )


def load_rubric_task_ids(results: list[TaskResult]) -> list[TaskResult]:
    return [r for r in results if r.rubric is not None]


def build_report(results: list[TaskResult]) -> dict[str, Any]:
    scored = [r for r in results if r.passed is not None]
    successes = [r for r in results if r.success]
    priced = [r.cost_usd for r in successes if isinstance(r.cost_usd, (int, float))]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": [r.__dict__ for r in results],
        "summary": {
            "total": len(results),
            "scriptable": len(scored),
            "needs_human_review": len(results) - len(scored),
            "success_rate": (len(successes) / len(scored)) if scored else None,
            "cost_per_successful_task": (sum(priced) / len(priced)) if priced else None,
            "total_cost_usd": sum(r.cost_usd for r in results if isinstance(r.cost_usd, (int, float))) or None,
            "total_elapsed_s": sum(r.elapsed_s for r in results),
        },
    }
