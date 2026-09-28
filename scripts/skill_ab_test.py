#!/usr/bin/env python3
"""A/B test a candidate skill against the eval baseline (issue #16): run the same task set with
the skill enabled vs. disabled, on the same model, and report whether it regresses any outcome.

This is the promotion criterion issue #16 asks for, made runnable rather than only documented:
promote a candidate only if enabling it does not regress task outcomes. It does not run itself
automatically (no auto-promotion) - a maintainer runs this and decides, per the issue's own
"evaluated criteria are a maintainer decision, not something to invent" instruction. Also gated on
issue #13's eval-baseline existing at all (it does now: evals/lib.py), which this issue's own
dependency note anticipated.

    python scripts/skill_ab_test.py --skill skills/my_candidate.md
    python scripts/skill_ab_test.py --skill skills/my_candidate.md --tasks bugfix-off-by-one,feature-cart-validation --repeat 3

Makes real, billed provider calls (twice per task per repeat: with and without the skill) - same
cost caveat as scripts/run_evals.py and scripts/live_check.py. Not run in CI.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.lib import Task, load_tasks, run_resume_task, run_task  # noqa: E402


def install_skill(skill_path: Path):
    def _install(workdir: Path) -> None:
        dest_dir = workdir / ".motion" / "skills"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(skill_path, dest_dir / skill_path.name)
    return _install


async def run_variant(task: Task, provider_id: str, skill_path: "Path | None") -> bool:
    """One attempt at one task, with the skill installed if given. Returns whether it passed - only
    scriptable tasks (not rubric-scored ones) can feed a pass/fail A/B comparison at all."""
    runner = run_resume_task if task.category == "resume" else run_task
    workdir = Path(tempfile.mkdtemp(prefix=f"skill-ab-{task.id}-"))
    try:
        result = await runner(
            task, provider_id=provider_id, workdir=workdir,
            before_run=install_skill(skill_path) if skill_path else None,
        )
        return bool(result.success)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def main_async(args: argparse.Namespace) -> int:
    skill_path = Path(args.skill).resolve()
    if not skill_path.is_file():
        print(f"no such skill file: {skill_path}", file=sys.stderr)
        return 2

    only = args.tasks.split(",") if args.tasks else None
    tasks = [t for t in load_tasks(include_held_out=False, only=only) if t.scriptable]
    if not tasks:
        print("no scriptable tasks to compare (rubric-scored tasks can't feed a pass/fail A/B)", file=sys.stderr)
        return 2

    print(f"A/B testing {skill_path.name} against {len(tasks)} task(s), {args.repeat} repeat(s) each, provider={args.provider or '(config default)'}\n")
    regressions = []
    for task in tasks:
        without = [await run_variant(task, args.provider, None) for _ in range(args.repeat)]
        with_skill = [await run_variant(task, args.provider, skill_path) for _ in range(args.repeat)]
        w_rate = sum(without) / len(without)
        s_rate = sum(with_skill) / len(with_skill)
        verdict = "REGRESSION" if s_rate < w_rate else ("improved" if s_rate > w_rate else "no change")
        print(f"  {task.id:32s} without={without}  with_skill={with_skill}  {verdict}")
        if s_rate < w_rate:
            regressions.append(task.id)

    print()
    if regressions:
        print(f"DO NOT PROMOTE: regressed on {len(regressions)}/{len(tasks)} task(s): {', '.join(regressions)}")
        return 1
    print(f"No regression across {len(tasks)} task(s) x {args.repeat} repeat(s) - safe to consider promoting.")
    print(f"  /skill promote {skill_path.stem}   (this script never promotes automatically)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill", required=True, help="path to the candidate skill's .md file")
    p.add_argument("--tasks", default=None, help="comma-separated eval task ids (default: all non-held-out scriptable tasks)")
    p.add_argument("--repeat", type=int, default=1, help="attempts per task per variant, to see past single-run noise (default 1)")
    p.add_argument("--provider", default=None, help="provider/model id (default: config.yml's default)")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
