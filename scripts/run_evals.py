#!/usr/bin/env python3
"""Run the task-evaluation baseline (issue #13, docs/evals.md).

Makes real, billed provider calls - not run in CI (see evals/lib.py and docs/evals.md for why).

Examples:
  python scripts/run_evals.py --list
  python scripts/run_evals.py --provider ollama-cloud/deepseek-v4-flash
  python scripts/run_evals.py --task bugfix-off-by-one --task feature-cart-validation
  python scripts/run_evals.py --include-held-out --keep-workdirs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.lib import RESULTS_DIR, Task, build_report, load_tasks, run_resume_task, run_task  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default=None, help="provider/model id (default: config.yml's default)")
    p.add_argument("--task", action="append", dest="tasks", help="run only this task id (repeatable)")
    p.add_argument("--include-held-out", action="store_true", help="also run tasks marked held_out: true")
    p.add_argument("--keep-workdirs", action="store_true", help="don't delete the temp workspace after each task")
    p.add_argument("--list", action="store_true", help="list tasks and exit, without running anything")
    p.add_argument("--out", default=None, help="report path (default: evals/results/<timestamp>.json)")
    return p.parse_args(argv)


def print_task_list(tasks: list[Task]) -> None:
    for t in tasks:
        kind = "script" if t.scriptable else "rubric"
        flag = " [held-out]" if t.held_out else ""
        print(f"  {t.id:32s} {t.category:12s} {kind:7s}{flag}  {t.description.strip().splitlines()[0]}")


async def main_async(args: argparse.Namespace) -> int:
    tasks = load_tasks(include_held_out=args.include_held_out, only=args.tasks)
    if not tasks:
        print("no matching tasks", file=sys.stderr)
        return 2
    if args.list:
        print_task_list(tasks)
        return 0

    results = []
    for task in tasks:
        runner = run_resume_task if task.category == "resume" else run_task
        workdir = Path(tempfile.mkdtemp(prefix=f"eval-{task.id}-"))
        print(f"-> {task.id} ({task.category})...", end=" ", flush=True)
        result = await runner(task, provider_id=args.provider, workdir=workdir, keep_workdir=args.keep_workdirs)
        if not args.keep_workdirs:
            shutil.rmtree(workdir, ignore_errors=True)
        results.append(result)
        if result.passed is None:
            print(f"needs human review (rubric: {result.rubric})")
        else:
            print("PASS" if result.success else "FAIL")

    report = build_report(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{report['timestamp'].replace(':', '')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    s = report["summary"]
    print()
    if s["success_rate"] is not None:
        print(f"Scriptable tasks: {s['scriptable']}/{s['total']} (success rate {s['success_rate']:.0%})")
    if s["needs_human_review"]:
        print(f"Needs human review (rubric-scored): {s['needs_human_review']} - see each task's `rubric` file.")
    if s["cost_per_successful_task"] is not None:
        print(f"Cost per successful task: ${s['cost_per_successful_task']:.4f}")
    if s["total_cost_usd"] is not None:
        print(f"Total cost: ${s['total_cost_usd']:.4f}")
    print(f"Total wall time: {s['total_elapsed_s']:.1f}s")
    print(f"Report written to {out_path}")
    return 0


def main(argv=None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
