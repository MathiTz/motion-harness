#!/usr/bin/env python3
"""Run the memory/compaction recall-quality scenarios (issue #18) and print a report.

    python scripts/eval_memory_quality.py
    python scripts/eval_memory_quality.py --scenario conflicting_memories
    python scripts/eval_memory_quality.py --provider ollama-cloud/deepseek-v4-flash

Two of the three scenarios make real, billed provider calls (see evals/memory_quality.py's
docstring); the compaction one always does (it calls MotionAgent.summarize for real - a scripted
summary would tell us nothing about real summarization quality). Not run in CI.
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

from evals.memory_quality import SCENARIOS, build_report  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    names = [args.scenario] if args.scenario else list(SCENARIOS)
    results = []
    for name in names:
        fn = SCENARIOS[name]
        workdir = Path(tempfile.mkdtemp(prefix=f"memquality-{name}-"))
        print(f"-> {name}...")
        try:
            result = await fn(workdir, args.provider)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        results.append(result)
        print(f"   relevance:  {'n/a' if result.relevance_ok is None else ('OK' if result.relevance_ok else 'BAD')}  {result.relevance_finding}")
        print(f"   downstream: {'n/a' if result.downstream_ok is None else ('OK' if result.downstream_ok else 'BAD')}  {result.downstream_finding}")

    report = build_report(results)
    out_path = Path(args.out) if args.out else Path(tempfile.gettempdir()) / f"memory-quality-{report['timestamp'].replace(':', '')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nTotal cost: ${report['total_cost_usd']:.4f}")
    print(f"Report written to {out_path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None, help="run only this scenario (default: all)")
    p.add_argument("--provider", default=None, help="provider/model id (default: config.yml's default)")
    p.add_argument("--out", default=None, help="report path (default: a temp file)")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
