#!/usr/bin/env python3
"""Compare two saved trajectories (issue #19): before/after a harness change, or two versions of
the same task run against different providers - reports step/token/time/cost deltas and which tool
calls appear in one run but not the other.

    python scripts/compare_trajectories.py before.json after.json

Takes any file `/trajectory save` produced (with or without `full`) or a doc built by
core.trajectory.to_json directly. Cost is estimated from the built-in catalog's pricing for the
provider/model recorded in each file (core.pricing.estimate_cost); a model this harness has no
pricing for reports cost as "n/a", not a guess.

Does not need the eval-baseline's pass/fail data (a separate issue) to be useful on its own - if a
trajectory file happens to carry a "task_result" field (e.g. from evals/lib.py), the outcome is
reported too, but its absence doesn't block anything here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def estimated_cost(doc: Dict[str, Any]) -> Optional[float]:
    from core.catalog import BUILTIN_CATALOG
    from core.pricing import estimate_cost

    provider = doc.get("provider") or ""
    if "/" not in provider:
        return None
    base_id, model = provider.split("/", 1)
    options = (BUILTIN_CATALOG.get(base_id, {}).get("models", {}) or {}).get(model)
    if not options:
        return None
    s = doc.get("summary", {})
    return estimate_cost(options, s.get("prompt_tokens", 0), s.get("completion_tokens", 0))


def call_set(doc: Dict[str, Any]) -> set:
    """(name, args-preview) pairs across every step - the args preview is already truncated
    (core.trajectory.preview_args), which is enough to recognize the "same call" across two runs
    without needing full arguments."""
    return {
        (t["name"], t.get("args", ""))
        for step in doc.get("steps", [])
        for t in step.get("tools", [])
    }


def fmt_delta(before: float, after: float, fmt: str = "{:.0f}") -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    pct = f" ({sign}{delta / before:.0%})" if before else ""
    return f"{fmt.format(before)} -> {fmt.format(after)} ({sign}{fmt.format(delta)}){pct}"


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    lines: List[str] = []
    bv, av = before.get("harness_version", "unknown"), after.get("harness_version", "unknown")
    bp, ap = before.get("provider", "?"), after.get("provider", "?")
    lines.append(f"before: harness {bv}, provider {bp}")
    lines.append(f"after:  harness {av}, provider {ap}")
    if bv == av and bp == ap:
        lines.append("(same harness version and provider - comparing noise, not a real change)")
    lines.append("")

    bs, as_ = before.get("summary", {}), after.get("summary", {})
    lines.append(f"steps:              {fmt_delta(bs.get('steps', 0), as_.get('steps', 0))}")
    lines.append(f"tool calls:         {fmt_delta(bs.get('tool_calls', 0), as_.get('tool_calls', 0))}")
    lines.append(f"prompt tokens:      {fmt_delta(bs.get('prompt_tokens', 0), as_.get('prompt_tokens', 0))}")
    lines.append(f"completion tokens:  {fmt_delta(bs.get('completion_tokens', 0), as_.get('completion_tokens', 0))}")
    lines.append(f"model time (s):     {fmt_delta(bs.get('model_seconds', 0), as_.get('model_seconds', 0), '{:.1f}')}")
    lines.append(f"tool time (s):      {fmt_delta(bs.get('tool_seconds', 0), as_.get('tool_seconds', 0), '{:.1f}')}")

    bc, ac = estimated_cost(before), estimated_cost(after)
    if bc is not None and ac is not None:
        lines.append(f"estimated cost ($): {fmt_delta(bc, ac, '{:.4f}')}")
    else:
        lines.append("estimated cost ($): n/a (no pricing for one or both providers/models)")

    before_calls, after_calls = call_set(before), call_set(after)
    only_before = before_calls - after_calls
    only_after = after_calls - before_calls
    if only_before or only_after:
        lines.append("")
        lines.append("call-set differences:")
        for name, args in sorted(only_before):
            lines.append(f"  - {name}({args})  [only in before]")
        for name, args in sorted(only_after):
            lines.append(f"  + {name}({args})  [only in after]")
    else:
        lines.append("")
        lines.append("call-set differences: none (same tool calls in both runs)")

    b_result, a_result = before.get("task_result"), after.get("task_result")
    if b_result is not None or a_result is not None:
        lines.append("")
        lines.append(f"task outcome: {b_result!r} -> {a_result!r}" + ("  ⚠️ CHANGED" if b_result != a_result else ""))

    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", help="the earlier trajectory JSON")
    p.add_argument("after", help="the later trajectory JSON")
    args = p.parse_args(argv)
    print(compare(load(args.before), load(args.after)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
