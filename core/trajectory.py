"""The trajectory of a turn: one record per model step saying how long it took, how many
tokens it sent and produced, and which tools it called with how large a result.

Records are plain dicts so they serialize straight to JSON:

  {"turn": 2, "agent": "lead" | "sub:<label>", "step": 3,
   "duration_s": 12.4, "ttft_s": 1.7,
   "prompt_tokens": 22409, "completion_tokens": 1985,   # None if the provider reported no usage
   "context_tokens_est": 21800, "text_chars": 0, "reasoning_chars": 6300,
   "tools": [{"name": "read_file", "args": "path=ui/tui.py offset=1",
              "ok": true, "result_chars": 12000, "duration_s": 0.01}]}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ARG_PREVIEW_CHARS = 100
LONG_VALUE = 48


def preview_args(args: Dict[str, Any], limit: int = ARG_PREVIEW_CHARS) -> str:
    """Compact ``k=v`` view of tool arguments (long values, e.g. file contents, are elided)."""
    parts: List[str] = []
    for key, value in (args or {}).items():
        text = value if isinstance(value, str) else str(value)
        text = text.replace("\n", " ")
        if len(text) > LONG_VALUE:
            text = f"{text[:LONG_VALUE - 10]}…({len(text)} chars)"
        parts.append(f"{key}={text}")
    out = " ".join(parts)
    return out if len(out) <= limit else out[: limit - 1] + "…"


def fmt_tok(n: Optional[int]) -> str:
    if n is None:
        return "  ?"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def fmt_chars(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def turn_records(records: List[Dict[str, Any]], turn: Optional[int] = None) -> List[Dict[str, Any]]:
    """Records of one turn (default: the most recent turn that has any)."""
    if not records:
        return []
    if turn is None:
        turn = max(r.get("turn", 0) for r in records)
    return [r for r in records if r.get("turn", 0) == turn]


def totals(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = [t for r in recs for t in r.get("tools", [])]
    model_s = sum(r.get("duration_s") or 0 for r in recs)
    tool_s = sum(t.get("duration_s") or 0 for t in calls)
    prompts = [r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None]
    return {
        "steps": len(recs),
        "tool_calls": len(calls),
        "model_seconds": round(model_s, 2),
        "tool_seconds": round(tool_s, 2),
        "prompt_tokens": sum(prompts),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
        "usage_reported": bool(prompts),
        "result_chars": sum(t.get("result_chars", 0) for t in calls),
    }


def insights(recs: List[Dict[str, Any]]) -> List[str]:
    """Plain-language findings about where the cost came from."""
    out: List[str] = []
    lead = [r for r in recs if r.get("agent", "lead") == "lead"]
    prompts = [(r["step"], r["prompt_tokens"]) for r in lead if r.get("prompt_tokens") is not None]
    if len(prompts) >= 3:
        first, last = prompts[0][1], prompts[-1][1]
        avg = sum(p for _, p in prompts) // len(prompts)
        if last > 2 * max(first, 1):
            out.append(
                f"prompt tokens grew {fmt_tok(first)} → {fmt_tok(last)} over {len(prompts)} steps (avg {fmt_tok(avg)}): "
                "everything the agent has read stays in context and is re-sent on every step"
            )
        else:
            out.append(f"each step re-sent ~{fmt_tok(avg)} prompt tokens (first {fmt_tok(first)}, last {fmt_tok(last)})")
    big = sorted(
        ((t["result_chars"], t["name"], t.get("args", ""), r["step"]) for r in recs for t in r.get("tools", [])),
        reverse=True,
    )[:3]
    big = [b for b in big if b[0] >= 4000]
    if big:
        out.append("largest tool results: " + "; ".join(f"{n} {a[:40]} = {fmt_chars(c)} chars (step {s})" for c, n, a, s in big))
    t = totals(recs)
    wall = t["model_seconds"] + t["tool_seconds"]
    if wall > 0 and t["model_seconds"] / wall > 0.85 and t["steps"] >= 3:
        reasoning = sum(r.get("reasoning_chars", 0) for r in recs)
        hint = " Reasoning models spend most of each step thinking; `/effort low` shortens that." if reasoning else ""
        out.append(f"{t['model_seconds'] / wall:.0%} of the time was the model generating, only {t['tool_seconds']:.1f}s was tools.{hint}")
    if t["steps"] >= 8:
        base = min((p for _, p in prompts), default=0)
        out.append(
            f"{t['steps']} steps: each one re-sends at least the base prompt (~{fmt_tok(base)}); asking for several "
            "independent tool calls in one step is the biggest saver"
        )
    seen: Dict[tuple, int] = {}
    for r in recs:
        for tl in r.get("tools", []):
            seen[(tl["name"], tl.get("args", ""))] = seen.get((tl["name"], tl.get("args", "")), 0) + 1
    repeats = [(c, k) for k, c in seen.items() if c >= 2]
    if repeats:
        c, (n, a) = max(repeats)
        out.append(f"repeated call: {n} {a[:50]} ran {c}×")
    return out


def render(recs: List[Dict[str, Any]], title: str = "") -> str:
    """The plain-text table shown by /trajectory (also what gets copied)."""
    if not recs:
        return "No steps recorded yet. Run a turn, then use /trajectory."
    t = totals(recs)
    tok = (
        f"{t['prompt_tokens']:,} prompt + {t['completion_tokens']:,} completion tokens"
        if t["usage_reported"] else "token usage not reported by the provider"
    )
    lines = [
        f"{title or 'Trajectory'} · {t['steps']} steps · {t['tool_calls']} tool calls · "
        f"{t['model_seconds'] + t['tool_seconds']:.1f}s ({t['model_seconds']:.1f}s model, {t['tool_seconds']:.1f}s tools) · {tok}",
        f"{'#':>3}  {'who':<14} {'time':>6} {'ttft':>5} {'prompt':>7} {'output':>7}  tools",
    ]
    for r in recs:
        who = r.get("agent", "lead")
        who = "lead" if who == "lead" else who[:14]
        ttft = f"{r['ttft_s']:.1f}s" if r.get("ttft_s") is not None else "-"
        tools = "; ".join(
            f"{'✗ ' if not t.get('ok', True) else ''}{t['name']}({t.get('args', '')[:40]}) → {fmt_chars(t.get('result_chars', 0))}"
            for t in r.get("tools", [])
        ) or ("(answer)" if r.get("text_chars") else "")
        lines.append(
            f"{r['step']:>3}  {who:<14} {r.get('duration_s', 0):>5.1f}s {ttft:>5} "
            f"{fmt_tok(r.get('prompt_tokens')):>7} {fmt_tok(r.get('completion_tokens')):>7}  {tools}"
        )
    found = insights(recs)
    if found:
        lines.append("Where it went:")
        lines += [f"  • {f}" for f in found]
    return "\n".join(lines)


def to_json(records: List[Dict[str, Any]], *, turn: Optional[int] = None, provider: str = "",
            system_prompt: Optional[str] = None, messages: Optional[list] = None) -> Dict[str, Any]:
    recs = turn_records(records, turn) if turn is not None or records else []
    doc: Dict[str, Any] = {
        "version": 1, "provider": provider, "turn": (recs[0].get("turn") if recs else None),
        "summary": totals(recs), "insights": insights(recs), "steps": recs,
    }
    if messages is not None:
        doc["system_prompt"] = system_prompt
        doc["messages"] = messages
    return doc
