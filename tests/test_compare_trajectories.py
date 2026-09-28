"""scripts/compare_trajectories.py: cross-run/cross-version trajectory comparison (issue #19)."""
import json
import subprocess
import sys
from pathlib import Path

import core.trajectory as traj
from scripts.compare_trajectories import call_set, compare, estimated_cost

REPO = Path(__file__).resolve().parents[1]


def rec(step, prompt, completion=20, tools=(), duration=1.0):
    return {
        "turn": 1, "agent": "lead", "step": step, "duration_s": duration, "ttft_s": 0.2,
        "prompt_tokens": prompt, "completion_tokens": completion, "context_tokens_est": prompt,
        "text_chars": 0, "reasoning_chars": 0,
        "tools": [{"name": n, "args": a, "ok": True, "result_chars": 100, "duration_s": 0.01} for n, a in tools],
    }


def doc(prompt_tokens, tools=(), provider="ollama-cloud/deepseek-v4-flash", **extra):
    d = traj.to_json([rec(1, prompt_tokens, tools=tools)], turn=1, provider=provider)
    d.update(extra)
    return d


def test_call_set_extracts_name_and_args_preview_pairs():
    d = doc(1000, tools=[("read_file", "path=a.py"), ("grep", "pattern=x")])
    assert call_set(d) == {("read_file", "path=a.py"), ("grep", "pattern=x")}


def test_estimated_cost_resolves_pricing_from_the_builtin_catalog():
    d = doc(1_000_000, provider="ollama-cloud/deepseek-v4-flash")
    d["summary"]["completion_tokens"] = 0
    cost = estimated_cost(d)
    assert cost == 0.22  # input_mtok for this model, per 1M prompt tokens


def test_estimated_cost_is_none_for_an_unpriced_or_malformed_provider():
    assert estimated_cost(doc(1000, provider="totally-unknown-provider/model-x")) is None
    assert estimated_cost(doc(1000, provider="no-slash-at-all")) is None


def test_compare_reports_step_token_and_time_deltas():
    before = doc(2000, tools=[("read_file", "path=a.py")])
    after = doc(1500, tools=[("grep", "pattern=x")])
    report = compare(before, after)
    assert "prompt tokens:      2000 -> 1500 (-500) (-25%)" in report
    assert "- read_file(path=a.py)  [only in before]" in report
    assert "+ grep(pattern=x)  [only in after]" in report


def test_compare_flags_identical_provider_and_version_as_likely_noise():
    same = doc(1000)
    assert "comparing noise" in compare(same, same)


def test_compare_shows_no_call_set_differences_when_the_same_tools_ran():
    before = doc(1000, tools=[("read_file", "path=a.py")])
    after = doc(900, tools=[("read_file", "path=a.py")])
    assert "call-set differences: none" in compare(before, after)


def test_compare_surfaces_a_changed_task_outcome_when_present():
    before = doc(1000, task_result="fail")
    after = doc(1000, task_result="pass")
    report = compare(before, after)
    assert "task outcome: 'fail' -> 'pass'" in report and "CHANGED" in report


def test_compare_is_silent_about_task_outcome_when_neither_side_has_it():
    assert "task outcome" not in compare(doc(1000), doc(1000))


# ── CLI, against two small fixture files ────────────────────────────────────

def test_cli_prints_a_comparison_given_two_saved_trajectory_files(tmp_path):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(doc(2000, tools=[("read_file", "path=a.py")])))
    after_path.write_text(json.dumps(doc(1000, tools=[("grep", "pattern=x")])))
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_trajectories.py"), str(before_path), str(after_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "prompt tokens:      2000 -> 1000" in proc.stdout
    assert "only in before" in proc.stdout and "only in after" in proc.stdout
