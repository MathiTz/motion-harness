"""Headless subcommands (core/cli_commands.py): driven through `main.main(argv)` the way a shell or
CI would, with every provider-touching piece replaced - no network, no billed calls. The point of
these is the contract: parseable `--json` on stdout, human progress off stdout, and exit codes."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as motion_main
from core.session import SessionStore, state_dir
from core.skill_state import Provenance, write_candidate


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """A throwaway config.yml so `tracking on|off` never touches the real one."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("track_interactions: false\n")
    monkeypatch.setenv("MOTION_CONFIG", str(cfg))
    return cfg


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    return path


def run(capsys, *argv):
    code = motion_main.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def run_json(capsys, *argv):
    code, out, err = run(capsys, *argv, "--json")
    return code, json.loads(out), err


# ── argument parsing ─────────────────────────────────────────────────────────

def test_a_subcommand_does_not_clobber_top_level_options():
    """argparse subparsers write their own defaults over already-parsed values: `--provider` given
    before the subcommand must survive, and an absent option must not appear as an explicit None."""
    args = motion_main.build_parser().parse_args(["--provider", "p/m", "--workspace", "/w", "skill", "list"])
    assert args.provider == "p/m" and args.workspace == "/w"
    args = motion_main.build_parser().parse_args(["skill", "list", "--provider", "q", "--json"])
    assert args.provider == "q" and args.json_output is True


def test_headless_prompt_still_wins_over_subcommands_absent():
    args = motion_main.build_parser().parse_args(["-p", "hi", "--resume", "20260101-000000"])
    assert args.prompt == "hi" and args.resume == "20260101-000000" and args.command is None


# ── skill ────────────────────────────────────────────────────────────────────

def _candidate(ws: Path, name="refactor_parser") -> Path:
    path = state_dir(ws, "skills") / f"{name}.md"
    write_candidate(path, "# Refactor the parser\n1. step\n", Provenance(task_id="t", prompt="refactor the parser"))
    return path


def test_skill_list_hides_candidates_until_all(capsys, ws):
    _candidate(ws)
    code, data, _ = run_json(capsys, "skill", "list", "--workspace", str(ws))
    assert code == 0 and "refactor_parser" not in {s["name"] for s in data["skills"]}
    assert data["awaiting_review"] == 1
    code, data, _ = run_json(capsys, "skill", "list", "--all", "--workspace", str(ws))
    row = next(s for s in data["skills"] if s["name"] == "refactor_parser")
    assert row["status"] == "candidate" and row["verified"] is False and row["saved_by_hand"] is False


def test_skill_candidates_and_show_expose_review_details(capsys, ws):
    _candidate(ws)
    code, data, _ = run_json(capsys, "skill", "candidates", "--workspace", str(ws))
    assert [c["name"] for c in data["candidates"]] == ["refactor_parser"]
    assert data["candidates"][0]["from_prompt"] == "refactor the parser"
    code, data, _ = run_json(capsys, "skill", "show", "refactor_parser", "--workspace", str(ws))
    assert code == 0 and "Refactor the parser" in data["content"] and data["status"] == "candidate"


def test_skill_show_unknown_name_is_a_usage_error(capsys, ws):
    code, out, err = run(capsys, "skill", "show", "nope", "--workspace", str(ws))
    assert code == 2 and out == "" and "skill not found" in err


class _StubSynth:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    async def promote(self, path):
        self.calls.append(("promote", path))
        return self.ok

    def reject(self, path):
        self.calls.append(("reject", path))
        return self.ok

    def rollback(self, path):
        self.calls.append(("rollback", path))
        return self.ok


def _stub_agent(monkeypatch, synth):
    closed = []
    agent = SimpleNamespace(synthesizer=synth, memory=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr("core.headless.build_agent", lambda provider, workspace=None: agent)
    return closed


@pytest.mark.parametrize("action", ["promote", "reject", "rollback"])
def test_skill_lifecycle_actions_go_through_the_synthesizer(capsys, ws, monkeypatch, action):
    path = _candidate(ws)
    synth = _StubSynth()
    closed = _stub_agent(monkeypatch, synth)
    code, data, _ = run_json(capsys, "skill", action, "refactor_parser", "--workspace", str(ws))
    assert code == 0 and data == {"skill": "refactor_parser", "action": action, "ok": True}
    assert synth.calls == [(action, str(path))] and closed == [True]


def test_skill_lifecycle_failure_exits_nonzero_with_a_reason(capsys, ws, monkeypatch):
    _candidate(ws)
    _stub_agent(monkeypatch, _StubSynth(ok=False))
    code, data, _ = run_json(capsys, "skill", "rollback", "refactor_parser", "--workspace", str(ws))
    assert code == 1 and data["ok"] is False and "previous version" in data["reason"]


def test_skill_ab_test_json_reports_verdict_and_never_promotes(capsys, ws, monkeypatch, tmp_path):
    import scripts.skill_ab_test as ab

    async def regress(task, provider, skill_path):
        return skill_path is None  # passes without the skill, fails with it

    monkeypatch.setattr(ab, "run_variant", regress)
    skill = tmp_path / "cand.md"
    skill.write_text("# cand\n")
    code, data, err = run_json(capsys, "skill", "ab-test", "--skill", str(skill), "--tasks", "bugfix-off-by-one")
    assert code == 1 and data["verdict"] == "do_not_promote" and data["regressions"] == ["bugfix-off-by-one"]
    assert "A/B testing" in err  # progress went to stderr, stdout stayed pure JSON


# ── eval ─────────────────────────────────────────────────────────────────────

def test_eval_list_json_is_parseable(capsys):
    code, data, _ = run_json(capsys, "eval", "list")
    assert code == 0 and data and {"id", "category", "scoring", "held_out"} <= set(data[0])
    assert not any(t["held_out"] for t in data)
    code, with_held, _ = run_json(capsys, "eval", "list", "--include-held-out")
    assert len(with_held) > len(data)


def test_eval_run_json_has_summary_and_keeps_progress_off_stdout(capsys, monkeypatch, tmp_path):
    import scripts.run_evals as re_
    from evals.lib import TaskResult

    async def fake_run(task, *, provider_id, workdir, keep_workdir=False, **_):
        return TaskResult(task.id, task.category, task.held_out, True, True, None, None, 0.01, 1.0, 2, 3, "p/m", "m")

    monkeypatch.setattr(re_, "run_task", fake_run)
    monkeypatch.setattr(re_, "run_resume_task", fake_run)
    out_path = tmp_path / "report.json"
    code, data, err = run_json(capsys, "eval", "run", "--task", "bugfix-off-by-one", "--out", str(out_path))
    assert code == 0 and data["summary"]["total"] == 1 and data["summary"]["success_rate"] == 1.0
    assert data["report_path"] == str(out_path) and out_path.exists()
    assert "bugfix-off-by-one" in err and "PASS" in err


def test_eval_run_unknown_task_is_a_usage_error(capsys):
    code, _, err = run(capsys, "eval", "run", "--task", "does-not-exist")
    assert code == 2 and "no matching tasks" in err


def test_eval_memory_rejects_an_unknown_scenario(capsys):
    code, _, err = run(capsys, "eval", "memory", "--scenario", "nope")
    assert code == 2 and "unknown scenario" in err


def test_eval_memory_json_runs_the_requested_scenario(capsys, monkeypatch):
    import scripts.eval_memory_quality as emq
    from evals.memory_quality import ScenarioResult

    async def fake(workdir, provider_id):
        return ScenarioResult("fake", "rel", True, "down", True, cost_usd=0.01)

    monkeypatch.setattr(emq, "SCENARIOS", {"fake": fake})
    monkeypatch.setattr("evals.memory_quality.SCENARIOS", {"fake": fake})
    code, data, err = run_json(capsys, "eval", "memory", "--scenario", "fake")
    assert code == 0 and data["scenarios"][0]["name"] == "fake" and "-> fake" in err


# ── trajectory ───────────────────────────────────────────────────────────────

def _traj(path: Path, steps: int, tool_args="a.py") -> Path:
    path.write_text(json.dumps({
        "harness_version": "1.0", "provider": "p/m",
        "summary": {"steps": steps, "tool_calls": 1, "prompt_tokens": 100 * steps, "completion_tokens": 10},
        "steps": [{"tools": [{"name": "read_file", "args": tool_args}]}],
    }))
    return path


def test_trajectory_compare_json(capsys, tmp_path):
    a, b = _traj(tmp_path / "a.json", 2), _traj(tmp_path / "b.json", 5, tool_args="b.py")
    code, data, _ = run_json(capsys, "trajectory", "compare", str(a), str(b))
    assert code == 0
    assert data["metrics"]["steps"] == {"before": 2, "after": 5, "delta": 3}
    assert data["only_in_before"] == [{"name": "read_file", "args": "a.py"}]
    assert data["only_in_after"] == [{"name": "read_file", "args": "b.py"}]


def test_trajectory_compare_text_and_bad_input(capsys, tmp_path):
    a = _traj(tmp_path / "a.json", 2)
    code, out, _ = run(capsys, "trajectory", "compare", str(a), str(a))
    assert code == 0 and "steps:" in out
    code, _, err = run(capsys, "trajectory", "compare", str(a), str(tmp_path / "missing.json"))
    assert code == 2 and "cannot read after trajectory" in err


def test_trajectory_list(capsys, ws):
    d = state_dir(ws, "trajectories")
    (d / "turn-1.json").write_text("{}")
    code, data, _ = run_json(capsys, "trajectory", "list", "--workspace", str(ws))
    assert code == 0 and [Path(t["file"]).name for t in data["trajectories"]] == ["turn-1.json"]


# ── sessions ─────────────────────────────────────────────────────────────────

def test_sessions_list_and_show(capsys, ws):
    store = SessionStore(ws, "20260101-000000")
    store.append({"prompt": "what is 2+2", "response": "4"})
    code, data, _ = run_json(capsys, "sessions", "list", "--workspace", str(ws))
    assert code == 0 and data["sessions"][0]["id"] == "20260101-000000" and data["sessions"][0]["turns"] == 1
    code, data, _ = run_json(capsys, "sessions", "show", "20260101-000000", "--workspace", str(ws))
    assert code == 0 and data["turns"][0]["response"] == "4"


def test_sessions_show_missing_is_a_usage_error(capsys, ws):
    code, _, err = run(capsys, "sessions", "show", "nope", "--workspace", str(ws))
    assert code == 2 and "no session named" in err


def test_sessions_list_empty_workspace(capsys, ws):
    code, data, _ = run_json(capsys, "sessions", "list", "--workspace", str(ws))
    assert code == 0 and data["sessions"] == []


# ── memory ───────────────────────────────────────────────────────────────────

def _seed_memory(ws: Path, *texts: str) -> Path:
    from memory.db import EMBEDDING_DIM, MemoryChunk, MemoryDB

    path = state_dir(ws) / "memory.db"
    db = MemoryDB(str(path))
    for t in texts:
        db.add_memory(MemoryChunk(content=t, embedding=[0.0] * EMBEDDING_DIM, metadata={}, mem_type="DOC"))
    db.close()
    return path


def test_memory_search_never_creates_a_database(capsys, ws):
    code, data, _ = run_json(capsys, "memory", "search", "anything", "--workspace", str(ws))
    assert code == 0 and data["results"] == []
    assert not (ws / ".motion" / "memory.db").exists()


def test_memory_search_falls_back_to_keyword_when_no_provider_can_be_built(capsys, ws, monkeypatch):
    _seed_memory(ws, "The deploy script lives in scripts/ship.sh", "unrelated note about lunch")

    def boom(provider, workspace=None):
        raise RuntimeError("no provider configured")

    monkeypatch.setattr("core.headless.build_agent", boom)
    code, data, _ = run_json(capsys, "memory", "search", "deploy script", "--workspace", str(ws))
    assert code == 0 and data["semantic"] is False
    assert data["results"] and "ship.sh" in data["results"][0]["content"]
    assert data["results"][0]["weak"] is False  # keyword hits carry no cosine score to be "weak"


def test_memory_stats(capsys, ws):
    _seed_memory(ws, "one", "two")
    code, data, _ = run_json(capsys, "memory", "stats", "--workspace", str(ws))
    assert code == 0 and data["memories"] == 2 and data["exists"] is True
    code, data, _ = run_json(capsys, "memory", "stats", "--workspace", str(ws / "other"))
    assert data["exists"] is False and data["memories"] is None


# ── tracking ─────────────────────────────────────────────────────────────────

def test_tracking_on_off_status_persist_to_config(capsys, ws, isolated_config):
    code, data, _ = run_json(capsys, "tracking", "status", "--workspace", str(ws))
    assert code == 0 and data["tracking"] is False
    code, data, _ = run_json(capsys, "tracking", "on", "--workspace", str(ws))
    assert data["tracking"] is True and "track_interactions: true" in isolated_config.read_text()
    code, data, _ = run_json(capsys, "tracking", "status", "--workspace", str(ws))
    assert data["tracking"] is True
    code, data, _ = run_json(capsys, "tracking", "off", "--workspace", str(ws))
    assert data["tracking"] is False


# ── check ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("passes,expected", [(True, 0), (False, 1)])
def test_check_json_and_exit_code(capsys, monkeypatch, passes, expected):
    import scripts.live_check as lc

    seen = []

    async def fake_run_provider(provider_id, cm, rows=None, log=None):
        seen.append(provider_id)
        rows.append({"provider": provider_id, "check": "streaming", "status": "pass" if passes else "fail", "detail": "x"})
        print("human line", file=log)
        return passes

    monkeypatch.setattr(lc, "run_provider", fake_run_provider)
    code, data, err = run_json(capsys, "check", "fake/model")
    assert code == expected and seen == ["fake/model"] and data["ok"] is passes
    assert data["checks"][0]["status"] == ("pass" if passes else "fail")
    assert "human line" in err


# ── headless --resume ────────────────────────────────────────────────────────

def _recording_factory(history_seen):
    from tests.test_headless import factory, text

    make = factory([text("ok")])

    def build():
        agent = make()
        real_run = agent.run

        async def run(prompt, **kw):
            history_seen.append(kw.get("history"))
            return await real_run(prompt, **kw)

        agent.run = run
        return agent
    return build


async def test_resume_passes_prior_turns_as_history_and_records_the_new_one(ws):
    from core.headless import run_headless

    store = SessionStore(ws, "20260101-000000")
    store.append({"prompt": "my name is Ada", "response": "hi Ada"})
    seen = []
    out = io.StringIO()
    code = await run_headless(
        "what is my name?", workspace=str(ws), out=out, err=io.StringIO(),
        agent_factory=_recording_factory(seen),
        history=[{"role": "user", "content": "my name is Ada"}, {"role": "assistant", "content": "hi Ada"}],
        session_store=store,
    )
    assert code == 0 and out.getvalue() == "ok\n"
    assert seen[0][0]["content"] == "my name is Ada"
    turns = SessionStore.load(ws, "20260101-000000")
    assert [t["prompt"] for t in turns] == ["my name is Ada", "what is my name?"]
    assert turns[1]["response"] == "ok" and not turns[1].get("interrupted")


def test_resume_context_requires_an_existing_session(ws):
    from core.headless import HeadlessUsageError, _resume_context

    with pytest.raises(HeadlessUsageError, match="no session named"):
        _resume_context(SimpleNamespace(resume="nope", workspace=str(ws)))


def test_resume_context_only_records_when_tracking_is_on(ws, isolated_config):
    from core.headless import _resume_context

    SessionStore(ws, "s1").append({"prompt": "p", "response": "r"})
    history, store = _resume_context(SimpleNamespace(resume="s1", workspace=str(ws)))
    assert history == [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}]
    assert store is None  # tracking is off in the fixture config: never write a transcript the user opted out of
    isolated_config.write_text("track_interactions: true\n")
    _, store = _resume_context(SimpleNamespace(resume="s1.jsonl", workspace=str(ws)))
    assert store is not None and store.session_id == "s1"


def test_an_interrupted_prior_turn_is_not_replayed_as_an_answer(ws):
    from core.headless import _resume_context

    store = SessionStore(ws, "s2")
    store.start_turn("started but never finished")  # no turn_end: the process died
    history, _ = _resume_context(SimpleNamespace(resume="s2", workspace=str(ws)))
    assert history == [{"role": "user", "content": "started but never finished"}]
