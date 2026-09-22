"""Model failover: switching to a configured fallback provider when the primary is unavailable."""
from pathlib import Path

import httpx
import pytest

from core.providers import ProviderError
from tests.test_agent_loop import Scripted, call, make_agent, run, text


def arm(agent, *fallbacks, ids=None):
    """Give the agent fallback ids and a builder that returns the prepared providers by id."""
    ids = ids or [f"fb{i + 1}" for i in range(len(fallbacks))]
    table = dict(zip(ids, fallbacks))
    agent.fallback_ids = list(ids)
    agent.provider_builder = lambda pid: table.get(pid)
    agent.failovers = []
    return agent


def failing(exc):
    return Scripted([exc])


@pytest.mark.parametrize("error", [
    ProviderError("HTTP 503 from x: overloaded", status_code=503),
    ProviderError("HTTP 429 from x: rate limited", status_code=429),
    ProviderError("HTTP 401 from x: bad key", status_code=401),
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("refused"),
])
async def test_availability_errors_switch_to_the_fallback_and_finish_the_turn(tmp_path: Path, error):
    primary, backup = failing(error), Scripted([text("Answered by the backup.")])
    agent = arm(make_agent(primary), backup)
    resp, chunks, traces = await run(agent, workspace=str(tmp_path))
    assert resp == "Answered by the backup." and agent.provider is backup
    assert len(primary.requests) == 1 and len(backup.requests) == 1
    assert any("switching to fb1" in c for c in chunks) and any(s == "failover" for s, _ in traces)
    assert agent.failovers[0][1] == "fb1"


@pytest.mark.parametrize("error", [
    ProviderError("HTTP 400 from x: bad request", status_code=400),
    ProviderError("HTTP 404 from x: no such model", status_code=404),
    ProviderError("HTTP 422 from x: unprocessable", status_code=422),
])
async def test_request_errors_do_not_fail_over_because_any_provider_would_reject_them(tmp_path: Path, error):
    backup = Scripted([text("should not run")])
    agent = arm(make_agent(failing(error)), backup)
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert "request failed" in resp and not backup.requests and agent.failovers == []


async def test_without_fallbacks_nothing_changes(tmp_path: Path):
    agent = make_agent(failing(httpx.ReadTimeout("slow")))
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert "timed out" in resp


async def test_fallbacks_are_tried_in_order_and_unusable_ones_are_skipped(tmp_path: Path):
    good = Scripted([text("third time lucky")])
    agent = make_agent(failing(ProviderError("HTTP 500", status_code=500)))
    agent.fallback_ids = ["no-key", "unknown", "good"]
    table = {"good": good}
    agent.provider_builder = lambda pid: table.get(pid)                       # the first two are unusable
    agent.failovers = []
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert resp == "third time lucky" and sum(s == "failover_skipped" for s, _ in traces) == 2


async def test_running_out_of_fallbacks_reports_the_last_error(tmp_path: Path):
    second = failing(ProviderError("HTTP 502 from y: bad gateway", status_code=502))
    agent = arm(make_agent(failing(ProviderError("HTTP 500 from x", status_code=500))), second)
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert "HTTP 502" in resp and agent.provider is second and len(agent.failovers) == 1


async def test_the_switch_happens_midway_through_a_tool_loop_and_keeps_the_work(tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello")
    primary = Scripted([call("1", "read_file", path="f.txt"), ProviderError("HTTP 503", status_code=503)])
    backup = Scripted([text("Finished: the file says hello.")])
    agent = arm(make_agent(primary), backup)
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert resp.startswith("Finished")
    seen = str(backup.requests[0]["messages"])
    assert "f.txt" in seen and "hello" in seen                                # the backup received the tool result already gathered


async def test_the_switch_is_sticky_for_later_turns(tmp_path: Path):
    backup = Scripted([text("first"), text("second")])
    agent = arm(make_agent(failing(httpx.ConnectError("down"))), backup)
    await run(agent, "one", workspace=str(tmp_path))
    resp, _, _ = await run(agent, "two", workspace=str(tmp_path))
    assert resp == "second" and agent.provider is backup and len(agent.failovers) == 1


async def test_switching_to_a_provider_without_native_tools_falls_back_to_the_text_protocol(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    xml_backup = Scripted([text('<motion_tool>{"name":"read_file","arguments":{"path":"f.txt"}}</motion_tool>'), text("read via xml")],
                          native=False)
    agent = arm(make_agent(failing(ProviderError("HTTP 503", status_code=503))), xml_backup)
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert resp == "read via xml" and xml_backup.requests[0]["tools"] in (None, [])


async def test_subagents_follow_a_failover(tmp_path: Path):
    from tests.test_subagents import Router, task

    backup = Router(lead=[task("1", "alpha"), text("lead done")], subs={"alpha": [text("sub report")]})
    primary = failing(ProviderError("HTTP 503", status_code=503))
    agent = arm(make_agent(primary), backup)
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert resp == "lead done" and len(backup.sub_requests) == 1


def test_configure_agent_reads_the_list_and_builds_only_usable_providers(monkeypatch):
    from core.agent_config import configure_agent, make_provider_builder

    class FakeCM:
        def get_provider_config(self, pid):
            if pid == "nope":
                raise ValueError("unknown")
            return {"endpoint": "https://api.example.com/v1", "provider_type": "cloud", "options": {"model": "m"}, "name": pid}

        def has_api_key(self, pid):
            return pid == "with-key"

    class A: pass
    a = configure_agent(A(), {"fallback_providers": ["with-key", "no-key", "nope"]}.get, FakeCM())
    assert a.fallback_ids == ["with-key", "no-key", "nope"]
    assert a.provider_builder("with-key") is not None
    assert a.provider_builder("no-key") is None and a.provider_builder("nope") is None
    assert configure_agent(A(), {"fallback_providers": "solo"}.get).fallback_ids == ["solo"]
    assert configure_agent(A(), {}.get).fallback_ids == [] and configure_agent(A(), {}.get).provider_builder is None
