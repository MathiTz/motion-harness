"""Cost estimation and its display."""
import pytest

from core.pricing import estimate_cost, format_cost, turn_cost
from core.providers import StreamEvent
from tests.test_agent_loop import text
from tests.test_tui_flow import _text_of, send, tui_app, wait_idle


def test_estimate_uses_per_million_prices():
    opts = {"input_mtok": 0.22, "output_mtok": 0.66}
    assert estimate_cost(opts, 1_000_000, 1_000_000) == pytest.approx(0.88)
    assert estimate_cost(opts, 2000, 500) == pytest.approx(0.00077)
    assert estimate_cost(opts, 0, 0) == 0


def test_unpriced_models_are_none_not_guessed():
    assert estimate_cost({}, 100, 100) is None
    assert estimate_cost({"input_mtok": 1.0}, 100, 100) is None            # both prices required
    assert estimate_cost({"input_mtok": None, "output_mtok": None}, 1, 1) is None  # catalog "unknown"
    assert estimate_cost({"input_mtok": "x", "output_mtok": 1}, 1, 1) is None


def test_local_is_free_and_cloud_uses_pricing():
    assert turn_cost("local", {}, 10_000, 10_000) == 0.0
    assert turn_cost("cloud", {"input_mtok": 1, "output_mtok": 2}, 1000, 1000) == pytest.approx(0.003)
    assert turn_cost("cloud", {}, 1000, 1000) is None


def test_format_cost():
    assert format_cost(None) == "n/a" and format_cost(0) == "$0"
    assert format_cost(0.00001) == "<$0.0001" and format_cost(0.0123) == "$0.0123"
    assert format_cost(12.5) == "$12.50"


def test_catalog_prices_flow_into_config():
    from core.config import ConfigManager

    opts = ConfigManager().get_provider_config("ollama-cloud/deepseek-v4-flash")["options"]
    assert (opts["input_mtok"], opts["output_mtok"]) == (0.22, 0.66)


async def test_status_shows_turn_and_session_cost_for_priced_cloud_models(tmp_path, monkeypatch):
    usage = [StreamEvent("usage", usage={"prompt_tokens": 100_000, "completion_tokens": 10_000, "total_tokens": 110_000})]
    async with tui_app(tmp_path, monkeypatch, [text("ok") + usage, text("ok2") + usage]) as (app, pilot):
        app.provider.config.provider_type = "cloud"
        app.provider.config.options.update(input_mtok=1.0, output_mtok=2.0)  # $0.10 + $0.02 per turn
        await send(app, pilot, "one")
        await wait_idle(app, pilot)
        status = _text_of(app.screen.query_one("#chat_status_text"))
        assert "$0.1200" in status                       # this turn (and, being the first, the session)
        assert app.state.session_metrics["estimated_cost_usd"] == pytest.approx(0.12)
        await send(app, pilot, "two")
        await wait_idle(app, pilot)
        assert app.state.session_metrics["estimated_cost_usd"] == pytest.approx(0.24)
        assert "$0.2400" in _text_of(app.screen.query_one("#chat_status_text"))


async def test_unpriced_cloud_model_says_so_instead_of_showing_nothing(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        app.provider.config.provider_type = "cloud"      # no pricing options
        await send(app, pilot, "hello")
        await wait_idle(app, pilot)
        assert app.state.session_metrics["unpriced_turns"] == 1
        assert "cost n/a" in _text_of(app.screen.query_one("#chat_status_text"))


async def test_local_models_never_report_a_cost(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        await send(app, pilot, "hello")
        await wait_idle(app, pilot)
        assert app.state.last_turn_metrics["estimated_cost_usd"] == 0.0
        assert "cost n/a" not in _text_of(app.screen.query_one("#chat_status_text"))
        assert app.state.session_metrics["unpriced_turns"] == 0


async def test_new_session_resets_cost_counters(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        app.provider.config.provider_type = "cloud"
        await send(app, pilot, "hello")
        await wait_idle(app, pilot)
        app.state.new_session()
        assert app.state.session_metrics["unpriced_turns"] == 0 and app.state.session_metrics["estimated_cost_usd"] == 0.0


def test_user_model_entry_does_not_erase_catalog_pricing_or_context_window():
    """Regression: a user's `models: x: {temperature, max_tokens}` used to replace the
    catalog entry wholesale, silently dropping prices and context_window."""
    from core.catalog import merge_catalog

    merged = merge_catalog({"ollama-cloud": {"models": {"deepseek-v4-flash": {"temperature": 0.2, "max_tokens": 9999}}}})
    m = merged["ollama-cloud"]["models"]["deepseek-v4-flash"]
    assert m["temperature"] == 0.2 and m["max_tokens"] == 9999            # user wins where they set it
    assert m["input_mtok"] == 0.22 and m["context_window"] == 1_000_000  # catalog fills the rest
    assert "glm-5.2" in merged["ollama-cloud"]["models"]                  # other catalog models remain
    custom = merge_catalog({"ollama-cloud": {"models": {"my-new-model": {"temperature": 1}}}})
    assert custom["ollama-cloud"]["models"]["my-new-model"] == {"temperature": 1}


def test_every_catalog_model_has_pricing_and_a_context_window():
    """Claude and OpenAI models used to have neither: cost showed 'n/a' for every user of them, and the
    agent fell back to a 32k context window even for models with far larger real ones. CLI delegates
    (provider_type "cli") are exempt: their "model" is a synthetic marker, not a real billable one -
    Claude Code reports its own actual cost per turn, and Codex reports none (subscription usage)."""
    from core.catalog import BUILTIN_CATALOG

    missing = [
        f"{pid}/{name}" for pid, cfg in BUILTIN_CATALOG.items() if cfg.get("provider_type") != "cli"
        for name, opts in (cfg.get("models") or {}).items()
        if "input_mtok" not in opts or "output_mtok" not in opts or "context_window" not in opts
    ]
    assert missing == []


def test_claude_and_openai_prices_are_sane_and_distinct_by_tier():
    from core.catalog import BUILTIN_CATALOG

    claude = BUILTIN_CATALOG["claude"]["models"]
    openai = BUILTIN_CATALOG["openai"]["models"]
    # output is always pricier than input, and nothing is free or absurd
    for name, opts in {**claude, **openai}.items():
        assert 0 < opts["input_mtok"] <= opts["output_mtok"] <= 1000, name
        assert opts["context_window"] >= 100_000, name
    # a flagship costs more than its mini/haiku sibling
    assert claude["claude-opus-5"]["input_mtok"] > claude["claude-haiku-4-5"]["input_mtok"]
    assert openai["gpt-5"]["input_mtok"] > openai["gpt-5-nano"]["input_mtok"]
