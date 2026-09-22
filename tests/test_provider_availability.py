"""Provider availability: CLI delegates are 'usable' based on PATH (not a stored key), an unavailable
provider explains why instead of silently disappearing, and the model picker shows it that way too."""
import os
import stat
import sys
from pathlib import Path

import pytest

from core.catalog import BUILTIN_CATALOG
from core.cli_delegate import clear_detection_cache
from core.config import ConfigManager
from core.providers import ModelConfig, ProviderFactory


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Puts a real, executable stand-in for `claude`/`codex` on PATH for the duration of a test - the
    same approach test_cli_delegate.py uses, so availability is exercised for real, not by mocking
    internals that a refactor could silently stop matching."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def install(name: str) -> None:
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\nimport sys; sys.exit(0)\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    clear_detection_cache()
    yield install
    clear_detection_cache()


@pytest.fixture(autouse=True)
def _no_real_binaries(monkeypatch):
    """However this test runner's own PATH is set up, start every test from a clean slate: neither
    delegate is available until a test's own fake_bin() says otherwise."""
    monkeypatch.setenv("PATH", "")
    clear_detection_cache()
    yield
    clear_detection_cache()


# ── catalog shape ────────────────────────────────────────────────────────────

def test_catalog_has_both_cli_delegates_with_a_delegate_key():
    for pid, key in (("claude-cli", "claude-cli"), ("codex-cli", "codex-cli")):
        cfg = BUILTIN_CATALOG[pid]
        assert cfg["provider_type"] == "cli" and "endpoint" in cfg
        model = next(iter(cfg["models"].values()))
        assert model["delegate"] == key


# ── has_api_key / availability ───────────────────────────────────────────────

def test_cli_delegate_availability_follows_path_not_a_stored_key(fake_bin):
    cm = ConfigManager()
    assert cm.has_api_key("claude-cli") is False and cm.has_api_key("codex-cli") is False
    fake_bin("claude")
    clear_detection_cache()                                       # a fresh PATH lookup, not the cached "missing" result
    assert cm.has_api_key("claude-cli") is True and cm.has_api_key("codex-cli") is False


def test_local_and_unknown_provider_types_are_unaffected():
    cm = ConfigManager()
    assert cm.has_api_key("local-llama") is True                 # unrelated existing behavior, unchanged


# ── unavailable_reason ────────────────────────────────────────────────────────

def test_available_provider_has_no_reason(fake_bin):
    fake_bin("claude")
    assert ConfigManager().unavailable_reason("claude-cli") == ""
    assert ConfigManager().unavailable_reason("local-llama") == ""


def test_cli_delegate_reason_names_the_actual_binary_and_the_login_command():
    cm = ConfigManager()
    reason = cm.unavailable_reason("claude-cli")
    assert "claude" in reason and "claude login" in reason and "PATH" in reason
    reason2 = cm.unavailable_reason("codex-cli")
    assert "codex" in reason2 and "codex login" in reason2


def test_api_key_reason_names_the_env_var_and_auth_command():
    reason = ConfigManager().unavailable_reason("claude")
    assert "motion auth login claude" in reason and "CLAUDE_API_KEY" in reason and "Ctrl+A" in reason


def test_api_key_reason_offers_the_cli_delegate_only_when_it_is_actually_available(fake_bin):
    cm = ConfigManager()
    assert "switch to" not in cm.unavailable_reason("claude")     # claude-cli not available: no false promise
    fake_bin("claude")
    clear_detection_cache()
    assert "switch to" in cm.unavailable_reason("claude") and "Claude Code" in cm.unavailable_reason("claude")


# ── ProviderFactory wiring ────────────────────────────────────────────────────

def test_provider_factory_builds_the_right_delegate_class():
    from core.cli_delegate import ClaudeCLIProvider, CodexCLIProvider

    cfg = ConfigManager().get_provider_config("claude-cli")
    p = ProviderFactory.get_provider(ModelConfig(
        name=cfg.get("name"), endpoint=cfg["endpoint"], provider_type=cfg["provider_type"], options=cfg.get("options", {}),
    ))
    assert isinstance(p, ClaudeCLIProvider) and p.is_delegate and p.native_tools is False

    cfg2 = ConfigManager().get_provider_config("codex-cli")
    p2 = ProviderFactory.get_provider(ModelConfig(
        name=cfg2.get("name"), endpoint=cfg2["endpoint"], provider_type=cfg2["provider_type"], options=cfg2.get("options", {}),
    ))
    assert isinstance(p2, CodexCLIProvider)


def test_provider_factory_rejects_an_unknown_delegate_key():
    with pytest.raises(ValueError, match="Unknown CLI delegate"):
        ProviderFactory.get_provider(ModelConfig(name="x", endpoint="cli://x", provider_type="cli", options={"delegate": "nope"}))


# ── TUI model picker ──────────────────────────────────────────────────────────

async def test_unavailable_providers_show_with_a_reason_instead_of_vanishing(tmp_path, monkeypatch):
    from tests.test_tui_flow import tui_app, wait_for_screen
    from tests.test_agent_loop import text as text_step
    import ui.tui as tui

    async with tui_app(tmp_path, monkeypatch, [text_step("hi")]) as (app, pilot):
        await pilot.press("ctrl+o")
        screen = await wait_for_screen(app, pilot, tui.ModelDialog)
        labels = [_row_label(w) for w in screen.query(tui.ModelOption)]
        assert any("claude-cli" == item.full_id and not item.available for item in screen.query(tui.ModelOption)) or \
            any("Claude Code" in label and ("PATH" in label or "API key" in label) for label in labels)
        unavailable = [item for item in screen.query(tui.ModelOption) if not item.available]
        assert unavailable and all("unavailable" in item.classes for item in unavailable)


async def test_selecting_an_unavailable_provider_explains_rather_than_switching(tmp_path, monkeypatch):
    from tests.test_tui_flow import tui_app, wait_for_screen
    from tests.test_agent_loop import text as text_step
    import ui.tui as tui

    async with tui_app(tmp_path, monkeypatch, [text_step("hi")]) as (app, pilot):
        before = app.state.current_provider_id
        await pilot.press("ctrl+o")
        screen = await wait_for_screen(app, pilot, tui.ModelDialog)
        target = next(w for w in screen.query(tui.ModelOption) if not w.available)
        notifications = []
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notifications.append(msg))
        screen._select(target)
        assert notifications and ("PATH" in notifications[0] or "API key" in notifications[0])
        assert app.state.current_provider_id == before                # nothing actually switched


def _row_label(item) -> str:
    for child in item.children:
        text = getattr(child, "renderable", None)
        if text is not None:
            return str(text)
    return ""
