"""Unit tests for the semantic-version bumping script used by the
version-release GitHub Action."""
import importlib.util
import os
import sys

import pytest


def _load_module():
    here = os.path.dirname(__file__)
    script = os.path.normpath(os.path.join(here, "..", ".github", "scripts", "bump_version.py"))
    spec = importlib.util.spec_from_file_location("bump_version", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bump_mod():
    return _load_module()


def test_parse_roundtrip(bump_mod):
    major, minor, patch, pre, build = bump_mod.parse("0.2.0-beta.1")
    assert (major, minor, patch, pre, build) == ("0", "2", "0", "beta.1", None)


def test_parse_rejects_garbage(bump_mod):
    with pytest.raises(ValueError):
        bump_mod.parse("not-a-version")


def test_core_bump_types(bump_mod):
    assert bump_mod.bump_core("major", "1", "2", "3") == "2.0.0"
    assert bump_mod.bump_core("minor", "1", "2", "3") == "1.3.0"
    assert bump_mod.bump_core("patch", "1", "2", "3") == "1.2.4"


def test_bump_commits_feat(bump_mod, monkeypatch):
    monkeypatch.setattr(
        bump_mod.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": "feat: new thing\n"})(),
    )
    assert bump_mod.bump_from_commits() == "minor"


def test_bump_commits_fix_defaults_patch(bump_mod, monkeypatch):
    monkeypatch.setattr(
        bump_mod.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": "fix: a bug\n"})(),
    )
    assert bump_mod.bump_from_commits() == "patch"


def test_bump_commits_breaking(bump_mod, monkeypatch):
    monkeypatch.setattr(
        bump_mod.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": "fix!: break things\n"})(),
    )
    assert bump_mod.bump_from_commits() == "major"


def test_script_from_version_string(bump_mod, monkeypatch):
    monkeypatch.setattr(bump_mod.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "feat: x\n"})())
    # bump_from_commits returns minor and pre exists -> core bumps + pre resets to beta.0
    old = "0.2.0-beta.1"
    major, minor, patch, pre, _ = bump_mod.parse(old)
    new_core = bump_mod.bump_core(bump_mod.bump_from_commits(), major, minor, patch)
    # emulate main() logic
    new_pre = "beta.0"
    assert f"{new_core}-{new_pre}" == "0.3.0-beta.0"
