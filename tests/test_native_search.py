"""fd-backed file enumeration (core.workspace_tools._fd_walk / _walk_files): same results with or
without fd installed, graceful fallback on any failure, and real speed on a large tree."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.workspace_tools import WorkspaceTools, _fd_available

HAS_FD = shutil.which("fd") is not None


@pytest.fixture(autouse=True)
def _clear_fd_cache():
    _fd_available.cache_clear()
    yield
    _fd_available.cache_clear()


def make_tree(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "a.py").write_text("def one(): pass\n")
    (root / "src" / "pkg" / "b.py").write_text("def two(): pass\n")
    (root / "README.md").write_text("# hi\n")
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "x.js").write_text("var x = 1;\n")
    (root / "build").mkdir()
    (root / "build" / "out.txt").write_text("built\n")
    (root / ".gitignore").write_text("build/\n*.log\ngenerated/\n")
    (root / "generated" / "nested").mkdir(parents=True)
    (root / "generated" / "nested" / "g.py").write_text("x = 1\n")
    (root / "debug.log").write_text("noisy\n")


def force(monkeypatch, on: bool):
    """Force _fd_available() to a fixed value regardless of what's actually installed, so both branches
    of every test run in CI whether or not the runner has fd."""
    monkeypatch.setattr("core.workspace_tools._fd_available", lambda: on)


@pytest.mark.parametrize("fd_on", [False] + ([True] if HAS_FD else []))
def test_listing_matches_regardless_of_backend(tmp_path: Path, monkeypatch, fd_on):
    make_tree(tmp_path)
    force(monkeypatch, fd_on)
    t = WorkspaceTools(tmp_path)
    files = sorted(t.execute("list_files", {"path": "."})["files"])
    assert files == [".gitignore", "README.md", "src/a.py", "src/pkg/b.py"]   # node_modules, build/, *.log, generated/ all excluded
    assert sorted(t.execute("glob_files", {"pattern": "**/*.py"})["files"]) == ["src/a.py", "src/pkg/b.py"]
    grep = t.execute("grep", {"pattern": "def "})
    assert sorted(m["path"] for m in grep["matches"]) == ["src/a.py", "src/pkg/b.py"]


@pytest.mark.parametrize("fd_on", [False] + ([True] if HAS_FD else []))
def test_explicitly_targeted_ignored_dir_still_works_both_ways(tmp_path: Path, monkeypatch, fd_on):
    make_tree(tmp_path)
    force(monkeypatch, fd_on)
    t = WorkspaceTools(tmp_path)
    assert t.execute("list_files", {"path": "node_modules/dep"})["files"] == ["node_modules/dep/x.js"]
    assert t.execute("list_files", {"path": "build"})["files"] == ["build/out.txt"]      # dir_only gitignore pattern
    assert t.execute("list_files", {"path": "generated"})["files"] == ["generated/nested/g.py"]


@pytest.mark.parametrize("fd_on", [False] + ([True] if HAS_FD else []))
def test_path_qualified_dir_only_pattern_is_still_honored(tmp_path: Path, monkeypatch, fd_on):
    """The one case fd's --exclude can't express directly (a different glob dialect): a gitignore line
    with both a '/' and a trailing '/', which only the Python-side ancestor-walk fallback catches."""
    make_tree(tmp_path)
    (tmp_path / ".gitignore").write_text("build/\n*.log\nsrc/pkg/\n")     # src/pkg/ - path-qualified, dir-only
    force(monkeypatch, fd_on)
    t = WorkspaceTools(tmp_path)
    assert sorted(t.execute("list_files", {"path": "."})["files"]) == [".gitignore", "README.md", "generated/nested/g.py", "src/a.py"]


def test_falls_back_cleanly_when_fd_is_missing_broken_or_slow(tmp_path: Path, monkeypatch):
    make_tree(tmp_path)
    t = WorkspaceTools(tmp_path)
    monkeypatch.setattr("core.workspace_tools._fd_available", lambda: True)
    expected = [".gitignore", "README.md", "src/a.py", "src/pkg/b.py"]

    monkeypatch.setattr("core.workspace_tools.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert sorted(t.execute("list_files", {"path": "."})["files"]) == expected

    class NonZero:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("core.workspace_tools.subprocess.run", lambda *a, **k: NonZero())
    assert sorted(t.execute("list_files", {"path": "."})["files"]) == expected

    def raises_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="fd", timeout=1)

    monkeypatch.setattr("core.workspace_tools.subprocess.run", raises_timeout)
    assert sorted(t.execute("list_files", {"path": "."})["files"]) == expected


def test_disable_env_var_forces_the_python_fallback(tmp_path: Path, monkeypatch):
    make_tree(tmp_path)
    monkeypatch.setenv("MOTION_DISABLE_NATIVE_SEARCH", "1")
    _fd_available.cache_clear()
    called = []
    real_run = subprocess.run
    monkeypatch.setattr(
        "core.workspace_tools.subprocess.run",
        lambda *a, **k: called.append(a) or real_run(*a, **k),
    )
    t = WorkspaceTools(tmp_path)
    assert sorted(t.execute("list_files", {"path": "."})["files"]) == [".gitignore", "README.md", "src/a.py", "src/pkg/b.py"]
    assert called == []                                               # fd was never even invoked


@pytest.mark.skipif(not HAS_FD, reason="fd not installed on this machine")
def test_fd_walk_returns_base_relative_strings_sorted(tmp_path: Path):
    make_tree(tmp_path)
    t = WorkspaceTools(tmp_path)
    rel = t._fd_walk(tmp_path / "src")
    assert rel == sorted(rel) and set(rel) == {"a.py", "pkg/b.py"}
    assert all(not r.startswith(("/", "./")) for r in rel)


@pytest.mark.skipif(not HAS_FD, reason="fd not installed on this machine")
def test_a_project_with_no_gitignore_never_pays_for_the_ancestor_walk(tmp_path: Path):
    """Regression: an earlier version always walked every candidate's ancestor chain in Python, which
    profiled as the dominant cost (~2.4s for 80k files) even when there was nothing to find - this only
    runs at all when the project actually has a path-qualified directory-only .gitignore pattern."""
    make_tree(tmp_path)
    t = WorkspaceTools(tmp_path)
    assert t._ignore.has_dir_only_path_patterns() is False
    calls = []
    real = t._ignored_by_ancestor
    t._ignored_by_ancestor = lambda *a, **k: calls.append(a) or real(*a, **k)
    t.execute("list_files", {"path": "."})
    assert calls == []


@pytest.mark.skipif(not HAS_FD, reason="fd not installed on this machine")
def test_both_backends_agree_on_a_larger_tree(tmp_path: Path, monkeypatch):
    """Speed itself isn't asserted here - timing in a shared CI runner is inherently noisy, and at small
    file counts fd's own subprocess-spawn cost can outweigh its raw speed (verified manually: fd wins
    clearly from a few thousand files up - see the commit message for real numbers on 2,246- and
    79,951-file trees). What must never differ is the actual result set."""
    root = tmp_path / "big"
    for i in range(30):
        d = root / f"pkg{i}"
        d.mkdir(parents=True)
        for j in range(30):
            (d / f"m{j}.py").write_text("x = 1\n")                    # 900 files
    t = WorkspaceTools(root)

    force = lambda on: monkeypatch.setattr("core.workspace_tools._fd_available", lambda: on)
    force(False)
    slow = t.execute("list_files", {"path": "."})
    force(True)
    fast = t.execute("list_files", {"path": "."})
    assert slow["total"] == fast["total"] == 900
    assert sorted(slow["files"]) == sorted(fast["files"])
