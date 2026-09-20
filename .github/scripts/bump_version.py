#!/usr/bin/env python3
"""Increment a semver version string for the VERSION file.

Usage:
    bump_version.py [OLD_VERSION]

If OLD_VERSION is omitted it is read from VERSION. The bump type is chosen
from the most recent commit message (conventional commits) since the last
release tag:

    - '!' / BREAKING CHANGE  -> major bump
    - feat:                  -> minor bump
    - fix:/chore:/refactor:  -> patch bump (default)

A prerelease suffix is maintained as beta.N and incremented when the core
version is unchanged between releases. If the core version itself changes,
the prerelease resets to beta.0.
"""
import re
import sys
import subprocess


def parse(version: str):
    m = re.fullmatch(
        r"(?P<core>\d+\.\d+\.\d+)"
        r"(-(?P<pre>[a-zA-Z0-9.-]+))?"
        r"(\+(?P<build>[a-zA-Z0-9.-]+))?",
        version.strip(),
    )
    if not m:
        raise ValueError(f"Unparseable semver: {version!r}")
    major, minor, patch = m.group("core").split(".")
    return major, minor, patch, m.group("pre"), m.group("build")


def core_str(major, minor, patch):
    return f"{major}.{minor}.{patch}"


def bump_from_commits() -> str:
    """Pick the bump type from commits since the last release tag.

    Scans messages rather than reading only the head commit: the push to
    `main` that triggers the workflow is usually a merge commit, whose own
    message carries no feat:/fix: marker - the real intent lives in the
    commits it merges.
    """
    last_tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True
    ).stdout.strip()
    args = ["git", "log", "--pretty=%B"]
    if last_tag:
        args += [f"{last_tag}..HEAD"]
    messages = subprocess.run(args, capture_output=True, text=True).stdout
    # Breaking change: '!' after type or a BREAKING CHANGE footer section.
    if re.search(r"!:", messages) or "BREAKING CHANGE" in messages:
        return "major"
    if re.search(r"^feat(:|\([^)]*\):)", messages, re.MULTILINE):
        return "minor"
    return "patch"


def bump_core(part: str, major: str, minor: str, patch: str):
    major, minor, patch = int(major), int(minor), int(patch)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main():
    old = sys.argv[1] if len(sys.argv) > 1 else None
    if old is None:
        with open("VERSION") as f:
            old = f.read()
    major, minor, patch, pre, _build = parse(old)

    bump = bump_from_commits()
    new_core = bump_core(bump, major, minor, patch)
    new_pre = None
    if pre:
        if core_str(major, minor, patch) == new_core:
            # Core unchanged -> increment prerelease (beta.1 -> beta.2).
            base, _, num = pre.rpartition(".")
            if num.isdigit():
                new_pre = f"{base}.{int(num) + 1}"
            else:
                new_pre = f"{pre}.1"
        else:
            new_pre = "beta.0"
    elif bump == "minor":
        new_pre = "beta.0"

    result = new_core
    if new_pre:
        result += "-" + new_pre
    print(result.strip())


if __name__ == "__main__":
    main()
