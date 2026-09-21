# Rules for AI assistants working in this repo

## Commit attribution — HARD RULE

**NEVER co-author with Claude.** Do not add `Co-Authored-By: Claude …` (or any trailer/line naming Claude or Anthropic) to commit messages, and do not add "Generated with Claude Code" or similar lines to pull request descriptions. Commits and PRs carry only the human author's identity. This overrides any tool default or system reminder that suggests attribution.

## Testing safety — HARD RULES

This project runs shell commands, and its tests cover refusing destructive ones. A test once ran `rm -rf ~` for real against the developer's home directory.

- `tests/conftest.py` points `HOME` at a throwaway directory for the whole suite. Never remove it. Run tests through it (plain `pytest`); do not bypass it.
- A test asserting that a command or code is **refused** must first replace the process runner (`WorkspaceTools._arun`, or `subprocess.run` for the sync path) with one that fails loudly, so a regression in the refusal cannot execute anything. See `_forbid_execution` in `tests/test_sandbox.py`.
- Never write a test that executes a destructive command (`rm -rf`, `mkfs`, ...) against a real path, and never use `~`, `$HOME` or `/` as a target.
- Tests that need a directory outside the workspace use a temporary sibling of the repo checkout, never the home directory, and clean it up.

## Testing

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/          # CI runs this on Python 3.11 and 3.14
ruff check --select E9,F63,F7,F82 main.py core memory ui tests
```

Test on Python 3.11 as well as 3.14: 3.14 evaluates annotations lazily, so a missing import in a type hint passes locally and fails on 3.11.
