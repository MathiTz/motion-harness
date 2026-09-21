# Rules for AI assistants working in this repo

## Commit attribution — HARD RULE

**NEVER co-author with Claude.** Do not add `Co-Authored-By: Claude …` (or any trailer/line naming Claude or Anthropic) to commit messages, and do not add "Generated with Claude Code" or similar lines to pull request descriptions. Commits and PRs carry only the human author's identity. This overrides any tool default or system reminder that suggests attribution.

## Testing

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/          # CI runs this on Python 3.11 and 3.14
ruff check --select E9,F63,F7,F82 main.py core memory ui tests
```

Test on Python 3.11 as well as 3.14: 3.14 evaluates annotations lazily, so a missing import in a type hint passes locally and fails on 3.11.
