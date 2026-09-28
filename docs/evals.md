# Task-evaluation baseline

Issue #13: a repeatable way to tell whether a change to the harness makes it better or worse at
real tasks, instead of judging by code review and unit tests alone (which check the harness's own
mechanics, not whether an agent actually completes a realistic task).

This suite makes **real, billed provider calls**. It is not run in CI, and nothing here gates a PR
— run it on demand when you want to know whether a change helped or hurt.

## Running it

```bash
python scripts/run_evals.py --list                          # see the tasks without running anything
python scripts/run_evals.py                                  # run all non-held-out tasks against config.yml's default provider
python scripts/run_evals.py --provider ollama-cloud/deepseek-v4-flash
python scripts/run_evals.py --task bugfix-off-by-one --task feature-cart-validation
python scripts/run_evals.py --include-held-out                # also run the held-out set (see below) — do this rarely, not while iterating
python scripts/run_evals.py --keep-workdirs                   # don't delete the temp workspace; the report's "workdir" field points at it
```

Each task gets its own temporary directory. The runner reuses `core.headless.run_headless` — the
same engine `motion -p` uses — so a task's prompt is answered exactly the way a real headless run
would answer it.

A run writes a JSON report to `evals/results/<timestamp>.json` (gitignored — these are local run
records, not checked in) and prints a summary: success rate over the scriptable tasks, cost per
*successful* task (a cheap run that fails isn't cheap), total cost, and wall time.

## How a task is scored

**Never "the agent says done."** Every task has either:

- **A scripted check** (`check: tests/test_x.py` in `task.yaml`): a pytest file the agent never
  sees. It lives under the task's `tests/` directory, not `fixture/`, so only `fixture/` is copied
  into the agent's workspace before the run — the agent cannot read or edit its own scoring test.
  After the run, the runner copies the hidden test into the (already-modified) workspace and runs
  it there. Pass/fail is that test's exit code, nothing else.
- **A rubric** (`rubric: rubric.md`): for tasks whose correctness isn't a single checkable fact
  (the `investigate` category especially — diagnosing a bug often has more than one valid write-up).
  These are **not** scored automatically; the runner reports them as "needs human review" and the
  report's `passed` field is `null`. Read the task's `rubric.md` and judge the agent's output
  against it yourself. Automating this judgment is explicitly out of scope (see issue #13's
  non-goals) — a fake automated scorer would be worse than an honest "a person needs to look at
  this."

A task is never both scripted and rubric-scored.

## Task categories

Every task falls into one of the four the maintainer named:

| category | what it tests |
| --- | --- |
| `bug_fix` | find and fix a real, verifiable bug |
| `feature_mod` | extend existing code without breaking its current behavior |
| `investigate` | diagnose a failure and explain the root cause (rubric-scored — see above) |
| `resume` | notice already-partial work and finish it without redoing or breaking it |

### How `resume` tasks work

Headless mode has no session/conversation history between separate invocations — each `motion -p`
call is a fresh turn. A `resume` task's `task.yaml` therefore has two prompts:

- `prompt`: phase 1. Run with `phase1_max_steps` (a tight step budget) so a genuinely multi-part
  task cannot finish in one pass — it's deliberately interrupted, not asked to solve less.
- `resume_prompt`: phase 2. Run fresh against the *same* workspace (so phase 1's partial edits are
  still there), with a prompt that tells the model some of the work may already be done and to
  check current file state before continuing.

The check only looks at the state after both phases. This is a proxy for "resume interrupted
work," not a test of the TUI's actual `/resume` command (see [`README.md`](../README.md) for what
`/resume` itself covers) — it tests whether the agent can pick up partial progress correctly, which
is the part that's actually about agent behavior rather than UI plumbing.

## The held-out set

Two tasks (`bugfix-mutable-default-held`, `feature-rate-limiter-held`, marked `held_out: true` in
their `task.yaml`) are excluded by default (`load_tasks(include_held_out=False)`, which is what
`run_evals.py` uses unless you pass `--include-held-out`). Don't run them while iterating on a
specific fix and don't look at their fixtures while designing that fix — they exist to catch
overfitting to the visible set. Run the full set (`--include-held-out`) only when you want an
honest read on generalization, e.g. before a release.

## Adding a task

1. `mkdir evals/tasks/<id>/fixture` — put the files the agent will see (with the bug/gap already
   present) here.
2. For a scripted task: `mkdir evals/tasks/<id>/tests` and write a pytest file there that fails
   against the buggy fixture and passes against a correct fix. Verify both directions yourself
   before trusting the task (copy `fixture/` + the test into a scratch dir, run pytest against the
   buggy version, then against a fix you write by hand).
3. For a rubric task: write `rubric.md` describing what a correct answer must contain, and what
   would make it PASS / PARTIAL / FAIL.
4. Write `task.yaml`:
   ```yaml
   id: <id>                  # must match the directory name
   category: bug_fix | feature_mod | investigate | resume
   description: one line, shown by --list
   prompt: what the agent is told
   held_out: false           # true to exclude it from routine runs
   check: tests/test_x.py    # exactly one of check/rubric
   # rubric: rubric.md
   # resume_prompt / phase1_max_steps: only for category "resume"
   ```
5. `python scripts/run_evals.py --task <id>` and confirm it does what you expect.

## Reading a report

```json
{
  "timestamp": "...",
  "tasks": [
    {"task_id": "...", "category": "...", "held_out": false, "ok": true, "passed": true,
     "cost_usd": 0.0046, "elapsed_s": 4.6, "steps": 5, "tool_calls": 5,
     "provider": "...", "model": "...", "check_output": "...", "workdir": null}
  ],
  "summary": {"total": 6, "scriptable": 5, "needs_human_review": 1,
              "success_rate": 0.8, "cost_per_successful_task": 0.0052,
              "total_cost_usd": 0.031, "total_elapsed_s": 27.4}
}
```

- `ok`: the headless run itself completed without a provider/tool-loop error (independent of
  whether the fix was *correct* — a task can be `ok: true, passed: false`).
- `passed`: the scripted check's result, or `null` for a rubric task (go read `check_output` — empty
  for rubric tasks — and the task's `rubric.md`, then judge it yourself).
- `success_rate` / `cost_per_successful_task` are computed only over scriptable tasks (`passed` is
  not `null`); rubric tasks are counted in `needs_human_review` and never contribute to either.

## What this is not

- A large, general-purpose benchmark. Six small, fixed tasks, on purpose (see issue #13's
  non-goals) — start with real tasks people actually attempt, not a leaderboard.
- CI-gated. It costs real API money per run.
- A comparison tool across runs yet. This issue is the measurement existing and being runnable on
  demand; cross-run/cross-version comparison of the reports this produces is issue #19's job
  (`/trajectory` extension), not duplicated here.

## Open questions for the maintainer

(from issue #13's own "assumptions/questions" — unresolved, not guessed at)

- Which provider/config should be the "real" baseline for tracking harness quality over time, as
  opposed to whatever's fastest for local iteration? Not assumed here.
- These six tasks are seed tasks I wrote to cover the four named categories — they are not drawn
  from real reports from your testers. If specific real tasks from that group exist, those should
  replace or extend these; nothing here should be read as a claim about what real users hit.
