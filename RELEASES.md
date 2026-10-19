# Motion Harness Releases

This file tracks the evolution of Motion Harness. We follow Semantic Versioning (SemVer).

## 🧪 Beta Releases (Pre-release)
Current Phase: **Stability & Bug Squashing**

### Unreleased: Cobalt theme, a calmer window, a turn card, a session rail, and a launchpad

First five steps of a UI redesign (design: one page per state, in the redesign canvas shared with the maintainer). Nothing here changes behaviour, keybindings or headless output.

- **Cobalt**, a new blue-ground theme and the new default for anyone who hasn't picked one. Each colour has one meaning: gold is you and Build mode (can change files), blue is the agent and Plan mode (read-only), cyan is reserved for learned things (memory, skills), orange only ever means "needs a look". A saved `default_theme` is respected, so existing choices don't change.
- Contrast was checked, not assumed: Cobalt2's own line colour is 1.07:1 against its ground and would vanish in a terminal, and its blue is 3.6:1 as text, so Cobalt uses a lighter hairline and a text tint. Those numbers are pinned by tests.
- **Status bar** instead of a clock: workspace, git branch, and the current mode with what it allows ("Build · can edit files" / "Plan · read-only").
- **Hint bar** instead of the stock footer: only the keys that matter now (idle vs. a turn running).
- **Mode colours flipped**: Build is the primary colour and Plan the secondary (it was Build blue, Plan orange, which made Plan look like a warning). This applies to every theme.
- Muted system text no longer has an extra `dim` on top of the muted colour, and the context panel is separated by a hairline instead of a blank gutter.
- Themes gained `$blue-text` and `$error-text` variables (defaulting to their own secondary/error colours), through Textual's `Theme(variables=...)`.
- **Turn card:** while a turn runs, a card lists finished operations (✓, ✗ if one failed) and what is running right now (●), instead of three dim lines. When the turn ends the process collapses to a line that says what was done ("read 3 files · edited cart.py · ran 1 command") rather than "4 steps"; it still expands on click.
- **Done strip** under the answer: what changed ("✓ Done · 1 file changed (+3 −4) · 4.9s · $0.005", or "⚠ Done with 1 failed tool call") with View diff, Undo edit, Trace and Copy answer chips. Each chip prints the key or command that does the same thing. The chips act on "the last turn", so they are removed when the next turn starts: an old strip can never undo a newer turn's edits. One undo per strip.
- A write refused in Plan mode used to leave no trace in the chat; it now shows up as "1 blocked" and is never counted as an edit.
- Deliberate deviation from the design board: the board had "↑ focuses the actions", but ↑ already recalls prompt history in the composer, so the chips are clickable and teach the existing keys (`/diff`, `/undo`, F8, F9) instead.
- **Session rail** replaces the Context panel, which mostly repeated the chat. It shows what the chat doesn't: the **plan**, **files touched** this session (+/−, newest first), **recalled** memory for the current turn with its age and raw similarity (a weak match is flagged, so a 0.41 guess no longer looks like a 0.82 match), the per-turn **budget** as bars against each limit that is set (or "no limit set · /budget" when none is), and **safety**: whether the command sandbox is confining anything. Memory recall and the sandbox state reach the UI through two small additions to existing trace events (`items` on `memory_recall_done`, `active` on `sandbox`); nothing else about them changed.
- The rail hides itself below 110 columns so it never squeezes the chat; `Ctrl+B` still shows it, and an explicit hide is not overridden when the terminal widens.
- Recall and budget use belong to a turn, so they are cleared when the next turn starts rather than showing the previous turn's as the current one. `/new` clears files touched.
- Removed with the old panel: the "Recent turns" list and the "Current turn" checklist (the turn card now shows the latter in the chat).
- **Launchpad:** an empty session opens on four starting moves (Explore this repo, Fix something, Review my changes, Pick up where I left off) and a row of live facts (memory notes in this project, skills awaiting review, history on/off, budget set or not), replacing the one-line tip. Each card says whether it is read-only or can edit files. A card sets the mode and fills the composer for you to edit; it never sends anything. Cards that can't work say why: "not a git repository", "No uncommitted changes to review", "No saved sessions yet", "History is off · /tracking on turns it on". Gone at the first message; back on `/new`.
- Deliberate deviation from the design board: the board had "1–4 start", but digits are not shortcuts. A first message such as "3 tests are failing" must reach the model as typed (covered by a test), so cards are clicked.
- Not in these steps: the guided dialogs (approval explainer, grouped palette, next-step chips) are a separate step.

### v0.16.0-beta.0: headless subcommands — [#39](https://github.com/MathiTz/motion-harness/pull/39)

The features added over the last cycle were reachable only from the TUI (`/skill`, `/resume`, `/tracking`, `/trajectory`) or as loose scripts under `scripts/`, which made them unusable from CI or another program. They are now `motion` subcommands with machine-readable output; released as v0.16.0-beta.0.

- `motion skill list|show|candidates|promote|reject|rollback|ab-test` — the full candidate → active review loop from a shell. `ab-test` returns a verdict and never promotes; promotion stays a separate, explicit call.
- `motion eval list|run|memory` — the task baseline and memory-quality scenarios, with the report on stdout under `--json`.
- `motion trajectory list|compare`, `motion sessions list|show`, `motion memory search|stats`, `motion tracking on|off|status`, `motion check` — inspection, consent and provider checks.
- `motion -p ... --resume ID` — continue a saved session headlessly. The turn is recorded only if interaction tracking is on, so it never writes a transcript you opted out of.
- Contract: `--json` puts exactly one JSON document on stdout (progress goes to stderr); exit codes are 0 ok, 1 the operation did not succeed, 2 usage error. `motion memory search` never creates a database as a side effect of searching.
- Not changed: the billed commands (`eval run`, `eval memory`, `skill ab-test`, `check`) make real provider calls exactly as the scripts they wrap always did; nothing here runs them automatically, and CI does not run them.

### v0.15.2-beta.0

This entry covers PRs [#21](https://github.com/MathiTz/motion-harness/pull/21)–[#37](https://github.com/MathiTz/motion-harness/pull/37), a single work cycle driven by a review of the codebase's own real gaps plus direct user reports, each one verified against the actual code before being acted on (not assumed). The versions between v0.1.0-beta.1 and this one were never written up here — this file had gone stale; it starts tracking accurately from here rather than reconstructing history it doesn't have firsthand knowledge of.

**Crash recovery & safety**
- A `run_command` child process used to survive the harness itself being killed uncatchably (`kill -9`, OOM, a crash) and keep running as an orphan — reproduced for real, then fixed with a small supervisor (`core/command_watchdog.py`) that wraps every spawned command. [#22](https://github.com/MathiTz/motion-harness/pull/22)
- A turn that crashed, was cancelled, or errored used to vanish from the session transcript with no trace. Every exit path now records what happened, and an interrupted turn shows up explicitly on `/resume` instead of disappearing. [#22](https://github.com/MathiTz/motion-harness/pull/22)
- `/undo` and `/resume`'s actual boundaries (what they do and don't cover) are now documented plainly rather than left to be discovered. [#22](https://github.com/MathiTz/motion-harness/pull/22)
- Command output (stdout/stderr, and large tool results generally) is now bounded head-plus-tail instead of head-only, so a long build log's error at the end doesn't get silently cut off. [#24](https://github.com/MathiTz/motion-harness/pull/24)

**Evaluation infrastructure (new)**
- A repeatable task-evaluation baseline (`evals/`, `scripts/run_evals.py`): six fixed tasks across bug-fix, feature-modification, investigation and resume-after-interruption, each scored by a hidden check the agent can't see or edit, with a held-out subset to catch overfitting. [#29](https://github.com/MathiTz/motion-harness/pull/29)
- A provider/MCP compatibility matrix (`docs/compatibility.md`) distinguishing what's been verified against a real, live service from what's only proven against a mock — plus three new live checks (cancellation, usage-reporting accuracy, real failover-error classification) and a test against an independently-maintained third-party MCP server, not just an in-house one. [#31](https://github.com/MathiTz/motion-harness/pull/31)
- `/trajectory save full` now redacts secret-shaped values by default (API keys, bearer tokens, common vendor key formats) instead of writing the raw system prompt and every message to disk unredacted; saved trajectories record the harness version; `scripts/compare_trajectories.py` diffs two runs. [#32](https://github.com/MathiTz/motion-harness/pull/32)
- A measured evaluation of memory recall and compaction quality, run for real against a live model (`docs/memory-quality.md`, `scripts/eval_memory_quality.py`) — not just mechanics tests. Found a real gap (see Memory below) and confirmed compaction correctly preserves an early constraint across a long, mostly-filler session. [#34](https://github.com/MathiTz/motion-harness/pull/34)

**Memory**
- Memory is now scoped per workspace (`<workspace>/.motion/memory.db`) instead of one database shared across every project — a fact recorded on one project could previously surface, unscoped, on an unrelated one. An explicit `memory_path` config option is the documented opt-in for the old shared-DB behavior. [#30](https://github.com/MathiTz/motion-harness/pull/30)
- A recalled memory used to be pasted into the prompt as bare text, with no signal of how confident the match was or how old it was — a weak guess read exactly like a strong match. Every recalled memory now shows its age and, for a semantic match, its raw confidence, explicitly flagged when weak. [#37](https://github.com/MathiTz/motion-harness/pull/37)
- Two memories that disagree (an old decision and the one that superseded it) used to be a near-coin-flip in ranking, with no recency signal at all — measured live, the stale one could win. Near-tied conflicts are now broken by recency. [#37](https://github.com/MathiTz/motion-harness/pull/37)

**Skills**
- A synthesized skill used to go live the moment it was written, indexed for recall immediately, with no check that the turn it came from had actually succeeded at anything. Skills now go through a real candidate → active lifecycle with provenance, version history and rollback (`/skill candidates`, `promote`, `reject`, `rollback`); `scripts/skill_ab_test.py` A/B-tests a candidate against the eval baseline before promotion. [#33](https://github.com/MathiTz/motion-harness/pull/33)

**Provider reliability**
- `ANTHROPIC_API_KEY` — the exact variable name `.env.example` and every doc told users to set — wasn't recognized by the code that gates whether a provider is even attempted; it checked `CLAUDE_API_KEY` instead. A user who followed the docs exactly would see the provider locked and be told to set a variable nothing else mentions. Fixed. [#31](https://github.com/MathiTz/motion-harness/pull/31)

**UI**
- The reply area showed "No response content." in bold italic while a tool was still running — read like an error. Replaced with a live activity line that's replaced by the real answer once it streams. [#25](https://github.com/MathiTz/motion-harness/pull/25)
- Long reasoning made scrolling stutter badly — profiled for real and traced to a layout pass re-running on every streamed token; fixed by painting live content without triggering one. The trace panel and status line now read in plain sentences instead of `name.status` event codes. [#27](https://github.com/MathiTz/motion-harness/pull/27)
- The model's thinking and the steps it took used to disappear once a turn finished. The "thought for Ns" summary line is now clickable to expand the full process back open. [#28](https://github.com/MathiTz/motion-harness/pull/28)
- A model chosen in the app used to reset to the provider's default on the next launch, because a `.env` template value was silently outranking the saved choice. Fixed; the model you pick now stays picked. [#23](https://github.com/MathiTz/motion-harness/pull/23)
- The token-usage readout summed every request of a turn into one number, making it look far larger than comparable tools reporting a single context size. Now shows context size, summed input (with the cached portion), and output separately. [#23](https://github.com/MathiTz/motion-harness/pull/23)

**Docs & process**
- README's clone URL and an unverifiable "absolute theoretical minimum" performance claim were corrected; the Caveman whitespace-compression protocol was rewritten to be genuinely reversible (it wasn't). [#21](https://github.com/MathiTz/motion-harness/pull/21)
- A lightweight process for tracking real user reports through to a maintainer- or reporter-confirmed retest — not just a shipped PR, which is evidence of effort, not evidence the problem is actually resolved for that person. [#36](https://github.com/MathiTz/motion-harness/pull/36)

### v0.1.0-beta.1
- Initial architectural release.
- Implemented Hybrid Memory (FTS5 + Vector).
- Implemented Caveman Compression.
- Implemented Parallel Orchestrator.
- Implemented Textual TUI.
- Dockerized distribution and global CLI alias.

## 🚀 Stable Releases
No stable releases yet. The first stable version (`v1.0.0`) will be released after the beta cycle is complete and integration tests pass 100% of the time.
