# Motion Harness Releases

This file tracks the evolution of Motion Harness. We follow Semantic Versioning (SemVer).

## 🧪 Beta Releases (Pre-release)
Current Phase: **Stability & Bug Squashing**

### v0.15.2-beta.0 (Current)

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
