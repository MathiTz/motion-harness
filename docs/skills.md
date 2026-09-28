# The Skills Engine: Self-Learning in Motion

Motion Harness is designed to move from "Zero-Shot" to "Experienced." The Skills Engine is the mechanism that enables this evolution.

## 🎓 What is a Skill?

In Motion Harness, a **Skill** is a crystallized procedural trajectory. Instead of the agent having to "figure out" how to perform a complex multi-step task every time, it refers to a pre-validated Skill document.

Skills are stored as `.md` files in the workspace and are automatically indexed into the Hybrid Memory.

## 🔄 The Crystallization Loop

The process of transforming a successful task into a skill happens in four stages:

### 1. Trajectory Analysis
When a task is marked as `SUCCESS`, the harness analyzes the interaction history. It identifies the core sequence of actions, the tools used, and the constraints encountered.

### 2. Procedural Extraction
The synthesis engine strips away the specific data of the task and extracts the **general procedure**. 
- *Example*: If the agent successfully debugged a race condition in `core/db.py`, the engine extracts the "Race Condition Debugging Workflow" rather than the specific fix for that one file.

### 3. Skill Formalization
The extracted procedure is formatted into a structured Markdown skill, including:
- **Trigger**: When to use this skill.
- **Procedure**: Step-by-step execution logic.
- **Verification**: How to know the skill was applied successfully.

### 4. Integration — with a review gate (issue #16)
The skill is written to disk, but **not** integrated immediately: it starts as an unevaluated **candidate**, invisible to every live turn (the system prompt's skill index, `use_skill`, and Hybrid Recall all only ever see **active** skills). See "Lifecycle: candidate → active" below for what promotes it from there. This replaced an earlier version of this feature that made a synthesized skill live the moment it was written, with no check that the turn it came from had actually succeeded at anything — `auto_skill_synthesis` also stays off by default (`/synthesize on` to opt in) regardless of this lifecycle.

## Lifecycle: candidate → active

Every synthesized skill gets a metadata sidecar (`<name>.meta.json`, next to `<name>.md`) tracking its status, version history and provenance (which trajectory produced it, when, against which model). A hand-saved skill (`/skill save`) has no sidecar and is treated as active immediately — this lifecycle is about synthesizer output specifically, not skills you wrote yourself.

- **`/skill candidates`** — list skills awaiting review, with their provenance.
- **`/skill promote <name>`** — make a candidate active: from this point it's indexed for recall and shown in listings.
- **`/skill reject <name>`** — remove it from recall (de-indexing an already-promoted one too) without deleting the file; it stays on disk, in its own history, for audit.
- **`/skill rollback <name>`** — restore the previous version of a skill (content + metadata), de-indexing whatever was currently promoted. Regenerating a skill with the same name never silently loses the prior version: it's pushed onto that skill's own history first.

**Promotion criterion.** A skill's provenance carries a `verified` flag - `false` until something has actually checked that using the skill doesn't hurt task outcomes. `/synthesize`'s own success signal is a real one (the turn used a tool and produced an answer — not the unconditional `True` this feature used to hardcode), but that is not the same as verifying the skill *helps*. The honest way to check that is `scripts/skill_ab_test.py`:

```bash
python scripts/skill_ab_test.py --skill skills/my_candidate.md
python scripts/skill_ab_test.py --skill skills/my_candidate.md --tasks bugfix-off-by-one,feature-cart-validation --repeat 3
```

It runs the [eval-baseline task set](evals.md) twice per task — once with the candidate installed, once without — on the same provider, and reports whether enabling it regresses any outcome. Promotion is still a human decision (`/skill promote`) made by reading that report; nothing here auto-promotes. This needs the eval-baseline's task set to exist (it does, as of issue #13) and makes real, billed provider calls, so it's a maintainer/local check, not something run in CI.

## 🛠️ Manual Skill Creation

Users can also manually create skills to "teach" the agent specific preferences or complex project-specific workflows.

**Skill Template:**
```markdown
# Skill: [Skill Name]
**Trigger**: [When this skill should be activated]
**Context**: [Required environment/files]
**Procedure**:
1. [Step 1]
2. [Step 2]
...
**Verification**: [Expected outcome]
```

Adding a file following this template to your workspace will immediately expand the agent's capabilities.
## Where skills live and how the agent uses them

Skills are markdown files, looked up in this order (first match wins):

1. `<workspace>/.motion/skills/` — project skills (`/skill save <name>` writes here)
2. `<workspace>/skills/` — legacy location, still read
3. `<harness>/skills/` — auto-synthesized skills shared across projects

The agent sees an index of available skills (name + first line) in its system prompt and loads one on demand with the `use_skill` tool. Manage them with `/skill list`, `/skill show <name>`, `/skill save <name>` (saves the last reply) and `/skill delete <name>`.
