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

### 4. Integration
The skill is saved to the local skill library. The next time the agent encounters a similar problem, the **Hybrid Recall** system pulls this skill into the prompt, allowing the agent to "remember" the correct procedure.

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
