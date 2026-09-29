---
name: User report
about: Something a real person hit while using Motion Harness - a task that was harder than it should have been, a bug, a confusing behavior.
title: "[report] "
labels: user-report
assignees: ""
---

<!--
This template exists to close a specific loop: report -> change -> PR -> confirmed retest.
A PR that claims to fix this is evidence of effort, not evidence the product got better for this
person - only a retest, done by the reporter or the maintainer, is that evidence. See
CONTRIBUTING.md's "User-reported problems" section for the norm this template implements.
-->

## Who reported this, and how

<!-- Name/handle if they're okay being named, or "anonymous" / "relayed by <name>" if not.
     How it reached you: direct message, in person, a support channel, etc. -->

## Task they were attempting

<!-- What were they actually trying to get done? Not the symptom - the goal. -->

## What friction they hit

<!-- The actual problem, as close to their own words as possible. A quote or a transcript excerpt
     is worth far more here than a paraphrase - paraphrasing is where a report quietly turns into
     a different, tidier problem than the one that was actually reported. -->

## What changed

<!-- Filled in once a fix exists. Link the PR(s). -->

- PR:
- Summary of the change:

---

## Retest — maintainer or reporter only

<!--
Do not fill this in as the agent that wrote the code change. Completing the code and its tests is
not evidence this report is resolved - only the person who hit the problem (or the maintainer,
confirming with them) can say whether it actually got easier. Leave this section blank until a
real retest happens; an agent marking this "done" defeats the entire point of the template.
-->

- [ ] The reporter retried the same task
- **Date retested:**
- **Same friction, less friction, or gone entirely?**
- **Did they keep using Motion afterward?**
- **Notes:**
