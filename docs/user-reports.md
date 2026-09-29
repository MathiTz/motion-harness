# User reports: report → change → PR → retest

Every row starts from a real report filed with [`.github/ISSUE_TEMPLATE/user-report.md`](../.github/ISSUE_TEMPLATE/user-report.md). A PR shipping doesn't move a row to "confirmed" — only a retest recorded by the reporter or the maintainer does. See the README's "User-reported problems" section for the norm this implements.

Add a row when you file a report; update it when a PR ships; only the reporter or maintainer updates the **Retest** column.

| Report | Task attempted | PR(s) | Shipped | Retest |
| --- | --- | --- | --- | --- |
| [#35](https://github.com/MathiTz/motion-harness/issues/35) | Using Motion for a coding task; compared token/tool-call usage against opencode for the same task | [#25](https://github.com/MathiTz/motion-harness/pull/25), [#26](https://github.com/MathiTz/motion-harness/pull/26)/[#28](https://github.com/MathiTz/motion-harness/pull/28), [#27](https://github.com/MathiTz/motion-harness/pull/27) | 2026-09-27 | ⏳ awaiting reporter retest |

This table starts with one entry, filed for real from this session's own conversation (not invented — see [#35](https://github.com/MathiTz/motion-harness/issues/35) for the reporter's own words) precisely so the fields above are proven usable, not just theoretical, per this feature's own acceptance criteria.
