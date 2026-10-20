# Token usage, caching and cost (issue #35)

Issue [#35](https://github.com/MathiTz/motion-harness/issues/35) reported "10x more tokens (220k vs 20k) and 144 vs 6 tool calls compared with opencode". The earlier PRs fixed the *visibility* complaints; the usage comparison itself was never measured. This page is that measurement, what it found, what changed because of it, and what it did **not** settle.

## What was measured

One task, one model, run through both tools on 2026-10-19, in a throwaway copy of this repo (read-only/plan mode, nothing modified):

> Where and how is the per-turn budget (max steps, tokens, cost, seconds) enforced? Name the files and functions involved and explain the flow from config to the point where a turn is stopped. Do not modify anything.

Model: `ollama-cloud/deepseek-v4.1-flash` for both. opencode v2.0.18 (`opencode run --format json --agent plan`); ours `motion -p --plan --output-format json`. Cached-token counts for ours come from the provider's own usage reports (`prompt_tokens_details.cached_tokens`), the same field opencode reads.

| | opencode (1 run) | Motion before (3 runs) | Motion after (3 runs) |
| --- | --- | --- | --- |
| Model steps | 29 | 28 / 24 / 26 | 18 / 10 / 16 |
| Tool calls | 41 (36 of them `shell`) | ~100 in the one run counted | not counted per run |
| Prompt tokens, all requests summed | 1.24M | 405k / 421k / 465k | 341k / 142k / 276k |
| of which served from the provider's cache | 96% | 75% / 77% / 78% | 87% / 72% / 83% |
| **Fresh (full-price) prompt tokens** | **54k** | **101k / 98k / 103k** | **46k / 39k / 48k** |
| Fresh tokens per step | ~1.9k | 3.6k / 4.1k / 4.0k (avg 3.9k) | 2.5k / 3.9k / 3.0k (avg 3.2k) |
| Context in one request at the end | ~60k | ~31k | not recorded |
| Wall time | 75 s | 63 s (the one run timed) | not timed |
| Cost | $0.016 (opencode's own figure) | $0.088 (our estimate, see below) | |

## What it says

1. **The headline "10x" was largely a metric, not a 10x difference in work.** opencode also sends over a million prompt tokens on this task; 96% of them are cache hits, so its *fresh* total is 54k. A "tokens used" figure means different things in each tool: summed over every request, context size, or fresh tokens only. We showed the first and called it "tokens"; the TUI's readout was already split into context / summed / cached in #23.
2. **Our cost estimate overstated the cost.** It priced every prompt token at the full input price, cached or not: $0.088 for a run that opencode priced at $0.016. Cached tokens are now tracked and priced separately when the model has a `cached_mtok` price. The catalog has none, because no published cached price for these models was available to me and a guessed discount would be worse than an upper bound; set `cached_mtok` in `config.yml` for your model to get an accurate figure.
3. **Our caching was weaker than opencode's (77% vs 96%), and part of that was ours to fix.** The loop trimmed old tool output on *every* step, which rewrote messages the previous request had just sent in full and so invalidated the provider's cached prefix from that point. For a provider that reports cached tokens it now waits until the conversation is large (about 20k estimated tokens, never more than a third of the window), trims in one batch, and then leaves the prefix alone. A provider that reports no caching is trimmed every step, as before.
4. **The fix is real but modest.** Across three runs each, the cache hit rate went from 77% to 81% on average and fresh tokens per step from about 3.9k to 3.2k (about -18%). That is n=3 on a noisy model: the step counts also differ between the two groups (the model took a different path), which is why the table normalises per step. Treat it as "moves in the right direction", not a measured constant.
5. **Not settled: we still pay about 1.7x opencode's fresh tokens per step.** The likely reasons are that we issue more tool calls per step (about 3.4 against 1.4) and return larger results; I have not isolated which. The 41 vs 100+ tool-call gap is real. opencode also has a code-execution tool (`execute`) that lets the model do many operations in one call; Motion has no equivalent, and building one is a separate, larger piece of work that this page does not claim to have done.

## What changed

- Per-step `cached_tokens` is recorded in trajectories, summed in their totals, and reported in the headless JSON (`usage.cached_tokens`). `/trajectory` now says what share of prompt tokens were cached, or that none were reported over a long turn.
- Cache-aware trimming policy (item 3), remembered across turns once a provider is seen to cache.
- `estimate_cost` / `turn_cost` take cached tokens and an optional per-model `cached_mtok`.

## Reproducing it

```bash
# ours, with the cached share per step (needs a provider key; costs a few cents)
motion -p "<the question>" --plan --output-format json --workspace <copy of the repo>
motion trajectory compare before.json after.json

# opencode, same model
opencode run --format json -m ollama-cloud/deepseek-v4.1-flash --agent plan "<the question>"
```

opencode's per-step `tokens.input` / `tokens.cache.read` are in the `step_finish` events of its JSON stream.

## Caveats

One task, one model, one machine, three runs per arm, and an LLM whose step count varies by a factor of two between runs. The opencode figures are a single run. Nothing here says Motion is cheaper or dearer than opencode in general. The reporter has not retested; per [docs/user-reports.md](user-reports.md) that is the only thing that moves #35 to "confirmed".
