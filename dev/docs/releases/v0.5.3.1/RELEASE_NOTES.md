# Roampal Core v0.5.3.1 — 2026-04-23

**Release Date:** 2026-04-23
**Type:** Hotfix. Changes:

- Fix facts extraction greedy regex that swallows markdown code fences,
  causing `JSON.parse()` to fail with `Unrecognized token '\``' on every
  exchange scored by qwen3.6 and similar models that wrap output in
  ```` ```json ```` blocks
- Fix summary/scoring path JSON extraction fragility: non-greedy regex
  stopped at the first `}` (inside nested `memory_scores` objects or
  inside string values containing `}`), causing `JSON Parse error: Expected '}'`
  on responses with nested fields
- Refactor both lanes onto a shared `extractJson()` helper that tries
  whole-text parse → greedy match → non-greedy multi-match validation,
  handling backtick fences, nested objects, and embedded brace
  characters in one place
- Bump scoring `max_tokens` from 2000 → 4000 and facts `max_tokens` from
  1000 → 2000 to give verbose reasoning models (qwen3.6, glm) headroom
  to finish JSON output without truncation

## Summary

Three bugs, all in the OpenCode plugin's JSON extraction logic. Two are
parser-fragility issues that surfaced with qwen3.6-class local models;
one is a token-budget issue that compounded the parser problems.

**Bug 1 — facts extraction backtick fences.** The facts extraction path
used a greedy regex `\{[\s\S]*\}` to find JSON in model responses,
which captures trailing backtick code fences (```` ``` ````) when
models like qwen3.6 wrap their output in markdown blocks. `JSON.parse()`
then chokes on the leading backtick → every facts extraction fails
silently with `SyntaxError: Unrecognized token '\``'`. Present since
v0.4.8 when the separate facts-extraction lane was introduced.

**Bug 2 — summary/scoring path nested-brace fragility.** The
summary/scoring path used a non-greedy single-match regex
`\{[\s\S]*?\}` against the model's content field. Non-greedy stops at
the **first** `}` it finds, which on responses containing nested
objects (e.g., `{"summary":"x", "memory_scores":{"id1":"worked"}, ...}`)
or strings containing literal `}` characters captures an incomplete
fragment. `JSON.parse()` then fails with `Expected '}'`. Latent until
qwen3.6 started returning per-memory scores in `memory_scores: {...}`
nested objects, exposing the fragility in real exchanges.

**Bug 3 — max_tokens too low for verbose reasoning models.** The
scoring call's `max_tokens=2000` was sometimes too low for qwen3.6's
verbose summaries — the model's output got truncated mid-string with
no closing brace, so no regex (greedy, non-greedy, or anything else)
could find valid JSON to parse. Bumping to 4000 gives the model
enough headroom to finish.

The root-cause refactor consolidates JSON extraction into one
`extractJson()` helper used by both lanes (scoring + facts), so any
future parser hardening lands in one place.

## Implementation

### `roampal.ts` — shared `extractJson()` helper

**File:** `roampal/plugins/opencode/roampal.ts`

```typescript
// Robust JSON extraction:
//   1. Strip markdown code fences
//   2. Try parsing the whole stripped text directly (handles nested
//      objects like memory_scores, and strings containing { or })
//   3. Fallback to greedy match (whole outer object even with nesting)
//   4. Fallback to non-greedy multi-match validated last-to-first
function extractJson(raw: string): unknown | null {
  if (!raw) return null
  const stripped = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim()
  try { return JSON.parse(stripped) } catch { /* fall through */ }
  const greedy = stripped.match(/\{[\s\S]*\}/)
  if (greedy) {
    try { return JSON.parse(greedy[0]) } catch { /* fall through */ }
  }
  const all = [...stripped.matchAll(/\{[\s\S]*?\}/g)]
  for (let i = all.length - 1; i >= 0; i--) {
    try { return JSON.parse(all[i][0]) } catch { /* try next */ }
  }
  return null
}
```

### Scoring path

```typescript
let result: any = extractJson(contentText)
if (!result && reasoningText) {
  result = extractJson(reasoningText)
  if (result) debugLog(`scoreExchange: extracted JSON from reasoning field`)
}
if (!result) {
  debugLog(`scoreExchange no JSON from ${target.label}: ...`)
  break  // bad output — try next model
}
```

Replaces the prior single-match regex on `contentText` plus a hand-rolled
multi-match-validate fallback on `reasoningText`. Both paths now go
through the same helper. The hand-rolled fallback (lines 738-752 in
v0.5.3) is removed — the new helper does the same job for both fields.

### Facts path

```typescript
const factsResult: any = extractJson(factsContent)
if (factsResult) {
  // ... existing fact-filtering and /record-outcome POST unchanged
}
```

Replaces the prior greedy match (which had the backtick-swallowing bug)
plus an inline strip step.

### Token-budget bumps

- Scoring call (line 690): `max_tokens: 2000` → `max_tokens: 4000`
- Facts call (line 960): `max_tokens: 1000` → `max_tokens: 2000`

## Impact if unfixed

- **Bug 1** — facts extraction silently fails on every exchange scored
  by fence-wrapping models. User-level facts never land in persistent
  memory for those models; the error is logged as `non-fatal` so
  invisible to the user.
- **Bug 2** — every scoring attempt against a response containing
  nested objects fails. Summary lane drops the exchange. With qwen3.6's
  per-memory scoring (which always returns `memory_scores: {...}`),
  scoring breaks entirely after a few memories accumulate.
- **Bug 3** — verbose responses from larger models get truncated. Even
  with the parser fix, no valid JSON can be extracted from a response
  that ends mid-string.

## Testing

Manual end-to-end with qwen3.6-35b-a3b on LM Studio:

- Pre-fix log evidence (v0.5.3): `FACTS error (non-fatal): SyntaxError:
  JSON Parse error: Unrecognized token '\``'` and `scoreExchange error:
  Expected '}'` on every exchange.
- Post-fix log evidence: `scoreExchange SUCCESS: <outcome> via
  sidecar(qwen3.6-35b-a3b)` + `SUMMARY stored: N chars` + `FACTS: N
  facts extracted` (when the model produces well-formed output).

The bare-array tolerance tests from v0.5.3 are unaffected — the
extraction-helper refactor preserves the same parse-then-fall-back
ordering for arrays-vs-objects.

## Sidecar model recommendation

**Prefer a non-thinking sidecar model.** Reasoning-by-default models emit
their chain-of-thought into a separate `reasoning_content` field while
the structured answer goes in `content`. With longer summaries the model
sometimes runs over its token budget mid-reason, leaving `content`
truncated or empty. The hotfix bumps `max_tokens` to 4000 for scoring +
2000 for facts to give thinking models room to finish, and the
`extractJson()` helper recovers from most partial outputs — but
model-side truncation ultimately can't be fixed in the plugin. Pick a
non-thinking model in `roampal sidecar setup` to eliminate this class
of failure entirely.

## Files touched

- `roampal/plugins/opencode/roampal.ts` — `extractJson()` helper added,
  scoring + facts paths refactored onto it, scoring/facts max_tokens
  bumped
- `pyproject.toml` — version 0.5.3 → 0.5.3.1
- `roampal/__init__.py` — `__version__` 0.5.3 → 0.5.3.1
- `dev/docs/releases/v0.5.3.1/RELEASE_NOTES.md` — this file (gitignored)

No Python runtime code changed. No tests added (TypeScript plugin has
no unit tests in the repo, consistent with existing pattern).
