# v0.3.4 Release Notes

**Date:** 2026-02-08
**Platforms:** OpenCode (plugin + server-side scoring prompt)
**Fixes:** GitHub Issue #1

## Summary

Three fixes:
1. **Garbled UI fix** — scoring prompt injection uses deep cloning instead of in-place mutation
2. **Double-scoring fix** — sidecar scoring deferred to `session.idle` so it only runs if the main LLM didn't score
3. **Scoring prompt fix** — main LLM now asked to score both exchange outcome AND cached memories (was previously only memories)

## Fix 1: Garbled Text in Input Box (Issue #1)

**Problem:** `messages.transform` mutated message objects that OpenCode holds references to for UI rendering, causing scoring prompt text to appear in the input box.

**Fix:** Deep clone the user message before modifying it. The clone goes to the LLM, the original stays untouched in the UI.

```typescript
// Before (broken): mutates UI-visible object
textPart.text = prompt + "\n\n" + textPart.text

// After (fixed): clone → modify clone → replace in array
const cloned = JSON.parse(JSON.stringify(msg))
cloned.parts[textPartIndex].text = prompt + "\n\n" + cloned.parts[textPartIndex].text
output.messages[i] = cloned
```

## Fix 2: Prevent Double-Scoring Memories

**Problem:** Sidecar scoring ran in `chat.message` (before the main LLM), applying uniform scores to all surfaced memories. Then the main LLM could also call `score_response` with per-memory scores. Both scored the same memories — double-scoring.

The `mainLLMScored` flag only prevented this on the *next* cycle, not the current one.

**Fix:** Moved sidecar scoring from `chat.message` to `session.idle` (after the main LLM is done). At that point, `mainLLMScored` is definitive:

```
BEFORE (double-scoring):
  chat.message → sidecar scores N-1 (uniform)     ← FIRST
  messages.transform → scoring prompt for N-1
  main LLM → calls score_response for N-1          ← SECOND (double-score!)

AFTER (no double-scoring):
  chat.message → cache scoring data only
  messages.transform → scoring prompt for N-1
  main LLM → may call score_response for N-1
  session.idle → check mainLLMScored:
    if YES → skip sidecar (main LLM provided per-memory scores)
    if NO  → run sidecar (uniform fallback)
```

## Fix 3: Main LLM Now Scores Exchange + Memories

**Problem:** `build_scoring_prompt_simple()` hardcoded `outcome: "unknown"` and only asked the main LLM to score cached memories. The exchange outcome was left entirely to the sidecar. If the main LLM scored (setting `mainLLMScored`), the sidecar was skipped — meaning the exchange never got an outcome score.

**Fix:** Updated the scoring prompt to include the previous exchange and ask for a real outcome, matching Claude Code's prompt structure. The main LLM can now score both the exchange outcome AND per-memory scores. Sidecar remains as pure backup.

```
REQUIRED: Score the previous exchange before responding.

Previous:
- User asked: "..."
- You answered: "..."

Memories in your context last turn:
- mem_id: "content..."

Based on the user's follow-up, score the exchange:
- "worked" = user satisfied, moves on
- "failed" = user corrects you, says wrong
...

Call score_response with:
  outcome: "worked" or "failed" or "partial" or "unknown"
  memory_scores: {"mem_id": "___", ...}
```

## Files Changed

| File | Change |
|------|--------|
| `roampal/plugins/opencode/roampal.ts` | Deep clone in `messages.transform`, moved sidecar scoring from `chat.message` to `session.idle` with `mainLLMScored` gate |
| `roampal/hooks/session_manager.py` | `build_scoring_prompt_simple()` now includes previous exchange and asks for real outcome (was hardcoded `"unknown"`) |
| `roampal/__init__.py` | Version bump 0.3.3.2 -> 0.3.4 |
| `pyproject.toml` | Version bump 0.3.3.2 -> 0.3.4 |

## Testing

- 457 unit tests + 31 CLI tests pass
- Live tested: clone approach confirmed working (input box stays clean)
- Live tested: sidecar scoring confirmed working via debug log
