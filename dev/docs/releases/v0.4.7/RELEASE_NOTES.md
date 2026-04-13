# v0.4.7 Release Notes

**Platforms:** Claude Code (MCP tools), OpenCode (sidecar plugin)
**Theme:** Compaction recovery, sidecar resilience, recency metadata, unified memory formatting
**Status:** Complete

---

## Overview

Four areas fixed:

1. **Compaction recovery timing** — `session.idle` could clear the recovery flag before the user's next message. Fixed with a **generation counter**.

2. **Sidecar resilience** — `autoSummarize` competed with scoring for Ollama's single inference slot, causing timeouts. Circuit breaker lockout was 30 minutes. `scoringBroken` flag was a death spiral. All fixed.

3. **Recency metadata missing** — `_search_all()` never called `_add_recency_metadata()`, and that function only checked `timestamp` (not `created_at`). Both fixed.

4. **Unified metadata format** — recent exchanges (cold start + compaction) used bare-bones format. Now matches `_format_mem()` with full `[id:]`, collection, wilson, uses, last outcome.

---

## Changes

### 1. Compaction recovery: generation counter (`compactionGen`)
**File:** `roampal/plugins/opencode/roampal.ts`

`session.compacted` increments `compactionGen` and stores the recovery flag with the current generation. `session.idle` only clears flags with `gen < compactionGen` (strictly less — current-gen flags survive until consumed by `chat.message` pre-fetch).

### 2. Cold start: `session.created` sets recovery flag
**File:** `roampal/plugins/opencode/roampal.ts`

First message in a new session pre-fetches the 4 most recent exchange summaries for continuity.

### 3. `session.compacted` handler restored
**File:** `roampal/plugins/opencode/roampal.ts`

`session.compacted` IS a real OpenCode event (confirmed in logs). Sets recovery flag with gen counter after compaction completes.

### 4. Sidecar resilience: autoSummarize gated
**File:** `roampal/plugins/opencode/roampal.ts`

**Root cause:** `autoSummarize` calls Ollama via the server. `scoreExchange` also calls Ollama. Ollama processes one request at a time. When both run in `session.idle`, autoSummarize queues first, scoring waits behind it, 30s timeout expires → circuit breaker trips → sidecar dead.

**Fix:** `autoSummarize` only runs when `!scoringBroken && !pendingScoring` — scoring always gets priority.

### 5. Circuit breaker cooldown: 30min → 2min
**File:** `roampal/plugins/opencode/roampal.ts`

`CIRCUIT_BREAKER_COOLDOWN_MS` changed from 1,800,000 (30 min) to 120,000 (2 min). One timeout no longer kills the sidecar for the entire session.

### 6. `scoringBroken` auto-reset
**File:** `roampal/plugins/opencode/roampal.ts`

**Root cause:** Once `scoringBroken=true`, it never reset because scoring had to succeed to flip it back — but it couldn't succeed while broken. Death spiral.

**Fix:** `scoringBroken` auto-resets to `false` at the start of each exchange's scoring attempt. Every exchange gets a fresh chance.

### 7. Collection scope expanded
**Files:** `roampal/plugins/opencode/roampal.ts`, `roampal/cli.py`

`collections: ["working"]` → `["working", "history", "patterns"]` for recent exchanges search. Promoted summaries remain relevant.

### 8. `_search_all()` missing recency metadata
**File:** `roampal/backend/modules/memory/search_service.py`

Added `self._add_recency_metadata(paginated_results)` to `_search_all()`.

### 9. `_add_recency_metadata()` timestamp fallback
**File:** `roampal/backend/modules/memory/search_service.py`

Now checks `metadata.get("timestamp") or metadata.get("created_at")`.

### 10. Unified metadata format for recent exchanges
**Files:** `roampal/plugins/opencode/roampal.ts`, `roampal/cli.py`

Recent exchanges now use the same `_format_mem()` format as normal context: `• content [id:doc_id] (age, collection, wilson:N%, used:Nx, last:outcome)`. Applied to cold start pre-fetch, compaction injection, and CLI `cmd_context`.

### 11. Remove noun_tags from facts
**File:** `roampal/server/main.py`

Facts no longer get regex-extracted noun_tags when stored. Facts are retrieved via cosine similarity, not tag-routing — tags on facts were noise.

### 12. Migration: strip noun_tags from existing facts
**File:** `roampal/server/main.py`

Server startup migration scans all facts in working/history/patterns and clears their `noun_tags` field. One-time cleanup for v0.4.5-v0.4.6.1 data.

### 13. autoSummarize: tags + threshold bump
**File:** `roampal/server/main.py`

`_do_auto_summarize_one()` now calls `extract_tags()` on the summary after summarizing (no fact extraction — sidecar handles facts at exchange time). Candidate threshold bumped from 400 to 500 chars. Tag extraction failure is non-fatal — summarization still succeeds. autoSummarize is cleanup for oversized memories the sidecar missed.

### 14. Debug logging: full recent exchanges content dump
**File:** `roampal/plugins/opencode/roampal.ts`

Pre-fetch debug log now dumps the full `RECENT EXCHANGES` content for verification.

---

## Data Structures

```typescript
let compactionGen = 0
const includeRecentOnNextTurn = new Map<string, { flag: boolean; gen: number }>()
```

- `compactionGen`: incremented by `session.compacted`
- `session.created` sets flag with current gen (cold start)
- `session.idle` clears only if `recEntry.gen < compactionGen`
- `scoringBroken` auto-resets each exchange

---

## Verification

1. **Cold start:** New session → first message includes RECENT EXCHANGES with full metadata
2. **Compaction:** Compact → next message includes RECENT EXCHANGES with proper timestamps
3. **Sidecar:** Scoring succeeds, summaries stored, no circuit breaker trips
4. **Recency:** All results show "just now" / "N minutes ago" — no jumps
5. **Metadata:** Recent exchanges show `[id:...]`, collection, wilson, used, last
6. **Fact tags:** New facts stored without noun_tags; existing facts stripped on restart

---

## Tests

509 passed (roampal/backend/modules/memory/tests/)

New tests added:
- `test_auto_summarize_extracts_tags` — verifies tag extraction during auto-summarize (no facts)
- `test_auto_summarize_tags_extraction_failure_nonfatal` — verifies extraction failure doesn't break summarization
- `test_add_recency_metadata_uses_created_at_fallback` — verifies created_at fallback when timestamp missing
- `test_add_recency_metadata_prefers_timestamp_over_created_at` — verifies timestamp takes precedence
