# v0.4.8 Release Notes

**Platforms:** OpenCode (sidecar plugin), server (shared)
**Theme:** Sidecar reliability + benchmark alignment
**Status:** Complete

**Claude Code impact:** None. Changes 1, 2, 4 are OpenCode-only (roampal.ts). Change 3 (fact tags) is in the shared server but is additive — puts tags back on facts.

---

## Overview

Mirror the benchmark's proven sidecar architecture. Four changes:

1. **Remove autoSummarize** — eliminates Ollama contention that caused 20% scoring failures
2. **Remove storeExchange** — sidecar fails = nothing stored, no transcript pollution
3. **Restore noun_tags on facts** — v0.4.7 incorrectly removed them; benchmark tags all content
4. **Split sidecar into 2 LLM calls** — match benchmark: score_exchange then extract_facts. Tags handled server-side at store time.

---

## Architecture: Benchmark vs v0.4.8

**Benchmark flow per exchange (roampal-labs/benchmark/runner.py):**
```
1. sidecar_score()         → {summary, outcome, memory_scores}     60s timeout, temp 0
2. sidecar_extract_facts() → [atomic facts]                        30s timeout, temp 0
3. strategy.store(summary) → extract_tags() on summary at store    30s timeout, temp 0
4. strategy.store(fact)    → extract_tags() on each fact at store  30s timeout, temp 0
```
Sequential. No concurrency. No background tasks. Tags extracted per-content at store time, not as a separate sidecar call.

**v0.4.8 flow per exchange (roampal.ts session.idle):**
```
1. scoreExchange()    → {summary, outcome, memory_scores}     60s timeout, temp 0
2. extractFacts()     → [atomic facts]                         30s timeout, temp 0
3. Store summary via /stop → server store_working() tags it
4. Store facts via /record-outcome → server store_working() tags each
```
Same sequence. Same timeouts. Server handles tagging via `store_working()`'s built-in `extract_tags` — no extra LLM call from the plugin needed.

**Note on contention:** API models (DeepSeek, OpenAI, etc.) handle concurrent requests natively — contention is a non-issue. For local models (Ollama, LM Studio), sequential calls with no competing background tasks (autoSummarize removed) means clean queueing with no timeouts.

---

## Changes

### 1. Remove autoSummarize
**Files:** `roampal/plugins/opencode/roampal.ts`, `roampal/server/main.py`

**Why:** autoSummarize runs in `session.idle` and hits Ollama via the server. Ollama processes one request at a time. When autoSummarize runs, scoring requests queue behind it → 30s timeout → circuit breaker trips → sidecar dead. Caused 20% scoring failure rate in v0.4.7.

**What to remove:**
- `autoSummarizeOldMemory()` function in roampal.ts
- The `autoSummarize` call in `session.idle`
- `/api/memory/auto-summarize-one` endpoint in server/main.py
- `_do_auto_summarize_one()` and `_auto_summarize_one_memory()` in server/main.py
- Background task spawn in stop_hook handler (server/main.py)
- Related tests in test_v036_changes.py (TestAutoSummarizeEndpoint)

**Why it's safe:** Sidecar produces ~200-350 char summaries. No new oversized memories. autoSummarize returned `"no_candidates"` in testing — dead code.

### 2. Remove storeExchange
**File:** `roampal/plugins/opencode/roampal.ts`

**Why:** `storeExchange()` stores raw user+assistant text to `/stop` BEFORE scoring. If sidecar fails, the raw transcript persists in the DB. The sidecar already stores its own summary via `/stop` (with conversation_id, fingerprint, outcome, noun_tags, memory_type). `storeExchange()` is redundant.

**What changed:**
- `storeExchange()` function removed (stored raw text to ChromaDB)
- Replaced with `lifecycle_only: true` call to `/stop` — registers exchange in JSONL for scoring prompts and calls `set_completed()`, but does NOT store raw text to ChromaDB
- Without `/stop` call, `set_completed()` never fires → `scoringRequired` stays false → sidecar never runs

**New behavior:** Sidecar succeeds → summary stored via sidecar's own `/stop` call. Sidecar fails → nothing in ChromaDB (JSONL still has the exchange for scoring prompts).

### 3. Restore noun_tags on facts
**File:** `roampal/server/main.py` (shared — affects both clients)

**Why:** Benchmark `entity_routed.py:248-250` extracts LLM tags for ALL content at store time. v0.4.7 removed tags from facts. Tagged facts participate in TagCascade retrieval.

**Changes:**
1. Remove the `noun_tags=None` override on fact storage — let `store_working()` handle tagging via its built-in `extract_tags` (same as benchmark: tags extracted from fact content at store time)
2. Remove v0.4.7 tag-stripping migration (server/main.py:524-548)
3. Add v0.4.8 re-tag migration: LLM extraction via sidecar `extract_tags()` for facts with `noun_tags="[]"`, regex fallback if sidecar unavailable

### 4. Split sidecar into 2 LLM calls
**File:** `roampal/plugins/opencode/roampal.ts`

**Why:** Matches benchmark architecture. Each call has a focused prompt → better quality from small models. If one fails, the other can still succeed.

**Current (v0.4.7):** One prompt asks for `{summary, outcome, noun_tags, facts, memory_scores}`.

**New (v0.4.8):** Two sequential calls:
1. `scoreExchange()` → `{summary, outcome, memory_scores}` — 60s timeout, temp 0
2. `extractFacts()` → `[fact1, fact2, ...]` — 30s timeout, temp 0

Tags are NOT extracted by the plugin. The server's `store_working()` extracts tags from each piece of content at store time — matching exactly how the benchmark's `entity_routed.py:store()` calls `extract_tags()` internally.

**Noun tags for the exchange summary** come from the server-side tagging when the summary is stored via `/stop`. No separate plugin LLM call needed.

---

## Benchmark Reference

| Function | File | Timeout | Returns |
|---|---|---|---|
| `sidecar_score()` | runner.py:221 | 60s | summary, outcome, memory_scores |
| `sidecar_extract_facts()` | runner.py:173 | 30s | list of atomic facts |
| `extract_tags()` | entity_routed.py:74 | 30s | list of noun tags (called at store time) |

All use `temperature: 0`. No retries. Exceptions caught and skipped.

---

## Expected Impact

- **Scoring reliability:** ~100% (no contention, sequential calls)
- **Scoring quality:** Better (focused prompts vs overloaded combined prompt)
- **Fact retrieval:** Improved (tags enable TagCascade routing)
- **DB cleanliness:** No raw transcripts
- **Complexity:** Reduced (~200 lines autoSummarize removed)

---

## Verification

1. No `autoSummarize` in plugin debug log
2. No `Stored exchange` in plugin debug log (only `SUMMARY stored`)
3. 2 separate LLM calls per exchange visible in debug log (score → facts)
4. Facts stored with noun_tags (check working collection metadata)
5. 10+ exchanges with 0 circuit breaker trips
6. Re-tag migration: facts with `noun_tags="[]"` now have tags
7. Claude Code: score_memories still works, facts get tags

---

## Tests

- Remove TestAutoSummarizeEndpoint tests
- Add: sidecar failure → no exchange stored
- Add: facts stored with noun_tags
- Add: 2-call sidecar flow (score → facts)
- Add: re-tag migration
- Verify Claude Code MCP tools unaffected
