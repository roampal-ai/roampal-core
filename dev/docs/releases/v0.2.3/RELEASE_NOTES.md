# roampal-core v0.2.3 Release Notes

**Release Date:** 2026-01-05
**Type:** Performance Fix - Outcome Scoring Speed

---

## Summary

Critical performance fix for outcome scoring. Reduces `score_response` latency from 10+ seconds to <500ms.

**The Problem:** Users experiencing "0 memories updated" messages due to MCP timeout while scoring operations ran too long in the background.

**The Fix:** Defer heavy KG learning to background, remove duplicate operations, increase timeout safety margin.

---

## What Changed

### 1. Deferred KG Learning (outcome_service.py)

Heavy knowledge graph operations now run in the background after returning the score result.

**Before (synchronous, blocking):**
```
score_response() → 10+ seconds
├── update score metadata (fast)
├── KG routing update #1 (slow)
├── concept extraction (slow)
├── KG routing update #2 (duplicate!)
├── problem-solution tracking (slow)
└── return result
```

**After (fast path + deferred):**
```
score_response() → <500ms
├── update score metadata (fast)
├── schedule background task
└── return result immediately

Background (async):
├── KG routing update (single call)
├── concept extraction (only on success)
└── problem-solution tracking
```

### 2. Removed Duplicate KG Routing

`outcome_service.py` was calling `update_kg_routing()` twice:
- Line 96: First call (correct)
- Line 263: Duplicate call in `_update_kg_with_outcome()` (removed)

### 3. Skip Heavy Processing on Failure

Concept extraction and pattern tracking now skip on `failed` outcomes. No point building patterns from things that didn't work.

### 4. MCP Timeout Increase

`server.py` timeout increased from 5s to 15s as safety margin. The fast path should complete in <500ms, but background tasks may still be running.

### 5. Skip Session File Scan (main.py)

**Root cause of 6.6s delay:** The `/api/record-outcome` endpoint was calling `get_most_recent_unscored_exchange()` which scans ALL session files on disk.

```
Before:
score_response() → 6.6s
├── scan ALL .jsonl files in sessions_dir  ← O(n×m) file I/O!
├── read every line in every file
├── find most recent unscored exchange
└── (actual scoring is fast)
```

**The fix:** Skip the scan entirely. The doc_ids we need are already in:
- `request.related` (provided directly by LLM)
- `_search_cache` (from previous search)

The file scan was redundant busy work.

---

## Technical Details

### outcome_service.py Changes

```python
async def record_outcome(self, doc_id, outcome, ...):
    # FAST PATH: Score update + metadata (synchronous)
    metadata = self._update_score(doc_id, outcome)

    # DEFER: Heavy KG learning runs in background
    # Fire-and-forget - don't await
    if outcome == "worked" and self.kg_service:
        asyncio.create_task(
            self._deferred_kg_learning(doc_id, metadata, context)
        )

    return metadata  # Return immediately
```

### Removed Duplicate Call

```diff
# In _update_kg_with_outcome():
- await self.kg_service.update_kg_routing(problem_text, collection_name, outcome)
# Already called at top of record_outcome()
```

### Conditional Processing

```python
# Skip expensive operations for failed outcomes
if outcome == "worked" and problem_text and solution_text:
    # Extract concepts, build relationships
    ...
elif outcome == "failed":
    # Just track failure, no pattern building
    self.kg_service.update_success_rate(doc_id, outcome)
```

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/outcome_service.py` | Defer KG learning, remove duplicate call |
| `roampal/server/main.py` | Skip session file scan, defer Action KG updates |
| `roampal/mcp/server.py` | Increase timeout 5s → 15s |
| `pyproject.toml` | Version 0.2.2 → 0.2.3 |
| `roampal/__init__.py` | Version bump |

---

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| `score_response` latency | 10-30s | <500ms |
| MCP timeout errors | Frequent | Rare |
| "0 memories updated" bug | Common | Fixed |
| Background KG learning | N/A | Async |

---

## Testing

### 1. Before/After Timing Test

```python
import time
import asyncio

async def test_score_timing():
    start = time.time()
    result = await memory.record_outcome("working_xxx", "worked")
    elapsed = time.time() - start
    print(f"record_outcome: {elapsed:.2f}s")
    return elapsed

# Before fix: 10-30s
# After fix: <0.5s
```

### 2. MCP Timeout Test

```bash
# Use Claude Code, trigger a score
# Before: "0 memories updated" (timeout)
# After: "1 memory updated" (fast enough)
```

### 3. Verify Background Learning Still Happens

```python
# Score a memory, wait 5s, check KG was updated
await memory.record_outcome("working_xxx", "worked")
await asyncio.sleep(5)  # Let background task complete

# Verify KG routing learned
kg = memory.kg_service.knowledge_graph
assert "routing_weights" in kg  # KG still updated
```

### 4. Quick Manual Smoke Test

1. Start server
2. Score something via MCP tool
3. Confirm it returns in <1s
4. Check logs show "Background learning" happened after

### 5. Unit Tests

```bash
pytest roampal/backend/modules/memory/tests/test_outcome_service.py
```

---

## Upgrade Instructions

```bash
pip install --upgrade roampal
```

No migration needed. Existing memories preserved.

---

## Design Principle

From DDIA (Designing Data-Intensive Applications):

> "Every transaction must be small and fast."

> "It takes just one slow call to make the entire end-user request slow."

The synchronous KG learning was violating both principles. This fix applies the fire-and-forget pattern for background learning while keeping the user-facing path fast.

