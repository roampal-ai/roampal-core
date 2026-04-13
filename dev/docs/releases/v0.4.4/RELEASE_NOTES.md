# v0.4.4 Release Notes

**Platforms:** Claude Code, OpenCode
**Theme:** Async parallelization — eliminate sequential bottlenecks

---

## Planned

### 1. Parallel collection searches in `_search_collections`
**File:** `roampal/backend/modules/memory/search_service.py` (lines 348-370)

**Problem:** `_search_collections()` iterates collections in a sequential `for` loop — each `await _search_single_collection()` waits for the previous one to complete. With 4-5 collections, this adds 50-150ms per search call.

**Fix:** Replace sequential loop with `asyncio.gather()`:
```python
coll_results = await asyncio.gather(*(
    self._search_single_collection(coll_name, query_embedding, processed_query, limit, metadata_filters)
    for coll_name in valid_collections
), return_exceptions=True)
```
Handle per-collection exceptions individually so one failure doesn't block the rest.

**Impact:** ~50-150ms saved per search call. Compounds with optimization #2 (3 searches × 4 collections = 12 sequential calls → 3 parallel batches of 4).

---

### 2. Parallel search calls in `get_context_for_injection`
**File:** `roampal/backend/modules/memory/unified_memory_system.py` (lines 1007-1032)

**Problem:** Three independent `self.search()` calls run sequentially:
- working_results (line 1007)
- history_results (line 1017)
- other_results (line 1028)

These searches are fully independent — each targets different collections with different limits.

**Fix:** Run all three concurrently:
```python
working_task = self.search(query=query, limit=1, collections=["working"])
history_task = self.search(query=query, limit=1, collections=["history"])
other_task = self.search(query=query, limit=4, collections=other_collections)

working_results, history_results, other_results = await asyncio.gather(
    working_task, history_task, other_task
)
```

**Impact:** ~100-200ms saved per turn. Context injection latency drops from 3× single-search time to ~1× (bounded by slowest search).

---

### 3. Parallel per-memory scoring in `record_outcome`
**File:** `roampal/server/main.py` (lines 1270-1275)

**Problem:** Per-memory scoring loop awaits each `record_outcome()` sequentially:
```python
for doc_id, mem_outcome in request.memory_scores.items():
    await _memory.record_outcome(doc_ids=[doc_id], outcome=mem_outcome)
```
Typically 3-4 memories scored per exchange.

**Fix:**
```python
outcome_tasks = []
for doc_id, mem_outcome in request.memory_scores.items():
    if mem_outcome in ["worked", "failed", "partial", "unknown"]:
        outcome_tasks.append(_memory.record_outcome(doc_ids=[doc_id], outcome=mem_outcome))
        doc_ids_scored.append(doc_id)
if outcome_tasks:
    await asyncio.gather(*outcome_tasks)
```

**Impact:** 3-4× speedup on scoring endpoint. Reduces blocking time for CLI clients calling `score_memories`.

---

### 4. Parallel Action KG background updates
**File:** `roampal/server/main.py` (lines 1132-1156)

**Problem:** `_bg_action_kg_update()` loops sequentially over doc_ids for `record_action_outcome()`, then sequentially over collections for `_update_kg_routing()`.

**Fix:**
```python
# Parallel action outcome recording
action_tasks = [_memory.record_action_outcome(action) for action in actions]
await asyncio.gather(*action_tasks)

# Parallel routing updates
routing_tasks = [
    _memory._update_kg_routing(cached_query, collection, outcome)
    for collection in collections_updated
]
await asyncio.gather(*routing_tasks)
```

**Impact:** Background task completes 2-5× faster. Less contention on ChromaDB during concurrent requests.

---

### 5. Parallel collection adapter initialization
**File:** `roampal/backend/modules/memory/unified_memory_system.py` (lines 454-461)

**Problem:** 5 ChromaDB collection adapters initialize sequentially on startup:
```python
for short_name, chroma_name in collection_mapping.items():
    adapter = ChromaDBAdapter(...)
    await adapter.initialize()  # Sequential!
```

**Fix:**
```python
adapters = []
for short_name, chroma_name in collection_mapping.items():
    adapter = ChromaDBAdapter(...)
    adapters.append((short_name, adapter))

await asyncio.gather(*(adapter.initialize() for _, adapter in adapters))
for short_name, adapter in adapters:
    self.collections[short_name] = adapter
```

**Impact:** Server startup 3-5× faster for collection init phase. One-time cost but noticeable on cold start.

---

### 6. Parallel startup cleanup
**File:** `roampal/backend/modules/memory/unified_memory_system.py` (lines 567-571)

**Problem:** Two cleanup calls run sequentially after garbage collection:
```python
await self._promotion_service.cleanup_old_working_memory(max_age_hours=24.0)
await self._promotion_service.cleanup_old_history(max_age_hours=720.0)
```

**Fix:**
```python
await asyncio.gather(
    self._promotion_service.cleanup_old_working_memory(max_age_hours=24.0),
    self._promotion_service.cleanup_old_history(max_age_hours=720.0),
)
```

**Impact:** Minor (~50-100ms on startup). Clean win since the two cleanups are independent.

---

### 7. Show full memory metadata in KNOWN CONTEXT
**File:** `roampal/backend/modules/memory/unified_memory_system.py` (`_format_context_injection`)

**Problem:** Memory injection only showed Wilson score if `wilson >= 0.7` ("proven") or effectiveness > 0. The LLM had no idea how reliable each memory was — just saw content + age + collection. The `normalize_memory()` standardization from v0.3.5 computes all the fields but the display code never caught up.

**Fix:** Always show scoring metadata when available:
- **Scored memories:** `(2h, working, wilson:72%, used:3x, last:worked)`
- **Unscored memories:** `(5m, working)` — just age + collection
- **Books:** `(3d, books, reference)` — static reference material, no scoring

Also updated KNOWN CONTEXT header to explain tag meanings inline: `wilson:N%` = reliability confidence, `used:Nx` = times retrieved, `last:worked/failed/partial/unknown` = whether memory was *helpful* last time (not whether a task succeeded). Prevents LLMs from misinterpreting "worked" as task completion.

**Impact:** LLM can make informed decisions about memory reliability. A memory with `wilson:30%, used:8x, last:failed` gets treated differently than `wilson:85%, used:5x, last:worked`.

---

## Carried Forward

### From v0.4.3
- Add negative example to `record_response` tool description

---

## Files Modified

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/search_service.py` | `_search_collections()`: sequential loop → `asyncio.gather` |
| `roampal/backend/modules/memory/unified_memory_system.py` | `get_context_for_injection()`: 3 sequential searches → `asyncio.gather`; adapter init parallelized; startup cleanup parallelized; `_format_context_injection()`: full metadata display |
| `roampal/server/main.py` | `record_outcome`: per-memory scoring parallelized; `_bg_action_kg_update`: action + routing updates parallelized |

---

## Verification

```bash
# Run tests
pytest roampal/backend/modules/memory/tests/ -v

# Verify server starts correctly
roampal start --dev && sleep 3 && roampal status --dev

# Verify search still works
curl -s http://localhost:3838/api/search -X POST -H "Content-Type: application/json" -d '{"query": "test", "limit": 3}' | python -m json.tool

# Verify scoring endpoint
curl -s http://localhost:3838/api/record-outcome -X POST -H "Content-Type: application/json" -d '{"outcome": "worked", "conversation_id": "test"}' | python -m json.tool
```
