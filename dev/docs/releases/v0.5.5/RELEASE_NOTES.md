# Roampal Core v0.5.5 — 2026-04-27

**Release Date:** 2026-04-27
**Type:** Bugfix release. Changes:

- Fix issue #8: new memories stop being generated after GUI deletion of memory_bank entries
- Replace ChromaDB hard delete with soft delete (status=archived) for memory_bank — HNSW index doesn't support true deletion, so "deleted" entries remain in the vector graph and match as duplicates
- Add `status != "archived"` filter to `_find_duplicate_fact()` dedup queries
- Fix `_get_count()` capacity check to exclude archived entries from the 500-item limit
- Add `include_archived` flag to `/api/search` endpoint — allows searching archived entries when needed (default: excluded)
- Replace hard-skip mutex with async request queue in OpenCode plugin, eliminating dropped scoring requests that cause consecutive failures and unscored exchanges (~50% of exchanges were being dropped with slow localhost models like qwen3:1.7b)
- Fix sidecar summary contamination: wrap exchange and memory block in distinct delimiters in the OpenCode `score_exchange` prompt so small scoring models (qwen3:1.7b et al.) stop laundering injected memory text into stored summaries

## Summary

One bug: after deleting memories via the GUI, no new memories are generated until `_completion_state.json` is manually deleted. The root cause is that ChromaDB's `collection.delete(ids)` doesn't actually remove vectors from the HNSW index — it marks them as deleted but they remain queryable. Combined with dedup having no status filter for memory_bank, this creates a situation where "deleted" entries still match during duplicate detection and block new fact storage.

**Symptom:** User deletes one or more memory_bank entries through the OpenCode Desktop GUI. Subsequent turns score exchanges normally, but no new facts appear in Recent/All Types views. Deleting `~/.roampal/data/mcp_sessions/_completion_state.json` restores memory generation temporarily.

### Root cause: ChromaDB hard delete doesn't actually delete from HNSW

The current flow for GUI deletion is:

1. GUI calls `/api/memory-bank/archive` with the memory content
2. `delete_memory_bank(content)` → `_memory_bank_service.archive(content)`
3. Archive searches for the doc_id via semantic match, then calls `self.delete(doc_id)`
4. `delete()` does `collection.delete(ids=[doc_id])` — ChromaDB marks it deleted but **the vector stays in the HNSW graph**

This is a known limitation of ChromaDB's HNSW index (documented since v0.4.1.2 as "phantom entries"). The existing phantom filter at `chromadb_adapter.py:267-269` only catches cases where BOTH document AND metadata are `None`. Deleted memory_bank entries still have valid metadata, so they pass through the filter and appear in query results.

### How this breaks new memory storage

When a new fact is stored via `store_memory_bank()`, `_find_duplicate_fact()` runs first (`unified_memory_system.py:932`). It queries ALL memory_bank documents with no status filter (`FACT_DEDUP_FILTERS["memory_bank"] = None`). If the new fact semantically matches a "deleted" entry that's still in the HNSW index, dedup returns the old doc_id and skips creating a new document. The API returns success, but nothing was actually stored — the returned ID points to the deleted entry.

### Secondary issue: capacity count includes deleted entries

`_get_count()` at `memory_bank_service.py:450` calls `self.collection.collection.count()`, which counts ALL ChromaDB documents including "deleted" ones. After enough deletions, the 500-item limit is reached and all new stores fail with `ValueError`.

## Impact if unfixed

- **Silent memory loss.** Every new fact that semantically matches a deleted entry is deduped away. No error shown — API returns success with a doc_id pointing to the old deleted document.
- **Capacity exhaustion.** Deleted entries count toward the 500-item limit, blocking all new writes after enough deletions.
- **No self-recovery.** Deleting `_completion_state.json` resets scoring state temporarily, but doesn't fix the underlying dedup problem — eventually the same deleted entries match again.

## Implementation

### `memory_bank_service.py` — soft delete replaces hard delete

**File:** `roampal/backend/modules/memory/memory_bank_service.py`

The fundamental change: `archive()` no longer calls ChromaDB's `delete()`. Instead it sets `metadata["status"] = "archived"` on the existing document. This is reliable, reversible, and works with HNSW's limitations.

```python
async def archive(
    self,
    content: str,
    reason: str = "llm_decision"
) -> bool:
    """
    Soft-delete memory by setting status=archived in metadata.

    v0.5.5: Replaced ChromaDB hard delete with soft delete. HNSW index doesn't
    support true deletion — collection.delete() marks entries as deleted but they
    remain queryable in the vector graph, causing phantom matches during dedup.
    Soft delete is reliable, reversible, and consistent with how working/history/
    patterns collections handle archived memories.

    Args:
        content: Content to find and archive (semantic match)
        reason: Why archiving

    Returns:
        Success status
    """
    # Search for the memory by content to find its doc_id
    if self.search_fn:
        results = await self.search_fn(
            query=content,
            collections=["memory_bank"],
            limit=5
        )

        # Find best match - look for exact or close content match among ACTIVE entries only
        doc_id = None
        for r in results:
            meta = r.get("metadata", {})
            # v0.5.5: Skip already-archived entries when searching for target
            if meta.get("status") == "archived":
                continue
            r_content = r.get("content") or meta.get("content", "")
            if content in r_content or r_content in content:
                doc_id = r.get("id")
                break

        if not doc_id and results:
            # Fall back to top active result
            for r in results:
                if r.get("metadata", {}).get("status") != "archived":
                    doc_id = r.get("id")
                    break
    else:
        doc_id = content

    if not doc_id:
        logger.warning(f"Could not find memory to archive: {content[:50]}...")
        return False

    # v0.5.5: Soft delete — update metadata instead of ChromaDB hard delete
    doc = self.collection.get_fragment(doc_id)
    if not doc:
        logger.warning(f"Memory {doc_id} not found for archiving")
        return False

    metadata = doc.get("metadata", {})
    metadata["status"] = "archived"
    metadata["archived_at"] = datetime.now().isoformat()
    metadata["archive_reason"] = reason

    self.collection.update_fragment_metadata(doc_id, metadata)
    logger.info(f"Soft-deleted memory_bank item {doc_id}: {reason}")
    return True
```

The `delete()` method (hard delete) remains for permanent cleanup operations but is no longer called by the archive flow. It's kept for `cleanup_archived()` which runs during maintenance.

### `unified_memory_system.py` — add status filter to memory_bank dedup

**File:** `roampal/backend/modules/memory/unified_memory_system.py`

```python
# v0.5.3: Pre-store fact dedup constants (ported from Desktop v0.3.2)
self.FACT_DEDUP_DISTANCE_THRESHOLD = 0.32
self.FACT_DEDUP_TIERS = ("working", "history", "patterns", "memory_bank")

# Per-tier filter. working/history/patterns hold mixed memory types
# (exchanges, summaries, facts) so scan only facts. memory_bank is
# fact-shaped by construction and doesn't tag rows with memory_type,
# so scan unfiltered — EXCEPT for archived status.
self.FACT_DEDUP_FILTERS = {
    "working": {"memory_type": "fact"},
    "history": {"memory_type": "fact"},
    "patterns": {"memory_type": "fact"},
    # v0.5.5: CRITICAL FIX — filter out archived entries during dedup.
    # Without this, _find_duplicate_fact() matches against soft-deleted memories
    # and silently blocks new fact storage. Root cause of issue #8.
    "memory_bank": {"status": {"$ne": "archived"}},  # was None
}
```

### `memory_bank_service.py` — exclude archived from capacity count

**File:** `roampal/backend/modules/memory/memory_bank_service.py`

Batch-fetch all metadatas in one ChromaDB call, then filter in Python. Avoids O(n) round trips per ID.

```python
def _get_count(self) -> int:
    """Get current active item count (excludes archived entries)."""
    # v0.5.5: collection.count() includes ALL documents including archived ones,
    # which inflates the capacity check and blocks new writes after deletion.
    # Batch-fetch all metadatas in one call, then filter in Python.
    try:
        result = self.collection.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        return sum(
            1 for m in metadatas
            if m and m.get("status", "active") != "archived"
        )
    except Exception as e:
        logger.warning(f"Could not get memory_bank active count: {e}")
        return 0
```

### `unified_memory_system.py` — startup phantom migration

**File:** `roampal/backend/modules/memory/unified_memory_system.py`

Added to `_startup_cleanup()` after the existing score-based cleanup (line 679). This is a one-time migration pass that runs every startup but only has work on first run post-fix.

```python
# v0.5.5: memory_bank phantom migration — remove IDs from pre-fix hard deletes.
# ChromaDB's collection.delete() marks entries as deleted but leaves the ID in
# list_all_ids(). get_fragment() returns None for these phantoms because both
# document and metadata are gone. They're already broken (no content, no vector),
# so it's safe to remove them from the index permanently.
if self._memory_bank_service:
    try:
        all_ids = self._memory_bank_service.collection.list_all_ids()
        phantom_ids = []
        for doc_id in all_ids:
            if not self._memory_bank_service.collection.get_fragment(doc_id):
                phantom_ids.append(doc_id)

        if phantom_ids:
            self._memory_bank_service.collection.delete_vectors(phantom_ids)
            logger.info(f"v0.5.5 migration: removed {len(phantom_ids)} phantom entries from memory_bank")
    except Exception as e:
        logger.warning(f"Startup cleanup error for memory_bank phantoms: {e}")
```

**Why not also clean archived entries at startup?** Archived entries (`status=archived`) are intentionally soft-deleted — they're reversible via `restore()`. Cleaning them at startup would make soft delete effectively permanent, removing the distinction between LLM archive (reversible) and user hard delete (permanent). Instead, `cleanup_archived()` remains available for manual invocation or capacity-pressure triggers in `_get_count()` when approaching the 500-item limit.

### `search_service.py` — verify memory_bank status filter coverage

**File:** `roampal/backend/modules/memory/search_service.py`

The tag cascade path already filters archived for memory_bank at line 361 (`status_filter = {"status": {"$ne": "archived"}}`). The cosine fill path at line 532-540 also adds the filter. Verify these paths cover all query routes through SearchService. No code change needed if coverage is confirmed — just document it.

## Testing

Manual end-to-end reproduction of issue #8:

1. Add 5+ facts to memory_bank via GUI or CLI (`roampal add "fact text"`)
2. Delete 3+ of them via the OpenCode Desktop GUI (triggers `archive()`)
3. Attempt to add new facts with similar content to deleted ones
4. **Pre-fix:** New facts silently fail — `_find_duplicate_fact()` matches deleted entry in HNSW, returns old doc_id, no new document created
5. **Post-fix:** New facts store correctly — soft-deleted entries have `status=archived`, dedup filter excludes them, fresh documents are created

Automated verification:
- Unit test: `_find_duplicate_fact()` with mixed active/archived entries in memory_bank returns `None` when only archived matches exist
- Unit test: `archive()` sets `status=archived` instead of calling ChromaDB delete
- Unit test: `_get_count()` returns correct count after archiving entries
- Integration test: full cycle of add → archive → add similar fact succeeds
- Startup migration: verify phantom IDs (in `list_all_ids()` but not in `get_fragment()`) are removed on first post-fix startup

## Scoring mutex → async queue

### Symptom

With localhost sidecar models like qwen3:1.7b taking ~28s per scoring call, any exchange that completes within that window is dropped by a hard-skip mutex (`scoringInFlight`). Plugin debug log shows `scoreExchange SKIP — already in flight` → deferred retry also fails → payload dropped entirely. Result: ~50% of exchanges unscored during active sessions, consecutive failures cascade into persistent "failed" state where the system prompt loses Roampal memory context entirely.

### Root cause

The `scoringInFlight` boolean at `roampal.ts:153/570-572` was designed to prevent 429 rate-limit pile-ups from cloud models. With localhost sidecar (Ollama), there are no 429s — the mutex only serializes and drops. A second `session.idle` event never reaches the retry logic; it's dropped at the gate before any attempt is made.

### Fix

Replace hard-skip mutex with async request queue in `roampal.ts`. Split into `_scoreExchangeViaLLM()` (internal, no queuing) and `scoreExchangeViaLLM()` (public wrapper with queue). When scoring is already running, pending requests enqueue and execute sequentially after the current call finishes. No requests are dropped — they all eventually run in order.

```typescript
// v0.5.4.1: Replace hard-skip mutex with async queue.
let scoringQueueRunning = false
let scoringQueue: _ScoringQueueItem[] = []
let lastQueuedResult: boolean | undefined

async function scoreExchangeViaLLM(
  sessionId, currentUserMessage, exchange, memories
): Promise<boolean> {
  if (scoringQueueRunning) {
    debugLog(`scoreExchange QUEUED — waiting for in-flight call to finish`)
    await new Promise<void>((resolve) => {
      scoringQueue.push({ sessionId, currentUserMessage, exchange, memories, resolve })
    })
    return lastQueuedResult!
  }

  scoringQueueRunning = true
  try {
    const result = await _scoreExchangeViaLLM(sessionId, currentUserMessage, exchange, memories)
    lastQueuedResult = result
    return result
  } finally {
    scoringQueueRunning = false
    while (scoringQueue.length > 0) {
      const next = scoringQueue.shift()!
      lastQueuedResult = await _scoreExchangeViaLLM(
        next.sessionId, next.currentUserMessage, next.exchange, next.memories
      )
      next.resolve()
    }
  }
}
```

### Testing

Manual end-to-end with qwen3:1.7b on Ollama — pre-fix log showed `scoreExchange SKIP — already in flight` → consecutive failures on every other exchange. Post-fix shows `scoreExchange QUEUED — 1 waiting` → original call completes → queued call starts → success. No dropped payloads, no consecutive failures from collisions.

## Sidecar summary contamination fix

### Symptom

In live observation of an OpenCode ghost-profile session today, the qwen3:1.7b sidecar produced summaries that contained verbatim text lifted from previously-stored memories rather than describing the actual exchange. Three consecutive summaries opened with the identical sentence ("User and assistant discussed Roampal memory system issues, confirming deletion of `_completion_state.json` fixes GUI deletion problems...") even though the underlying exchanges were different. A 63-character exchange produced a multi-clause summary referencing technical implementations that never came up in the conversation.

The fact extractor on the same model worked perfectly — 9 of 9 sampled facts were sharp, accurate, and grounded in the exchange (e.g., *"Resetting `_completion_state.json` temporarily resolves the issue"*, *"Dedup filter ignores archived entries (no status filter for memory_bank)"*). Only the summary call leaked.

### Root cause

The `score_exchange` call in `roampal.ts:scoreExchangeViaLLM()` does three things in one shot: write a summary, decide an outcome, and score each retrieved memory. To score memories, those memories must be in the prompt — so the prompt pasted them in as plain bullets:

```
These memories were injected into your context for this exchange:
- id1: "content..."
- id2: "content..."
```

No delimiter. No "this is reference material" hint. Small scoring models can't distinguish that block from the exchange-to-summarize body — they pattern-match the longest, most-structured text in the prompt and emit it back as the summary. Once a contaminated summary lands in the working-memory bank, it gets retrieved next turn, fed back into the prompt, and re-emitted — a self-reinforcing pollution loop.

### Fix

Wrap the exchange and the memory block in distinct delimiter tags, and explicitly scope the summary instruction to the exchange-only block:

```typescript
// Memory block — fenced with "do not summarize" hint
const memorySection = memories?.length
  ? `\n<memories_to_score>\nThese are stored memories — score them below. DO NOT summarize, quote, or reference their content in the "summary" field.\n${memories.map(m => `- ${m.id}: "${m.content}"`).join("\n")}\n</memories_to_score>\n`
  : ""

// Exchange body — fenced with explicit role labels
const scoringPrompt = `<exchange_to_summarize>
USER: "${exchange.user.slice(0, 8000)}"
ASSISTANT: "${exchange.assistant.slice(0, 8000)}"
USER_FOLLOW_UP: "${currentUserMessage.slice(0, 8000)}"
</exchange_to_summarize>
${memorySection}
...
SUMMARY (under 2000 chars): Summarize ONLY the content inside <exchange_to_summarize>. Do NOT include, mention, or paraphrase anything from <memories_to_score>. ...
```

Cost: zero — same call, same model, same tokens (slightly more from the delimiters). Memory scoring still has the memories visible. Summary scope is now syntactically explicit.

### Why this works for small models

Tiny instruct-tuned models follow delimiter-scoped instructions far better than they follow free-form ones. `<exchange_to_summarize>...</exchange_to_summarize>` reads as a closed unit; the model's attention naturally weights the in-block content higher when generating per a "summarize what's in this block" instruction. The memory bullets fall outside that fence and lose their pull as candidate summary content.

If contamination persists after this fix on smaller models, the next step would be splitting `score_exchange` into two calls (summary first, no memories in prompt; then a separate score-only call with memories). That's strictly more expensive and reserved as a fallback if Option A here under-delivers.

### Testing plan

Manual reproduction:
1. Configure OpenCode sidecar to a small model (qwen3:1.7b, qwen2.5:3b)
2. Have several exchanges on a single topic
3. Observe `roampal_plugin_debug.log` `scoreExchange raw` lines and the ghost-profile `roampal_working` collection in `chromadb`
4. **Pre-fix:** summaries open with identical templates lifted from prior memories
5. **Post-fix:** summaries reference only the actual exchange content; consecutive summaries diverge in wording

Automated verification (future): a unit test that builds the sidecar prompt with synthetic memories whose content is impossible to confuse with the exchange (e.g., random UUID strings), runs it against a small local model, and asserts the returned summary contains none of those UUIDs.

## Files touched

- `roampal/backend/modules/memory/memory_bank_service.py` — `archive()` changed from hard delete to soft delete (metadata update); `_get_count()` rewritten to exclude archived; `delete()` retained for cleanup only
- `roampal/backend/modules/memory/unified_memory_system.py` — `FACT_DEDUP_FILTERS["memory_bank"]` changed from `None` to `{"status": {"$ne": "archived"}}`; `_startup_cleanup()` gains phantom migration pass for memory_bank (removes IDs where `list_all_ids()` returns an ID but `get_fragment()` returns None)
- `roampal/plugins/opencode/roampal.ts` — `scoringInFlight` mutex replaced with async request queue (`scoringQueue`) to prevent dropped scoring requests; prompt: wrap exchange + memory block in `<exchange_to_summarize>` / `<memories_to_score>` delimiters with explicit "summary scope = exchange only" instruction
- `pyproject.toml` — version 0.5.4 → 0.5.5
- `roampal/__init__.py` — `__version__` 0.5.4 → 0.5.5
- `dev/docs/releases/v0.5.5/RELEASE_NOTES.md` — this file (gitignored)

## Simple Explanation

**The Problem:** When you delete a memory through the GUI, Roampal tries to remove it from ChromaDB using `collection.delete()`. But ChromaDB's underlying search index (HNSW) doesn't actually support true deletion — it marks entries as deleted but they stay in the graph and can still be found by queries. So when you try to save a new memory that's similar to one you "deleted," the duplicate-checking code finds the old entry still sitting there, thinks "this already exists," and skips saving the new one. Your memories stop being saved after you delete any of them.

**The Fix:** Instead of trying to hard-delete (which ChromaDB can't do properly), we switch to soft deletion — just mark entries with `status: archived` in their metadata, then tell all query paths to skip archived entries. This is reliable, reversible (you can restore deleted memories), and consistent with how other collections already handle archiving.

On first startup after this fix, Roampal also cleans up "phantom" entries from old hard deletes — IDs that ChromaDB still lists but have no actual content behind them. These are harmless debris from the broken delete path; removing them frees up index space and prevents any future confusion.
