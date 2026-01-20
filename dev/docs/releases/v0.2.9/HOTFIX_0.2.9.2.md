# roampal-core v0.2.9.2 Hotfix

**Release Date:** 2026-01-20
**Type:** Hotfix (Critical)

---

## Issue

**Exchange scoring broken since v0.2.3** - The `outcome` parameter in `score_response` was not being applied to the previous exchange's doc_id.

### Root Cause

v0.2.3's performance fix removed the exchange scoring to eliminate a slow O(n×m) session file scan:

```python
# v0.2.3 removed this from /api/record-outcome:
previous = await _session_manager.get_most_recent_unscored_exchange()
if previous and previous.get("doc_id"):
    await _session_manager.mark_scored(...)
    doc_ids_scored.append(previous["doc_id"])
```

The methods `get_most_recent_unscored_exchange()` and `mark_scored()` still exist but became dead code.

### Impact

When calling `score_response(outcome="worked", memory_scores={...})`:

1. ✅ `memory_scores` items get scored individually (working)
2. ❌ The **exchange itself** never gets the `outcome` applied (broken)
3. All exchanges sit at s:0.5 forever in working memory

This broke a core feature - exchanges never learned from user feedback. The scoring prompt asks the LLM to "Score the previous exchange" but the exchange's own doc_id was never scored.

---

## Fix

### server/main.py - Re-enable exchange scoring with O(1) cache lookup

**Location:** `/api/record-outcome` endpoint (line 950-967)

```python
# v0.2.9.2: Score the exchange itself with the outcome
# This was accidentally removed in v0.2.3's performance fix
exchange_doc_id = None
if request.outcome in ["worked", "failed", "partial"]:
    # Get exchange doc_id from session manager's cache (O(1) lookup)
    previous = _session_manager._last_exchange_cache.get(conversation_id)
    if not previous:
        # Try most recent cache entry (handles session ID mismatch)
        for cid, exc in _session_manager._last_exchange_cache.items():
            if not exc.get("scored", False):
                previous = exc
                break
    if previous and previous.get("doc_id") and not previous.get("scored", False):
        exchange_doc_id = previous["doc_id"]
        await _memory.record_outcome(doc_ids=[exchange_doc_id], outcome=request.outcome)
        # Mark as scored in session manager
        await _session_manager.mark_scored(conversation_id, exchange_doc_id, request.outcome)
        logger.info(f"Scored exchange {exchange_doc_id} with outcome={request.outcome}")
```

**Key difference from v0.2.2:**
- Uses `_last_exchange_cache` (O(1)) instead of `get_most_recent_unscored_exchange()` (O(n×m) file scan)
- Maintains the v0.2.3 performance improvement while restoring functionality

---

## Why This Wasn't Caught

1. **Surfaced memories still scored** - `memory_scores` path worked, so scoring looked functional
2. **Key takeaways still worked** - `record_response` creates takeaways starting at 0.7
3. **Dead code not obvious** - Functions existed but nothing called them
4. **No test coverage** - No test checked "did the exchange doc_id get scored"
5. **Silent regression** - 6 versions (v0.2.3 → v0.2.9.1) shipped with broken exchange scoring

---

## Test Cases

```python
async def test_exchange_scoring():
    """score_response should apply outcome to the exchange doc_id."""
    # Setup: Create an exchange
    exchange_doc_id = await memory.store_working(
        content="User: hello\n\nAssistant: hi there",
        metadata={"turn_type": "exchange"}
    )
    session_manager._last_exchange_cache["test_session"] = {
        "doc_id": exchange_doc_id,
        "scored": False
    }

    # Act: Call score_response
    await record_outcome(RecordOutcomeRequest(
        conversation_id="test_session",
        outcome="worked",
        memory_scores={}
    ))

    # Assert: Exchange should have the score applied
    doc = await memory.get("working", exchange_doc_id)
    assert doc["metadata"]["score"] > 0.5  # worked = +0.2
    assert session_manager._last_exchange_cache["test_session"]["scored"] == True
```

---

## Version Bump

- `pyproject.toml`: 0.2.9.1 → 0.2.9.2
- `roampal/__init__.py`: 0.2.9.1 → 0.2.9.2

---

## Summary

| Location | Change |
|----------|--------|
| `server/main.py:950-967` | Re-enable exchange scoring with O(1) cache lookup |

---

## Related

- v0.2.3 RELEASE_NOTES.md documents the performance fix that caused this
- `mark_scored()` and `get_most_recent_unscored_exchange()` in session_manager.py are no longer dead code
