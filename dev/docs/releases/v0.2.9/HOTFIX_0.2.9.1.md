# roampal-core v0.2.9.1 Hotfix

**Release Date:** 2026-01-17
**Type:** Hotfix (Critical)

---

## Issue

**memory_bank memories can be permanently deleted** when their outcome score drops below the deletion threshold (0.2).

### Root Cause

`_handle_deletion()` in `promotion_service.py` has no protection for permanent collections:

```python
# promotion_service.py:257-282
async def _handle_deletion(self, doc_id, collection, metadata, score):
    """Handle deletion of low-scoring memories."""
    # NO CHECK FOR PERMANENT COLLECTIONS!

    if score < deletion_threshold:
        self.collections[collection].delete_vectors([doc_id])  # Deletes memory_bank!
```

### Impact

When memory_bank memories are surfaced in context and rated negatively:

1. Memory surfaced → LLM rates it with failed outcome
2. Score drops: 0.5 - 0.3 = 0.2 (at threshold)
3. Second failed outcome: 0.2 - 0.3 = 0.0 (below threshold)
4. `handle_promotion()` calls `_handle_deletion()`
5. **Memory permanently deleted**

This contradicts the documented design where memory_bank is a **permanent** collection:

| Collection | Score Δ | Wilson | Lifecycle |
|------------|---------|--------|-----------|
| memory_bank | ✓ (useless) | ✓ (ranking) | **permanent** |
| books | ✗ skip | ✗ | **permanent** |

User identity, preferences, and critical facts stored in memory_bank could be lost after just 2 negative ratings.

---

## Fix

### promotion_service.py - Protect permanent collections

**Location:** `_handle_deletion()` (line 257-282)

```python
# Before (buggy)
async def _handle_deletion(
    self,
    doc_id: str,
    collection: str,
    metadata: Dict[str, Any],
    score: float
):
    """Handle deletion of low-scoring memories."""
    # Calculate age
    age_days = 0
    # ... deletion logic with no collection check

# After (fixed)
async def _handle_deletion(
    self,
    doc_id: str,
    collection: str,
    metadata: Dict[str, Any],
    score: float
):
    """Handle deletion of low-scoring memories."""
    # v0.2.9.1 FIX: memory_bank and books are PERMANENT - never delete based on score
    # Scores are only used for Wilson ranking, not lifecycle management
    if collection in ("memory_bank", "books"):
        logger.debug(f"[PROMOTION] Skipping deletion for {collection} (permanent collection)")
        return

    # Calculate age
    age_days = 0
    # ... rest unchanged
```

---

## Why This Happened

The v0.2.8 Wilson scoring fix (`outcome_service.py:100-104`) removed an early return that was *accidentally* protecting memory_bank from the full scoring flow. Once scoring started working properly, the deletion threshold could be triggered.

The books collection was already protected with an explicit early return in `outcome_service.py:104-106`, but memory_bank was not.

---

## Test Cases

```python
# Test: memory_bank should never be deleted via score
async def test_memory_bank_deletion_protection():
    """memory_bank memories should never be deleted regardless of score."""
    service = PromotionService(collections, embed_fn)

    # Simulate very low score that would normally trigger deletion
    await service._handle_deletion(
        doc_id="memory_bank_abc123",
        collection="memory_bank",
        metadata={"timestamp": "2026-01-01"},
        score=0.0  # Far below deletion threshold
    )

    # Memory should still exist
    doc = collections["memory_bank"].get_fragment("memory_bank_abc123")
    assert doc is not None, "memory_bank memory was incorrectly deleted"

# Test: books should also be protected
async def test_books_deletion_protection():
    """books memories should never be deleted regardless of score."""
    service = PromotionService(collections, embed_fn)

    await service._handle_deletion(
        doc_id="books_xyz789",
        collection="books",
        metadata={},
        score=0.0
    )

    doc = collections["books"].get_fragment("books_xyz789")
    assert doc is not None, "books memory was incorrectly deleted"

# Test: working/history/patterns still get deleted normally
async def test_working_deletion_still_works():
    """Non-permanent collections should still be deletable."""
    service = PromotionService(collections, embed_fn)

    await service._handle_deletion(
        doc_id="working_test123",
        collection="working",
        metadata={"timestamp": "2026-01-01"},  # Old enough for 0.2 threshold
        score=0.1  # Below threshold
    )

    doc = collections["working"].get_fragment("working_test123")
    assert doc is None, "working memory should have been deleted"
```

---

## Version Bump

- `pyproject.toml`: 0.2.9 → 0.2.9.1
- `roampal/__init__.py`: 0.2.9 → 0.2.9.1

---

## Summary

| Location | Change |
|----------|--------|
| `promotion_service.py:264` | Add permanent collection guard before deletion logic |

---

## Related

- Desktop v0.3.0 has the same fix applied
- Benchmark test `test_memory_bank_score_immutability()` in `test_learning_sabotage.py` would have caught this
