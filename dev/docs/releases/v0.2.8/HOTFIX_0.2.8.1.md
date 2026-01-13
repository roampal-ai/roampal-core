# roampal-core v0.2.8.1 Hotfix

**Release Date:** 2026-01-12
**Type:** Hotfix

---

## Issue

`unknown` outcomes are being silently treated as `partial` due to else-branch fallthrough in `_calculate_score_update()`.

### Root Cause

```python
# outcome_service.py:283-287
if outcome == "worked":
    # ...
elif outcome == "failed":
    # ...
else:  # assumes partial, but catches unknown too!
    score_delta = 0.05 * time_weight
    success_delta = 0.5  # BUG: unknown gets +0.5 success
```

### Impact

When per-memory scoring sends `unknown` outcomes through the MCP fallback:
- **Expected**: No score change (unknown = didn't use this memory)
- **Actual**: Treated as `partial` (+0.05 score, +0.5 success_delta)

This corrupts Wilson score accuracy - memories marked "didn't use" still get positive weight.

---

## Fix

### 1. outcome_service.py - Guard invalid outcomes

**Location:** `_calculate_score_update()` (line 255-289)

```python
# Before (buggy)
def _calculate_score_update(
    self,
    outcome: str,
    current_score: float,
    uses: int,
    time_weight: float
) -> tuple:
    if outcome == "worked":
        score_delta = 0.2 * time_weight
        new_score = min(1.0, current_score + score_delta)
        uses += 1
        success_delta = 1.0
    elif outcome == "failed":
        score_delta = -0.3 * time_weight
        new_score = max(0.0, current_score + score_delta)
        uses += 1
        success_delta = 0.0
    else:  # partial - BUG: catches unknown too
        score_delta = 0.05 * time_weight
        new_score = min(1.0, current_score + score_delta)
        uses += 1
        success_delta = 0.5

    return score_delta, new_score, uses, success_delta

# After (fixed)
def _calculate_score_update(
    self,
    outcome: str,
    current_score: float,
    uses: int,
    time_weight: float
) -> tuple:
    if outcome == "worked":
        score_delta = 0.2 * time_weight
        new_score = min(1.0, current_score + score_delta)
        uses += 1
        success_delta = 1.0
    elif outcome == "failed":
        score_delta = -0.3 * time_weight
        new_score = max(0.0, current_score + score_delta)
        uses += 1
        success_delta = 0.0
    elif outcome == "partial":  # v0.2.8.1: explicit partial
        score_delta = 0.05 * time_weight
        new_score = min(1.0, current_score + score_delta)
        uses += 1
        success_delta = 0.5
    else:
        # v0.2.8.1: Guard - unknown or invalid outcomes don't affect score
        logger.warning(f"Unexpected outcome '{outcome}' - no score change")
        return 0.0, current_score, uses, 0.0

    return score_delta, new_score, uses, success_delta
```

### 2. mcp/server.py - Filter unknowns in fallback

**Location:** MCP fallback path (line 822-826)

```python
# Before (buggy)
if memory_scores:
    for doc_id, mem_outcome in memory_scores.items():
        result = await _memory.record_outcome([doc_id], mem_outcome)
        scored_count += result.get("documents_updated", 0)

# After (fixed)
if memory_scores:
    for doc_id, mem_outcome in memory_scores.items():
        # v0.2.8.1: Skip unknowns - they mean "didn't use this memory"
        if mem_outcome == "unknown":
            continue
        result = await _memory.record_outcome([doc_id], mem_outcome)
        scored_count += result.get("documents_updated", 0)
```

### 3. server/main.py - Filter unknowns in FastAPI endpoint

**Location:** `/api/record-outcome` endpoint (same pattern)

```python
# v0.2.8: Per-memory scoring
if memory_scores:
    for doc_id, mem_outcome in memory_scores.items():
        # v0.2.8.1: Skip unknowns - they mean "didn't use this memory"
        if mem_outcome == "unknown":
            continue
        result = await _memory.record_outcome([doc_id], mem_outcome)
        scored_count += result.get("documents_updated", 0)
```

---

## Test Cases

```python
# Test 1: unknown should not affect score
def test_unknown_outcome_no_score_change():
    initial_score = 0.5
    score_delta, new_score, uses, success_delta = service._calculate_score_update(
        "unknown", initial_score, 0, 1.0
    )
    assert score_delta == 0.0
    assert new_score == initial_score
    assert success_delta == 0.0

# Test 2: partial should still work
def test_partial_outcome_explicit():
    initial_score = 0.5
    score_delta, new_score, uses, success_delta = service._calculate_score_update(
        "partial", initial_score, 0, 1.0
    )
    assert score_delta == 0.05
    assert success_delta == 0.5
```

---

## Version Bump

- `pyproject.toml`: 0.2.8 → 0.2.8.1
- `roampal/__init__.py`: 0.2.8 → 0.2.8.1

---

## Summary

| Location | Change |
|----------|--------|
| `outcome_service.py:283` | `else:` → `elif outcome == "partial":` |
| `outcome_service.py:288` | Add `else:` guard returning no change |
| `mcp/server.py:824` | Filter `if mem_outcome == "unknown": continue` |
| `server/main.py` | Same filter in FastAPI endpoint |
