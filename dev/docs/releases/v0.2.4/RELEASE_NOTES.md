# roampal-core v0.2.4 Release Notes

**Release Date:** 2026-01-05
**Type:** Bug Fix - MCP Scoring Reliability

---

## Summary

Fixes reliability issue with the `related` parameter in `score_response` and ensures full doc_ids are displayed for selective scoring.

**The Problem:** When FastAPI returned 0 scored memories (cache timing mismatch), the MCP fallback path incorrectly filtered the cache by `related` instead of using `related` directly.

**The Fix:** Fallback now uses `related` doc_ids directly when provided.

---

## What Changed

### 1. Fixed `related` Parameter Handling (server.py)

The fallback path now correctly uses `related` doc_ids when the FastAPI primary path returns 0.

**Before (broken):**
```python
# Bug: Filtered cache BY related instead of USING related
doc_ids = [d for d in doc_ids if d in related]
```

**After (fixed):**
```python
# If related provided, use directly (don't filter cache)
if related is not None and len(related) > 0:
    doc_ids = related
elif session_id in _mcp_search_cache:
    cached = _mcp_search_cache[session_id]
    doc_ids = cached.get("doc_ids", [])
else:
    doc_ids = []
```

### 2. Full Doc ID Display (server.py)

Doc IDs are now shown in full instead of truncated, enabling use with the `related` parameter.

**Before:**
```python
id_str = f" [id:{doc_id[:8]}]"  # Truncated - unusable
```

**After:**
```python
id_str = f" [id:{doc_id}]"  # Full ID for related param
```

---

## Technical Details

### Scoring Flow

```
score_response(outcome, related=["id1", "id2"])
│
├── PRIMARY PATH (FastAPI /api/record-outcome)
│   └── Works correctly with related parameter
│
└── FALLBACK PATH (MCP cache) ← FIX HERE
    └── Triggers when FastAPI returns 0
    └── Now uses related directly instead of filtering
```

### When Does Fallback Trigger?

1. FastAPI cache empty (timing mismatch between MCP and FastAPI)
2. Session ID not found in FastAPI cache
3. FastAPI server unavailable

### Protected Collections

`memory_bank` and `books` are protected from scoring - they return `None` immediately.

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/mcp/server.py` | Fixed fallback `related` handling, full doc_id display |
| `pyproject.toml` | Version 0.2.3 → 0.2.4 |
| `roampal/__init__.py` | Version bump |

---

## Usage

### Selective Scoring (recommended)
```python
# Score specific memories that were actually relevant
score_response(outcome="worked", related=["working_abc123", "history_def456"])
```

### Score All Cached (legacy)
```python
# Omit related to score all memories from previous search
score_response(outcome="worked")
```

### Score None
```python
# Empty array scores nothing
score_response(outcome="worked", related=[])
```

---

## Upgrade Instructions

```bash
pip install --upgrade roampal
```

No migration needed. Existing memories preserved.

---

## Impact

- **Primary path (FastAPI):** Was already working correctly
- **Fallback path (MCP):** Now fixed for edge cases
- **Doc ID visibility:** Now usable for selective scoring
