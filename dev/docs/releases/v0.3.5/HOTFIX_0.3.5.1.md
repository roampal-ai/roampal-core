# roampal-core v0.3.5.1 Hotfix

**Release Date:** 2026-02-10
**Type:** Hotfix

---

## Issue

KNOWN CONTEXT injection only displays 3 memories instead of the 4 allocated by the v0.3.2 4-Slot Context Injection system.

### Root Cause

v0.3.2 introduced 4-slot allocation in `_build_context_injection()`:

```python
# Line 1024-1026: allocates 4 memories
top_memories = reserved_working + reserved_history + valid_results[:2]
# 1 working + 1 history + 2 best matches = 4 total
```

But `_format_context_injection()` was never updated to display all 4:

```python
# Line 1097: only shows 3 of the 4 fetched
for mem in memories[:3]:  # BUG: should be [:4]
```

The 4th memory was fetched, had its ID cached for scoring, but was never shown in KNOWN CONTEXT.

### Impact

- Users see 3 memories in KNOWN CONTEXT instead of 4
- The 4th memory (usually a high-quality best-match) is fetched but silently dropped from display
- Scoring still receives all 4 IDs, so the 4th memory gets scored despite never being shown to the LLM

---

## Fix

### unified_memory_system.py - Display all 4 allocated memories

**Location:** `_format_context_injection()` (line 1097)

```python
# Before (buggy)
for mem in memories[:3]:

# After (fixed)
for mem in memories[:4]:
```

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py:1097` | `memories[:3]` → `memories[:4]` |
| `roampal/__init__.py:18` | `0.3.5` → `0.3.5.1` |
| `pyproject.toml:7` | `0.3.5` → `0.3.5.1` |

---

## Version Bump

- `pyproject.toml`: 0.3.5 → 0.3.5.1
- `roampal/__init__.py`: 0.3.5 → 0.3.5.1

---

## Summary

One-line fix. The 4-slot allocation from v0.3.2 fetched 4 memories but the display loop only iterated over 3. Now all 4 are shown in KNOWN CONTEXT as originally intended.
