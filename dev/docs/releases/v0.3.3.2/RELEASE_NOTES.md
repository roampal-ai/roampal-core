# v0.3.3.2 Release Notes

**Date:** 2026-02-06
**Platforms:** Claude Code, OpenCode, Cursor

## Summary

Adds relative timestamps to the KNOWN CONTEXT injection block. Memories now show how old they are (e.g., "2d", "5h", "3m") so the AI can weigh recency when using context.

## Changes

### 1. Relative Timestamps in KNOWN CONTEXT

**Before:**
```
• User prefers concise responses (85% proven, memory_bank)
```

**After:**
```
• User prefers concise responses (2d, 85% proven, memory_bank)
```

The `_relative_time()` helper converts ISO timestamps to human-readable relative ages:
- `1y`, `2mo`, `5d`, `3h`, `15m`, `now`
- Gracefully returns empty string if timestamp is missing or unparseable
- Uses `timestamp` field for working/history collections, `created_at` for memory_bank

## Files Changed

| File | Change |
|------|--------|
| `roampal/__init__.py` | Version bump 0.3.3.1 -> 0.3.3.2 |
| `pyproject.toml` | Version bump 0.3.3.1 -> 0.3.3.2 |
| `roampal/backend/modules/memory/unified_memory_system.py` | Added `_relative_time()` helper, timestamp extraction in `_format_context_injection()` |
