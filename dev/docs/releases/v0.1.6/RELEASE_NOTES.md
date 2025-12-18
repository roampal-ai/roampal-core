# roampal-core v0.1.6 Release Notes

**Release Date:** December 2025

## Summary

Two critical bug fixes: MCP score_response fallback and book management lazy initialization.

## Bug Fixes

### 1. score_response returning "0 memories updated"

The MCP's `score_response` tool was always returning 0 because:
1. MCP calls FastAPI's `/api/record-outcome` endpoint
2. FastAPI looks up its own cache (populated by hooks, not MCP)
3. FastAPI returns 200 with `documents_scored: 0`
4. MCP trusted this response and never fell back to its own cache

**Fix:** Fallback now runs when FastAPI returns 0, not just on exception.

### 2. Book management commands failing

`roampal books` showed empty and `roampal remove` always failed with "No book found", even though books were successfully ingested.

**Root cause:** Lazy initialization race condition. The `list_books()` and `remove_book()` methods checked `.collection is None` without first calling `_ensure_initialized()`.

**Fix:** Added `await books_collection._ensure_initialized()` before accessing the collection.

### 3. Missing API endpoints

CLI commands tried to call `/api/books` and `/api/remove-book` which didn't exist.

**Fix:** Added both endpoints to server/main.py.

## Files Changed

```
roampal/mcp/server.py                                    | ~5 lines (fallback fix)
roampal/backend/modules/memory/unified_memory_system.py  | ~10 lines (lazy init fix)
roampal/server/main.py                                   | +40 lines (new endpoints)
```

## Impact

- MCP scoring now works (learning loop functional)
- `roampal books` lists ingested books correctly
- `roampal remove <title>` removes books correctly
- CLI and server stay in sync

## Previous Version

- v0.1.5: DEV/PROD isolation, batch working memory cleanup
