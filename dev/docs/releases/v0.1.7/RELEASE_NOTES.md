# roampal-core v0.1.7 Release Notes

**Release Date:** December 2025

## Summary

Working memory cleanup fix + update notifications + archive_memory fix. Old memories with only `created_at` now get properly cleaned up, users are notified when new versions are available, and archive_memory MCP tool now works.

## New Features

### 1. Update Notifications

Users are now notified when a newer version of roampal is available on PyPI:

**CLI Notifications:**
When running `roampal init`, `roampal status`, or `roampal stats`, users see:
```
⚠️  Update available: 0.1.8 (you have 0.1.7)
    Run: pip install --upgrade roampal
```

**Hook Injection (for Claude Code users):**
On cold start (first message of session), if an update is available, Claude receives:
```xml
<roampal-update>
A newer version of roampal is available: 0.1.8 (user has 0.1.7).
IMPORTANT: Briefly mention this to the user. Say something like:
"Quick note: roampal 0.1.8 is available. Run `pip install --upgrade roampal` to update."
Only mention once per conversation.
</roampal-update>
```

**MCP Notifications (backup):**
When `get_context_insights()` is called, the update notice is also included in the response.

- Version check hits PyPI API with 2-second timeout
- Cached per server session to avoid repeated checks
- Fails silently on network issues (non-blocking)
- Hook injection provides explicit instruction to Claude to tell the user

## Bug Fixes

### 1. Working memory cleanup never ran (lazy initialization)

**Critical:** Startup cleanup was never actually executing due to ChromaDB's lazy initialization. Collections were not initialized when cleanup ran, so it silently returned 0 every time.

**Impact:** ALL versions before v0.1.7 had no working cleanup. Working memories accumulated forever.

**Fix:** Collections are now explicitly initialized before cleanup runs:
```python
# In unified_memory_system.py _startup_cleanup():
await adapter._ensure_initialized()  # Force collection init
if not adapter.collection:
    continue
```

### 2. Working memory not decaying for old memories (timestamp fallback)

Working memories older than 24 hours were not being cleaned up because:
1. Cleanup looked for `timestamp` field in metadata
2. Old memories only had `created_at` (always set by `store_working`)
3. If `timestamp` missing, `_calculate_age_hours()` returned 0.0 (looks brand new)
4. Items never matched the >24hr cleanup threshold

**Impact:** Even if cleanup ran, old memories without `timestamp` would never be found.

**Fix:** Cleanup now falls back to `created_at` when `timestamp` is missing:
```python
# Before:
timestamp_str = metadata.get("timestamp", "")

# After:
timestamp_str = metadata.get("timestamp") or metadata.get("created_at", "")
```

**Note:** Both fixes are required - #1 makes cleanup run, #2 makes it find old memories.

### 3. archive_memory MCP tool always returned "Memory not found"

The `archive_memory` MCP tool never worked - it always returned "Memory not found for archiving" even when the memory existed.

**Root cause:** The tool passed `content` (text to archive) but `memory_bank_service.archive()` expected a `doc_id`. It never searched for the memory first.

**Impact:** Users could not archive memories via the MCP tool (archive_memory was completely broken).

**Fix:** `memory_bank_service.archive()` now:
1. Accepts `content` parameter (text to match)
2. Searches for the memory semantically using `search_fn`
3. Gets the `doc_id` from search results
4. Archives by `doc_id`

```python
# Before: Expected doc_id, received content - never matched
async def archive(self, doc_id: str, reason: str = "llm_decision") -> bool:
    doc = self.collection.get_fragment(doc_id)  # Always None

# After: Searches for memory by content first
async def archive(self, content: str, reason: str = "llm_decision") -> bool:
    results = await self.search_fn(query=content, collections=["memory_bank"])
    doc_id = results[0].get("id")  # Get actual doc_id
    # Then archive by doc_id
```

## Files Changed

```
roampal/backend/modules/memory/memory_bank_service.py | ~40 lines changed
  - archive(): Now accepts content, searches semantically, then archives by doc_id

roampal/backend/modules/memory/unified_memory_system.py | 1 line changed
  - Pass search_fn to MemoryBankService so archive() can search

roampal/backend/modules/memory/promotion_service.py | 2 lines changed
  - Line 324: batch_promote_working() timestamp fallback
  - Line 458: cleanup_old_working_memory() timestamp fallback

roampal/cli.py | ~45 lines added
  - check_for_updates(): PyPI version check
  - print_update_notice(): CLI update notification
  - Added calls in cmd_init, cmd_status, cmd_stats

roampal/mcp/server.py | ~45 lines added
  - _check_for_updates(): PyPI version check (cached)
  - _get_update_notice(): MCP update notification
  - Injected into get_context_insights response

roampal/server/main.py | ~60 lines added
  - _check_for_updates(): PyPI version check (cached)
  - _get_update_injection(): Hook update injection with directive
  - Injected on cold start with clear instructions for Claude
```

## Impact

- Old working memories (pre-timestamp) will be cleaned on next startup
- Edge cases where `timestamp` is missing now handled gracefully
- `created_at` is ALWAYS present, making cleanup more robust
- Users now see update notifications (both CLI and MCP users)
- archive_memory MCP tool now works (was completely broken before)

## Upgrade Notes

After upgrading, restart your MCP server or roampal service. The startup cleanup will automatically remove old working memories (>24 hours) that were previously stuck.

## Previous Version

- v0.1.6: MCP score_response fallback, book lazy init fix
