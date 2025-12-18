# roampal-core v0.1.8 Release Notes

**Release Date:** December 2025

## Summary

Hotfix for archive_memory - now performs hard delete instead of soft archive.

## Bug Fixes

### 1. archive_memory MCP tool didn't actually remove memory from collection

**Issue:** After v0.1.7 fixed the `archive_memory` MCP tool to find memories by content, it still wasn't actually removing them. The tool returned "Memory archived successfully" but the memory remained in the collection with `status: "archived"`.

**Root cause:** The `archive()` method was doing a soft delete (setting `status: "archived"` in metadata) instead of a hard delete (actually removing from the vector store).

**Impact:** Users thought memories were deleted but they remained in the collection, potentially appearing in search results with `include_archived=True`.

**Fix:** Changed `archive()` to call `delete()` which performs a hard delete via `delete_vectors()`:

```python
# Before (soft delete - memory stayed in collection):
metadata["status"] = "archived"
metadata["archive_reason"] = reason
metadata["archived_at"] = datetime.now().isoformat()
self.collection.update_fragment_metadata(doc_id, metadata)

# After (hard delete - memory actually removed):
return await self.delete(doc_id)
```

## Files Changed

```
roampal/backend/modules/memory/memory_bank_service.py | ~15 lines changed
  - archive(): Now calls delete() instead of soft-archiving

roampal/backend/modules/memory/tests/unit/test_memory_bank_service.py | ~10 lines changed
  - TestArchive: Updated to expect delete_vectors instead of update_fragment_metadata
```

## Testing

All 23 memory bank service tests pass:
- TestMemoryBankServiceInit: 2 passed
- TestStore: 3 passed
- TestUpdate: 3 passed
- TestArchive: 2 passed
- TestSearch: 3 passed
- TestRestore: 2 passed
- TestDelete: 2 passed
- TestListAll: 3 passed
- TestStats: 1 passed
- TestIncrementMention: 2 passed

## Upgrade Notes

After upgrading, restart your MCP server or Claude Code. The `archive_memory` tool will now actually remove memories from the collection.

## Previous Version

- v0.1.7: Working memory cleanup fix, update notifications, archive_memory semantic search fix
