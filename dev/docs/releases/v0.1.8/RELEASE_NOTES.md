# roampal-core v0.1.8 Release Notes

**Release Date:** December 2025

## Summary

Renamed `archive_memory` → `delete_memory` and made it actually delete.

## Changes

### 1. Renamed MCP tool: `archive_memory` → `delete_memory`

The tool was called "archive" but users expected it to remove memories. Now it:
- Is named `delete_memory` (clearer intent)
- Actually deletes the memory from the collection (hard delete)

### 2. Hard delete instead of soft archive

**Before:** Setting `status: "archived"` in metadata (memory stayed in collection)
**After:** Calling `delete_vectors()` (memory actually removed)

## Files Changed

```
roampal/mcp/server.py
  - Renamed tool: archive_memory → delete_memory
  - Updated messages: "deleted successfully" / "not found for deletion"

roampal/backend/modules/memory/memory_bank_service.py
  - archive(): Now calls delete() instead of soft-archiving

roampal/cli.py, README.md, ARCHITECTURE.md
  - Updated tool name references
```

## Upgrade Notes

**Breaking change:** The MCP tool is now `delete_memory` instead of `archive_memory`.

After upgrading, restart Claude Code. The tool will appear as `delete_memory` and will actually remove memories from the collection.

## Previous Version

- v0.1.7: Working memory cleanup fix, update notifications, archive_memory semantic search fix
