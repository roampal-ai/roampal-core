# roampal-core v0.1.9 Release Notes

**Release Date:** December 2025

## Summary

Fixed critical bug where MCP connected to wrong ChromaDB collection, plus improved search output formatting.

## Changes

### 1. ChromaDB adapter collection name fix (CRITICAL)

`initialize()` was ignoring the collection name passed to constructor and defaulting to `loopsmith_memories`.

**Before:** Collections like `roampal_memory_bank` were never loaded - all searches returned 0 results
**After:** Constructor's `collection_name` is respected when `initialize()` is called without args

```python
# Fix in chromadb_adapter.py
if self.collection_name and collection_name == DEFAULT_COLLECTION_NAME:
    collection_name = self.collection_name  # Use constructor value
else:
    self.collection_name = collection_name
```

### 2. Explicit adapter initialization in UMS

Added `await adapter.initialize()` call in the collection setup loop to ensure adapters are fully initialized before use.

### 3. Better search result formatting

- Added `_humanize_age()` - shows "2d", "3h" instead of raw ISO timestamps
- Added `_format_outcomes()` - shows `[YYN]` for last 3 outcomes
- Search results now include doc_id prefix for reference
- Updated collection descriptions with lifecycle info

**New format:**
```
1. [patterns] (2d, s:1.0, [YYY]) [id:abc12345] Content here...
```

**Old format:**
```
1. [patterns] (score:1.00, uses:5, last:worked) Content here...
```

### 4. CLI terminal compatibility

Changed update notification emoji from `⚠️` to `[!]` for terminals that don't render emoji.

## Files Changed

```
roampal/backend/modules/memory/chromadb_adapter.py
  - Lines 146-150: Collection name fix

roampal/backend/modules/memory/unified_memory_system.py
  - Line 212: Added await adapter.initialize()

roampal/mcp/server.py
  - Lines 57-97: Added _humanize_age() and _format_outcomes()
  - Updated search result formatting
  - Updated tool descriptions

roampal/cli.py
  - Line 99: Changed emoji to [!]
```

## Upgrade Notes

No breaking changes. After upgrading, restart Claude Code. Collections will now load correctly and search results will show improved formatting.

## Previous Version

- v0.1.8: archive_memory → delete_memory (hard delete)
