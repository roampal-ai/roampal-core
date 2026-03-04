# v0.4.0 Release Notes

**Platforms:** Claude Code, OpenCode, Core Backend
**Theme:** Cross-platform compatibility audit and backend data integrity fixes

---

## Overview

Comprehensive audit of roampal-core against OpenCode v1.2.x and Claude Code v2.1.x, covering the MCP server, Claude Code hooks, OpenCode plugin, and the full backend data layer (ChromaDB adapter, promotion service, outcome service, search service).

- 1 critical, 5 high, 8 medium, 10 low issues identified
- All Python backend fixes implemented and verified (527 tests passing)
- OpenCode plugin TypeScript changes deferred to v0.4.1

---

## Critical

### 1. ChromaDB `hnsw:space` metadata mismatch

**File:** `chromadb_adapter.py`

`query_vectors()` passed `metadata={"hnsw:space": "l2"}` to `get_or_create_collection`, but `initialize()` creates collections without this metadata. This inconsistency can cause silent distance metric mismatches depending on ChromaDB version.

**Fix:** Removed the inconsistent metadata from `query_vectors()` so both paths use the same collection settings.

---

## High

### 2. `search_books()` calls nonexistent adapter method

**File:** `search_service.py`

`search_books()` called `self.collections["books"].query()`, but `ChromaDBAdapter` exposes `query_vectors()`. Any books search via MCP would raise `AttributeError` at runtime.

**Fix:** Generate embedding first, then call `query_vectors()`.

### 3. Promoted document scoring silently dropped

**File:** `outcome_service.py`, `promotion_service.py`

`record_outcome` matched documents by ID prefix only. When a document was promoted (e.g., `working_abc` → `history_abc`), the old ID no longer existed and scores were silently discarded.

**Fix:** Added `original_id` metadata during promotion (both `_promote_working_to_history` and `_promote_history_to_patterns`). Added fallback lookup by `original_id` in the promotion target collection.

### 4. Promotion race condition

**File:** `promotion_service.py`

`handle_promotion` (called from outcome service) did not acquire `_promotion_lock`, but `_do_batch_promotion` did. Both could operate on the same document simultaneously.

**Fix:** Wrapped `handle_promotion` in the same `_promotion_lock` via an inner method pattern.

### 5. Fragile promotion ID derivation

**File:** `promotion_service.py`

Promotion methods used `.replace("working_", "history_")` for ID derivation, which could corrupt IDs containing the substring elsewhere.

**Fix:** Changed to `f"history_{doc_id.split('_', 1)[1]}"` across all promotion paths (3 specific methods + batch promotion).

### 6. `book_` vs `books_` prefix mismatch

**File:** `unified_memory_system.py`, `outcome_service.py`

`store_book` created IDs with `book_` prefix, but outcome scoring expected `books_` (matching the collection name). Book documents never matched prefix-based lookup.

**Fix:** Changed prefix to `books_`. Added dual-prefix guard in outcome service for backward compatibility with existing data.

---

## Medium

### 7. Stop hook ignores `last_assistant_message`

**File:** `stop_hook.py`

Claude Code v2.1+ sends `last_assistant_message` directly in hook input JSON. The hook was parsing the full transcript file instead.

**Fix:** Read `last_assistant_message` from input data first, with transcript parsing as fallback for older versions.

### 8. Timezone mismatch in `_calculate_age_hours`

**File:** `promotion_service.py`

`datetime.now()` (naive) subtracted from timezone-aware timestamps caused a silent `TypeError`, returning age=0. Documents with TZ-aware timestamps were never cleaned up.

**Fix:** Use `datetime.now(timezone.utc)` and normalize naive timestamps to UTC before comparison.

### 9. Batch promotion race condition

Resolved as part of issue #4 — `handle_promotion` now acquires the shared `_promotion_lock`.

### 10. Batch promotion missing text fallback

**File:** `promotion_service.py`

`_do_batch_promotion` only read `metadata.get("text", "")`. Documents using the `"content"` key produced empty embeddings.

**Fix:** Multi-key fallback: `text or metadata.get("content") or doc.get("content", "")`.

### 11–13. OpenCode plugin improvements (deferred)

- Dead `experimental.session.compacting` handler in event switch (should be top-level hook)
- Missing `XDG_CONFIG_HOME` in plugin config path lookup
- Sidecar config never refreshed mid-session

Deferred to v0.4.1.

### 14. Zen model list mismatch

**File:** `sidecar_service.py`

Sidecar and plugin had different free model lists. Synchronized to `["glm-4.7-free", "kimi-k2.5-free", "gpt-5-nano"]`.

---

## Low

### 15–16. OpenCode plugin cleanup (deferred)

- `sessionOnboarded` set never cleaned on session end
- URL construction via fragile `.replace("/hooks", "")`

Deferred to v0.4.1.

### 17. Unbounded `mcp` dependency

**File:** `pyproject.toml`

Pinned to `mcp>=1.0.0,<2.0.0` to prevent breaking changes from a future major version.

### 18. Invalid `default: None` in MCP tool schema

**File:** `mcp/server.py`

Removed `"default": None` from `collections` and `sort_by` properties — `None` is not valid JSON Schema.

### 19. Duplicate `import os`

**File:** `chromadb_adapter.py`

Consolidated to a single import at the top of the file.

### 20. `update_fragment_metadata` re-upserts full embedding

**File:** `chromadb_adapter.py`

Changed from `upsert()` (which re-sends the embedding) to `collection.update()` for metadata-only changes.

### 21. Books recency boost checks wrong field

**File:** `search_service.py`

Changed from `upload_timestamp` (nonexistent) to `metadata.created_at` with top-level fallback.

### 22. Hook docblocks show old format

Cosmetic only. Not fixed in this release.

### 23. `_search_all` fetches up to 100k items per collection

Needs design decision on pagination approach. Not fixed in this release.

### 24. Incorrect error message in delete endpoint

**File:** `server/main.py`

Changed "Error archiving" to "Error deleting memory" to match the actual operation.

---

## Carried Forward

### 25. Cold start exchange summary truncation

Bumped from 200 to 300 chars in a prior release. Full removal deferred — needs token budget analysis.

### 26. Ghost entry self-healing

Ghost cleanup implemented in promotion and cleanup loops. Ghost log downgraded to DEBUG. Adapter-level self-healing deferred.

### 27. Content graph relationships deserialization ✅

**File:** `content_graph.py`

Added `isinstance(rels, dict)` guard to prevent `.items()` errors when relationships deserialize as a list.

### 28. "Document not found" log noise ✅

**File:** `outcome_service.py`

Downgraded from `logger.warning` to `logger.debug` to reduce console noise during normal scoring operations.

### 29–30. OpenCode hook adoption (deferred)

- `tool.definition` hook for hiding scoring tool description when sidecar is active
- `tool.execute.after` for in-plugin score detection (replacing HTTP round-trip)

Deferred to v0.4.1.

### 31. Content graph relationship traversal in search

Entity relationships are stored but not yet used during search. Low priority.

### 32. Naive timezone in tests ✅

Test files updated to use `datetime.now(timezone.utc)` to match production behavior.

---

## Upstream Changes to Monitor

| Change | Impact |
|--------|--------|
| `enableAllProjectMcpServers` deprecation (Claude Code v2.1.63) | Old boolean causes infinite hang; users must migrate to `enabledMcpjsonServers` array |
| Opus 4.6 medium effort default (Claude Code v2.1.68) | Fewer tool calls per turn by default |
| `defer_loading` API error (Claude Code #27065) | Upstream bug, not roampal-related |
| Orphaned MCP process fix (OpenCode v1.2.16) | MCP subprocesses properly terminate on exit |

---

## Files Modified

### Python Backend

| File | Changes |
|------|---------|
| `chromadb_adapter.py` | Remove hnsw:space mismatch; consolidate imports; use `update()` for metadata |
| `search_service.py` | Fix `search_books()` method call; fix books recency boost field |
| `outcome_service.py` | Add `original_id` fallback lookup; dual prefix guard; downgrade log level |
| `promotion_service.py` | Fix timezone; split-based ID derivation; write `original_id`; add promotion lock; text fallback |
| `unified_memory_system.py` | Fix `book_` → `books_` prefix |
| `content_graph.py` | Guard relationships deserialization |
| `stop_hook.py` | Read `last_assistant_message` from input data |
| `sidecar_service.py` | Sync Zen model list |
| `server/main.py` | Fix error message |
| `mcp/server.py` | Remove invalid `default: None` from schemas |
| `pyproject.toml` | Pin `mcp<2.0.0`; version bump to 0.4.0 |
| `__init__.py` | Version bump to 0.4.0 |

### Tests Updated

| File | Changes |
|------|---------|
| `test_promotion_service.py` | Use UTC-aware timestamps in age calculation test |
| `test_search_service.py` | Use `metadata.created_at` in books recency boost test |

---

## Verification

**527 tests passing. 0 failures.**

### Critical/High
- [x] ChromaDB collections use consistent metadata
- [x] `search_books()` uses correct adapter method
- [x] Promoted documents retain `original_id` in metadata
- [x] `handle_promotion` acquires shared promotion lock
- [x] Book IDs use `books_` prefix matching collection name
- [x] Promotion IDs use split-based derivation

### Medium
- [x] Stop hook reads `last_assistant_message` from input data
- [x] `_calculate_age_hours` handles TZ-aware timestamps correctly
- [x] Batch promotion uses multi-key text fallback
- [x] Zen model lists synchronized

### Low
- [x] `mcp` dependency pinned to `<2.0.0`
- [x] All 527 tests pass
- [x] No spurious "Document not found" warnings
- [x] Invalid JSON Schema defaults removed

### Deferred to v0.4.1
- [ ] OpenCode plugin: compacting hook, XDG_CONFIG_HOME, config refresh, session cleanup
- [ ] OpenCode plugin: `tool.definition` and `tool.execute.after` hook adoption
