# v0.4.9.1 Release Notes

**Date:** 2026-04-14
**Type:** Hotfix
**Platforms:** Claude Code, OpenCode, Cursor

## Summary

Critical hotfix for server deadlock on startup and leftover regex tag fallback that contradicted the v0.4.9 LLM-only tag extraction decision.

## Changes

### 1. Remove v0.4.8 Migration Block (Server Deadlock Fix)

**Problem:** The v0.4.8 fact re-tag migration ran on every server boot inside the async `lifespan()` function. It pulled up to 10,000 facts per collection (working, history, patterns) via synchronous `collection.get()` and `collection.update()` calls. With large datasets (40k+ tags), this blocked the uvicorn event loop — the server bound the port but never started serving requests. All hook calls (UserPromptSubmit, Stop) timed out silently.

**Symptoms:** Server process listening on port 27182 but all requests time out. Hooks report "Roampal hook error: timed out" with non-blocking exit code 1. Health endpoint unresponsive. No request logs after startup.

**Fix:** Removed the migration block entirely from `server/main.py` lifespan. The migration is no longer needed — users who need to retag should use `roampal retag`.

### 2. Remove Regex Tag Fallback from Store Paths

**Problem:** Three store paths still had regex tag extraction as a fallback when `noun_tags` were not passed by the caller. This contradicted the v0.4.9 decision: LLM-only tags, no regex fallback (matching benchmark behavior where LLM failure returns `[]`).

**Affected paths:**
- `store_working()` — elif branch called `_tag_service.extract_tags()` (regex)
- `store_memory_bank()` — elif branch called `_tag_service.extract_tags()` (regex)
- Direct store to patterns/history — unconditionally called `_tag_service.extract_tags()` (regex)

**Fix:** Removed all three regex fallback branches. If `noun_tags` are passed (from LLM via score_memories/record_response), they're stored. If not passed, no tags are extracted. Users can run `roampal retag` to backfill tags on existing memories using their sidecar LLM.

## Files Changed

| File | Change |
|------|--------|
| `roampal/__init__.py` | Version bump 0.4.9 -> 0.4.9.1 |
| `pyproject.toml` | Version bump 0.4.9 -> 0.4.9.1 |
| `roampal/server/main.py` | Removed v0.4.8 migration block from lifespan() |
| `roampal/backend/modules/memory/unified_memory_system.py` | Removed regex fallback from store_working(), store_memory_bank(), and direct store path |
