# v0.4.1 Release Notes

**Platforms:** Claude Code, OpenCode, Core Backend
**Theme:** Linux stability, performance, error UX

---

## Overview

Linux reliability and runtime stability fixes driven by real user feedback (Arch Linux, OpenCode TUI). Addresses the root causes of server crashes, "not responding" errors, and OOM kills. Also includes performance improvements for large memory stores, sidecar-only scoring architecture for OpenCode, a timezone bug fix in memory age display, and dead code cleanup.

---

## Critical

### 1. `embed_text()` blocks asyncio event loop
**File:** `embedding_service.py`

`model.encode()` is CPU-bound (50-200ms) called inside `async def` with no `asyncio.to_thread()`. Blocks the entire event loop during encoding. Health checks from plugin/MCP time out, conclude server is dead, attempt restart → port conflicts → "not responding" errors. **Most likely root cause of user-reported instability.**

**Fix:** Wrapped `self.model.encode()` in `await asyncio.to_thread()` for both `embed_text()` and `embed_texts()`.

---

## High

### 2. MCP server restart timeout too short
**File:** `mcp/server.py`

`_ensure_server_running(timeout=3)` gave only 3 seconds. Hooks and plugin get 15 seconds. Return value was **ignored** — code proceeded to `_api_call()` regardless, causing raw `httpx.ConnectError` exceptions. **Root cause of "Roampal not responding" user reports.**

**Fix:** Bumped timeout to 15s (parity with hooks/plugin), check return value, return friendly "server is restarting" message on failure.

### 3. Cap `_search_all` fetch limit
**File:** `search_service.py`

`_search_all` fetched up to 100,000 items per collection via `collection.get(limit=100000)`, even when returning only 5-20 results.

**Fix:** Capped at 1,000 per collection.

### 4. Graceful `sentence-transformers` import failure
**File:** `embedding_service.py`

Top-level `from sentence_transformers import SentenceTransformer` had no try/except. If PyTorch fails to install (common on Arch with system package conflicts), the entire server crashed with a raw ImportError.

**Fix:** Wrapped import in try/except, set `EMBEDDING_AVAILABLE = False`, clear error message on use.

### 5. Server `lifespan()` startup error handling
**File:** `server/main.py`

`lifespan()` had no try/except around `UnifiedMemorySystem()` init. If ChromaDB fails, embedding model download fails, or any core dep has issues, the server crashed with no guidance.

**Fix:** Wrapped initialization in try/except with "what failed + how to fix" messages.

### 6. BM25 index rebuild loads ALL documents into memory
**File:** `chromadb_adapter.py`

`_build_bm25_index()` loaded ALL documents from ChromaDB with no limit. On memory-constrained Linux, triggers OOM killer.

**Fix:** Capped at 2,000 documents per rebuild.

---

## Medium

### 7. Respect `XDG_DATA_HOME` on Linux
**Files:** `cli.py`, `unified_memory_system.py`, `server/main.py`

Data directory hardcoded `~/.local/share/roampal/data` on Linux. `XDG_DATA_HOME` was not respected.

**Fix:** Check `$XDG_DATA_HOME` (default `~/.local/share`) across all three path resolution sites.

### 8. Non-atomic `_completion_state.json` writes
**File:** `session_manager.py`

`open("w")` truncates immediately. Concurrent `get-context` and `stop_hook` can race, producing partial/empty JSON.

**Fix:** Write to temp file, then `os.replace()` (atomic on both Linux and Windows).

### 9. `ChromaDBAdapter.__del__()` event loop conflict
**File:** `chromadb_adapter.py`

Created a new `asyncio.new_event_loop()` inside `__del__` during GC. Can conflict with the running loop and leave SQLite WAL files locked.

**Fix:** Removed `__del__`, added explicit `await adapter.cleanup()` during lifespan shutdown.

### 10. User-facing "server restarting" messages
**Files:** `mcp/server.py`, `user_prompt_submit_hook.py`

Hook showed alarming stderr ("killed stale server process"). MCP returned cryptic httpx errors.

**Fix:** Replaced with calm "Roampal: restarting server..." messages.

### 11. `_humanize_age()` timezone offset in context injection
**File:** `unified_memory_system.py`

`_humanize_age()` treated naive local timestamps as UTC, then compared against `datetime.now(timezone.utc)`. On any non-UTC system, memory ages were inflated by the timezone offset (e.g. 1h-old memory showed as "8h" on UTC-7).

**Fix:** Strip timezone info and compare naive local time to `datetime.now()`, matching the approach already used in the MCP server's copy of the function.

---

## Low

### 12. Dead cross-encoder code cleanup
**File:** `search_service.py`

Cross-encoder reranking code (~80 lines) was dead — `reranker` was never passed to `SearchService`. Removed `_rerank_with_cross_encoder` method, `reranker` parameter, and conditional calls. Redundant since v0.3.7 (Wilson scoring replaced it).

### 13. Normalize macOS detection
**File:** `unified_memory_system.py`

Changed `os.uname().sysname == 'Darwin'` to `sys.platform == 'darwin'` for consistency with `cli.py` and `server/main.py`.

### 14. Sidecar-only scoring on OpenCode
**Files:** `mcp/server.py`, `roampal.ts`

`score_memories` was always registered in the MCP tool list. On OpenCode, the sidecar handles scoring silently — but the model could see the tool, read its detailed description, and call it unprompted on every turn, causing double-scoring.

**Fix:** Scoring on OpenCode is now sidecar-only with no main LLM fallback:
- `score_memories` tool is never registered on OpenCode (`_hide_score_tool` flag in MCP server)
- Plugin no longer injects scoring prompt into system context when sidecar breaks
- Plugin no longer checks if main model scored via `check-scored` HTTP endpoint
- When sidecar is down, a `[IMPORTANT — roampal scoring: BROKEN]` warning is shown to the user suggesting `roampal sidecar setup`
- Scoring is paused until sidecar recovers (auto-recovery on next successful sidecar call)
- Tool remains available on Claude Code (scores via hook prompt)

---

## Deferred to v0.4.2

- **ONNX Runtime embedding backend** — Needs RAM benchmarking and `sentence-transformers>=3.2.0` version bump
- **OpenCode plugin: Fix `experimental.session.compacting` hook** — Handler incorrectly placed inside event switch; should be top-level hook
- **OpenCode plugin: `tool.execute.after` for score detection** — Replace HTTP round-trip with in-plugin detection

---

## Files Modified

| File | Changes |
|------|---------|
| `embedding_service.py` | `asyncio.to_thread()` for encode; graceful import with `EMBEDDING_AVAILABLE` flag |
| `search_service.py` | Cap `_search_all` at 1000; remove dead cross-encoder code (~80 lines) |
| `mcp/server.py` | Timeout 3s→15s; check return value; friendly error message; calm log messages; hide `score_memories` on OpenCode |
| `roampal.ts` | Remove scoring prompt injection and check-scored fallback; sidecar-only scoring |
| `server/main.py` | Lifespan try/except with guidance; XDG_DATA_HOME; explicit adapter cleanup on shutdown |
| `chromadb_adapter.py` | BM25 index capped at 2000 docs; `__del__` removed |
| `session_manager.py` | Atomic JSON writes via temp file + `os.replace()` |
| `cli.py` | XDG_DATA_HOME support |
| `unified_memory_system.py` | XDG_DATA_HOME; `sys.platform` for macOS; removed `reranker=None` from SearchService init; fix `_humanize_age` timezone offset |
| `user_prompt_submit_hook.py` | Calm "restarting server..." messages |
| `test_search_service.py` | Removed reranker test |
| `test_unified_memory_system.py` | Updated 2 cross-encoder wiring tests |

---

## Verification

**526 tests passing. 0 failures.**

### Critical/High
- [x] `embed_text()` uses `asyncio.to_thread()` for CPU-bound encode
- [x] MCP restart timeout bumped to 15s with return value check
- [x] `_search_all` capped at 1,000 per collection
- [x] `sentence-transformers` import wrapped in try/except
- [x] `lifespan()` has try/except with user-friendly error messages
- [x] BM25 index rebuild capped at 2,000 documents

### Medium
- [x] `XDG_DATA_HOME` respected in all 3 Linux path resolution sites
- [x] `_completion_state.json` writes are atomic via `os.replace()`
- [x] `__del__` removed; explicit cleanup wired to lifespan shutdown
- [x] Restart messages are calm and user-friendly
- [x] `_humanize_age()` returns correct relative ages on non-UTC systems

### Low
- [x] Cross-encoder dead code removed (~80 lines + reranker param)
- [x] macOS detection normalized to `sys.platform`
- [x] Sidecar-only scoring on OpenCode: tool hidden, no fallback prompt, BROKEN warning on failure
