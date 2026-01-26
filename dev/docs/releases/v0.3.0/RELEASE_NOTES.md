# Roampal Core v0.3.0 Release Notes

## Overview
Resilience release - fixes silent hook server crashes that leave users without context injection.

## Changes

### 1. Fix Hook Server Silent Crash ✓
**Problem:** Hook server crashes silently with `[Errno 22] Invalid argument` when PyTorch embedding model state becomes corrupted. Users don't see any error - hooks just stop injecting context.

**Root cause:** Three compounding issues:
1. `_start_fastapi_server()` only checks if port is in use, not if server is healthy
2. `/api/health` endpoint doesn't test embedding functionality - returns 200 even when embeddings broken
3. `_ensure_server_running()` (health check + restart) only called for `score_response`, not search operations
4. `user_prompt_submit_hook.py` always exits 0 (success) even on errors - silent failure

**Why embeddings break:** PyTorch state corruption when parent process (Claude Code) restarts while child (FastAPI) has model loaded. Common on Windows due to subprocess spawning semantics.

**Fix implemented:**
1. **Health endpoint tests embeddings** - `/api/health` now calls `embed_text("health check")` and returns 503 if it fails
2. **Server check before all MCP tools** - `_ensure_server_running()` called at start of `call_tool()` handler, restarts FastAPI if unhealthy
3. **Hook exits non-zero on failure** - Users see error message instead of silent failure. Clear message for 503 (embedding corruption)

**Files changed:**
- `roampal/mcp/server.py:637` - Added `_ensure_server_running()` to tool handler entry point
- `roampal/server/main.py:1044-1070` - Health endpoint tests embeddings, returns 503 if broken
- `roampal/hooks/user_prompt_submit_hook.py:147-159` - Exit 1 on errors with clear messages

**How it works now:**
1. FastAPI embedding corrupts → `/api/health` returns 503
2. Hook sees 503 → exits 1 with "Will auto-restart on next MCP tool call"
3. User sees hook error → knows something is wrong
4. User sends another message → MCP tool triggers `_ensure_server_running()`
5. Server auto-restarts → hooks work again

**Testing:**
1. Start Claude Code with roampal hooks working
2. Restart Claude Code (kill and relaunch)
3. Verify hooks still inject context (no silent failure)
4. Check server auto-restarts if corrupted