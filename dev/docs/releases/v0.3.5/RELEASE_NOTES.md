# v0.3.5 Release Notes (PLANNED)

**Status:** Design only — not yet implemented
**Platforms:** Claude Code, Cursor (hook-based clients)
**Feature:** Sidecar scoring for hook-based clients

## Problem

Claude Code and Cursor rely entirely on the main LLM calling `score_response` via MCP. The scoring prompt is injected by the UserPromptSubmit hook, but the main LLM sometimes ignores it. When that happens, the exchange goes unscored — there's no fallback.

OpenCode already has a sidecar (independent free LLM call via Zen API) that catches these misses. Hook-based clients need the same safety net.

Server logs show the gap clearly:
```
Soft enforce: record_response not called for <conversation_id>
```

## Current Flow (no sidecar)

```
1. UserPromptSubmit hook
   → calls /api/hooks/get-context
   → server returns context + scoring prompt (if scoring needed)
   → scoring prompt injected into LLM context

2. Main LLM responds
   → SHOULD call score_response MCP tool (sometimes doesn't)

3. Stop hook
   → calls /api/hooks/stop
   → server stores exchange, checks if score_response was called
   → if not called: logs "Soft enforce" warning, does nothing else
   → exchange goes unscored
```

## Proposed Flow (with sidecar)

```
1. UserPromptSubmit hook (unchanged)
   → calls /api/hooks/get-context
   → server returns context + scoring prompt

2. Main LLM responds
   → may or may not call score_response

3. Stop hook (CHANGED)
   → calls /api/hooks/stop
   → server checks if score_response was called
   → NEW: if NOT scored, server returns sidecar_scoring payload:
       {
         "should_block": false,
         "scoring_complete": false,
         "needs_sidecar": true,
         "sidecar_data": {
           "conversation_id": "...",
           "exchange": { "user": "...", "assistant": "..." },
           "memories": [{ "id": "...", "content": "..." }, ...],
           "current_user_message": "..."
         }
       }
   → stop_hook.py sees needs_sidecar=true
   → makes independent LLM call to Zen free model
   → posts result to /api/record-outcome
```

## Implementation Plan

### Server Changes (roampal/server/main.py)

**`/api/hooks/stop` endpoint:**

Currently returns `StopHookResponse(stored, doc_id, scoring_complete, should_block, block_message)`.

Add to the response model:
- `needs_sidecar: bool = False`
- `sidecar_data: Optional[dict] = None`

When `scoring_was_required and not scored_this_turn`:
1. Build `sidecar_data` from `_last_exchange_cache[conversation_id]` (previous exchange) and `_search_cache[conversation_id]` (cached memory doc_ids)
2. Set `needs_sidecar = True`
3. Keep `should_block = False` (never block the user's flow)

The server already has all the data — `_last_exchange_cache` stores the previous user/assistant messages, and `_search_cache` stores the doc_ids of memories surfaced last turn.

### Stop Hook Changes (roampal/hooks/stop_hook.py)

After receiving the response from `/api/hooks/stop`:

```python
if result.get("needs_sidecar"):
    sidecar_data = result["sidecar_data"]
    outcome = score_via_free_llm(sidecar_data)
    if outcome:
        # Post back to server
        post_to_server(f"{server_url}/api/record-outcome", {
            "conversation_id": sidecar_data["conversation_id"],
            "outcome": outcome,
            "memory_scores": {}  # uniform — sidecar can't do per-memory
        })
```

New function `score_via_free_llm()`:
- Uses the same Zen free model API (OpenAI-compatible `/chat/completions`)
- Same model fallback chain as OpenCode plugin: trinity-large-preview-free → kimi-k2.5-free → phi-4-free, etc.
- Same scoring prompt format (exchange + follow-up → JSON `{"outcome": "worked"}`)
- Timeout: 6 seconds per model, 15 seconds total
- Pure Python with `urllib.request` (no dependencies — stop hook is dependency-free by design)

### Zen API Key Distribution

The stop hook runs as a subprocess — it doesn't have access to the OpenCode plugin's `ZEN_API_KEY` constant or the MCP server's config.

Options:
1. **Environment variable** (`ROAMPAL_ZEN_KEY`) — simplest, user sets it once
2. **Server passes it in `sidecar_data`** — server reads from config, sends to hook
3. **Hardcoded free-tier key** — Zen free models may not require auth (check API)
4. **Server config endpoint** — hook calls `/api/config/zen-key` to fetch it

Recommendation: Option 2 (server passes it). The server already knows the Zen config. The hook doesn't need to store secrets. The key only travels over localhost.

### Memory Scoring Behavior

Same as OpenCode sidecar:
- Sidecar scores the **exchange outcome** only (worked/failed/partial/unknown)
- All cached memories that turn inherit the exchange outcome (uniform scoring)
- If the main LLM DID score via `score_response`, sidecar is skipped entirely (main LLM provides per-memory precision)
- No double-scoring possible — `needs_sidecar` is only true when `score_response` wasn't called

### Timing

The stop hook already runs after the main LLM is done and after the 500ms race-condition wait for in-flight `score_response` calls. The sidecar adds ~2-6 seconds to the stop hook but runs in the background from the user's perspective (Claude Code doesn't block on the stop hook's exit for this).

**Important:** The stop hook currently exits 0 immediately on success. The sidecar HTTP call must complete before `sys.exit(0)` — it's synchronous Python, so this happens naturally. The user won't notice the delay because the stop hook runs after the LLM has already finished responding.

## Files to Change

| File | Change |
|------|--------|
| `roampal/server/main.py` | Add `needs_sidecar` + `sidecar_data` to `StopHookResponse` model and `/api/hooks/stop` endpoint logic |
| `roampal/hooks/stop_hook.py` | Add `score_via_free_llm()` function, handle `needs_sidecar` in response, post result to `/api/record-outcome` |
| `roampal/__init__.py` | Version bump → 0.3.5 |
| `pyproject.toml` | Version bump → 0.3.5 |

## Testing Plan

- Unit test: mock `/api/hooks/stop` returning `needs_sidecar=true` → verify `score_via_free_llm` is called
- Unit test: mock `/api/hooks/stop` returning `needs_sidecar=false` → verify sidecar is NOT called
- Unit test: Zen API timeout/failure → verify stop hook still exits 0 (never blocks user)
- Integration test: full round-trip — skip `score_response` → stop hook fires → sidecar scores → verify exchange has outcome in memory
- Live test: have Claude Code ignore scoring prompt → check server logs for sidecar scoring instead of "Soft enforce" warning

## Open Questions

1. **Should sidecar also work for Cursor?** — Same hook system, should work identically. Cursor 1.7+ uses the same stop hook format.
2. **Rate limiting** — Zen free models have rate limits. If user is chatting rapidly, sidecar calls could pile up. Add a mutex (same as OpenCode's `scoringInFlight` flag) or skip if last sidecar was <10s ago.
3. **Fallback to server-side sidecar?** — Instead of the hook making the LLM call, the server could do it. Pro: centralizes logic. Con: adds a dependency (httpx/aiohttp) to the server and couples scoring to server availability.
