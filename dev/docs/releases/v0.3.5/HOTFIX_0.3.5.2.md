# roampal-core v0.3.5.2 Hotfix

**Release Date:** 2026-02-11
**Type:** Hotfix (Critical)

---

## Issue

OpenCode plugin double-scores every exchange. When the main LLM calls `score_response` via MCP (providing granular per-memory scores), the sidecar fires anyway and overwrites those scores with blanket outcomes from a fallback model (trinity-large-preview-free).

### Root Cause

The plugin's client-side detection of `score_response` calls via `message.part.updated` never worked. The `mainLLMScored` flag was always `false` because:

1. MCP tool calls appear as `type=tool` with `part.tool` set (not `part.name` or `part.toolName`)
2. The original detection checked `part.name === "score_response"` which was always `undefined`
3. Even after broadening the check, `message.part.updated` fires for MCP tool calls BUT the detection still doesn't match because the tool name includes the server prefix: `roampal-core_score_response`

This bug was present since v0.3.2 and was documented as "fixed" in v0.3.2 and v0.3.4 release notes, but was never actually verified — zero instances of the detection log line in the entire log history.

### Impact

- Every exchange scored TWICE: once by the main LLM (per-memory), once by the sidecar (blanket)
- Sidecar scores overwrite the LLM's granular per-memory differentiation
- All 4 memories receive identical outcomes instead of individual scores
- Affects all OpenCode users since v0.3.2

---

## Fix

### Approach: Server-Side Scoring Detection

Since MCP `score_response` calls the HTTP server's `/api/record-outcome`, the server already knows when scoring happens via `session_manager.set_scored_this_turn()`. Added a new endpoint for the plugin to query before firing the sidecar.

### Change 1: main.py - New `check-scored` endpoint (line 617)

```python
@app.get("/api/hooks/check-scored")
async def check_scored(conversation_id: str = ""):
    """Check if score_response was already called for this conversation this turn.
    Used by OpenCode plugin to skip sidecar if main LLM already scored."""
    if not _session_manager:
        return {"scored": False}
    scored = _session_manager.was_scored_this_turn(conversation_id)
    return {"scored": scored}
```

### Change 2: main.py - Moved injection_map resolution before outcome check (line 1061)

Previously the injection_map resolution (MCP session ID -> plugin session ID) was inside the `if request.outcome in ["worked", "failed", "partial"]` block, meaning "unknown" outcomes wouldn't resolve the plugin session ID. Moved it before the outcome check so `set_scored_this_turn()` fires for ALL outcomes.

```python
# v0.3.5.2: Resolve the plugin session ID via injection_map BEFORE outcome check.
exchange_conv_id = conversation_id
if request.memory_scores and _injection_map:
    for doc_id in request.memory_scores.keys():
        injection = _injection_map.get(doc_id)
        if injection:
            exchange_conv_id = injection["conversation_id"]
            _session_manager.set_scored_this_turn(exchange_conv_id, True)
            logger.info(f"Resolved conversation {exchange_conv_id} via injection_map (doc_id={doc_id})")
            break
```

### Change 3: roampal.ts - Server-side check before sidecar (line 928)

```typescript
// v0.3.5.2: Server-side check — ask the server if score_response was already called.
let serverSaysScored = false
try {
    const checkResp = await fetch(`${ROAMPAL_HOOK_URL}/check-scored?conversation_id=${encodeURIComponent(sid)}`, {
        signal: AbortSignal.timeout(3000)
    })
    if (checkResp.ok) {
        const checkData = await checkResp.json() as { scored: boolean }
        serverSaysScored = checkData.scored
    }
} catch (err) {
    debugLog(`session.idle: check-scored failed (${err}), assuming not scored`)
}

if (mainLLMScored.get(sid) || serverSaysScored) {
    debugLog(`session.idle: Main LLM already scored — skipping sidecar for ${sid} (client=${mainLLMScored.get(sid) || false}, server=${serverSaysScored})`)
} else {
    // Main LLM didn't score — run sidecar
    debugLog(`session.idle: Main LLM did NOT score — running sidecar for ${sid}`)
    // ... existing sidecar logic ...
}
```

### Change 4: roampal.ts - Diagnostic logging for tool call discovery (line 846)

```typescript
// DEBUG: Log RAW part data BEFORE any guards filter it out
if (part && part.type !== "text") {
    debugLog(`part.updated RAW: type=${part.type}, name=${part.name}, toolName=${part.toolName}, tool=${part.tool}, sessionID=${part.sessionID}, messageID=${part.messageID}, keys=${Object.keys(part).join(",")}`)
}
```

This logging confirmed that OpenCode MCP tool calls appear as `type=tool` with `part.tool=roampal-core_score_response`.

---

## Verification

Log evidence (line 1891 of `roampal_plugin_debug.log`):

```
[2026-02-11T00:30:27.231Z] session.idle: Main LLM already scored — skipping sidecar for ses_3bc185662ffe1UYz5Mork50E76 (client=false, server=true)
```

- `client=false`: Client-side detection still doesn't work (kept as belt-and-suspenders for future fix)
- `server=true`: Server-side check confirmed scoring happened
- No `scoreExchange` lines after this — sidecar was correctly skipped

### Timing flow:
1. `chat.message`: `scoringRequired=true`, scoring data cached
2. LLM calls `score_response` via MCP -> server sets `scored_this_turn=true`
3. `session.idle` fires -> plugin calls `check-scored` -> returns `{"scored": true}`
4. Sidecar skipped. LLM's per-memory scores preserved.

---

## Deployment Notes

- **Stale `.pyc` issue**: After editing `main.py`, the `__pycache__/main.cpython-310.pyc` file was stale. The new endpoint registered in Python source but was NOT loaded by `roampal start` until the `.pyc` was deleted. This is because `roampal start` imports the module and Python uses the cached bytecode when the source modification time matches. Fix: delete `__pycache__/*.pyc` after edits, or restart Python fresh.
- **Plugin deployment**: The plugin source (`roampal/plugins/opencode/roampal.ts`) must be copied to `~/.config/opencode/plugins/roampal.ts`. Running `roampal init --opencode` does NOT overwrite if it says "already installed". Use `cp` or `--force` to deploy updates.

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/server/main.py:617-624` | New `GET /api/hooks/check-scored` endpoint |
| `roampal/server/main.py:1061-1072` | Moved injection_map resolution before outcome check |
| `roampal/plugins/opencode/roampal.ts:846-849` | Diagnostic logging for non-text parts |
| `roampal/plugins/opencode/roampal.ts:928-959` | Server-side check before sidecar scoring |

---

## Summary

The OpenCode plugin has been double-scoring every exchange since v0.3.2. The main LLM's granular per-memory scores were consistently overwritten by blanket sidecar scores. Client-side detection of MCP tool calls via `message.part.updated` was the wrong approach entirely — the event fires but the field names don't match. Server-side detection via a new `check-scored` endpoint is robust because the MCP tool call necessarily goes through the HTTP server.
