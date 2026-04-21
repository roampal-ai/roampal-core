# Roampal Core v0.4.9.4 Release Notes

**Release Date:** April 16, 2026  
**Priority:** Hotfix  
**Type:** Plugin enhancement  

## Summary

This hotfix addresses [Issue #4](https://github.com/roampal-ai/roampal-core/issues/4) by implementing automatic subagent filtering in the OpenCode plugin, and fixes a critical bug where sidecar scoring failed with 400 errors for OpenAI-hosted models.

Subagent conversations are no longer sent to the sidecar for summarization and scoring, preventing low-signal task exchanges from polluting the memory store. The `think: false` Ollama flag is now only sent to local/Zen targets, fixing silent sidecar failures for users with OpenAI API keys.

## Problem

When using OpenCode with multi-agent workflows (primary agent + subagents via the `task` tool), Roampal's hooks fired for **all** sessions equally. Subagent exchanges — short-lived, narrowly scoped task completions — were:

1. **Fetching context** from the server (wasted HTTP round-trip per subagent message)
2. **Injecting memory** into subagent system prompts (wasted tokens, no benefit)
3. **Capturing exchanges** and sending them to the sidecar for summarization (wasted LLM calls)
4. **Storing summaries** as working memories (polluting retrieval with low-signal content)
5. **Scoring memories** against subagent outcomes (skewing Wilson scores with irrelevant data)

The v0.4.2 debounce (1.5s idle timeout) was a timing-based workaround that reduced but did not eliminate the problem.

### Bug: `think: false` breaks OpenAI sidecar

The scoring request body included `think: false` — an Ollama-specific field to suppress chain-of-thought. The comment claimed "Non-Ollama servers ignore unknown fields," but **OpenAI's API rejects unknown fields with HTTP 400**. This caused sidecar scoring to silently fail for all users with `ROAMPAL_SIDECAR_URL` pointing to OpenAI (including `gpt-4o-mini`). Every scoring attempt returned 400, exhausted retries, and set `scoringBroken=true`.

**Fix:** `think: false` is now only included for Ollama/Zen targets (URLs containing `localhost`, `127.0.0.1`, or `opencode.ai/zen`). Applied to both the scoring call and the fact extraction call.

## Solution

Mode-based subagent detection using OpenCode's `client.app.agents()` API, with guards at every hook entry point.

### Detection Strategy

**Primary:** Query the agent definitions API at plugin load time. Each agent has a `mode` property (`"primary"` or `"subagent"`) set by OpenCode — this is authoritative regardless of agent naming conventions.

**Cache refresh:** If an unknown agent name appears in `chat.message` (agents added after plugin load), the cache is refreshed automatically.

**Fallback heuristic:** If `client.app.agents()` is unavailable (older OpenCode versions), falls back to name-based pattern matching (`"subagent"`, `"task-"`, `"task_"` prefixes). This is a best-effort safety net, not the primary mechanism.

### Hook Guards

| Hook | Guard | Effect |
|------|-------|--------|
| `chat.message` | `isSubagent(input.agent)` | Skips context fetch, exchange tracking, scoring data cache |
| `experimental.chat.system.transform` | `subagentSessions.has(sessionId)` | Skips memory context injection |
| `session.idle` (event) | `subagentSessions.has(sid)` | Skips exchange capture, sidecar scoring, summary storage |
| `session.deleted` (event) | Cleanup | Removes session from `subagentSessions` tracking set |

### Session Tracking

`session.idle` and other event hooks do not receive agent information. To bridge this gap, `chat.message` (which receives `input.agent`) marks subagent sessions in a `subagentSessions` Set. Downstream hooks check this set instead of re-querying the agent API.

### Configuration

**Default behavior:** Subagents are filtered out (no action required).

**Opt-in capture:** Set `ROAMPAL_ALLOW_SUBAGENTS=1` to allow subagent exchange capture.

Can be set via environment variable or in `opencode.json`:

```json
{
  "mcp": {
    "roampal-core": {
      "environment": {
        "ROAMPAL_ALLOW_SUBAGENTS": "1"
      }
    }
  }
}
```

The env var is read from both `opencode.json` MCP config and `process.env` (same pattern as `ROAMPAL_SIDECAR_*` variables).

## Files Modified

- `roampal/plugins/opencode/roampal.ts` — Subagent detection, hook guards, session tracking, config loading

## Technical Details

### Changes to `roampal.ts`

1. **Config loader** (`_loadSidecarConfig`): Added `allowSubagents` field, reads `ROAMPAL_ALLOW_SUBAGENTS` from opencode.json and process.env.

2. **State management**: Added `subagentSessions` Set (tracks which sessions belong to subagents) and `cachedAgentModes` Map (caches agent name-to-mode mapping from `client.app.agents()`).

3. **Plugin initialization**: Calls `client.app.agents()` at load time to populate the agent mode cache.

4. **`isSubagent()` helper**: Checks `cachedAgentModes` first (authoritative mode-based detection), falls back to name-based heuristic if the agent isn't in the cache.

5. **`chat.message` guard**: Early return for subagent sessions — no context fetch, no exchange tracking, no scoring data cache. Refreshes agent cache on unknown agent names.

6. **`system.transform` guard**: Early return for subagent sessions — no memory context or scoring status injected.

7. **`session.idle` guard**: Early return for subagent sessions — no exchange capture, no sidecar call, no summary storage.

8. **`session.deleted` cleanup**: Removes session from `subagentSessions` set to prevent memory leaks.

### Prior Art

The detection pattern follows the [plannotator plugin](https://github.com/backnotprop/plannotator) which uses the same `client.app.agents()` API to skip subagents in its system prompt transform hook.

### Relationship to v0.4.2 Debounce

The v0.4.2 debounce (1.5s idle timeout) remains as defense-in-depth for edge cases where OpenCode fires `session.idle` prematurely. With subagent filtering in place, the debounce primarily serves its original purpose of handling timing races, not subagent noise.

## Dependencies

- OpenCode v1.4.6+ (plugin API compatibility)
- `client.app.agents()` API (graceful degradation if unavailable)
- No changes to Python server components

## Related Issues

- **Issue #4:** Feature request: restrict memory capture to named agents
- **OpenCode #13334:** `session.idle` fires for subagent completions (addressed by v0.4.2 debounce)
- **v0.4.2:** Added debouncing workaround for subagent noise (now supplemented by proper filtering)

---

**Next Release:** v0.5.0 (planned major release with enhanced memory management)
