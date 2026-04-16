# Roampal Core v0.5.0 Release Notes

**Release Date:** April 2026  
**Type:** Minor release — plugin reliability, subagent filtering, scoring observability  

## Summary

v0.5.0 addresses five issues in the OpenCode plugin (`roampal/plugins/opencode/roampal.ts`):

1. **Subagent filtering** ([Issue #4](https://github.com/roampal-ai/roampal-core/issues/4)): Subagent exchanges no longer pollute the memory store
2. **OpenAI sidecar fix**: Removed `think: false` field that broke all non-Ollama sidecar targets
3. **Sidecar input cap**: Bumped fact extraction input from 800 chars to 16K (8K + 8K)
4. **Summary output cap**: Bumped scoring prompt summary limit from 300 chars to 2000 chars
5. **Scoring status visibility**: Replaced boolean `scoringBroken` with consecutive failure counter so persistent sidecar failures are surfaced to the user instead of silently masked

## Changes

### 1. Subagent Exchange Filtering (Issue #4)

**Problem:** When using OpenCode with multi-agent workflows (primary agent + subagents via the `task` tool), Roampal's hooks fired for all sessions equally. Subagent exchanges — short-lived, narrowly scoped task completions — were:

1. Fetching context from the server (wasted HTTP round-trip per subagent message)
2. Injecting memory into subagent system prompts (wasted tokens, no benefit)
3. Capturing exchanges and sending them to the sidecar for summarization (wasted LLM calls)
4. Storing summaries as working memories (polluting retrieval with low-signal content)
5. Scoring memories against subagent outcomes (skewing Wilson scores with irrelevant data)

The v0.4.2 debounce (1.5s idle timeout) was a timing-based workaround that reduced but did not eliminate the problem.

**Solution:** Mode-based subagent detection using OpenCode's `client.app.agents()` API.

**Detection strategy:**
- **Primary:** Query agent definitions API at plugin load time. Each agent has a `mode` property (`"primary"` or `"subagent"`) set by OpenCode — authoritative regardless of naming conventions.
- **Cache refresh:** If an unknown agent name appears in `chat.message` (agents added after plugin load), the cache is refreshed automatically.
- **Fallback heuristic:** If `client.app.agents()` is unavailable (older OpenCode versions), falls back to name-based pattern matching (`"subagent"`, `"task-"`, `"task_"` prefixes). Best-effort safety net, not the primary mechanism.

**Hook guards:**

| Hook | Guard | Effect |
|------|-------|--------|
| `chat.message` | `isSubagent(input.agent)` | Skips context fetch, exchange tracking; marks `primarySessions` |
| `experimental.chat.system.transform` | `subagentSessions.has(sessionId)` | Skips memory context injection |
| `session.idle` (pre-debounce) | `subagentSessions.has(sid)` | Skips debounce entirely for known subagents |
| `session.idle` (debounced) | `!primarySessions.has(sid)` | Skips exchange capture for sessions where `chat.message` never fired |
| `session.deleted` (event) | Cleanup | Removes session from `subagentSessions` and `primarySessions` |

**Session tracking (defense in depth):**

OpenCode does not fire `chat.message` for subagent child sessions — only event hooks (`message.part.updated`, `session.idle`) fire. This means the `subagentSessions` Set (populated in `chat.message`) may never be filled for subagents. To handle this, a `primarySessions` Set tracks sessions where `chat.message` DID fire. Inside the `session.idle` debounced callback, any session not in `primarySessions` is treated as a subagent and skipped.

Three layers of defense:
1. **`subagentSessions` check** (pre-debounce) — catches subagents identified by name/mode in `chat.message`
2. **`primarySessions` check** (in debounced callback) — catches subagents where `chat.message` never fired at all
3. **No `pendingScoringData`** — even if a subagent session reaches the scoring block, no scoring data was cached for it

In production testing, OpenCode did not fire `session.idle` for subagent sessions at all — the subagent's 36-tool-call, 6-minute run was completely invisible to the plugin. The `primarySessions` guard exists as a safety net for edge cases.

**Configuration:**
- Default: subagents filtered out (no action required)
- Opt-in: `ROAMPAL_ALLOW_SUBAGENTS=1` (env var or opencode.json MCP config)

**Prior art:** Detection pattern follows the [plannotator plugin](https://github.com/backnotprop/plannotator) which uses the same `client.app.agents()` API.

---

### 2. Remove `think: false` from Sidecar Requests

**Problem:** The scoring and fact extraction request bodies included `think: false` — an Ollama-specific field to suppress chain-of-thought reasoning. The code comment claimed "Non-Ollama servers ignore unknown fields," but **OpenAI's API rejects unknown fields with HTTP 400**. This also affects Azure OpenAI, Groq, Together, Mistral, and any other strict API.

**Impact:** Sidecar scoring silently failed for all users with `ROAMPAL_SIDECAR_URL` pointing to any non-Ollama provider. Every scoring attempt returned 400, exhausted retries, and the failure was masked by the auto-reset bug (see change #3 below). Users had no indication their sidecar was broken.

**Fix:** Removed `think: false` entirely from both LLM calls (scoring and fact extraction). The `/no_think` text prefix already present in the system and user message content handles reasoning model suppression for all providers — it's message content, not an API field, so no provider rejects it.

---

### 3. Sidecar Input Cap Bump (8K + 8K)

**Problem:** The fact extraction call truncated input to 500 chars (user) + 300 chars (assistant) = 800 chars total. This was benchmark-aligned (`runner.py` uses `user_msg[:500] + assistant_msg[:300]`) but too aggressive for production — long exchanges had most of their content stripped before the sidecar could extract facts from them.

**Fix:** Bumped to 8000 chars user + 8000 chars assistant (16K total). Even 3B parameter models handle 32K token context windows comfortably. The sidecar runs in the background after the main LLM responds, so larger input has no UX impact. The scoring call (summary + outcome) already sent full exchange text with no truncation — only the fact extraction call was capped.

---

### 4. Summary Output Cap Bump (300 → 2000 chars)

**Problem:** The scoring prompt instructed the sidecar LLM to produce summaries "under 300 chars." This was too short to capture meaningful exchange context — summaries were often truncated mid-thought or stripped of important detail.

**Fix:** Bumped the summary instruction from `<~300 chars>` to `<~2000 chars>`, matching `MAX_MEMORY_CHARS` used elsewhere in the system. Summaries now have room to capture the full story of an exchange without artificial truncation.

---

### 5. Scoring Status Visibility (Consecutive Failure Counter)

**Problem:** The v0.4.7 `scoringBroken` auto-reset created a cycle that masked persistent failures:

```
session.idle:
  1. scoringBroken = true (from previous failure)
  2. Auto-reset: scoringBroken = false ("give another chance")
  3. Try scoring → fails → queue for deferred retry
  4. Deferred retry fails → scoringBroken = true

system.transform (runs on next user message):
  → scoringBroken was just auto-reset to false at step 2
  → LLM sees [roampal scoring: initializing] — looks fine
  → User has no idea sidecar has been failing for the entire session
```

The auto-reset intent was correct (don't permanently kill scoring after one transient failure), but the implementation hid persistent failures from the user.

**Fix:** Replace `scoringBroken: boolean` with `consecutiveFailures: number`.

**New state machine:**

```
session.idle:
  - Scoring succeeds → consecutiveFailures = 0
  - Scoring fails (including deferred retry) → consecutiveFailures++
  - Always retry regardless of count (auto-reset intent preserved)

system.transform (status tag):
  - consecutiveFailures === 0 → [roampal scoring: ok via {model}]
  - consecutiveFailures === 1 → [roampal scoring: initializing]
                                  (one failure could be transient — silent)
  - consecutiveFailures >= 2  → [roampal scoring: failed ({N} consecutive failures).
                                  Check sidecar config or run "roampal sidecar setup".]
```

**Design principles:**
- **Never give up:** Scoring retries every exchange regardless of failure count. A broken sidecar that gets fixed mid-session recovers immediately.
- **Threshold of 2:** One failure is transient (rate limit, cold start, network blip). Two consecutive means a real problem (wrong API key, model rejected, malformed request). At 2+, the LLM sees the failure count and can inform the user.
- **Count resets on success:** A single success clears the counter. No lingering penalty from past failures.

**Implementation locations:**

| Location | Change |
|----------|--------|
| State management (top-level) | `let scoringBroken` → `let consecutiveFailures = 0` |
| `scoreExchangeViaLLM()` success paths | `scoringBroken = false` → `consecutiveFailures = 0` |
| `session.idle` deferred retry failure | `scoringBroken = true` → `consecutiveFailures++` |
| `session.idle` current scoring failure | Queue for retry (no change), `consecutiveFailures++` on deferred failure |
| `session.idle` auto-reset block | **Remove entirely** — counter doesn't need resetting |
| `system.transform` status tag | Switch on `consecutiveFailures` threshold (0 / 1 / 2+) |
| `SIDECAR_DISABLED` init | `scoringBroken = SIDECAR_DISABLED` → `consecutiveFailures = SIDECAR_DISABLED ? 2 : 0` |

## Testing

### Automated Tests

64 structural validation tests pass (`test_opencode_plugin.py`), including 24 new tests:

- **Subagent filtering (14):** `primarySessions`, `subagentSessions`, `isSubagent`, agent API timeout/optional chaining, config, session cleanup
- **Sidecar request body (2):** no `think: false` field, `/no_think` prefix retained
- **Consecutive failure counter (6):** definition, no `scoringBroken`, reset/increment, threshold, display
- **Sidecar caps (2):** 8K input, 2000 summary

### Manual Verification

Tested live with OpenCode + DeepSeek V3 (chat) + gpt-4o-mini (sidecar):

1. **Sidecar scoring:** 4/4 exchanges scored successfully (`scoreExchange SUCCESS`), 0 failures, `consecutiveFailures=0`
2. **Subagent filtering:** Spawned task agent (36 tool calls, 6 min). Zero subagent sessions reached `session.idle`. Zero garbage summaries stored. Zero wasted sidecar calls.
3. **Summary quality:** 424, 490, 707 chars (benefiting from 2000 cap)
4. **Fact extraction:** 1 + 1 + 4 = 6 facts extracted across 3 exchanges
5. **Agent API timeout:** `client.app.agents()` timed out at 3s, gracefully fell back to name heuristic

## Files Modified

- `roampal/plugins/opencode/roampal.ts` — All five changes (+149/-43 lines)
- `roampal/backend/modules/memory/tests/unit/test_opencode_plugin.py` — 24 new structural tests

## Dependencies

- OpenCode v1.4.6+ (plugin API compatibility)
- `client.app.agents()` API (graceful degradation if unavailable)
- No changes to Python server components

## Related Issues

- **Issue #4:** Feature request: restrict memory capture to named agents
- **OpenCode #13334:** `session.idle` fires for subagent completions (addressed by v0.4.2 debounce, now supplemented by subagent filtering)

## Migration Notes

- No action required — all changes are backward compatible
- Subagent filtering is automatic (opt-in via `ROAMPAL_ALLOW_SUBAGENTS=1`)
- Users with broken OpenAI sidecars will see scoring recover automatically after update
- Users will now be notified when sidecar is persistently failing (previously silent)
