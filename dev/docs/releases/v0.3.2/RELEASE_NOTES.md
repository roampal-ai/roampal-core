# Roampal Core v0.3.2 Release Notes

## Overview
Multi-client support — roampal now works with Claude Code and OpenCode through a shared single-writer server architecture. Context injection expanded to 4 slots (reserved working + reserved history + 2 best matches) to break the history scoring feedback loop.

## Changes

### 1. OpenCode Plugin Support
**New file:** `roampal/plugins/opencode/roampal.ts`

TypeScript plugin that integrates roampal with OpenCode via hooks:
- `chat.message` hook - captures user text, stores in `lastUserMessage` map, fetches context from server (split `scoring_prompt` + `context_only` fields), caches both for subsequent hooks. Does NOT modify `output.parts` (changes there are visible in UI).
- `experimental.chat.system.transform` hook - reads cached context from `cachedContext` Map (avoids double-fetching). Injects memory context into system prompt via `output.system.push()` (invisible in UI). Falls back to fresh fetch if cache miss.
- `experimental.chat.messages.transform` hook - reads cached scoring prompt, prepends it to the last user message's text part. Invisible in UI but AI sees it as part of the user message, matching Claude Code's UserPromptSubmit hook behavior.
- `event` hook - handles session lifecycle and message events:
  - `session.created` / `session.deleted` — session tracking and cleanup
  - `message.updated` (role=assistant) — tracks assistant message IDs
  - `message.part.updated` (type=text) — accumulates assistant text parts from `TextPart.text`
  - `message.part.updated` (type=tool-invocation, name=score_response) — detects main LLM scoring, sets `mainLLMScored` flag to skip independent scoring
  - `session.idle` — assembles assistant response from collected parts, sends exchange to `/api/hooks/stop`, fires independent scoring if main LLM didn't score

**Architecture:**
`chat.message` fires first — it stores user text, fetches context from server (split `scoring_prompt` + `context_only`), and caches both in a `cachedContext` Map. It does NOT modify `output.parts` (changes there are visible in the UI). `experimental.chat.system.transform` fires next — it reads from cache (O(1) lookup, no second HTTP call) and pushes memory context into the system prompt via `output.system.push()` (invisible in UI). `experimental.chat.messages.transform` fires last — it reads the cached scoring prompt and prepends it to the last user message's text part (invisible in UI, AI sees it as user instruction). The server returns split fields (`scoring_prompt` and `context_only`) alongside backward-compatible `formatted_injection` so Claude Code hooks continue working unchanged.

**Event structure fix (v0.3.2):** The initial plugin used `event.session_id || event.sessionId || event.id` to extract session IDs from events. None of these matched OpenCode's actual event structure, causing all event handlers to silently exit early. The correct patterns per event type:
- `session.created` / `session.deleted`: `event.properties.info.id`
- `message.updated`: `event.properties.info.sessionID` (+ `info.role`, `info.id`)
- `message.part.updated`: `event.properties.part.sessionID` (+ `part.messageID`, `part.type`, `part.text`)
- `session.idle`: `event.properties.sessionID`

Additionally, assistant response text was read from `msg.content` / `msg.parts` on `AssistantMessage`, but OpenCode delivers content through separate `message.part.updated` events with `TextPart` objects. Both bugs fixed — exchange capture and scoring now work.

### 2. CLI Integration
**File changed:** `roampal/cli.py`

Added:
- `configure_opencode()` function
- `--opencode` and `--claude-code` flags for `roampal init`
- Auto-detection of OpenCode installation (`~/.config/opencode/`)
- Plugin installation to `~/.config/opencode/plugins/roampal.ts`
- MCP config generation in `opencode.json`
- `PYTHONPATH` auto-computation for MCP environment (see fix #5 below)

### 3. Context Injection Fix
**Problem:** The initial plugin appended `<system-reminder>` tags directly to user message text via `output.parts[].text`. This content was visible in OpenCode's chat UI, causing massive vertical gaps and exposed memory context.

**Solution (v3 — split delivery via system prompt):**
Switched to `experimental.chat.system.transform` hook with a two-phase caching architecture:

1. `chat.message` fetches context from `/api/hooks/get-context` which returns split fields: `scoring_prompt` (the scoring block) and `context_only` (memory context without scoring). Both are cached in a `cachedContext` Map keyed by sessionID. Does NOT modify `output.parts`.
2. `system.transform` reads from the cache and injects memory context via `output.system.push(contextOnly)` — background knowledge at end of system prompt (invisible in UI).
3. `messages.transform` reads from the cache and prepends `scoringPrompt` to the last user message's text part — AI sees it as a direct instruction (invisible in UI).
4. No hook modifies `output.parts` in `chat.message` — all injection is through `system.transform` and `messages.transform`, invisible in the UI.

This matches the Claude Code experience where hooks inject via stdin/stdout and the content never appears in the chat window. The server returns both split fields and backward-compatible `formatted_injection` so Claude Code hooks continue unchanged.

### 4. Identity Prompt Improvements
**Issue identified:** The `<roampal-identity-missing>` prompt was being injected on cold start for 2+ weeks but Claude never acted on it due to passive language.

**Changes made:**
- Removed "consider" - passive hedge word
- Removed "No rush - just when it fits the conversation" - explicit permission to ignore

**Status:** Partially fixed. May still need more forceful redesign if Claude continues to ignore.

### 5. PYTHONPATH Fix for OpenCode MCP
**Problem:** OpenCode's MCP process failed with `ModuleNotFoundError: No module named 'roampal'`. Roampal is not pip-installed - it only imports because Python adds cwd to `sys.path`. Claude Code's MCP works because its cwd happens to be the project root. OpenCode runs from a different cwd (`C:\ROAMPAL` on Windows Desktop app).

**Fix:**
- Added `"PYTHONPATH": "<roampal-core-dir>"` to `opencode.json` MCP environment
- Updated `cli.py` `configure_opencode()` to auto-compute and include PYTHONPATH for future installs
- Verified: MCP now loads successfully (`toolCount=7`)

### 6. Shared Single-Writer Server Architecture

**Problem:** Each MCP process (Claude Code, OpenCode, Cursor) creates its own `UnifiedMemorySystem`, all opening the same ChromaDB files. The second process to connect gets errors:
```
ERROR: [ChromaDB] Query failed: Error creating hnsw segment reader: Nothing found on disk
```
Confirmed on `patterns` collection. Blocks multi-client usage.

**Solution — MCP as thin HTTP client:**

```
Before (broken):
  Claude Code → MCP → UnifiedMemorySystem ─┐
  Cursor      → MCP → UnifiedMemorySystem ─┼→ ChromaDB files (CONFLICT)
  OpenCode    → MCP → UnifiedMemorySystem ─┘
                 │
                 └→ FastAPI → UnifiedMemorySystem → ChromaDB files

After (clean):
  Claude Code → MCP (HTTP client) ──┐
  Cursor      → MCP (HTTP client) ──┼→ FastAPI → UnifiedMemorySystem → ChromaDB
  OpenCode    → MCP (HTTP client) ──┤   (port 27182)  (single instance)
                 + Plugin (HTTP) ───┘
```

One ChromaDB connection. All access serialized through FastAPI. Based on single-writer pattern from Designing Data-Intensive Applications.

**All three clients share the same MCP server (`server.py`) and hooks.** Only the config format and delivery mechanism differ:

| Aspect | Claude Code | Cursor | OpenCode |
|--------|------------|--------|----------|
| MCP | Same `server.py` | Same `server.py` | Same `server.py` |
| Hooks | stdin/stdout subprocess | stdin/stdout subprocess (`hooks.json`) | TypeScript plugin |
| Init | `roampal init` | `roampal init --cursor` | `roampal init --opencode` |

**What changed:**
- `roampal/mcp/server.py`: Removed `UnifiedMemorySystem` import, all 7 tools now proxy via HTTP to FastAPI through `_api_call()` helper. Each MCP instance gets a unique `_mcp_session_id` (`mcp_{uuid}`) for search cache isolation between clients. MCP process is now a lightweight HTTP client — no ChromaDB, PyTorch, or sentence-transformers loaded. Faster startup, smaller memory footprint.
- `roampal/server/main.py`: Added 2 new endpoints (`POST /api/record-response`, `POST /api/context-insights`), updated request models (`SearchRequest` gets `metadata_filters`/`sort_by`, `MemoryBankAddRequest` gets `always_inject`), wired `sort_by` through to `_memory.search()` (was on request model but not passed through), disabled parent process monitoring so FastAPI outlives any single MCP client, removed dead monitoring code (`_is_parent_alive`, `_monitor_parent_process`).

**What does NOT change (for this section):**
- Hook files (`user_prompt_submit_hook.py`, `stop_hook.py`) — already use FastAPI HTTP (but updated in section 7 for self-healing)
- OpenCode plugin (`roampal.ts`) — already uses FastAPI HTTP
- CLI (`cli.py`) — all three `configure_*` functions stay the same
- Claude Code / Cursor config files — unchanged

**Safety:** `_ensure_server_running()` runs before every tool call. If FastAPI is down, it auto-restarts and waits up to 3 seconds. All 7 tools now depend on FastAPI — but hooks already did, so this extends an existing proven pattern.

**Status:** Implemented in v0.3.2.

### 7. Hook Self-Healing (Server Auto-Restart)
**Files changed:** `roampal/hooks/user_prompt_submit_hook.py`, `roampal/hooks/stop_hook.py`, `roampal/plugins/opencode/roampal.ts`

**Problem:** Prior to v0.3.2, hooks had no self-healing. If FastAPI crashed or its PyTorch embedding state corrupted (`[Errno 22] Invalid argument`), the health check correctly returned 503 — but only `_ensure_server_running()` in `server.py` could restart it, and that only runs when an MCP tool is called. Hooks called FastAPI directly and failed silently every turn until a tool call happened to trigger recovery.

```
Before (v0.3.0):
  Hook → FastAPI (503/down) → print error, exit ← NO RECOVERY
  MCP tool → _ensure_server_running() → kill + restart ← ONLY RECOVERY PATH

After (v0.3.2):
  Hook → FastAPI (503/down) → _restart_server() → retry ← SELF-HEALING
  Plugin → FastAPI (503/down) → restartServer() → retry ← SELF-HEALING
  MCP tool → _ensure_server_running() → kill + restart ← STILL WORKS
```

**What changed:**

**Python hooks** (`user_prompt_submit_hook.py`, `stop_hook.py`): Both have a `_restart_server()` function (standalone, no roampal imports) that:
1. Finds and kills the stale process on the port (cross-platform: `netstat`+`taskkill` on Windows, `lsof`+`kill` on Unix)
2. Spawns a fresh `python -m roampal.server.main` subprocess
3. Polls `/api/health` for up to 15 seconds
4. Retries the original request on success

**OpenCode plugin** (`roampal.ts`): Has a `restartServer()` function with the same pattern:
1. Uses `child_process.execSync()` for cross-platform port killing
2. `child_process.spawn()` with `detached: true` for server launch
3. Polls `/api/health` for up to 15 seconds
4. Both `getContextFromRoampal()` and `storeExchange()` retry after restart
5. `_restartInProgress` flag prevents concurrent restart attempts

The `user_prompt_submit_hook` / `chat.message` is the critical path — it fires first every turn, so if the server is down, it recovers before context injection. The `stop_hook` / `session.idle` has the same logic for completeness but maintains its existing behavior of never blocking on errors.

**Status:** Implemented in v0.3.2.

### 8. Cursor Integration Status

Cursor integration is fully implemented in `cli.py` (`configure_cursor()`, lines 512-641):
- MCP config → `~/.cursor/mcp.json`
- Hooks config → `~/.cursor/hooks.json` (`beforeSubmitPrompt` + `stop`)

**Blocked by Cursor bug:** As of Cursor v2.4.7 (Jan 2026), the `agent_message` field in `beforeSubmitPrompt` hook response is not reaching the AI. Cursor team has acknowledged the bug. `stop` hook works. MCP tools work. Context injection does not.

When Cursor ships the hook fix, roampal's Cursor support goes live with zero code changes — the implementation is already correct.

### 9. 4-Slot Context Injection
**File changed:** `roampal/backend/modules/memory/unified_memory_system.py`

**Problem:** Context injection used 3 slots: 1 reserved working + 2 best matches from `[patterns, history, memory_bank]` ranked by Wilson score. In practice, history memories never surfaced because patterns and memory_bank items consistently outranked them. This created a feedback loop: history never appeared → never got scored → never improved → never appeared.

**Solution — reserved history slot + expanded best-match pool:**

```
Before (3 slots):
  1. working (reserved)
  2. best from [patterns, history, memory_bank]
  3. second best from [patterns, history, memory_bank]

After (4 slots):
  1. working (reserved)
  2. history (reserved — breaks scoring feedback loop)
  3. best from [working, patterns, history, memory_bank]
  4. second best from [working, patterns, history, memory_bank]
```

Key details:
- History gets its own dedicated search (`limit=1, collections=["history"]`), same pattern as working
- If no history match exists (new user), the slot falls through to the best-match pool
- Best-match pool now includes ALL collections except books (deduped against reserved slots)
- History can still win best-match slots too — up to 3 of 4 slots could be history if it's most relevant
- `always_inject` memory_bank items (identity, preferences) remain separate, outside the 4 scored slots
- Single-point change: all platforms (Claude Code, OpenCode, Cursor) benefit automatically

**Token impact:** ~150 extra tokens per turn (one additional memory). Full injection block goes from ~700 to ~850 tokens.

**Status:** Implemented in v0.3.2.

### 10. memory_bank Wilson Scoring Fix (80/20 Blend)
**Files changed:** `roampal/backend/modules/memory/unified_memory_system.py`, `roampal/backend/modules/memory/scoring_service.py`

**Problem:** v0.2.9 introduced an 80/20 blend for memory_bank ranking: after 3 uses, the score should be `0.8 * (importance × confidence) + 0.2 * wilson_score`. This was correctly implemented in `search_service.py` (`_apply_collection_boost()` and `_rerank_with_cross_encoder()`), but `SearchService` is never instantiated by `UnifiedMemorySystem`. The live search path uses inline scoring in UMS and `scoring_service.py`, both of which unconditionally overwrote with pure `importance × confidence`, ignoring Wilson entirely.

**Root cause — three scoring paths that disagreed:**

| Code path | memory_bank scoring | Actually used? |
|-----------|-------------------|----------------|
| `search_service.py` (v0.2.9) | 80% confidence + 20% Wilson after 3 uses | No — never instantiated by UMS |
| `scoring_service.py` (initial) | 100% confidence always | No — not called from UMS search |
| `unified_memory_system.py` (inline) | 100% confidence always | **Yes** — this is the live path |

**Fix:**
- `unified_memory_system.py` (inline search scoring): Now checks `uses >= 3` and blends `0.8 * base_quality + 0.2 * (success_count / uses)` matching search_service.py's implementation
- `scoring_service.py` (`calculate_final_score()`): Now blends `0.8 * quality + 0.2 * wilson_score` after 3 uses instead of unconditional overwrite

**Cold start protection preserved:** Below 3 uses, pure `importance × confidence` (no Wilson influence from sparse data).

**Status:** Implemented in v0.3.2.

### 11. Wilson Scoring for search() Path
**File changed:** `roampal/backend/modules/memory/unified_memory_system.py`

**Problem:** The `search()` method used an inline scoring formula with raw `metadata["score"]` (the running +0.2/-0.3 delta) for non-memory_bank collections, and a simple `success_count/uses` ratio for memory_bank. Meanwhile, `get_context_for_injection()` (the hook context path) used `ScoringService.calculate_final_score()` with proper Wilson confidence intervals and dynamic weight shifts. This meant:

| Code path | Scoring method | Wilson? | Dynamic weights? |
|-----------|---------------|---------|-----------------|
| `search()` (MCP search_memory) | Inline formula with raw score | No | No |
| `get_context_for_injection()` (hooks) | ScoringService with Wilson | Yes | Yes |

**Impact:** All benchmarks test the `search()` path — meaning they measured performance WITHOUT Wilson scoring. The learning plateau at 3 uses (30%→60%, then flat) was a direct consequence: the raw score formula saturates after one "worked" (+0.2) takes you from 0.5→0.7. Wilson would give more conservative scores with few observations and higher confidence as evidence accumulates.

**Fix:** Replaced the inline scoring block in `search()` with a call to `self._scoring_service.calculate_final_score()`, the same function used by `get_context_for_injection()`. Both paths now use:
- Wilson confidence intervals for learned scores
- Dynamic weight shifts based on memory maturity (NEW→EMERGING→ESTABLISHED→PROVEN)
- Memory bank 80/20 blend with real Wilson (not simple ratio)
- Fallback to inline formula if scoring service not yet initialized

**Before (raw score, no Wilson):**
```python
quality = float(metadata.get("score", 0.5))  # Raw running score
quality_boost = 1.0 - (quality * 0.8)
adjusted_distance = distance * quality_boost
final_rank_score = adjusted_similarity * (1.0 + quality)
```

**After (Wilson + dynamic weights):**
```python
scores = self._scoring_service.calculate_final_score(metadata, distance, coll_name)
# Uses Wilson CI, dynamic weights, memory_bank blend — same as hooks
```

**Status:** Implemented in v0.3.2.

### 12. Comprehensive Test Bitrot Fix (30/30)

**Problem:** The integration test `test_comprehensive.py` had 10 failures (20/30) due to API mismatches accumulated since the test was written.

**Root causes found and fixed:**

1. **record_outcome called with positional args** — `record_outcome(doc_id, "worked")` passed string as `doc_ids` (List[str] parameter), iterating over characters ('w', 'o', 'r', 'k'...). No outcomes were ever recorded. Fixed: use keyword args `record_outcome(doc_id=doc_id, outcome="worked")`.

2. **Books doc ID prefix** — `store_book()` generates `book_` prefix, test expected `books_`. Fixed assertion.

3. **verify_doc_exists** — Split on `_` mapped `book_abc_chunk_0` to collection `book` (doesn't exist). Fixed: `book_` prefix maps to `books` collection.

4. **Memory_bank dedup** — Test expected deduplication that doesn't exist (each store creates unique ID). Changed to test unique ID generation.

5. **Promotion async timing** — Promotion runs via `asyncio.create_task()` (fire-and-forget). Tests checked immediately. Fixed: added `await asyncio.sleep(0.5)`.

6. **history->patterns threshold** — v0.2.9 added `success_count >= 5` requirement. Test only did 3 outcomes. Fixed: pre-seed success_count + more outcomes.

7. **Content KG access** — `content_graph` is on `_kg_service`, not directly on UMS. Fixed all references.

8. **Entity extraction** — Not auto-triggered by `store_memory_bank()`. Test now manually calls `add_entities_from_text()`.

9. **Working 24h decay** — `cleanup_old_working_memory()` is on `_promotion_service`, not UMS. Fixed: use correct path, store with backdated timestamp.

10. **History 30d decay** — `clear_old_history()` doesn't exist. Now implemented (section 14) and test changed to verify actual 30d cleanup.

**Also moved** 13 mock-embedding integration tests from `dev/benchmarks/` to `dev/tests/integration/` to separate infrastructure tests from real-embedding benchmarks.

### 13. Cross-Encoder Reranking Wired Up
**Files changed:** `roampal/backend/modules/memory/unified_memory_system.py`

**Problem:** `SearchService` in `search_service.py` (768 lines) had full cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`, hybrid search (vector + BM25 RRF fusion), entity boost from Content KG, and KG-based routing — but was never instantiated by `UnifiedMemorySystem`. The `search()` method used inline `query_vectors()` (pure vector search) with basic Wilson scoring. Desktop's `SearchService` provided 96% accuracy vs Core's 1% on the same token efficiency benchmark.

**Fix:**
- `initialize()` now creates a `SearchService` instance with all services (scoring, routing, KG, embedding)
- Optionally loads `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` at startup — gracefully continues without it if model unavailable
- `search()` delegates to `SearchService.search()` which provides:
  - Hybrid search: vector + BM25 with Reciprocal Rank Fusion
  - Cross-encoder reranking: top-30 candidates re-scored by ms-marco model (when available)
  - Collection-specific boosts: patterns priority, memory_bank quality, books recency
  - Entity boost: Content KG quality-weighted entity matching
  - KG routing: intelligent collection selection when no collections specified
  - Known solution injection from Action-Effectiveness KG
- Ghost ID filtering and `sort_by` parameter preserved on top of SearchService results
- Inline scoring retained as fallback if SearchService fails or isn't initialized

**Impact:** `search()` now uses the same full pipeline as Desktop — hybrid search + cross-encoder + entity boost. The learning plateau at 60% Top-1 in the 4-way benchmark should improve since the cross-encoder adds ~10 percentage points and the hybrid search improves recall.

### 14. History 30-Day Cleanup
**Files changed:** `roampal/backend/modules/memory/promotion_service.py`, `roampal/backend/modules/memory/unified_memory_system.py`, `roampal/backend/modules/memory/outcome_service.py`

**Problem:** The MCP server docstring documented history as "30d scored" but no time-based cleanup existed. History items persisted indefinitely (only removed by score-based deletion). This contradicted the documented tier lifecycle:
- working: 24h auto-promote/delete
- history: 30d scored ← **NOT IMPLEMENTED**
- patterns: permanent scored
- memory_bank/books: permanent

**Fix:**
- Added `cleanup_old_history(max_age_hours=720.0)` to `PromotionService` — same pattern as `cleanup_old_working_memory()` but with 30-day (720h) cutoff
- Called on startup in `_startup_cleanup()` alongside working memory cleanup
- Called in `_deferred_learning()` batch cleanup (every 50 outcomes) alongside working cleanup
- Integration test updated: test 27 changed from "history persistence" to "history 30d cleanup" — stores a 31-day-old history item and verifies `cleanup_old_history()` removes it

### 15. Cross-Collection Search Test (31/31)
**File changed:** `dev/tests/integration/test_comprehensive.py`

Added test `test_2_6_cross_collection_search` — stores distinct Docker-related memories across 4 collections (working, history, patterns, memory_bank), then searches across all collections without specifying any. Verifies:
- Results come back from at least 2 different collections
- All results have `final_rank_score` and `collection` fields

## Known Issues

1. **Cursor context injection blocked** — Cursor v2.4.7 `agent_message` field not reaching the AI. MCP tools and stop hook work. Waiting on Cursor fix (section 8).
2. **Identity prompt partially effective** — `<roampal-identity-missing>` prompt improved but Claude may still not act on it. May need more forceful redesign (section 4).
3. **Cross-encoder optional** — If `sentence-transformers` can't load `ms-marco-MiniLM-L-6-v2`, search falls back to vector + Wilson only. No cross-encoder reranking in that case.

### 16. Hook/MCP Port Mismatch — Split Brain Bug

**Status:** Fixed (Bug 8). `_build_hook_command()` in `cli.py` wraps hook commands with `ROAMPAL_DEV` env var, ensuring hooks and MCP always target the same port.

**Problem:** Hooks and MCP can silently target different FastAPI servers, creating two independent ChromaDB databases that never sync. Exchanges stored by the stop hook become invisible to MCP `search_memory`, and MCP `record_response` key_takeaways become invisible to hook context injection.

**Root cause:** Port selection is configured in two independent places with no coordination:

| Component | Config location | Reads `ROAMPAL_DEV` from | Default port |
|-----------|----------------|--------------------------|-------------|
| MCP server | `.mcp.json` `env` block | `.mcp.json` `"ROAMPAL_DEV": "1"` | 27183 (dev) |
| UserPromptSubmit hook | `settings.json` command | System environment (not set) | 27182 (prod) |
| Stop hook | `settings.json` command | System environment (not set) | 27182 (prod) |

Claude Code launches hooks as subprocess commands that inherit the system environment — NOT the MCP `env` block. So `.mcp.json` can set `ROAMPAL_DEV=1` for MCP, but hooks never see it.

**Reproduction:**
```
1. curl -X POST http://127.0.0.1:27182/api/hooks/stop  → stores working_60fafbf1 ✓
2. curl -X POST http://127.0.0.1:27182/api/hooks/get-context "test probe"  → finds it ✓
3. MCP search_memory("test probe")  → hits port 27183 → NOT FOUND ✗
```

**Impact:** In any dev setup where `.mcp.json` has `ROAMPAL_DEV=1` but hooks don't:
- Stop hook stores exchanges on prod server (27182)
- MCP tools query dev server (27183)
- Context injection hook queries prod server (27182) — so surfaced memories are from prod
- Result: MCP `search_memory` can't find any hook-stored exchanges or context-injection memories. The AI appears to have amnesia when searching, but context injection works fine.

**Fix (mapped out, 3 options):**

**Option A — Single port enforcement in `roampal init` (recommended):**
`configure_claude_code()` should ensure hooks and MCP use the same mode. When generating `settings.json` hooks, prepend the env var to match `.mcp.json`:
```json
{
  "type": "command",
  "command": "C:\\...\\python.exe -m roampal.hooks.user_prompt_submit_hook",
  "env": { "ROAMPAL_DEV": "1" }
}
```
If Claude Code hooks support `env` blocks (needs verification), this is the cleanest fix. If not, the command can be wrapped: `cmd /c "set ROAMPAL_DEV=1 && python -m roampal.hooks.user_prompt_submit_hook"`.

**Option B — Hooks auto-detect MCP port:**
Hooks check both ports (27183 first, then 27182) and use whichever responds to `/api/health`. Simple, no config changes, but adds latency on first call if dev port is down.

**Option C — Shared config file:**
Both hooks and MCP read port from a shared file (e.g., `~/.roampal/config.json`). Single source of truth. More infrastructure but most robust.

**Immediate workaround:** Remove `"ROAMPAL_DEV": "1"` from `.mcp.json` so MCP uses prod port 27182, matching hooks. Or set `ROAMPAL_DEV=1` as a system environment variable so hooks see it too.

## Bugs Fixed During Testing

4. **OpenCode plugin path** — `cli.py` installed plugin to `~/.config/opencode/plugin/` (singular) but OpenCode loads from `plugins/` (plural). Plugin was never found. Fixed: `plugin_dir = config_dir / "plugins"`.
5. **KG deferred learning KeyError** — `knowledge_graph_service.py` `update_success_rate()` assumed routing pattern entries have `total`/`successes`/`failures`/`partials` keys, but existing entries use old schema (`collections_used`/`best_collection`). Fixed: initialize missing keys before incrementing.
6. **OpenCode session ID mismatch** — Plugin hooks use `ses_xxx` session IDs, but MCP tool calls (like `score_response`) use `mcp_xxx`. The stop hook's `scored_this_turn` check only matched the plugin session, so it always reported "record_response not called" even when the AI scored correctly. Fixed: stop hook now falls back to checking `mcp_*` sessions when the plugin session shows unscored.
7. **OpenCode scoring prompt ignored** — Scoring prompt was injected into the system prompt (via `experimental.chat.system.transform` with `output.system.unshift()`), but the AI treated it as optional guidance and consistently skipped calling `score_response`. Confirmed via metadata: all OpenCode exchanges had `last_outcome=none, score=0.5, outcome_history=none`. Fixed: scoring prompt now injected via `experimental.chat.messages.transform` — prepended to the last user message's text part (invisible in UI but AI sees it as user instruction), matching how Claude Code's UserPromptSubmit hook works. Context stays in system prompt via `system.transform`.
8. **Claude Code hooks missing ROAMPAL_DEV** — When running `roampal init --dev`, the `.mcp.json` env block got `ROAMPAL_DEV=1` but the hooks in `settings.json` didn't. Hooks run as subprocesses that don't inherit the MCP env block, causing split-brain: MCP writes to dev DB (27183), hooks write to prod DB (27182). Fixed: `cli.py` now uses centralized `_build_hook_command()` that wraps commands with env var (e.g., `cmd /c "set ROAMPAL_DEV=1 && python -m ..."` on Windows). Cursor already had this pattern; now all platforms use the same helper.
9. **Multi-session scoring mismatch** — MCP tools used a random session ID (`mcp_{uuid}`) while hooks used platform session IDs (`ses_xxx`). This caused scoring to use "most recent unscored" heuristic, which could score the wrong session's exchange when multiple windows are open. Fixed: (a) MCP now uses `"default"` session ID to trigger injection map lookup; (b) `main.py` added `_injection_map` that tracks which doc_ids were injected to which conversation; (c) `record_outcome` now resolves the correct conversation by looking up scored doc_ids in the injection map before falling back to timestamp heuristics.
10. **OpenCode plugin `export default` required** — Plugin file loaded successfully (TypeScript compiled, module imported) but the plugin function was never invoked and no hooks registered. Root cause: OpenCode requires `export default` to discover and call the plugin function. Named exports alone (`export const RoampalPlugin`) cause the file to load but the plugin entry point is never executed. Diagnosed via file-based logging (`appendFileSync`) since `console.log` from plugins is invisible in OpenCode's log files. Fixed: added `export default RoampalPlugin` to the plugin source.
11. **OpenCode event handler race condition** — `message.part.updated` events can arrive BEFORE `message.updated` events. The original code required assistant message IDs to be pre-registered via `message.updated` before accepting text parts — when parts arrived first, they were silently dropped (`NOT in assistantIds`). This caused intermittent exchange storage failures: some exchanges stored fine (when `message.updated` won the race), others were lost entirely (when `message.part.updated` arrived first). Fixed: `message.part.updated` now auto-registers the assistant message ID from `part.messageID` directly instead of requiring prior registration. The `message.updated` handler still registers IDs too (redundant but harmless).

**Note — `chat.params` toolChoice does not work:** Investigated using OpenCode's `chat.params` hook to force `toolChoice: { type: "tool", toolName: "score_response" }` via `output.options`, which would make scoring reliable across all models (not just Claude). Testing confirmed OpenCode ignores the `options` bag — `toolChoice` was set on all 3 test messages but `score_response` was only called once (by model choice, not force). OpenCode hard-codes `tool_choice: "auto"` internally. The correct fix is an upstream feature request to OpenCode to expose `toolChoice` in `chat.params`.

12. **OpenCode scoring: `scoringPromptSimple` never delivered** — `getContextFromRoampal()` had `scoringPromptSimple` in the return type but the main return path (line 196) only returned `scoringPrompt`, `contextOnly`, `injection`, `scoringRequired`. The two retry paths (503 and connection failure) included `scoringPromptSimple` but the happy path didn't. Result: `scoringPromptSimple` was always `undefined` in the cache, and `undefined || scoringPrompt` fell through to the full XML-heavy Claude prompt. Kimi was getting the wrong prompt the whole time. Fixed: added `scoringPromptSimple` to all three return paths.

### 17. Independent LLM Scoring for OpenCode

**Problem:** Non-Claude models (kimi, GPT, etc.) don't reliably follow tool-calling instructions for `score_response`. Even with simplified prompts, kimi called `get_context_insights` instead of `score_response`. The `chat.params` hook doesn't expose `toolChoice` to force it. Prompt-based scoring is unreliable for weaker models.

**Solution:** Independent LLM scoring call — same pattern as the Desktop app's `OutcomeDetector`. The plugin makes a separate, direct API call via Zen free models to score each exchange. No API key needed — uses OpenCode's built-in Zen proxy (`apiKey: "public"`), saving paid users' API credits.

**Architecture:**
```
chat.params hook (fires on every LLM call):
  └─ Captures provider details: { apiKey, baseURL, modelID, providerID, sdk }
     (from input.provider.key + input.model.api.url + input.model.api.npm)
     Falls back to provider.options.apiKey for Zen free tier ("public")

User sends message → chat.message hook fires
  ├─ Fetch context from server (existing)
  │   Server returns raw scoring data (full content, no truncation):
  │     scoring_exchange: {user: "full text", assistant: "full text"}
  │     scoring_memories: [{id: "doc_id", content: "full content"}]
  │
  ├─ Check: did main LLM call score_response LAST exchange? (mainLLMScored)
  │   YES → skip independent scoring
  │   NO  → fire scoreExchangeViaLLM() (awaited before main model):
  │          1. Always uses Zen free models (saves paid users' API credits)
  │          2. Model fallback chain: glm→kimi→gpt-5-nano (hardcoded fallback; dynamic discovery adds more at startup)
  │          3. If model returns 404/500 (removed), tries next automatically
  │          4. Prompt asks ONLY for exchange outcome (worked/failed/partial/unknown)
  │          5. All surfaced memories inherit the exchange outcome
  │          6. POST to /api/record-outcome with uniform memory_scores
  │
  └─ Cache context for system.transform + messages.transform

Main LLM responds → session.idle event fires
  ├─ Store exchange (existing)
  └─ Clear assistant tracking + cached context for next exchange
```

**Sequential scoring strategy:** The scoring call fires from `chat.message` and is **awaited** (blocking) before the main model starts. This guarantees the scoring call gets the rate limit window on free tier (Zen's limit is "one request at a time"). Adds ~2-3s latency on scoring turns only (not every turn — only when there's a previous exchange to score). A mutex (`scoringInFlight`) prevents multiple scoring calls from piling up. Consistent behavior across free tier and paid API users.

**Scoring strategy (two-tier):**
1. **Independent call (reliable fallback):** Scores exchange outcome only. All surfaced memories inherit that outcome. Over time, Wilson score interval handles noise — memories consistently surfaced in successful exchanges trend positive, irrelevant ones average out.
2. **Main LLM (precise, optional):** When the main LLM cooperates and calls `score_response` with per-memory scores, those precise scores take priority. The independent call is skipped entirely via `mainLLMScored` tracking.

**Tool invocation detection:** The `message.part.updated` event handler tracks tool invocations with `part.type === "tool-invocation"` and `part.name === "score_response"`. When detected, `mainLLMScored` is set for that session, and the independent scoring call is skipped at `session.idle`.

**API routing (simplified):**
Scoring always uses Zen free models, which are all OpenAI-compatible. Single code path: `/chat/completions` with `Bearer` auth. No SDK-adaptive routing needed (Anthropic branch removed). Handles reasoning models like `glm-4.7-free` that put output in `reasoning_content` field instead of `content`.

**Zen-always scoring (saves paid users' API credits):**
Scoring ALWAYS uses Zen free models, even for paid users. This saves API credits — the scoring call is a tiny ~300-token exchange that doesn't need a powerful model. Zen is built into every OpenCode installation.
- `zenBaseURL` defaults to `https://opencode.ai/zen/v1`, dynamically updated if "opencode" provider detected in `chat.params`
- `zenApiKey` defaults to `"public"` (Zen proxy accepts this for free-tier models)
- No user configuration needed — works identically for free and paid users
- Hardcoded fallback models: `glm-4.7-free`, `kimi-k2.5-free`, `gpt-5-nano`. Dynamic discovery at startup fetches all available free models from Zen `/models` endpoint (e.g., `minimax-m2.1-free`, `trinity-large-preview-free`)
- The main model's ID is filtered from the scoring model list to avoid rate limit collision on free tier

**Model fallback chain (resilient to Zen updates):**
The `ZEN_SCORING_MODELS` list is tried in order. On plugin load, the list is populated dynamically by fetching Zen's `/models` endpoint and filtering for models with "free" in the name. If discovery fails, a hardcoded fallback list (`glm-4.7-free`, `kimi-k2.5-free`, `gpt-5-nano`) is used. If a model returns 404 (removed by OpenCode update) or 500 (broken), the next model is tried automatically. This means OpenCode can add/remove free models without breaking scoring — as long as at least one model exists, scoring works. Each model gets up to 2 retry attempts for 429s (rate limits) with a 2-second cap before falling through to the next.

**Rate limit handling:**
Free-tier Zen proxy has tight per-model rate limits. Scoring fires from `chat.message` and is awaited before the main model starts. A mutex (`scoringInFlight`) prevents pile-ups. Model rotation ensures the scoring call uses a different model than the main conversation, avoiding rate limit collision.

**Key design decisions:**
- Awaited in `chat.message`: scoring completes BEFORE main model starts — guarantees rate limit window
- Sequential timing: ~2-3s latency on scoring turns only (free tier ~2-3s, paid API ~1-2s)
- Mutex: `scoringInFlight` flag ensures only one scoring call at a time (skip if one is already running)
- `mainLLMScored` persists across session.idle → read at NEXT chat.message, then cleared
- 6-second per-request timeout (AbortSignal.timeout), 8-second total scoring timeout (MAX_SCORING_TIME_MS)
- 429 retry (2 attempts per model, 2s backoff cap) for rate limit resilience
- Falls back gracefully if all models exhausted or scoring fails
- Always uses Zen free models (no API credits consumed, even for paid users)
- Dynamic model discovery at startup + hardcoded fallback chain: if a model is removed by OpenCode update, next one is tried
- Server returns full (untruncated) scoring data (`scoring_exchange`, `scoring_memories`)
- Independent call only asks for exchange outcome — simpler prompt, more reliable JSON parsing
- Per-memory scoring delegated to main LLM; fallback applies uniform exchange outcome to all surfaced memories

**Files changed:**
- `roampal/plugins/opencode/roampal.ts` — added `chat.params` hook for provider capture (with `model.api.url` + `options.apiKey` fallbacks for Zen + dynamic Zen URL capture), `scoreExchangeViaLLM()` function (always uses Zen free models, model fallback chain for resilience), `scoringInFlight` mutex, `mainLLMScored` tracking via tool invocation detection, scoring fires from `chat.message` awaited before main model, `pendingScoringPrompt` fallback for messages.transform (OpenCode doesn't pass sessionID)
- `roampal/server/main.py` — added `scoring_exchange` and `scoring_memories` fields to `GetContextResponse` (full content, no truncation), populated when `scoring_required=true`

### 18. Cross-Encoder Bypass for Scored Memories
**File changed:** `roampal/backend/modules/memory/search_service.py`

**Problem:** The 4-way benchmark revealed that Full Roampal (42% Top-1) performed *worse* than Outcomes Only (50% Top-1). Root cause: the cross-encoder reranker gave 40% weight to semantic similarity after Wilson scoring had already ranked results. This re-promoted semantically-close bad advice that outcome history had correctly demoted.

**Solution:** Cross-encoder weight drops to 0 for memories with `uses >= 3`. At that point, Wilson has enough outcome data to rank reliably — semantic similarity should not override learned rankings. Cold-start memories (< 3 uses) still get full CE blending since there's no outcome data yet.

```
Before (all memories):
  ce_weight = 0.4 (non-memory_bank) or 0.3 (memory_bank)
  → Wilson ranking partially overridden by semantic similarity

After (conditional):
  uses >= 3: ce_weight = 0.0 → Wilson ranking preserved
  uses < 3:  ce_weight = 0.4/0.3 → CE helps cold-start discovery
```

**Threshold consistency:** 3 uses matches all other Wilson thresholds in the system (Wilson blend in `scoring_service.py`, memory_bank quality blend in `search_service.py`).

**Expected impact:** Full Roampal should match or exceed Outcomes Only on adversarial benchmarks, since the reranker no longer undermines learned rankings. Real-world queries still benefit from CE on new memories.

**Status:** Implemented in v0.3.2.

## Usage

```bash
# Auto-detect installed tools and configure
roampal init

# Or configure explicitly
roampal init --claude-code
roampal init --cursor
roampal init --opencode

# Start the shared server (if not auto-started by MCP)
roampal start
```

## Testing

### Manual Testing

1. **Claude Code:** `roampal init` → open Claude Code → verify MCP tools + context injection
2. **Cursor:** `roampal init --cursor` → open Cursor → verify MCP tools work (hooks blocked by Cursor bug)
3. **OpenCode:** `roampal init --opencode` → open OpenCode → verify MCP tools + plugin context injection + exchange capture (check `[roampal] Stored exchange` in logs) + scoring prompts on next turn
4. **Concurrent:** Open two clients simultaneously → verify no ChromaDB errors

### Automated Tests (457 unit + 31 integration across 18+ test files)

Run: `python -m pytest roampal/backend/modules/memory/tests/unit/ -v`

**Key v0.3.2 test files (227 tests):**

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_mcp_server.py` | 59 | Helper functions, `_api_call` HTTP proxy, session ID format, port config, server lifecycle, **all 7 tool handlers** (search_memory, add_to_memory_bank, update_memory, delete_memory, score_response, record_response, get_context_insights), unknown tool, error handling, update notice |
| `test_opencode_plugin.py` | 42 | Structural validation of TypeScript plugin: exports, 3 hook handlers, 5 event types, event property paths (v0.3.2 fix), `restartServer()` self-healing, caching architecture, split delivery (`push`), port config, session state cleanup, exchange capture flow |
| `test_fastapi_endpoints.py` | 33 | All HTTP endpoints: health, get-context (split delivery fields), stop hook, search, memory-bank CRUD, record-outcome, record-response, context-insights, ingest/books |
| `test_unified_memory_system.py` | 39 | Core memory system: 4-slot context injection, memory_bank Wilson 80/20 blend (cold start, threshold, low success), **search() Wilson scoring** (proven doc outranks closer unproven, consistent with hooks path), **cross-encoder wiring** (SearchService creation, graceful fallback, delegation, reranker passthrough), collections, search, scoring, promotions, books, memory bank |
| `test_cli.py` | 31 | `configure_opencode()` (MCP config, PYTHONPATH, plugin install, idempotency), `configure_claude_code()` (hooks, permissions, MCP), `configure_cursor()` (MCP, hooks), `cmd_init()` (auto-detect, explicit flags), `is_dev_mode()`/`get_port()`/`get_data_dir()` helpers |
| `test_hooks.py` | 23 | Both hooks' stdin parsing, exit codes, context injection, self-healing (`_restart_server` kill+start+poll+timeout), transcript reading, update check caching |

**Pre-existing test files (230 tests across 12 files):** outcome_service (36), search_service (30), scoring_service (31), knowledge_graph (27), memory_bank (23), server_main (16), routing (15), context_service (13), embedding (11), exchange_scoring (11), promotion (11), schema_migration (6).

**v0.3.2 test additions:** `test_mcp_server.py` extended with 22 tool handler tests. `test_cli.py` and `test_opencode_plugin.py` are new files. `test_scoring_service.py` extended with 4 Wilson blend tests. `test_unified_memory_system.py` extended with 4 Wilson blend tests + 2 search() Wilson scoring tests (`TestSearchWilsonScoring`) + 5 cross-encoder wiring tests (`TestCrossEncoderWiring`).

## Files Changed

| File | Change |
|------|--------|
| `roampal/plugins/opencode/roampal.ts` | **NEW** - TypeScript plugin with hook-based architecture. v0.3.2 fixes: corrected event property paths (`event.properties.*`), added `message.part.updated` handler for assistant text capture, scoring prompt via `experimental.chat.messages.transform` (bug 7 — system prompt was ignored by AI), context in system prompt via `output.system.push()`, `cachedContext` Map for no-double-fetch, `restartServer()` self-healing on 503/connection failure with retry, `export default RoampalPlugin` (required for OpenCode to invoke the plugin function), race condition fix in `message.part.updated` (bug 11), `scoringPromptSimple` fix (bug 12), and **independent LLM scoring** (section 17) via `chat.params` Zen URL capture + always-Zen scoring in `chat.message` (awaited before main model). Single `/chat/completions` code path (no SDK-adaptive routing). Dynamic model discovery at startup + hardcoded fallback chain. 2-attempt 429 retry with 2s backoff cap. Deferred retry queue (`pendingScoring`) for guaranteed delivery. |
| `roampal/cli.py` | Added `configure_opencode()`, `--opencode`/`--claude-code` flags, PYTHONPATH computation. Fixed plugin directory: `plugin/` → `plugins/` (bug 4). **Bug 8 fix:** Added `_build_hook_command()` helper for centralized hook command building with ROAMPAL_DEV env var wrapping. Updated `configure_claude_code()` and `configure_cursor()` to use this helper. |
| `roampal/mcp/server.py` | **REWRITTEN** - Thin HTTP client proxying all 7 tools through FastAPI. Removed `UnifiedMemorySystem` import, added `_api_call()` helper. No more ChromaDB/PyTorch in MCP process. **Bug 9 fix:** Changed `_mcp_session_id` from random UUID to `"default"` to trigger injection map lookup for multi-session scoring. |
| `roampal/server/main.py` | Added `POST /api/record-response` + `POST /api/context-insights` endpoints, `RecordResponseRequest` + `ContextInsightsRequest` models, `metadata_filters`/`sort_by` on `SearchRequest` (wired through to `_memory.search()`), `always_inject` on `MemoryBankAddRequest`, removed dead parent monitoring code (`_is_parent_alive`, `_monitor_parent_process`). Stop hook: added MCP session fallback for `scored_this_turn` check (bug 6). **Bug 9 fix:** Added `_injection_map` for doc_id→conversation tracking, updated `get_context` to populate it, updated `record_outcome` to resolve conversation via injection map before timestamp fallback. |
| `roampal/hooks/user_prompt_submit_hook.py` | Added `_restart_server()` self-healing: on 503 or server down, kills stale process, starts fresh, retries request |
| `roampal/hooks/stop_hook.py` | Added `_restart_server()` self-healing: same pattern, maintains exit 0 on errors |
| `tests/unit/test_mcp_server.py` | Extended with 22 tool handler tests (all 7 tools + unknown + error handling) and update notice tests |
| `roampal/backend/modules/memory/unified_memory_system.py` | `get_context_for_injection()` updated: 3→4 slot allocation (reserved history slot + expanded best-match pool to all collections except books). `search()` **rewritten twice**: (1) section 11: replaced inline scoring with `ScoringService.calculate_final_score()` for Wilson + dynamic weights; (2) section 13: **delegates to `SearchService`** for hybrid search (vector + BM25), cross-encoder reranking, entity boost, and KG routing — with inline fallback. `initialize()` now creates `SearchService` + optional `CrossEncoder`. `_startup_cleanup()` calls `cleanup_old_history()` (section 14). Memory_bank uses real Wilson after 3 uses (section 10). |
| `roampal/backend/modules/memory/promotion_service.py` | Added `cleanup_old_history(max_age_hours=720.0)` — 30-day history lifecycle cleanup, same pattern as `cleanup_old_working_memory()` (section 14) |
| `roampal/backend/modules/memory/outcome_service.py` | `_deferred_learning()` batch cleanup now calls `cleanup_old_history()` alongside `cleanup_old_working_memory()` every 50 outcomes (section 14) |
| `roampal/backend/modules/memory/scoring_service.py` | `calculate_final_score()` fixed: memory_bank now blends 80% quality + 20% Wilson after 3 uses. Previously unconditionally overwrote with pure importance×confidence. |
| `tests/unit/test_cli.py` | **NEW** - 31 tests for CLI: `configure_opencode()`, `configure_claude_code()`, `configure_cursor()`, `cmd_init()`, helpers |
| `tests/unit/test_opencode_plugin.py` | **NEW** - 42 structural validation tests for TypeScript plugin |
| `tests/unit/test_scoring_service.py` | Extended with 4 memory_bank Wilson blend tests: cold start, under threshold, 80/20 after 3 uses, low success demotion |
| `tests/unit/test_unified_memory_system.py` | Extended with 4 memory_bank Wilson blend tests via `TestMemoryBankWilsonBlend` class + 2 search() Wilson scoring tests via `TestSearchWilsonScoring` class (proven doc outranks closer unproven, scoring consistent with hooks path) |
| `dev/tests/integration/test_comprehensive.py` | **FIXED + EXTENDED** — 10 API mismatches fixed (section 12). Added cross-collection search test (section 15). History test changed from persistence to 30d cleanup (section 14). 20/30 -> 31/31. |
| `dev/tests/integration/` | **NEW directory** — 13 mock-embedding integration tests moved from `dev/benchmarks/` (torture suite, edge cases, contradictions, etc.) |
| `README.md` | Updated for multi-client support: tagline, OpenCode install section, `--claude-code`/`--opencode` flags in commands, "Without Roampal" table header |
| `dev/benchmarks/README.md` | Rewritten: removed mock test descriptions (sections 1-12), kept real-embedding benchmarks only (sections 1-5), added integration test redirect |
| `dev/benchmarks/test_outcome_learning_ab.py` | Rewritten: improved A/B test methodology, cleaner scenario definitions, better mock integration (479→446 lines) |
| `dev/docs/releases/v0.2.2/RELEASE_NOTES.md` | Sanitized: personal info removed ("Logan" → "[NAME]") |
| `dev/docs/releases/v0.2.9/RELEASE_NOTES.md` | Minor heading clarification |
| `dev/docs/releases/v0.3.1/RELEASE_NOTES.md` | Sanitized: removed specific strategy doc reference |
| `roampal/backend/modules/memory/knowledge_graph_service.py` | `update_success_rate()`: initialize missing keys on old-schema routing pattern entries before incrementing (bug 5). |
| `roampal/__init__.py` | Version bump: 0.3.1 → 0.3.2 |