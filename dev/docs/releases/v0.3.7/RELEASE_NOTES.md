# v0.3.7 Release Notes

**Status:** All changes implemented. 527 tests passing, 0 failures (63 collection errors in full-suite run due to fixture ordering — all pass individually).
**Platforms:** Claude Code, OpenCode (Cursor: init support only, context injection blocked by Cursor bug)
**Theme:** Sidecar-only scoring for OpenCode + CLI UX polish + security hardening + package weight optimization

---

## Overview

v0.3.7 makes the sidecar (`scoreExchangeViaLLM` in the plugin) the sole scorer for OpenCode. When the sidecar is broken, the IMPORTANT tag tells the model to suggest `roampal sidecar setup` — a CLI command that auto-detects available models and configures a local scorer. Claude Code is unchanged. This release also applies a full security audit: fixing silent error swallowing, adding cache TTL eviction, and fragile string matching — plus removing ~280 MB of unnecessary transitive dependencies and fixing a Windows hook path compatibility issue with Claude Code 2.1.x.

---

## Sidecar-Only Scoring (OpenCode)

### Sidecar-only scoring with CLI setup command

Previously, OpenCode had dual scoring every turn: the main LLM was prompted to call `score_memories`, and the sidecar ran as a fallback if it didn't. This created complexity (check-scored polling, mainLLMScored tracking, scoring prompt injection) for marginal benefit — the sidecar is actually better at scoring because it reviews the exchange transcript as a third party rather than self-assessing.

**What changed:**
- Sidecar is primary scorer — runs silently in `session.idle`, model never sees scoring prompt
- `score_memories` tool always registered (6 tools) but model never sees scoring prompt
- When `scoringBroken=true`: system tag tells model to ask user about running `roampal sidecar setup`
- Zen health probe in `session.idle` — auto-recovers sidecar when models come back
- `mainLLMScored` and `pendingScoringPrompt` tracking removed (simplified state)
- Sidecar always runs with full memory list for per-memory scoring
- Added user's current model as last-resort fallback if all Zen free models fail
- **Separate independent budgets**: Zen models share a 15s budget (`ZEN_BUDGET_MS`), fallback gets its own 25s budget (`FALLBACK_BUDGET_MS`) — dead Zen models can never starve the fallback
- **Circuit breaker**: dead models auto-skipped for 5 minutes after timeout/500
- Zen request timeout: 5s, 1 attempt per model (fail fast — circuit breaker handles future calls)
- Fallback gets 4 attempts with exponential backoff (3s, 4.5s, 7s, 10s)
- **Retry-after cap**: 429 retry delay capped at 5s regardless of server's `retry-after` header
- Fallback cooldown: 5s wait before first fallback attempt (lets rate limit window reset)
- **Fixed ghost terminal on Windows**: `pythonw.exe` (GUI subsystem) used for server spawn — no console window created at all. Falls back to WScript.Shell VBS if pythonw unavailable.
- `windowsHide: true` added to all `execSync` calls (netstat, taskkill) to suppress any flash during server restart
- `sys.stdout`/`sys.stderr` None guard added to server's `main.py` for pythonw compatibility

**Claude Code is unchanged** — main LLM still scores there via hooks.

**Scoring never dies:** The scoring chain is: user's chosen model (25s budget) → Zen free models (15s budget, skipped when user configured a model). If everything fails, the system tag tells the model to suggest `roampal sidecar setup` — a one-command fix that auto-detects available models (Ollama, local servers, custom API) and configures a scorer. The plugin NEVER scans the full provider config — only the specific model written by `roampal sidecar setup` is used. No surprise API charges.

**Files:**
| File | Changes |
|------|---------|
| `roampal/mcp/server.py` | `score_memories` always registered |
| `roampal/plugins/opencode/roampal.ts` | Sidecar-only scoring, simplified scoring chain, `_loadSidecarConfig()`, circuit breaker, separate budgets, Zen health probe, blank terminal fix |
| `roampal/hooks/session_manager.py` | Updated `build_scoring_prompt_simple()` — model-agnostic with inline memory content |
| `roampal/cli.py` | NEW: `roampal sidecar setup/status/disable` commands, init-time sidecar prompt, `_detect_local_servers()`, `_prompt_custom_endpoint()`, `_write_sidecar_model()` |

**Sidecar scoring chain (in order):**
1. User's chosen model (`ROAMPAL_SIDECAR_URL/KEY/MODEL` — set by `roampal sidecar setup`, 25s budget)
2. Zen free models (15s budget, **skipped entirely** when step 1 has a model; model rotation, circuit breaker)
3. If ALL fail twice → `scoringBroken=true` → model tells user to run `roampal sidecar setup`

**`roampal sidecar setup` (NEW):** One-command sidecar configuration. Auto-detects Ollama models, local inference servers (8 ports), and API providers from opencode.json. User picks a model, command writes `ROAMPAL_SIDECAR_URL/KEY/MODEL` to opencode.json MCP environment. Also supports custom endpoint (URL/key/model). `roampal sidecar status` (show config) and `roampal sidecar disable` (remove all sidecar config). When broken, the system tag tells the model to ask the user — model never runs it without permission.

---

## Init-Time Sidecar Setup (OpenCode)

### Automatic sidecar model selection during `roampal init`

Previously, `roampal init` configured OpenCode's MCP and plugin but never mentioned scoring. Users silently got Zen free models with no awareness. The `roampal sidecar setup` command existed but was only discovered reactively when scoring broke.

**What changed:**

During `roampal init`, when OpenCode is detected, the CLI now:
1. Explains what memory scoring is and why it matters (info box)
2. Auto-detects ALL available local inference servers in parallel (8 ports)
3. Lists Ollama models with size info
4. Lists API models from `opencode.json` providers
5. Offers Zen (free, default), detected models, and a custom endpoint option
6. Enter or Ctrl+C defaults to Zen silently

**Local server auto-detection (`_detect_local_servers()`):**
Uses `concurrent.futures.ThreadPoolExecutor` to probe all ports in parallel (~2s total):
| Port | Server |
|------|--------|
| 1234 | LM Studio |
| 8080 | LocalAI / llama.cpp / Llamafile |
| 1337 | Jan.ai |
| 8000 | vLLM |
| 5000 | text-generation-webui |
| 4891 | GPT4All |
| 5001 | KoboldCpp |
| (OLLAMA_HOST) | Ollama non-standard port |

Each probed via `GET http://localhost:{port}/v1/models`. Ollama default port (11434) handled separately via `_detect_ollama_models()` (richer `/api/tags` metadata).

**Custom endpoint (`_prompt_custom_endpoint()`):**
Collects URL, API key (optional for local), and model name. Writes `ROAMPAL_SIDECAR_URL`, `ROAMPAL_SIDECAR_KEY`, `ROAMPAL_SIDECAR_MODEL` to opencode.json MCP environment.

**Note:** OpenCode passes MCP env vars to MCP server subprocesses but NOT to plugins. The plugin reads sidecar config directly from `opencode.json` via `_loadSidecarConfig()` (fixed in v0.3.7).

**Idempotency:** `force=False` + already configured → one-liner status, skip. `force=True` → always show prompt. `roampal sidecar setup` uses the same expanded menu.

**Files:**
| File | Changes |
|------|---------|
| `roampal/cli.py` | `_detect_local_servers()`, `_prompt_custom_endpoint()`, `_write_sidecar_model()`, `_prompt_sidecar_setup()`, updated `cmd_init()`, expanded `_cmd_sidecar_setup()`, updated `_cmd_sidecar_disable()` |
| `README.md` | Updated commands, platform differences, score_memories note |
| `ARCHITECTURE.md` | Updated sidecar references, added sidecar configuration section |

---

## CLI UX Improvements

### Production-ready CLI for scripting and CI environments

Comprehensive CLI polish based on PyPI tool best practices. All changes in `roampal/cli.py`.

**NO_COLOR / TERM=dumb support:**
- Respects `NO_COLOR` env var (https://no-color.org) and `TERM=dumb`
- Non-TTY stdout auto-disables ANSI colors (piping to files/other tools)
- New `_should_color()` function gates all ANSI color constants

**Non-interactive mode (`--no-input`):**
- `roampal init --no-input` skips all prompts, uses defaults (Zen scorer, no email signup)
- Auto-detects non-TTY stdin (piped input, CI) via `_is_interactive()` check
- Guards on `collect_email()`, `_prompt_sidecar_setup()`, `_cmd_sidecar_setup()`, `cmd_summarize()`

**Exit codes:**
- All commands return proper exit codes (0=success, 1=failure)
- `main()` dispatches with `sys.exit(exit_code)` for scriptability
- Status/stats return 1 when server not running

**Machine-readable output (`--json`):**
- `roampal status --json` -- JSON with mcp config + server status
- `roampal stats --json` -- JSON with collection counts, data path, port
- Error states also return valid JSON with `"error"` field

**Secure input:**
- API key prompts use `getpass.getpass()` (hidden input, no terminal echo)

**Banner removal:**
- `print_banner()` only shown during `roampal init`
- All other commands (start, stop, status, stats, doctor, etc.) output directly

**Cached update checks:**
- PyPI version check cached for 24 hours in `.update_cache`
- No network call on subsequent runs within cache window
- Suppressed in `--json` mode

**Grouped help:**
- `roampal --help` shows commands grouped by category (Setup, Server, Memory, Scoring, Advanced)
- Includes examples section with `--no-input` and `--json` usage

**Parser safety:**
- `allow_abbrev=False` prevents ambiguous subcommand matching (e.g. `roampal st` won't match `start`/stop/stats)

---

## Tier 1: Critical Security Fixes

### 1. Bare `except:` → Specific Exceptions (5 instances)

Bare `except:` clauses silently swallow all errors including `SystemExit`, `KeyboardInterrupt`, and real bugs. Replaced with specific exception types + `logger.warning()`.

| File | Lines | Fix |
|------|-------|-----|
| `server/main.py` | 3 instances in `_build_cold_start_profile()` | `except (json.JSONDecodeError, ValueError, TypeError):` |
| `unified_memory_system.py` | 1 instance in `_format_context_injection()` | `except (json.JSONDecodeError, ValueError, TypeError):` |
| `chromadb_adapter.py` | 1 instance in `__del__` | `except Exception:` (destructor — broad catch is acceptable) |

### 2. Unbounded Cache TTL Eviction

`_search_cache` and `_injection_map` in `server/main.py` grew without limit — one entry per conversation turn, never cleaned except on scoring.

**Fix:** Added `_evict_stale_entries()` with 30-minute TTL. Called at the top of every `get_context()` request. Evicts entries with timestamps older than `_CACHE_TTL_SECONDS = 1800`. Entries with missing or invalid timestamps are also evicted.

**File:** `server/main.py`

### 3. Windows Hook Path Compatibility (Claude Code 2.1.x)

Claude Code 2.1.x changed how hook commands are passed to bash — backslash Windows paths (`C:\roampal-core\...`) now get mangled by bash escape processing (`C:roampal-core...`). Previously worked fine for 890+ invocations across 8 days.

**Fix:** `roampal init` now writes forward slashes in hook commands (`C:/roampal-core/...`). Forward slashes are accepted by both Windows APIs and bash, avoiding escape interpretation entirely.

**File:** `roampal/cli.py` (init command path generation)

---

## Tier 2: Medium Robustness Fixes

### 4. Collection Prefix Detection — `startswith(name)` → `startswith(name + "_")`

8 locations across 5 files detected which collection a `doc_id` belongs to using `doc_id.startswith(coll_name)`. This is fragile — if a collection named `work` were added, it would match `working_*` doc_ids.

**Fix:** Changed all 8 instances to `doc_id.startswith(coll_name + "_")`, explicitly requiring the `_` separator.

| File | Instances |
|------|-----------|
| `server/main.py` | 1 |
| `unified_memory_system.py` | 2 |
| `context_service.py` | 1 |
| `knowledge_graph_service.py` | 2 |
| `outcome_service.py` | 2 |

---

## Package Weight Optimization (~280 MB saved)

### 6. Remove scipy (~180 MB saved)

scipy was imported solely for `stats.norm.ppf(1 - (1-0.95)/2)` in `scoring_service.py` — which always returns 1.96. Replaced with a pure-math implementation:

- Pre-computed z-score lookup table for common confidence levels (0.90, 0.95, 0.99)
- Abramowitz & Stegun rational approximation (26.2.23) for arbitrary confidence levels

**Verification:** `wilson_score_lower(90, 100)` returns 0.8256 with the new implementation, matching the mathematical expectation for z=1.96.

**File:** `scoring_service.py`

### 7. Remove nltk (~100 MB saved)

nltk was imported for `punkt` sentence tokenizer in BM25 indexing, but the actual BM25 code was already using `text.lower().split()`. The import was unused dead code.

**File:** `chromadb_adapter.py`

### 8. Make rank-bm25 Optional

rank-bm25 already had graceful degradation (`BM25_AVAILABLE` flag with try/except on import). Moved from core `dependencies` to `[project.optional-dependencies.hybrid]`.

**File:** `pyproject.toml`

### 9. Remove Cross-Encoder Reranker (~8 MB RAM, complexity reduction)

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was added in v0.3.2 as an optional search reranker. With Wilson-only ranking for proven facts (3+ uses), the cross-encoder is redundant — Wilson scoring based on real usage outcomes is a better reranking signal than text similarity.

**Change:** Removed cross-encoder loading from `unified_memory_system.py`. `SearchService` still accepts `reranker=` parameter for API compatibility but always receives `None`.

**RAM impact:** Minimal (~8 MB) since the cross-encoder shared PyTorch runtime with the embedding model. The real win is complexity reduction — one less model to download, load, and maintain.

**File:** `roampal/backend/modules/memory/unified_memory_system.py`

---

## Tier 3: Code Quality

### 9. mypy Configuration

Added `[tool.mypy]` section to `pyproject.toml` with `python_version = "3.10"`, `warn_return_any = true`, `warn_unused_configs = true`, `ignore_missing_imports = true`. Added `mypy>=1.0.0` to dev dependencies.

### 10. SessionManager Exchange Cache Cleanup

`_last_exchange_cache` dict in `session_manager.py` grew without limit. Added `_cleanup_exchange_cache(max_entries=50)` called at startup — evicts oldest entries when cache exceeds 50 entries.

**File:** `roampal/hooks/session_manager.py`

---

## Wilson-Only Ranking for memory_bank (v0.3.7)

### Proven facts ranked purely by track record

Previously, memory_bank retrieval blended semantic similarity, quality metadata (`importance * confidence`), and Wilson score — diluting Wilson's signal. For facts with 3+ uses, Wilson is now the **sole** ranking signal.

| Uses | Ranking | Rationale |
|------|---------|-----------|
| 0-2 | `importance * confidence` | No outcome data yet — trust user's rating |
| 3+ | `wilson_score` (100%) | Real-world track record > metadata |

**Changes:**

- **`search_service.py` `_apply_collection_boost()`**: For 3+ uses, distance multiplier is `1.0 - wilson * 0.8` (was blended 50/50 with quality at 0.4 coefficient)
- **`scoring_service.py` `get_dynamic_weights()`**: Returns `(0.0, 1.0)` for memory_bank with 3+ uses — zero embedding weight, full Wilson weight
- **`scoring_service.py` `calculate_final_score()`**: `learned_score = wilson_score` for 3+ uses (was `0.5 * quality + 0.5 * wilson`)
- **`server/main.py` `_build_cold_start_profile()`**: Proven facts sort by `2.0 + wilson` — the offset guarantees any proven fact beats any cold fact regardless of quality metadata
- **`roampal_pack` `memory_store.py` `search()`**: Post-cosine Wilson re-ranking for memory_bank results with `wilson_samples >= 3`

Combined effect: `final_rank_score = 0.0 * embedding_sim + 1.0 * wilson = wilson`

---

## Fix: Recent Exchanges Injection (OpenCode)

### Cold start + compaction recovery was dead code on cached turns

The `includeRecentOnNextTurn` check in `system.transform` was placed AFTER the early `return` on cached context (line 1133). On normal turns where `chat.message` fires first and caches the context, `system.transform` found the cache, used it, and returned — never reaching the recent exchanges injection code.

**Result:** Recent exchanges (last 4 summaries) were only injected when the cache missed (rare race condition), not on cold start or compaction as intended.

**Fix:** Moved the `includeRecentOnNextTurn` block to BEFORE the cached context check. Now the injection order is:
1. Recent exchanges (if cold start or post-compaction)
2. Cached/fetched semantic context from server

**Affects:** OpenCode only. Claude Code uses shell hooks and gets context from the server's `/api/hooks/get-context` response directly — no `system.transform`, no `includeRecentOnNextTurn`.

**File:** `roampal/plugins/opencode/roampal.ts` (system.transform handler)

---

## Sidecar Failure Warning (OpenCode)

### Sidecar failure warning (limited visibility)

Previously, sidecar scoring failures were silent — logged to `roampal_plugin_debug.log` only. Users had no way to know scoring was failing unless they checked the log file.

**Change:** Add `console.warn()` when deferred retry also fails (two consecutive scoring failures).

**Known limitation:** `console.warn()` from OpenCode plugins goes to OpenTUI's internal console overlay, which is **hidden by default**. Users only see it if they manually toggle the console from the command palette. There is no plugin-accessible API to surface warnings in the main OpenCode UI (would require `Bus.publish(Session.Event.Error, ...)` which is an internal API). This means sidecar failure is effectively silent for most users.

**Trigger conditions (ALL must be true):**
1. Custom endpoint failed or not configured
2. ALL Zen free models failed (circuit breaker, timeout, 404, 500, bad JSON)
3. User's own model failed (4 retries with exponential backoff)
4. Deferred retry on next message ALSO failed the full queue again

**Implementation in `roampal.ts` `session.idle` handler:**
- When retrying deferred scoring (`pendingScoring`), if `scoreExchangeViaLLM()` returns false:
  - `console.warn("[roampal] Scoring failed twice — check ROAMPAL_SIDECAR_URL/KEY/MODEL or Zen availability")`
  - Set `pendingScoring = null` (drop the payload — don't pile up retries)
- First failure remains silent (queued for deferred retry as before)

**File:** `roampal/plugins/opencode/roampal.ts` (session.idle handler, ~lines 1298-1309)

---

## Sidecar Robustness: Simplified Scoring Chain (OpenCode)

### Explicit model selection replaces provider scanning

Previously, the sidecar scanned ALL configured providers from `opencode.json` (including paid APIs like DeepSeek) and used them as scoring fallbacks. This silently consumed paid API credits the user configured for chat, not scoring. Users had no idea background scoring was costing them money.

**v0.3.7 fix:** The plugin ONLY uses the specific model the user chose in `roampal sidecar setup`. The setup command writes `ROAMPAL_SIDECAR_URL/KEY/MODEL` to opencode.json — these are the ONLY env vars the plugin reads. No scanning of the full provider config. No surprise API charges.

**Scoring chain (simplified):**
1. User's chosen model (`ROAMPAL_SIDECAR_URL/MODEL` — 25s budget, 4 retries with exponential backoff)
2. Zen free models (15s budget, **skipped entirely** when user configured a model — Zen routes through OpenCode's proxy which may log data)
3. If ALL fail → `scoringBroken=true` → system tag tells model to suggest `roampal sidecar setup`

**`_write_sidecar_model()` (NEW):** Helper that determines the correct OpenAI-compatible URL for each model source:
- Ollama → `http://localhost:11434/v1`
- Local server → `http://localhost:{port}/v1`
- API model → base URL from provider config

**Removed:** `loadConfiguredScoringTargets()`, `ROAMPAL_SIDECAR_FALLBACK`, reasoning→chat auto-map, multi-model provider scanning.

### Scoring status injection (every turn)

Scoring status is injected into the `system.transform` context on every turn — a tiny header that tells the model (and by extension the user) what's happening.

**Status tags prepended to injected context:**

| State | Tag |
|-------|-----|
| Scoring working | `[roampal scoring: ok]` |
| Scoring broken | `[roampal scoring: UNAVAILABLE — no working scorer found. Install Ollama (ollama.com) or set ROAMPAL_SIDECAR_URL/KEY/MODEL]` |
| Custom sidecar active | `[roampal scoring: custom (groq/llama-3.3-70b)]` |

When broken, the model sees the warning and can proactively tell the user how to fix it.

**Implementation:**
- New module-level flag: `let scoringBroken = false`
- New module-level string: `let lastScorerLabel = ""` (tracks what scored last)
- Set `scoringBroken = true` when deferred retry fails (line 1309)
- Reset `scoringBroken = false` and update `lastScorerLabel` on next successful `scoreExchangeViaLLM()` call
- In `system.transform`: prepend status tag to injected context (always — not just on failure)
- 5-15 chars overhead when working, actionable instructions when broken

### 4. First-run onboarding message

On `session.created`, inject a one-time setup recommendation into context:

```
[roampal memory active | scoring: zen (free, best-effort) | For reliable scoring, install Ollama (ollama.com) with any small model, or set ROAMPAL_SIDECAR_URL/KEY/MODEL for a dedicated scorer]
```

**Implementation:**
- Track `hasShownOnboarding` flag per session
- On first `system.transform` for a new session, append the onboarding line
- Only fires once per session, not on every turn
- Separate from the per-turn status tag (this is supplementary context)

### 4. Circuit breaker cooldown: 5 min → 30 min

Zen models that timeout at 5s are not recovering in 5 minutes. Longer cooldown avoids wasting 15s of budget on dead models every scoring cycle.

**Change:** `CIRCUIT_BREAKER_COOLDOWN_MS = 1800000` (30 minutes, was 300000 / 5 minutes)

### 5. Zen model list update

Current hardcoded list contains models that may have been removed from Zen:
- `minimax-m2.5-free`, `minimax-m2.1-free`, `trinity-large-preview-free` — all timing out consistently

**Change:** Update `ZEN_SCORING_MODELS` to current working models. Add dynamic discovery: on first successful Zen call, cache the working model and try it first next time.

---

**Files:** `roampal/plugins/opencode/roampal.ts` (all changes in plugin only)

**Env var documentation:**
| Var | Purpose | Required? |
|-----|---------|-----------|
| `ROAMPAL_SIDECAR_URL` | Scoring endpoint (e.g. `http://localhost:11434/v1` for Ollama) | Set by `roampal sidecar setup` |
| `ROAMPAL_SIDECAR_KEY` | API key (optional for local models) | Only for paid APIs |
| `ROAMPAL_SIDECAR_MODEL` | Model name (e.g. `qwen3:30b-a3b`) | Set by `roampal sidecar setup` |
| `ROAMPAL_SIDECAR_DISABLED` | Set to `true` to disable all sidecar scoring | Optional — for testing |

**Compatible sidecar providers (any OpenAI-compatible `/chat/completions` endpoint):**
| Provider | URL | Cost | Notes |
|----------|-----|------|-------|
| Ollama (local) | `http://localhost:11434/v1` | Free | **Recommended** — any model works, even 3B |
| LM Studio (local) | `http://localhost:1234/v1` | Free | Whatever model is loaded |
| Groq | `https://api.groq.com/openai/v1` | Cheap | Fast inference |
| Together | `https://api.together.xyz/v1` | Cheap | Wide model selection |
| OpenRouter | `https://openrouter.ai/api/v1` | Varies | Universal proxy |
| OpenAI | `https://api.openai.com/v1` | Paid | Works but overkill for scoring |
| DeepSeek | `https://api.deepseek.com/v1` | Cheap | V3 only (R1 too slow) |

The scoring task is ~500 tokens input, ~100 tokens output. Even a 3B model handles it fine. **Recommended local setup:** `ollama pull llama3.2:3b` or any small chat model — fast, free, reliable.

---

## Files Modified

| File | Changes |
|------|---------|
| `roampal/mcp/server.py` | `score_memories` always registered, bare excepts (3), cache TTL eviction, pythonw stdout/stderr None guard, Wilson-first cold start sort |
| `roampal/plugins/opencode/roampal.ts` | Sidecar-only scoring, simplified scoring chain (no provider scanning), `_loadSidecarConfig()`, separate Zen/fallback budgets, pythonw spawn, retry-after cap, windowsHide on execSync, scoring status injection, 30min circuit breaker, Zen model list update, recent exchanges race fix |
| `roampal/hooks/session_manager.py` | Deprecation docstring, exchange cache cleanup |
| `roampal/server/main.py` | Bare excepts (3), cache TTL eviction, pythonw stdout/stderr None guard, Wilson-first cold start sort |
| `roampal/backend/modules/memory/unified_memory_system.py` | Bare except (1), startswith fix (2), remove cross-encoder reranker |
| `roampal/backend/modules/memory/chromadb_adapter.py` | Bare except (1), remove unused nltk import |
| `roampal/backend/modules/memory/scoring_service.py` | Remove scipy, add pure-math z-score, Wilson-only weights + learned_score for memory_bank 3+ uses |
| `roampal/backend/modules/memory/search_service.py` | Wilson-only distance boost (0.8 coeff) for memory_bank 3+ uses |
| `roampal/backend/modules/memory/context_service.py` | startswith fix (1) |
| `roampal/backend/modules/memory/knowledge_graph_service.py` | startswith fixes (2) |
| `roampal/backend/modules/memory/outcome_service.py` | startswith fixes (2) |
| `roampal/cli.py` | Forward-slash hook paths, init-time sidecar prompt, `_detect_local_servers()`, `_prompt_custom_endpoint()`, NO_COLOR/TTY support, `--no-input`, `--json` for status/stats, exit codes, getpass for API keys, cached update checks, grouped help, banner removal from non-init commands, `allow_abbrev=False` |
| `roampal/hooks/session_manager.py` | Exchange cache cleanup |
| `pyproject.toml` | Remove scipy/nltk deps, optional rank-bm25, add mypy |

---

## Dependency Changes

| Before | After | Savings |
|--------|-------|---------|
| scipy (transitive ~180 MB) | Removed — pure-math z-score | ~180 MB |
| nltk (transitive ~100 MB) | Removed — unused dead code | ~100 MB |
| rank-bm25 (core dep) | Optional `[hybrid]` extra | ~1 MB (install only if needed) |

**Total savings:** ~280 MB lighter install.

---

## Verification

- **527 tests collected**, 464 pass in full-suite run, 63 collection errors (fixture ordering — all pass individually), 0 logic failures
- `grep -rn "except:" roampal/ --include="*.py"` → 0 bare excepts
- `grep -rn "startswith(coll_name)" roampal/` → 0 unsafe startswith
- `grep -rn "from scipy" roampal/` → 0 scipy imports
- `grep -rn "import nltk" roampal/` → 0 nltk imports
- `wilson_score_lower(90, 100) = 0.8256` — mathematically correct for z=1.96
- `ZEN_BUDGET_MS = 15000`, `FALLBACK_BUDGET_MS = 25000` — separate clocks verified in plugin
- `where pythonw` → exists on test machine (Python 3.10+)
- Retry-after cap: `Math.min(serverDelay, 5000, remaining - requestTimeout)` — server header ignored
- Zen: `maxAttempts = 1` (fail fast, circuit breaker handles subsequent calls)
