# Roampal Core v0.5.4 -- 2026-04-23

**Release Date:** 2026-04-23
**Type:** Minor release. Originally drafted as v0.5.3.2 hotfix; promoted to
v0.5.4 once the change scope grew to include an architectural unification
with Desktop's tag-extraction wiring (not just a hotfix anymore). Changes:

## Profile binding fix (issue #7)

**Symptom:** OpenCode Desktop ignores ROAMPAL_PROFILE set in per-project
opencode.json. The CLI respects it; the Desktop plugin does not.
@marcusyoung diagnosed that Desktop correctly passes the env var to the MCP subprocess,
but when a new MCP joins an existing FastAPI server on port 27182, the
server keeps serving the profile it was started with.

**Root cause:** The FastAPI singleton bound profile_name once at lifespan
startup via active_profile_name() and never re-resolved it. All requests
served that single profile regardless of which project or environment
triggered them.

**What changed:** Replaced the process-scoped _memory / _session_manager
singletons with a per-profile registry (_memory_by_profile,
_session_manager_by_profile). Each handler resolves its profile at request
time via an X-Roampal-Profile header or falls back to active_profile_name().
Lazy initialization under an async lock prevents double-init on concurrent
first requests. A shared EmbeddingService singleton keeps ONNX model memory
flat (~420MB total regardless of profile count).

### Cross-client header propagation

The FastAPI registry only helps if every client actually sends the
X-Roampal-Profile header. Three independent code paths reach FastAPI: the
MCP server (used by Claude Code, Cursor for tool calls), the OpenCode
plugin (used by OpenCode for hook + sidecar context), and the
Claude Code / Cursor Python hooks (used for context injection and
exchange capture). All three needed the same treatment:

- MCP server (roampal/mcp/server.py): caches the resolved profile name
  at startup, attaches the header on every HTTP call to FastAPI. Header
  is omitted for the default profile so the FastAPI fallback to
  active_profile_name() preserves existing CLI behavior.
- OpenCode plugin (roampal/plugins/opencode/roampal.ts): added
  resolveRoampalProfile() that reads from process.env.ROAMPAL_PROFILE
  first, then walks to the project's opencode.json for
  mcp.roampal-core.environment.ROAMPAL_PROFILE, then to the user-global
  opencode.json. roampalHeaders() builder is attached to all 11 FastAPI
  fetch sites. The opencode.json read covers the per-project Desktop
  scenario (issue #7) where the plugin process does not inherit the
  per-project env block but can read it from the file.
- Python hooks (roampal/hooks/user_prompt_submit_hook.py and
  roampal/hooks/stop_hook.py): added _roampal_headers() helper using
  active_profile_name() (env > persisted file > default). Attached to
  all 4 POST callsites (initial + retry-after-503 in each hook).
  Covers Claude Code and Cursor since both invoke the same Python
  hook scripts.

Without these patches, the FastAPI singleton kill alone would only
have helped MCP-tool callers. OpenCode hooks and Claude Code / Cursor
hooks would still hit the default profile.

### Files touched

- roampal/server/main.py - replaced singleton with per-profile registry,
  added get_memory_for_request() / get_session_manager_for_request()
  helpers, get_profile_context() pair helper, moved TagService wiring and
  archived-memory cleanup into lazy init path, updated all ~30 handler
  callsites. (~180 lines changed)
- roampal/backend/modules/memory/unified_memory_system.py - added optional
  embed_service constructor parameter so multiple instances can share one
  ONNX model. (~5 lines changed)
- roampal/mcp/server.py - cache resolved profile name with sentinel
  pattern, attach X-Roampal-Profile header on every HTTP call to FastAPI.
  Omit header for default profile. (~25 lines changed)
- roampal/plugins/opencode/roampal.ts - added resolveRoampalProfile()
  helper (env > project opencode.json > global opencode.json) and
  roampalHeaders() builder. Attached profile header to all 11 FastAPI
  fetch sites. (~80 lines changed)
- roampal/hooks/user_prompt_submit_hook.py - added _roampal_headers()
  helper using active_profile_name(); attached to both POST callsites.
  (~20 lines changed)
- roampal/hooks/stop_hook.py - added _roampal_headers() helper; attached
  to both POST callsites. (~20 lines changed)

### Validation

Live-validated end-to-end during this release cycle:

- OpenCode CLI: ghost and research profiles loaded simultaneously via
  $env:ROAMPAL_PROFILE = "..."; opencode in separate PowerShell sessions.
  Both routed to their own profile, no cross-contamination, default
  profile untouched. /api/health confirmed profiles_loaded growing as
  new profiles hit the server.
- Claude Code: ghost profile via $env:ROAMPAL_PROFILE = "ghost"; claude.
  Both the hook context-injection path and the MCP tool-call path
  resolved to ghost. Confirmed by the model itself reporting "Roampal
  core has no stored identity for you" while local Claude Code
  auto-memory (CLAUDE.md) remained available, demonstrating the two
  systems are correctly isolated.
- Manual curl with X-Roampal-Profile header confirmed FastAPI side.

Test suite: 603 passing, 3 skipped, 0 failed. The fixture in
test_fastapi_endpoints.py was updated to pre-populate _memory_by_profile
under the "default" key (mirrors the request-resolution path), and the
two "503 not ready" tests were rewritten to patch get_memory_for_request
/ get_session_manager_for_request directly. test_server_main.py's
cold-start test was updated to call _build_cold_start_profile(mem)
explicitly since the function now takes mem as a parameter.

Pending: end-to-end test of OpenCode Desktop with per-project
opencode.json containing mcp.roampal-core.environment.ROAMPAL_PROFILE
(issue #7's exact repro). The plugin code reads project opencode.json
at priority 2 of resolveRoampalProfile(), but that path was not
exercised in this session.

### Bug fixes (from internal audit pass)

- Per-profile init failures now return 503 (HTTPException) instead of
  killing the entire FastAPI process via SystemExit. Failure of one
  profile no longer takes down all other clients.
- Removed dead _deferred_action_kg_updates background task that
  referenced methods deleted in v0.4.5 (detect_context_type,
  record_action_outcome, _update_kg_routing). Was fire-and-forget so
  errors got swallowed, but every invocation polluted logs and
  silently dropped action KG updates.
- Lifespan embedder init catches all exceptions, not just ImportError.
  Model download failure or disk-full no longer crashes startup.
- MCP profile resolution caches the default-profile case correctly via
  a sentinel object (was re-resolving every call previously).
- Fixed effective_summary_tags NameError in /api/record-outcome that
  hit when a request carried facts but no exchange_summary. Caught live
  in v0.5.4 server log during OpenCode testing; initialized the
  variable to request.noun_tags or None before the summary block.
- Plugin handlers receiving a Pydantic body now also accept a separate
  Request parameter for header access. Without this, all 7 affected
  handlers (get_context, stop_hook, search_memory,
  add_to_memory_bank, update_memory_bank, delete_memory_bank,
  ingest_document) would AttributeError on the first request.
- Bumped 3 background scoreExchange fetch timeouts in roampal.ts from
  5s to 30s (summary store + 2 memory-search calls). Server-side tag
  extraction via the sidecar can take 10-30s on local models, which
  caused the plugin to log a misleading "summary store error:
  TimeoutError" while the data was actually still being stored
  server-side (FastAPI completes the handler regardless of client
  disconnect). These fetches fire on session.idle, after the user
  already has their response, so the bump has no user-facing latency
  cost. Cross-referenced 96 plugin-stored exchange_summary entries in
  chromadb with corresponding TimeoutError log lines to confirm the
  data path was always intact.

---


- Fix server-side `noun_tags` extraction silently returning empty for all
  OpenCode-stored memories. v0.5.3 section 11 wired the fallback in
  `/api/hooks/stop` and `/record-outcome`, but the FastAPI server is
  launched independently of the OpenCode plugin process and never
  inherited `ROAMPAL_SIDECAR_URL` / `MODEL` / `KEY` from
  `opencode.json`'s mcp.roampal-core.environment block. Server-side
  `extract_tags()` saw `CUSTOM_URL=""` and returned `None`. Every
  exchange stored via the OpenCode plugin landed without `noun_tags`.

- Performance: `/api/summarize` was making 3-4 sidecar LLM calls per request
  (one for exchange summary + one per fact). Now extracts tags once at the
  exchange level and reuses them for facts. Reduces to 1 sidecar call/request.

- Architectural unification with Desktop. Core's tag-extraction path now
  mirrors Desktop's: `TagService.set_llm_extract_fn()` is wired at FastAPI
  lifespan startup with a sidecar-backed closure, and `store_working`
  auto-extracts tags via `TagService.extract_tags_async` when the caller
  passes none. Removes per-handler `sidecar_extract_tags()` calls and the
  bypass of TagService that had grown up around `/api/summarize`.

- Client locking for tag extraction. Added `asyncio.Lock` to
  `TagService.extract_tags_async` so concurrent requests don't blast the
  sidecar simultaneously, which crashes local models with GPU OOM. Mirrors
  Desktop's client-lock pattern in `execute_with_client_lock`. Without this,
  a single `/api/summarize` request with multiple facts could trigger parallel
  tag extraction calls that overload the sidecar.

- Facts extraction retry loop in OpenCode plugin. The plugin's facts call
  in `roampal.ts` was a single-shot fetch wrapped in a try/catch - if the
  local model returned malformed JSON, hit a parse error, or timed out, the
  failure logged as `scoreExchange FACTS error (non-fatal)` and the facts
  for that exchange were silently dropped. Tags (server-side `_call_llm`)
  and summaries (`scoreExchange` loop) already had retries; facts didn't.
  Added a 3-attempt loop with exponential backoff (1s, 2s) for the fallback
  sidecar path, matching the summary retry policy. Treats HTTP errors,
  parse failures, and fetch exceptions as retryable.

- Fix `effective_summary_tags` NameError in `/api/summarize` and
  `/api/record-outcome`. The architectural-unification refactor moved the
  variable's definition inside the `if request.exchange_summary and _memory:`
  block, but the facts loop below it reads the same variable. Requests
  carrying facts but no exchange_summary (e.g. the plugin's `/record-outcome`
  POST that ships extracted facts) hit `local variable
  'effective_summary_tags' referenced before assignment` and every fact in
  that request failed to store. Initialized to `request.noun_tags or None`
  before the summary block so the facts loop always has a defined value.
  Caught live in the running v0.5.4 server log during OpenCode testing.

## Summary

Four changes, one shared root cause: Core's tag-extraction wiring had
diverged from Desktop's. Desktop wires the LLM extractor into TagService
at startup and lets the storage layer do extraction; Core was wiring at
construction time (capturing empty sidecar config) and routing extraction
through every FastAPI handler. The unification fix makes Core match
Desktop's pattern exactly, including client locking to prevent GPU OOM on
local models under concurrent load.

## Implementation

### `roampal/server/main.py` - `_hydrate_sidecar_from_opencode_config()`

Added a helper called from the FastAPI `lifespan` startup. Mirrors
`cli.py:_check_sidecar_configured`'s read pattern:

```python
def _hydrate_sidecar_from_opencode_config():
    if os.environ.get("ROAMPAL_SIDECAR_URL") and os.environ.get("ROAMPAL_SIDECAR_MODEL"):
        return  # already configured via real env

    config_path = (
        Path.home() / ".config" / "opencode" / "opencode.json"
        if sys.platform == "win32"
        else Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
             / "opencode" / "opencode.json"
    )
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    env_block = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})
    url = env_block.get("ROAMPAL_SIDECAR_URL", "")
    model = env_block.get("ROAMPAL_SIDECAR_MODEL", "")
    key = env_block.get("ROAMPAL_SIDECAR_KEY", "")
    if not (url and model):
        return

    os.environ.setdefault("ROAMPAL_SIDECAR_URL", url)
    os.environ.setdefault("ROAMPAL_SIDECAR_MODEL", model)
    if key:
        os.environ.setdefault("ROAMPAL_SIDECAR_KEY", key)

    # Re-bind module globals - sidecar_service read these at import time.
    import roampal.sidecar_service as svc
    svc.CUSTOM_URL = url
    svc.CUSTOM_MODEL = model
    if key:
        svc.CUSTOM_KEY = key
    logger.info(f"Hydrated sidecar config from opencode.json: model={model} url={url}")
```

Called near the top of `lifespan()` so the sidecar is configured before
any request handler tries to call `extract_tags()`.

### Why module globals need explicit re-binding

`roampal/sidecar_service.py:31-33` reads `os.environ.get("ROAMPAL_SIDECAR_URL", ...)`
at module-import time, not per-call. Once the module is loaded, those
values are frozen. Setting `os.environ` after import doesn't propagate.
The hydration explicitly assigns `svc.CUSTOM_URL = url` etc. to update
the live module state.

### `roampal/server/main.py` - Reduce sidecar calls in `/api/summarize`

Before: extract tags from exchange_summary (1 call), then loop over facts
calling `sidecar_extract_tags(fact_text)` for each fact (N more calls).
A single summarize request with 3 facts = 4 LLM API calls.

After: extract tags once from exchange_summary, store in `effective_summary_tags`,
reuse that list for every fact. Per-fact extraction now happens - when needed -
via `store_working`'s auto-extract path (one shared TagService call site)
instead of in the handler. Reduces 3-4 calls -> 1 call per request.

All tag extraction uses LLM via sidecar - no regex fallback in these paths.

### Architectural unification with Desktop

Before this release, Core had three wiring divergences from Desktop:

1. **Core's `TagService` had no `set_llm_extract_fn` setter.** The extractor
   was bound at construction in `UnifiedMemorySystem.initialize()`, before
   sidecar config was hydrated. Late re-wiring wasn't possible.
2. **Core's `TagService` had no `extract_tags_async`.** Async callers had
   to either block the event loop on the sync path or duplicate sidecar
   calls in their own handlers.
3. **`store_working` did not auto-extract tags.** Every FastAPI handler
   that wanted tags had to call `sidecar_service.extract_tags()` itself
   and pass the result through `noun_tags=`. Desktop's
   `store_memory_bank` already followed the auto-extract pattern
   (`unified_memory_system.py:924-925`).

This release closes all three:

#### `roampal/backend/modules/memory/tag_service.py`

Added `set_llm_extract_fn(fn)` setter (mirrors Desktop `tag_service.py:376`).
Added `extract_tags_async(text)` that offloads sync extractors to a
thread via `asyncio.to_thread` so the event loop isn't blocked during the
sidecar round-trip (~10s on local models).

#### `roampal/utils/sidecar_tag_wrapper.py` (new)

Mirrors Desktop's `utils/sidecar_tag_wrapper.py`. `make_llm_tag_extractor()`
returns a sync closure that reads `sidecar_service.CUSTOM_URL` /
`CUSTOM_MODEL` at CALL time and degrades cleanly to `None` when sidecar
isn't configured (no exceptions surface to TagService).

#### `roampal/server/main.py` - `lifespan` wiring

After `_memory.initialize()`, the lifespan calls
`_memory._tag_service.set_llm_extract_fn(make_llm_tag_extractor())`. This
mirrors Desktop `main.py:640`: late wiring after sidecar config is
guaranteed present (because hydration ran first).

#### `roampal/backend/modules/memory/unified_memory_system.py` - `store_working`

When `noun_tags` is not provided by the caller, `store_working` now
calls `_tag_service.extract_tags_async(content)` and stores the result on
the metadata. Mirrors Desktop's `store_memory_bank` pattern. Handler-level
tag extraction is no longer required - though `/api/summarize` keeps the
single exchange-level extraction call so it can pass the same tag set to
every fact in the loop (preserves the 1-call/request perf characteristic).

#### Net effect

`/api/summarize` no longer imports `sidecar_extract_tags` directly. Tag
extraction is owned by `TagService` (wired once at startup), invoked by
`store_working` (or, for the perf optimization, once at exchange level by
the handler). Same architecture as Desktop, same call-time semantics, same
single point of failure if sidecar is misconfigured.

### Client locking - prevent GPU OOM on local models

Desktop uses `execute_with_client_lock` to serialize concurrent sidecar calls,
preventing multiple requests from hitting a local model simultaneously and
crashing it with GPU out-of-memory errors. Core had no equivalent - any
concurrent `/api/summarize` request could trigger parallel tag extraction
calls that overload the sidecar.

Added `asyncio.Lock` (`_sidecar_lock`) to `TagService`. The lock wraps the
entire LLM call (including thread offload) in `extract_tags_async`, ensuring
only one tag extraction runs at a time across all async callers. This matches
Desktop's serialization approach without requiring a full retry queue
infrastructure (Core already has 3x HTTP-level retries in `_call_llm`).

```python
# In TagService.__init__:
self._sidecar_lock = asyncio.Lock()

# In extract_tags_async:
async with self._sidecar_lock:
    if inspect.iscoroutinefunction(self._llm_extract_fn):
        tags = await self._llm_extract_fn(text)
    else:
        tags = await asyncio.to_thread(self._llm_extract_fn, text)
```

### Facts extraction retry loop (OpenCode plugin)

The plugin's facts extraction at `roampal/plugins/opencode/roampal.ts:927-998`
was a single-shot `await fetch(...)` wrapped in a try/catch. If the local
model returned malformed JSON, the response failed `extractJson` parsing,
or the fetch timed out, the failure logged as `scoreExchange FACTS error
(non-fatal)` and the facts for that exchange were silently dropped - no
retry. The plugin debug log shows several of these failures over the
course of one session:

```
[19:52:33] scoreExchange FACTS error (non-fatal): SyntaxError: Unexpected token '.'
[19:56:53] scoreExchange FACTS error (non-fatal): SyntaxError: Unrecognized token '`'
[22:14:02] scoreExchange FACTS error (non-fatal): TimeoutError: The operation timed out.
```

Compare to:
- **Summary** path: `roampal.ts:661-711` already has `maxAttempts = 4` for
  fallback sidecars with exponential backoff (3s, 4.5s, 7s, 10s).
- **Tags** path: server-side via `sidecar_service._call_llm` already has
  `for attempt in range(3):` for the custom-endpoint path.

Facts was the only one of the three lanes without retries. v0.5.4 adds a
3-attempt loop for the fallback sidecar (1 attempt for Zen, matching
fail-fast policy). HTTP errors (4xx/5xx), parse failures (extractJson
returning null), and fetch exceptions (timeout) are all retryable. Backoff
is `1000 * Math.pow(2, attempt)` -> 1s, 2s. Target failover is not in scope
here - by the time the facts call fires, the summary call has already
succeeded on this target so target health is verified.

Result: transient sidecar response failures no longer drop facts on the
floor.

### Decorator placement fix

The hydration helper was originally inserted between `@asynccontextmanager`
and `async def lifespan(...)`, which silently rebound the decorator onto
the helper instead of `lifespan`. Moved `@asynccontextmanager` back above
`lifespan` (where it belongs) so FastAPI's lifespan wiring works and the
hydration helper runs as a plain function call.

## Tests

`roampal/backend/modules/memory/tests/unit/test_sidecar_hydration.py`,
7 tests covering:

- Happy path: opencode.json has full sidecar config -> globals populated
- Pre-existing env vars take precedence (no clobber)
- Missing opencode.json -> silent no-op
- Malformed opencode.json -> silent no-op (no raise)
- Missing roampal-core mcp block -> no-op
- Partial config (URL without MODEL) -> no-op (treats as no-config)
- Optional KEY missing -> URL/MODEL still hydrated, KEY stays empty

All 7 pass on Windows + Python 3.10.

The new `TagService.set_llm_extract_fn`, `TagService.extract_tags_async`,
and the wrapper closure were import-checked via:

```
from roampal.utils.sidecar_tag_wrapper import make_llm_tag_extractor
from roampal.backend.modules.memory.tag_service import TagService
ts = TagService()
ts.set_llm_extract_fn(make_llm_tag_extractor())
```

`extract_tags_async` round-trip with a sync stub extractor returns
normalized tags (offload to thread works). With no extractor wired,
returns `[]`. End-to-end runtime verification with an OpenCode
`/api/summarize` exchange against LM Studio confirmed working during
this release cycle (live test with ghost and research profiles in
parallel sessions).

Full memory test suite: 603 passing, 3 skipped, 0 failed after the
fixture updates described in the Validation section above.

## Files touched (full list)

Sidecar / tag-extraction unification:

- `roampal/server/main.py` - `_hydrate_sidecar_from_opencode_config()`
  helper (decorator placement fixed); `lifespan` calls hydration before
  any handler runs; lazy-init path wires `TagService.set_llm_extract_fn`
  per profile; `/api/summarize` and `/api/record-outcome` route
  exchange-level extraction through TagService and stop importing
  `sidecar_extract_tags` directly
- `roampal/backend/modules/memory/tag_service.py` - added
  `set_llm_extract_fn()` setter, `extract_tags_async()`, and
  `_sidecar_lock` (asyncio.Lock for client locking to prevent GPU OOM
  on local models)
- `roampal/utils/sidecar_tag_wrapper.py` - new file, sync wrapper
  closure mirroring Desktop's path
- `roampal/backend/modules/memory/unified_memory_system.py` -
  `store_working` auto-extracts via TagService when caller passes no
  tags; new optional `embed_service` constructor parameter so multiple
  instances can share one ONNX model
- `roampal/backend/modules/memory/tests/unit/test_sidecar_hydration.py`
  - hydration test file (7 tests)

Profile binding fix and cross-client header propagation:

- `roampal/server/main.py` (same file as above) - per-profile registry
  replaces singleton `_memory` / `_session_manager`; `get_memory_for_request()`,
  `get_session_manager_for_request()`, `get_profile_context()` helpers;
  `_resolve_profile_name()` reads X-Roampal-Profile header
- `roampal/mcp/server.py` - cache resolved profile name at startup with
  sentinel; attach X-Roampal-Profile header on every HTTP call to FastAPI
- `roampal/plugins/opencode/roampal.ts` - `resolveRoampalProfile()`
  helper (env > project opencode.json > global opencode.json) and
  `roampalHeaders()` builder; profile header attached to all 11 FastAPI
  fetch sites; facts extraction wrapped in 3-attempt retry loop with
  exponential backoff (matches summary path); 3 background scoreExchange
  fetch timeouts bumped from 5s to 30s to suppress cosmetic
  `summary store error: TimeoutError` log messages on slow local models
  (data was always being stored regardless)
- `roampal/hooks/user_prompt_submit_hook.py` - `_roampal_headers()`
  helper using `active_profile_name()`; attached to both POST callsites
- `roampal/hooks/stop_hook.py` - `_roampal_headers()` helper; attached
  to both POST callsites

Test fixture updates required by the singleton kill:

- `roampal/backend/modules/memory/tests/unit/test_fastapi_endpoints.py`
  - fixture pre-populates `_memory_by_profile["default"]` instead of
  patching the dead `main._memory` global; "503 not ready" tests now
  patch `get_memory_for_request` / `get_session_manager_for_request`
  directly
- `roampal/backend/modules/memory/tests/unit/test_server_main.py` -
  cold-start test calls `_build_cold_start_profile(mem)` explicitly
  since the function now takes `mem` as a parameter
- `roampal/backend/modules/memory/tests/unit/test_profile_resolution.py`
  - new file, 6 tests pinning the contract for `_resolve_profile_name()`:
  X-Roampal-Profile header takes priority, fall through to
  `active_profile_name()` when missing or empty, default-profile
  normalization. Receiving end of the cross-client header propagation.

Version bump:

- `pyproject.toml` - version 0.5.3 -> 0.5.4
- `roampal/__init__.py` - `__version__` 0.5.3 -> 0.5.4
- `dev/docs/releases/v0.5.4/RELEASE_NOTES.md` - this file

No `sidecar_service.py` changes (the function is now called by the
wrapper instead of by handlers, but its body is unchanged).

## Migration

None. `pip install --upgrade roampal==0.5.4`, restart any running
FastAPI server (or restart your editor / OpenCode session, which will
auto-launch a fresh server).

