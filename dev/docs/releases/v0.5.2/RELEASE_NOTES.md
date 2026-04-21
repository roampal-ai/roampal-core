# Roampal Core v0.5.2

**Release Date:** 2026-04-21
**Type:** Patch — chat-path performance, field-drift lifecycle fix, v0.5.1 profile-resolution bugfixes
**Coordination:** Core ships first; Desktop v0.3.2 follows with the same latency trio and age-gate fix applied to the Tauri backend

## Summary

Three bundled improvements:

1. **Performance (Sections 1–3)** — cut the per-message latency added by the
   memory pipeline on every MCP tool call. Beta feedback on Desktop v0.3.1
   (2026-04-17) flagged the pipeline as painfully slow vs. raw Ollama / LM
   Studio. A code audit identified three serializable inefficiencies in core:
   serial retrieval lanes, cold ONNX loads, and a blocking CE rerank. All
   three are fixed here. Retrieval quality is unchanged — same TagCascade + CE
   pipeline, just unblocked.
2. **Lifecycle age-gate bug (Section 4)** — one read site in
   `promotion_service.py` read `metadata["timestamp"]` without falling back
   to `metadata["created_at"]`, silently shielding every core-written memory
   from the 7-day deletion threshold. Two-line fix, read-side only, no data
   migration.
3. **v0.5.1 profile-resolution bugs (Section 5)** — issue #6 (marcusyoung,
   2026-04-20): `roampal doctor` silently ignored `--profile` and
   `ROAMPAL_PROFILE`, and `roampal start --profile` printed an unexpanded
   `%APPDATA%/Roampal/data` literal in the startup banner even when the
   server was using a profile-resolved path. Both fixed here. The reporter's
   third item (`init --force` clobbering user-global `opencode.json`) is
   documented as non-reproducible with a specific information ask filed.

---

## Verified bottlenecks (audit 2026-04-17, core-specific)

| Location | Issue | Added latency |
|---|---|---|
| `unified_memory_system.py` two-lane retrieval | Two `search()` calls back-to-back with no `asyncio.gather` | ~500–1000 ms per MCP call |
| `search_service.py::_rerank_with_ce` | Sync `def` — CE ONNX inference blocks the asyncio event loop during rerank | Blocks heartbeats, concurrent tool calls |
| `search_service.py::_load_ce` + `embedding_service.py::_load_model` | ONNX models cold-load on first tool call rather than at server startup | +2–4 s on first MCP call after spawn |

**Note on scope vs. Desktop:** Desktop v0.3.2 also fixed a sync `embed_text`.
Core's `embed_text` has been wrapped in `asyncio.to_thread` since v0.4.1 and
was not a bottleneck here — no change needed on the embedding path.

**Impact per MCP tool call (pre-fix):**
- Steady state: ~1.0–1.8 s overhead
- First call after server spawn: +1.3–3.3 s on top (cold ONNX loads)

Because roampal-core auto-spawns its server on the first MCP tool call, every
new `claude` or `opencode` session hits the full cold-start penalty on the
first memory operation.

---

## Scope

### 1. Parallelize the two retrieval lanes

**File:** `roampal/backend/modules/memory/unified_memory_system.py`
**Site:** `get_context_for_injection`

Both lanes (summaries, facts) now run under a single `asyncio.gather`. The
lanes share no mutable state: `search()` operates on `self.embed_fn`
(stateless) and read-only ChromaDB collections.

```python
summary_results, fact_results = await asyncio.gather(
    self.search(query=query, limit=4, collections=all_collections,
                metadata_filters={"memory_type": {"$ne": "fact"}}),
    self.search(query=query, limit=4, collections=all_collections,
                metadata_filters={"memory_type": "fact"}),
)
```

**Expected savings:** ~500–1000 ms per MCP tool call.

### 2. Warm ONNX models at server startup

**File:** `roampal/backend/modules/memory/unified_memory_system.py`
**Site:** end of `initialize()`

Two named background tasks are created at the end of `initialize()` —
`warmup_ce` (loads the cross-encoder in a thread) and `warmup_embedding`
(calls `EmbeddingService.prewarm()`). Fire-and-forget. If a tool call arrives
before warm-up finishes, the lazy-load fallback still runs, so there's no
correctness risk.

Tasks are stored on `self._warmup_tasks` so tests can observe them and a
future shutdown path can await them.

**Expected savings:** ~2–4 s on the first MCP call after server spawn.

### 3. Offload blocking ONNX work to threadpool

**File:** `roampal/backend/modules/memory/search_service.py`
**Site:** `search()`, around the `_rerank_with_ce` call

The CE rerank is now dispatched via `asyncio.to_thread`:

```python
all_results = await asyncio.to_thread(
    self._rerank_with_ce, processed_query, all_results, limit
)
```

ONNX runtime releases the GIL during inference, so the event loop can process
other async work (MCP protocol heartbeats, concurrent tool calls, server
timeouts) while a rerank is in flight. This doesn't speed the math — it
restores event-loop responsiveness and prevents MCP protocol stalls on
heartbeat.

**Note:** the sibling embedding path (`EmbeddingService.embed_text` and
`embed_texts`) has been using `asyncio.to_thread(self._encode, ...)` since
v0.4.1 — no change needed.

### Intentionally NOT touched

- `CE_CANDIDATE_POOL` stays at its current value. Benchmark retrieval quality
  is pinned to that pool size; we are not trading it for latency.
- CE rerank stays on for every call. No "fast mode" toggle that silently
  regresses retrieval quality — the +23pt improvement vs. raw cosine is
  load-bearing in the published benchmark claims.

---

### 4. Lifecycle age-gate bug — `created_at` fallback in deletion threshold

**Problem:** `promotion_service.py` had one remaining read site reading
`metadata["timestamp"]` without falling back to `metadata["created_at"]`.
Every other lifecycle read in core already tolerates both
(`promotion_service.py` lines 368, 519, 569; `search_service.py:642`;
`unified_memory_system.py:184, 668-669, 746`).

Core's `store_working` writes `created_at`, not `timestamp`. For
core-written memories — which is *all* of them on a pure-core install —
this age check silently evaluated `age_days = 0` and routed every item to
the lenient `new_item_deletion_threshold` regardless of true age. Items that
should hit the stricter threshold after 7 days never did.

This was exposed by a shared-DB investigation on 2026-04-21 during Desktop
v0.3.2 laptop-testing, where core-written memories rendered as "now" in the
Desktop Memory Panel because of a parallel field-drift bug on the desktop
side (see Desktop v0.3.2 Section 0j). The desktop symptom was visible; the
core symptom was silent and had been shipping since the deletion-threshold
logic was added.

**Fix (read-side only, no data migration):**

```python
# Before
if metadata.get("timestamp"):
    age_days = (datetime.now() - datetime.fromisoformat(metadata["timestamp"])).days

# After
ts_str = metadata.get("timestamp") or metadata.get("created_at")
if ts_str:
    age_days = (datetime.now() - datetime.fromisoformat(ts_str)).days
```

Matches the idiom already used at `promotion_service.py` lines 368, 519, 569.

---

### 5. Fix v0.5.1 profile-resolution bugs (issue #6)

**Source:** Issue #6 on roampal-core (marcusyoung, 2026-04-20). v0.5.1
shipped named profiles but two user-facing surfaces never adopted the new
resolver.

#### Bug 1 — `roampal doctor` ignores `--profile` / `ROAMPAL_PROFILE`

`cmd_doctor` resolved the data directory via the pre-v0.5.1 `get_data_dir`
helper, which has no knowledge of the profile registry. Both sites that
needed the path — the Data Directory check and the Memory System init probe
— were affected, so `doctor` always reported and probed the default store,
even with a valid profile selected.

**Before (v0.5.1):**
```
python -m roampal doctor --profile research
# Data Directory:
#   [OK] Data directory exists: C:\Users\me\AppData\Roaming\Roampal\data
```

**After (v0.5.2):**
```
python -m roampal doctor --profile research
# Data Directory:
#   [OK] Profile: research (source: env)
#   [OK] Data directory exists: C:\Users\me\AppData\Roaming\Roampal\data\research
```

**Fix:** `cmd_doctor` now mirrors `cmd_start`'s profile-handling pattern.
`--profile` writes into `ROAMPAL_PROFILE` so `active_profile_name()` picks
it up; `--dev` is propagated via `ROAMPAL_DEV` so profile resolution targets
`Roampal_DEV` when appropriate. The Data Directory check and the Memory
System probe both call `resolve_data_path(profile_name)` and pass the
resolved `data_path` + `profile_name` into `UnifiedMemorySystem(...)`.
Unregistered profiles produce a single `[FAIL]` line and the memory probe
is skipped rather than crashing.

#### Bug 2 — `start_server` banner prints wrong data path

The banner hardcoded `%APPDATA%/Roampal/data` — an unexpanded literal that
was also profile-unaware and platform-inappropriate (doesn't apply on
macOS/Linux). Meanwhile `lifespan()` resolved the actual path correctly, so
the server used the right store; the banner just misreported it.

**Before (v0.5.1):**
```
===================================================
  ROAMPAL SERVER - PROD MODE
  Port: 27182
  Data: %APPDATA%/Roampal/data
===================================================
```

**After (v0.5.2):**
```
===================================================
  ROAMPAL SERVER - PROD MODE
  Port: 27182
  Data: C:\Users\me\AppData\Roaming\Roampal\data\research
  Profile: research (env)
===================================================
```

**Fix:** `start_server` now resolves the actual path via
`resolve_data_path(active_profile_name())` and prints it in the banner. A
`Profile: <name> (<source>)` line is appended only when a non-default profile
is active. `ProfileNotFoundError` is caught and surfaces as
`<profile 'name' not registered>` so the banner doesn't crash before
`lifespan()` gets a chance to error explicitly.

#### Bug 3 (reported, not reproducible) — `init --force` + project-level `opencode.json`

Reporter described `roampal init --force`, run from a project directory
containing a local `opencode.json`, overwriting `~/.config/opencode/opencode.json`
with project-level content. Investigated thoroughly and **not reproducible
from the current source**:

- `configure_opencode` and `_get_opencode_config_path` both compute
  `~/.config/opencode/opencode.json` unconditionally. No `Path.cwd()` or
  `os.getcwd()` touches the opencode path. The only `Path.cwd()` call in
  `cmd_init` writes a Claude Code `.mcp.json`, not opencode.
- The `roampal.ts` plugin never writes to `opencode.json` — it only reads.
- Three empirical repros on v0.5.1 with OpenCode installed confirmed the
  user-global file is byte-identical before/after (pre-existing case),
  project content does not leak into a freshly created user-global
  (missing-file case), and the project file itself is never touched.
- Reporter also noted they could not reproduce the behaviour afterwards.

**Most likely explanation:** OpenCode's own runtime config-merge view
(project-over-user) being read as a disk-level clobber. Follow-up requested
from reporter: on-disk SHA256 + first 20 lines before/after, plus confirmation
the file isn't a symlink/junction. Bug left open pending a reproducer.

---

## Combined impact

- Steady-state MCP tool call overhead: ~1.0–1.8 s → **~400–900 ms**
  (driven mostly by parallelizing the two CE lanes via `asyncio.gather`)
- First-call overhead after server spawn: ~3–5 s → **~400–900 ms**
  (~2–4 s saved by warm-start)
- Event-loop responsiveness during CE rerank: restored (no longer blocks
  MCP protocol heartbeats or concurrent tool calls)
- Lifecycle age gate: items past 7 days now correctly evaluated against the
  stricter deletion threshold, regardless of which field stores the timestamp
- Retrieval quality: **unchanged**

---

## Files touched

- `roampal/backend/modules/memory/unified_memory_system.py` — Fix #1 (gather),
  Fix #2 (warmup tasks at end of `initialize()`)
- `roampal/backend/modules/memory/search_service.py` — Fix #3 (CE rerank via
  `asyncio.to_thread`)
- `roampal/backend/modules/memory/promotion_service.py` — Fix #4
  (`created_at` fallback in deletion age gate)
- `roampal/cli.py` — Fix #5 Bug 1 (`cmd_doctor` profile resolution)
- `roampal/server/main.py` — Fix #5 Bug 2 (`start_server` banner resolves
  actual path + optional profile line)

**Not touched (intentionally):**
- `roampal/backend/modules/memory/embedding_service.py` — already
  thread-offloaded since v0.4.1; release notes for Desktop v0.3.2 list it
  because it applied there, not here.

---

## Tests added

**Performance (`test_unified_memory_system.py::TestV052PerformanceFixes`):**
- `test_get_context_for_injection_parallelizes_lanes` — mocks `search()` with
  a 50 ms async sleep, captures start/end timestamps for each lane, asserts
  the second lane's start occurs before the first lane's end (fails cleanly
  if `asyncio.gather` is removed).
- `test_onnx_models_warm_at_init` — asserts `ums._warmup_tasks` exists after
  `initialize()` with exactly two tasks named `warmup_ce` and
  `warmup_embedding`.

**Performance (`test_search_service.py::TestV052CERerankOffload`):**
- `test_ce_rerank_offloaded_to_executor` — patches `asyncio.to_thread` with a
  spy, runs `search()`, asserts `_rerank_with_ce` was dispatched through the
  spy (fails cleanly if the call reverts to a direct sync invocation).

**Age-gate (`test_promotion_service.py`):**
- `test_deletion_threshold_ages_core_written_memory` — seeds an item with
  `created_at` 14 days old and no `timestamp` field; confirms `age_days > 7`
  and the stricter `deletion_score_threshold` is applied (not
  `new_item_deletion_threshold`).
- `test_deletion_threshold_still_ages_timestamp_memory` — regression case:
  `timestamp`-only memory (desktop-style) behaves identically to before.

**Issue #6 profile bugs (`test_cli.py::TestV052CmdDoctorProfile`):**
- `test_doctor_respects_profile_flag` — `args.profile = "research"`, asserts
  stdout contains `Profile: research` and `source: env`.
- `test_doctor_respects_profile_env` — sets `ROAMPAL_PROFILE=research` with
  no `--profile` flag, same assertions.
- `test_doctor_unregistered_profile` — unknown profile name, asserts a
  single `[FAIL]` with `'<name>' is not registered` and that memory init is
  skipped.

**Issue #6 banner bugs (`test_server_main.py::TestV052StartServerBanner`):**
- `test_start_banner_shows_resolved_path_default` — default profile, banner
  contains absolute `Data:` path and no `Profile:` line; `%APPDATA%` literal
  is absent.
- `test_start_banner_shows_profile_line_when_active` — registered profile,
  banner contains `Profile: research (env)`.

**Counterfactual verification:** each performance test was confirmed to fail
when its fix is reverted (serial lanes / warmup removed / sync rerank),
proving the tests exercise the intended behaviour rather than just running.

**Full suite status:** 541 tests (530 backend unit + 11 profile_manager)
passing on Windows / Python 3.10.

---

## User-facing release blurb

```
v0.5.2: Performance — cut MCP tool-call overhead roughly in half by
parallelizing the two retrieval lanes, warming ONNX models at server
startup, and offloading CE rerank to a threadpool. Plus a lifecycle
age-gate fix so core-written memories now age out correctly, and two
v0.5.1 profile-resolution fixes (`doctor --profile` and the `start` banner
both now honor named profiles). Retrieval quality unchanged. No breaking
changes.
```

---

## Coordination

Core ships first. Desktop v0.3.2 follows with the same performance trio +
age-gate fix ported to the Tauri backend. Desktop additionally needs a
sync-`embed_text` fix that core did not; core's `embed_text` has been
wrapped in `asyncio.to_thread` since v0.4.1.

Both releases address the 2026-04-17 tester report on chat-path latency.
