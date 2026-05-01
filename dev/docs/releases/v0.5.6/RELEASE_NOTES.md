# Roampal Core v0.5.6

**Release Date:** 2026-05-01
**Type:** Hardening / hygiene release (no new user-visible features)
**Triggered by:** Verification audit of v0.5.5.x — gaps not yet covered by shipped fixes

## Summary

v0.5.5 → v0.5.5.2 closed the headline bugs from issues #8, #10, #11. This release closes the *remaining* coverage gaps surfaced during a verification pass. Nothing here is blocking real users — these are the difference between "fix is in code" and "fix is provably robust under unusual conditions."

Items, by issue:

- **#11 (Windows install)** — items 1, 2, 3
- **#10 (Desktop profile)** — item 4 (closes a concurrency window the v0.5.5.1 fix missed)
- **#8 (memory_bank phantoms)** — items 5, 6, 7, 8, 9, 10, 11 (a mix of correctness hardening, observability, and lane enforcement)
- **Sidecar prompt alignment with benchmark** — items 12, 13, 14, 15, 16, 17, 18 (reduce drift between `roampal-labs/benchmark/runner.py` and the production OpenCode prompts so benchmark numbers translate to real-world quality)
- **Scoring mutex → async queue (v0.5.5 unfinished)** — item 19 (the v0.5.5 release notes documented an async-queue refactor that never landed; ship the actual code change plus per-session deferred-retry hardening)
- **MCP tool definition quality (TDQS audit)** — item 20+ (rewrite each MCP tool description in `roampal/mcp/server.py` against the Tool Definition Quality Score rubric; agents using these tools should score B+ on every dimension, not just Purpose)
- **OpenCode Go support in sidecar setup** — items 27, 28 (clarify free-Zen `[2]` wording so Go subscribers know it applies to them, and add an auto-detect `Use OpenCode Go` option that reads `auth.json` and configures their Go subscription as the sidecar)

---

## Item 1 — Regression test for `_install_plugin_file()` (Highest priority)

### What's missing

`roampal/cli.py:898-943` (`_install_plugin_file`) is the v0.5.5.2 helper that handles silent `shutil.copy` failures on Windows (OneDrive sync, antivirus scanning, Controlled Folder Access). It has three failure modes (raised exception, missing destination, size mismatch) and a manual `read_bytes/write_bytes` fallback. **None of those branches is exercised by a unit test today.**

The function is the load-bearing piece of the v0.5.5.2 fix. If a future refactor breaks the size-mismatch branch or the fallback path, the regression silently reverts users to the pre-v0.5.5.2 silent-failure mode — they see "Installed plugin" and have a 0-byte file.

### Implementation

Add `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` (or a sibling location next to existing CLI tests). Cases to cover with `pytest`'s `monkeypatch` + `tmp_path`:

1. **Happy path.** Real source file, `shutil.copy` works, dest matches source size — assert success message printed, dest exists, sizes equal.
2. **Copy raises `PermissionError`.** Monkeypatch `shutil.copy` to raise — assert error message printed naming Desktop, dest does not exist, function returns without crashing.
3. **Copy "succeeds" but destination missing.** Monkeypatch `shutil.copy` to be a no-op — assert "file disappeared after copy" message, no crash.
4. **Copy succeeds but writes 0 bytes.** Monkeypatch `shutil.copy` to write an empty file — assert fallback `write_bytes` runs and dest ends up at correct size.
5. **Copy succeeds but writes wrong size.** Monkeypatch to write half the source — assert fallback corrects it.
6. **Both copy and fallback `write_bytes` produce wrong size.** Monkeypatch both — assert "file size mismatch after fallback copy" error printed.

These are pure-Python tests, no Windows-only code paths needed (the function is platform-agnostic). Should run cleanly on Linux/macOS CI.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` (new) | 6 unit tests |

---

## Item 2 — Surface `%APPDATA%` unset on Windows instead of silently skipping

### What's missing

`roampal/cli.py:1093-1100` checks `os.environ.get("APPDATA", "")` and silently skips the dual-path install if it's empty:

```python
if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    if appdata:                                  # ← silent skip if unset
        alt_plugin_dir = Path(appdata) / "opencode" / "plugins"
        ...
```

If a Windows user runs `roampal init --force` from a non-standard shell (cygwin, MSYS2, certain CI runners, shells that strip env vars), `%APPDATA%` may not be set. The user gets the primary install only, no warning, and may still hit issue #11 if their OpenCode Desktop reads from `%APPDATA%`.

### Fix

Replace the silent skip with a one-line `print` so the user knows the dual-path fallback was skipped:

```python
if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        alt_plugin_dir = Path(appdata) / "opencode" / "plugins"
        alt_plugin_file = alt_plugin_dir / "roampal.ts"
        if alt_plugin_file != plugin_file:
            alt_plugin_dir.mkdir(parents=True, exist_ok=True)
            _install_plugin_file(plugin_source, alt_plugin_file)
    else:
        print(f"  {YELLOW}Skipped %APPDATA% fallback install — APPDATA env var is unset.{RESET}")
        print(f"  {YELLOW}If OpenCode can't find the plugin, set APPDATA and re-run, or copy manually.{RESET}")
```

Cost: two lines. Worth it because silent skip is the exact failure mode v0.5.5.1 → v0.5.5.2 moved away from.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/cli.py` | 1093-1100 | Add `else` branch with warning when `APPDATA` unset |

---

## Item 3 — Improve `_install_plugin_file()` error message to name actual likely causes

### What's missing

When `shutil.copy` raises an exception, the current error message at `cli.py:911-915` blames OpenCode Desktop:

```python
print(f"  {RED}Failed to install plugin: {e}{RESET}")
print(f"  {YELLOW}Close OpenCode Desktop and try again, or copy manually:{RESET}")
print(f"  {YELLOW}  cp {plugin_source} {plugin_dest}{RESET}")
```

But `PermissionError` on Windows can also come from:
- Read-only NTFS attribute set on existing `roampal.ts` (rare but happens after restoring from backup)
- Controlled Folder Access blocking write to `%APPDATA%` or `~/.config`
- Antivirus quarantining the source file mid-install

The "close OpenCode Desktop" message sends users down the wrong rabbit hole if their actual cause is one of the above. Marcus's v0.5.5.1 report ("Desktop closed and no electron/node processes running") is exactly this failure mode.

### Fix

Inspect the exception type/message and produce a more honest list of likely causes:

```python
except (OSError, PermissionError) as e:
    print(f"  {RED}Failed to install plugin: {e}{RESET}")
    print(f"  {YELLOW}Possible causes:{RESET}")
    print(f"  {YELLOW}  - OpenCode Desktop is running and holds a file lock{RESET}")
    print(f"  {YELLOW}  - Read-only attribute on existing {plugin_dest.name}{RESET}")
    print(f"  {YELLOW}  - Antivirus / Controlled Folder Access blocking write{RESET}")
    print(f"  {YELLOW}  - OneDrive sync quarantining the destination{RESET}")
    print(f"  {YELLOW}If those don't apply, copy manually:{RESET}")
    print(f"  {YELLOW}  cp {plugin_source} {plugin_dest}{RESET}")
    return
```

Same applies to the "file disappeared" branch at line 920-924 — it currently only blames antivirus/OneDrive but should also mention Controlled Folder Access.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/cli.py` | 911-915, 920-924 | Expand cause list in error messages |

---

---

## Item 4 — Refresh profile in `session.idle` scope (#10 follow-up)

### What's missing

v0.5.5.1's #10 fix re-resolves the profile inside `chat.message` via `refreshProfile(sessionID)`. That works for HTTP calls made directly inside `chat.message`. **But the plugin also makes profile-bound HTTP calls from `session.idle` scope, and `refreshProfile` is never called there.**

Affected call sites in `roampal/plugins/opencode/roampal.ts`:

| Line | Call | Scope |
|------|------|-------|
| 1803 | `fetch(.../stop, headers: roampalHeaders())` — lifecycle exchange register | session.idle |
| 1825 | `scoreExchangeViaLLM(pendingScoring.sessionId, ...)` — deferred retry | session.idle |
| 1853 | `scoreExchangeViaLLM(sid, ...)` — primary scoring path | session.idle |

All three read the singleton `_cachedProfile`. That global is only updated by `chat.message` for the *most recent* user message. On Desktop's singleton plugin, the failure pattern is:

1. User in project A sends a message → `chat.message` for session A → `refreshProfile(A)` → `_cachedProfile = profile-A`.
2. User switches to project B in the UI and sends a message → `chat.message` for session B → `refreshProfile(B)` → `_cachedProfile = profile-B`.
3. Session A's `session.idle` fires (debounced 1.5s after A's last activity, possibly delayed past step 2) → calls `scoreExchangeViaLLM(A, ...)` → all FastAPI calls go out with `X-Roampal-Profile: profile-B`.
4. **Result: A's exchange summary, scoring, and lifecycle markers land in profile B's ChromaDB collections.** Silent cross-profile contamination.

The deferred retry path (line 1825) is even worse — `pendingScoring.sessionId` may be from a session several exchanges old, but the profile used is whatever the latest `chat.message` set.

### Why this wasn't caught in v0.5.5.1 validation

The release notes' two-project test was sequential: send in A, switch to B, send in B. That validates the `chat.message` path. To trigger this bug you need a `session.idle` to fire for an old session *after* a `chat.message` for a different session has already shifted the global. Concurrency-dependent — easy to miss in manual repro, will eventually hit real users.

### Fix sketch

`refreshProfile()` needs to be called immediately before each `session.idle` HTTP call, keyed to the right session. Two options:

**Option A — refresh per session.idle handler.** Call `await refreshProfile(sid)` at the top of the `session.idle` handler (before line 1803). Covers lifecycle + current scoring. Doesn't cover deferred retry from a *different* session.

**Option B — refresh per-call with explicit session.** Pass `sessionID` into `scoreExchangeViaLLM` and `roampalHeaders` and have them resolve fresh. Eliminates the global cache for HTTP-call scope. More invasive but correct.

Option A is the minimum to close the reported gap. Option B is the architectural fix and what we should ship.

For deferred retry (line 1825), use `pendingScoring.sessionId` — that's the right session for that payload, not the current `sid`.

### Impact

Affects users who:
- Use OpenCode Desktop with multiple projects in one workspace
- Have distinct `ROAMPAL_PROFILE` settings per project
- Switch between projects rapidly enough that one project's `session.idle` overlaps with another project's `chat.message`

Symptoms are subtle: memories from project A occasionally appear in project B's profile, with no error or warning. Easy to mistake for "the profile fix didn't work" and reopen #10 — but the chat.message path *is* fixed; the gap is purely in idle-scope hooks.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/plugins/opencode/roampal.ts` | ~1780 | `await refreshProfile(sid)` at top of session.idle handler |
| `roampal/plugins/opencode/roampal.ts` | 1825 | `await refreshProfile(pendingScoring.sessionId)` before deferred retry |
| `roampal/plugins/opencode/roampal.ts` | 163 (optional) | Plumb sessionID into `roampalHeaders()` for explicit resolution |

---

---

## ~~Item 5~~ ✅ Item 5 — Run phantom migration after `cleanup_archived()`, not just at startup (#8 follow-up)

**Status: COMPLETED.** `_sweep_phantoms()` helper added to `MemoryBankService` and called from both `_startup_cleanup` and `cleanup_archived()`. 3 unit tests cover the post-cleanup sweep.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `memory_bank_service.py:416-439` | `_sweep_phantoms()` helper — list_all_ids + get_fragment None check + delete_vectors | ✅ Done |
| `memory_bank_service.py:470` | `cleanup_archived()` calls `_sweep_phantoms()` after `delete_vectors()` | ✅ Done |
| `unified_memory_system.py:690` | `_startup_cleanup()` now delegates phantom sweep to `_memory_bank_service._sweep_phantoms()` | ✅ Done |
| `tests/unit/test_memory_bank_service.py::TestCleanupArchivedSweepsPhantoms` | 3 tests: sweep invoked after delete, end-state phantom-free, no-op when index is well-behaved | ✅ Pass (40/40) |
| `tests/integration/test_phantom_cleanup_safety.py` | 2 end-to-end tests on real ChromaDB: (a) raw-delete + sweep leaves survivors intact, (b) cleanup_archived only removes archived entries — never touches active ones | ✅ Pass (2/2) |

### Live safety validation (2026-05-01, isolated `phantom_test_v056` profile, real ChromaDB)

To answer "does this ever delete something it shouldn't?": ran both scenarios end-to-end against the actual install. Results:
- Scenario A (force-delete 2 of 5 entries via raw `adapter.collection.delete()`, then `_sweep_phantoms()`): the raw delete **already cleaned the index** on the current ChromaDB version (no phantoms were created), `_sweep_phantoms()` returned `0`, all 3 survivors remained intact with original content.
- Scenario B (archive 1 of 5 via `svc.archive()`, then `cleanup_archived()`): `cleanup_archived` returned `1`, exactly the archived entry was removed, the other 4 entries remained intact with original content.

Net: on this ChromaDB build the phantom-creation path the v0.5.5 release notes warn about doesn't even reproduce — the cleanup logic operates on an empty problem set in steady state. The active-entry safety property holds in both scenarios. The `test_phantom_cleanup_safety.py` integration tests above lock both scenarios into CI so any regression on a future ChromaDB upgrade surfaces immediately.

### What's missing

`_startup_cleanup()` (`unified_memory_system.py:681-698`) sweeps phantom IDs (entries where `list_all_ids()` returns the ID but `get_fragment()` returns None) once per process boot. But `cleanup_archived()` (`memory_bank_service.py:332-359`) calls `delete_vectors()` to permanently remove archived entries — and that's the *exact* operation that creates phantoms in the first place. So every `cleanup_archived()` call introduces new phantoms that survive until the next server restart.

Today `cleanup_archived()` is only triggered manually (no auto-trigger yet). When item 8 below adds capacity-pressure auto-trigger, this becomes a real correctness issue: phantoms accumulate during a long-running server session without anyone noticing.

### Fix

Add a `_sweep_phantoms()` helper to `MemoryBankService` and call it from both:
- `_startup_cleanup()` — same as today
- `cleanup_archived()` — after the `delete_vectors()` call returns, sweep the IDs that were just deleted

```python
def _sweep_phantoms(self, hint_ids: Optional[List[str]] = None) -> int:
    """Remove HNSW-orphaned IDs. If hint_ids given, only check those."""
    candidates = hint_ids if hint_ids is not None else self.collection.list_all_ids()
    phantom_ids = [
        doc_id for doc_id in candidates
        if not self.collection.get_fragment(doc_id)
    ]
    if phantom_ids:
        self.collection.delete_vectors(phantom_ids)
    return len(phantom_ids)

def cleanup_archived(self) -> int:
    archived_ids = [...]   # existing logic
    if archived_ids:
        self.collection.delete_vectors(archived_ids)
        # v0.5.6: sweep the IDs we just deleted
        self._sweep_phantoms(hint_ids=archived_ids)
    return len(archived_ids)
```

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/memory_bank_service.py` | New `_sweep_phantoms()` helper; `cleanup_archived()` calls it after `delete_vectors` |
| `roampal/backend/modules/memory/unified_memory_system.py:681-698` | Replace inline phantom logic with call to `_memory_bank_service._sweep_phantoms()` |
| `roampal/backend/modules/memory/tests/unit/test_memory_bank_service.py` | Test that `cleanup_archived()` ends with no phantoms |

---

## ~~Item 6~~ ✅ Item 6 — Strengthen the chromadb_adapter phantom filter (#8 follow-up)

**Status: COMPLETED.** OR logic change + 4 integration tests for all mid-state cases.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `chromadb_adapter.py:267-275` | Changed AND → OR in phantom filter; catches doc=None, meta=None, or both | ✅ Done |
| `tests/integration/test_chromadb_integration.py` | Added 4 tests: doc=None+meta=cached, doc=cached+meta=None, both=None, valid entry passes | ✅ Pass (22/22) |

### What's missing

`chromadb_adapter.py:267-269` filters phantoms only when document AND metadata are both falsy:

```python
if result_document is None and (result_metadata is None or not result_metadata):
    continue
```

ChromaDB can leave entries in mid-states after a delete: doc cleared but metadata cached, or metadata cleared but doc cached. Both leak through this filter. The metadata-cached case is what the v0.5.5 release notes called out as the original bug: "Deleted memory_bank entries still have valid metadata, so they pass through the filter."

### Fix

For `memory_bank` results specifically, require a `status` field on every result. Memory_bank entries written post-fix all have `status="active"` (verified at `memory_bank_service.py:110`), so any result without it is either pre-fix (handled by item 7's backfill) or a mid-state phantom (skip it):

```python
# v0.5.6: tighten phantom filter — catch mid-state results
is_phantom = (
    result_document is None
    or (result_metadata is None or not result_metadata)
)
if is_phantom:
    continue
```

This catches the OR case (doc cleared OR metadata cleared) instead of only AND.

For collection-aware tightening, the per-collection caller (`search_service` for memory_bank) can additionally require `metadata.get("status")` to be present and not `"archived"`, but that's already done by the existing status filter on the where clause — no need to duplicate at adapter level.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/backend/modules/memory/chromadb_adapter.py` | 267-269 | Change `and` → `or` in phantom filter |
| `roampal/backend/modules/memory/tests/integration/test_chromadb_integration.py` | new | Integration test that confirms mid-state phantoms are filtered |

---

## ~~Item 7~~ ✅ Item 7 — Status backfill on startup for legacy memory_bank entries (#8 follow-up)

**Status: COMPLETED.** Backfill block added to `_startup_cleanup()` after phantom migration + 4 unit tests.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `unified_memory_system.py:_startup_cleanup` | Added backfill loop after phantom migration; sets status=active on entries missing it | ✅ Done |
| `tests/unit/test_unified_memory_system.py` | 4 tests: legacy gets status, modern skipped, empty collection no-op, phantoms (None) skipped | ✅ Pass (6/6 startup) |

### What's missing

The dedup filter `{"status": {"$ne": "archived"}}` (`unified_memory_system.py:386`) and the search-path filters (`search_service.py:361, 540`) all rely on ChromaDB's `$ne`-on-missing-field semantics being "missing != archived → include this row." Most ChromaDB versions do honor this, but it's version-specific and never explicitly verified.

Pre-v0.5.5 entries written before `add()` started setting `status="active"` (`memory_bank_service.py:110`) have no `status` field. If a future ChromaDB upgrade changes `$ne` semantics, those entries silently drop out of dedup/search results.

### Fix

One-shot backfill in `_startup_cleanup()` after the phantom migration runs:

```python
# v0.5.6: backfill status=active on legacy memory_bank entries
if self._memory_bank_service:
    try:
        all_ids = self._memory_bank_service.collection.list_all_ids()
        backfilled = 0
        for doc_id in all_ids:
            frag = self._memory_bank_service.collection.get_fragment(doc_id)
            if frag and "status" not in frag.get("metadata", {}):
                meta = frag.get("metadata", {}) or {}
                meta["status"] = "active"
                self._memory_bank_service.collection.update_fragment_metadata(doc_id, meta)
                backfilled += 1
        if backfilled:
            logger.info(f"v0.5.6 migration: backfilled status=active on {backfilled} legacy entries")
    except Exception as e:
        logger.warning(f"Status backfill error: {e}")
```

Idempotent — runs every startup but only does work the first time after upgrade.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py:_startup_cleanup` | Add backfill block after phantom migration |
| `roampal/backend/modules/memory/tests/unit/test_unified_memory_system.py` | Test legacy entries get `status=active`, post-fix entries are untouched |

---

## ~~Item 8~~ ✅ Item 8 — Auto-trigger `cleanup_archived()` under capacity pressure (#8 follow-up)

**Status: COMPLETED.** `_maybe_cleanup_archived()` in `store()` path + 4 unit tests.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `memory_bank_service.py` | Added `ACTIVE_THRESHOLD=400`, `ARCHIVED_RATIO_THRESHOLD=0.5`, `_maybe_cleanup_archived()`, called from `store()` before capacity check | ✅ Done |
| `tests/unit/test_memory_bank_service.py` | 4 tests: below threshold, low archived ratio, high archived ratio triggers cleanup, total=0 guard; fixed existing capacity test mock | ✅ Pass (37/37) |

### What's missing

`memory_bank` has a 500-item cap (`MAX_ITEMS = 500`). After v0.5.5, archived entries don't count toward the active-count cap (`_get_count()` excludes them). But they DO count toward the underlying ChromaDB collection size. Over time, archived entries accumulate forever — `cleanup_archived()` exists but is never called.

Two failure modes:
1. ChromaDB performance degrades as the collection grows past the active cap (HNSW with thousands of archived entries).
2. If a user does heavy archive-and-replace work, the working dataset grows unboundedly.

The release notes for v0.5.5 explicitly mentioned this: *"`cleanup_archived()` remains available for manual invocation or capacity-pressure triggers in `_get_count()` when approaching the 500-item limit."* That trigger was never wired up.

### Fix

In `MemoryBankService._enforce_capacity()` (or wherever capacity enforcement lives — `store()` path), check active count against a soft threshold and run cleanup before rejecting writes:

```python
ACTIVE_THRESHOLD = 400   # 80% of MAX_ITEMS
ARCHIVED_RATIO_THRESHOLD = 0.5   # cleanup if >50% of total is archived

def _maybe_cleanup_archived(self) -> None:
    active = self._get_count()
    if active < ACTIVE_THRESHOLD:
        return
    total = self.collection.collection.count()
    archived = total - active
    if total == 0 or archived / total < ARCHIVED_RATIO_THRESHOLD:
        return
    cleaned = self.cleanup_archived()
    if cleaned:
        logger.info(f"Auto-cleanup at capacity pressure: removed {cleaned} archived entries")
```

Call from `store()` before the cap check.

Combined with item 5, `cleanup_archived()` won't leave phantoms behind even when called automatically.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/memory_bank_service.py` | New `_maybe_cleanup_archived()`; called from `store()` |
| `roampal/backend/modules/memory/tests/unit/test_memory_bank_service.py` | Test capacity-pressure trigger fires above threshold and not below |

---

## ~~Item 9~~ ✅ Item 9 — Dedup observability (#8 follow-up)

**Status: COMPLETED.** INFO-level log in `_find_duplicate_fact()` + test with caplog.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `unified_memory_system.py:_find_duplicate_fact` | Added INFO log on dedup hit with tier, distance (3 decimal), matched id | ✅ Done |
| `tests/unit/test_unified_memory_system.py` | Test that verifies `[DEDUP] skip=tier distance=X.XXX id=...` is logged at INFO level | ✅ Pass (12/12 dedup) |

### What's missing

`_find_duplicate_fact()` (`unified_memory_system.py:494+`) silently returns the matching doc_id when it finds a similar fact, and the caller silently skips storage. There is no log line, no metric, no warning. If the dedup ever becomes over-aggressive again — different root cause, same silent-drop symptom — we'd be just as blind as we were for issue #8.

### Fix

One log line per dedup-skip with enough context to triage:

```python
if best_match:
    logger.info(
        f"Dedup skip: tier={best_match['tier']} "
        f"distance={best_match['distance']:.3f} "
        f"matched_id={best_match['id']} "
        f"new_fact={fact[:80]!r}"
    )
    return best_match["id"]
```

For long-running production observability, also expose a counter on `/api/memory-bank/health` (item 10): number of dedup-skips per collection in the last N minutes. A sudden spike correlates with whatever new bug just got introduced.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py:_find_duplicate_fact` | Log dedup hits |

---

## ~~Item 10~~ ✅ Item 10 — Lane enforcement on `MemoryBankService.delete()` (#8 follow-up)

**Status: COMPLETED.** Renamed to `delete_permanent(force=True)` + guard flag + tests.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `memory_bank_service.py` | Renamed `delete()` → `delete_permanent()`, added `force=False` kwarg, raises `RuntimeError` if not force=True | ✅ Done |
| `tests/unit/test_memory_bank_service.py` | Updated to `TestDeletePermanent`; 4 tests: success with force, failure path, missing flag raises, explicit False raises | ✅ Pass (33/33) |

### What's missing

`MemoryBankService` has two delete paths with very different safety profiles:

| Method | Behavior | Safe to call from |
|---|---|---|
| `archive(content)` | Soft-delete (sets `status=archived`) | User-facing GUI, anywhere |
| `delete(doc_id)` | Hard delete via `delete_vectors()` | Only `cleanup_archived()` |

The `delete()` method is public and only differentiated from `archive()` by a one-line docstring ("User permanently deletes memory"). A future engineer adding a "GDPR delete" or "really delete" feature could wire a GUI button straight to `delete()`, hit the HNSW phantom problem all over again, and reopen issue #8.

### Fix

Two changes:

**a) Rename for clarity:**
- `delete()` → `delete_permanent()`
- Update the one in-tree caller (`cleanup_archived()`) and tests

**b) Mark as restricted:**

Add a `_caller_is_safe` guard or simply require an explicit flag:

```python
async def delete_permanent(self, doc_id: str, *, force: bool = False) -> bool:
    """
    Permanently hard-delete a memory_bank entry.

    HNSW phantom risk: this leaves debris in the vector index that bypasses
    metadata filters until the next phantom sweep. ONLY call from contexts
    that follow up with a phantom sweep (cleanup_archived, explicit GDPR flow).

    For user-initiated deletes, call archive() instead.
    """
    if not force:
        raise RuntimeError(
            "delete_permanent() requires force=True. "
            "For user-facing deletes, use archive() instead."
        )
    # ... existing delete logic
```

`cleanup_archived()` passes `force=True` explicitly. New code paths can't accidentally fall into hard-delete without grepping the codebase and seeing the warning.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/memory_bank_service.py` | Rename `delete()` → `delete_permanent()`; add `force=True` requirement |
| `roampal/backend/modules/memory/tests/unit/test_memory_bank_service.py` | Update tests; add coverage for force-flag enforcement |

---

## ~~Item 11~~ ✅ Item 11 — Integration test for full archive-then-add cycle (#8 follow-up)

**Status: COMPLETED.** End-to-end repro of original bug #8 with real ChromaDB.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `tests/integration/test_archive_dedup_cycle.py` (new) | 5 tests: archive then re-add succeeds, active dedup still works, archive all then re-add, search excludes archived after cycle, search after archive+readd only shows new entry | ✅ Pass (5/5) |

### What's missing

Nothing — this closes the entire #8 chain (Items 5, 6, 7, 8, 9, 10, 11 all complete).

---

# Sidecar prompt alignment with benchmark

The benchmark in `roampal-labs/benchmark/runner.py` simulates production by firing three sidecar calls per turn (score, extract_facts, extract_tags). Production (`roampal/plugins/opencode/roampal.ts`) does the same shape, but the prompts have drifted. Items 12-18 close the drift so benchmark scores translate to live runs.

Source-of-truth files:
- Labs scoring prompt: `C:/roampal-labs/benchmark/runner.py:226-290` (`sidecar_score`)
- Labs facts prompt: `C:/roampal-labs/benchmark/runner.py:174-223` (`sidecar_extract_facts`)
- Labs tags prompt: `C:/roampal-labs/strategies/entity_routed.py:74-117` (`extract_tags`)
- Core OpenCode scoring prompt: `roampal/plugins/opencode/roampal.ts:685-708` (`scoringPrompt`)
- Core OpenCode facts prompt: `roampal/plugins/opencode/roampal.ts:1058-1074` (`factsPrompt`)
- Core Python helpers: `roampal/sidecar_service.py` (`summarize_only`, `extract_tags`, `extract_facts`, `summarize_and_score`)

---

## ~~Item 12~~ ✅ Item 12 — Align scoring output field names with benchmark (Highest priority of this group)

**Status: COMPLETED.** Prompt + parse sites aligned across TS and Python.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `roampal/plugins/opencode/roampal.ts` | 692, 806, 891, 897 — prompt schema, system message, parse sites renamed to `exchange_summary` / `exchange_outcome` | ✅ Done |
| `roampal/sidecar_service.py` | 774, 789 — docstring + prompt in `summarize_and_score()` aligned | ✅ Done |

### What's missing

Nothing — prompt schema, system message, and parse sites all aligned.

---

## ~~Item 13~~ ✅ Item 13 — Pin `temperature: 0` on all Python sidecar HTTP calls

**Status: COMPLETED.** All 6 backends pinned to deterministic output.

### Implemented
| File | Backend | Status |
|------|---------|--------|
| `_call_custom` (line 200) | `"temperature": 0` | ✅ Done |
| `_call_haiku` (line 251) | `"temperature": 0` | ✅ Done |
| `_call_zen` (line 305) | `"temperature": 0` | ✅ Done |
| `_call_ollama_native` (line 415) | `options: {"temperature": 0.0}` | ✅ Done |
| `_call_lmstudio` (line 465) | `"temperature": 0` | ✅ Done |
| `_call_anthropic_model` (line 819) | `"temperature": 0` | ✅ Done |

### What's missing

Nothing — all Python sidecar backends now pin `temperature: 0`. Tag extraction is deterministic across re-extracts, preventing inverted-index drift.

---

## ~~Item 14~~ ✅ Item 14 — Standardize summary-length phrasing across all summarize prompts

**Status: COMPLETED.** All summarize prompts use consistent phrasing.

### Implemented
| File | Line | Before → After |
|------|------|----------------|
| `sidecar_service.py` (summarize_and_score schema) | 793 | `<~300 chars>` → `<around 300 chars, 1-2 sentences if possible>` |
| `sidecar_service.py` (summarize_and_score instruction) | 795 | `(~300 chars)` → `(around 300 chars, 1-2 sentences if possible)` |
| `sidecar_service.py` (summarize_only prompt + schema) | 863/869 | Same update |
| `sidecar_service.py` (test_sidecar_scoring) | 1011/1013 | Same update |
| `roampal.ts` (scoring prompt schema + instruction) | 692/694 | `<~300 chars>` / `(under 300 chars)` → `<around 300 chars, 1-2 sentences if possible>` |

Both constraints work together: char count is the hard cap (matches downstream `--max-chars` default of 400), sentence count gives small models a natural-prose target to avoid fragmentary output.

---

## ~~Item 15~~ ✅ Item 15 — Restore the inference rule on facts prompt

**Status: COMPLETED.** Inference rule added to both TS and Python fact-extraction prompts.

### Implemented
| File | Line | Change |
|------|------|--------|
| `roampal.ts` (factsPrompt) | 1062 | Added `- Capture what can be inferred, not just what was explicitly said` |
| `sidecar_service.py` (extract_facts) | 962 | Same bullet added |

Small models now capture context-derived facts (e.g., "User pivoting from PM to ops/program manager roles") instead of only echoing verbatim statements.

---

## ~~Item 16~~ ✅ Item 16 — Restore varied GOOD examples on facts prompt

**Status: COMPLETED.** Facts prompts now have 6 GOOD + 3 BAD examples (matching labs).

### Implemented
| File | Line | Change |
|------|------|--------|
| `roampal.ts` (factsPrompt) | 1068-1074 | Added 4 personal-life GOOD examples + "The user asked a question" BAD example |
| `sidecar_service.py` (extract_facts) | 968-974 | Same examples added |

Examples now span technical (auth/JWT, TS/Zod), creative writing (chapter draft), sports (Lakers/LeBron), food (sourdough starter), and family life (Emma kindergarten). Small models pattern-match better with diverse examples.

---

## ~~Item 17~~ ✅ Item 17 — Fix `test_sidecar_scoring` to match what production actually fires

**Status: COMPLETED.** Test now fires 3 real production prompts in sequence.

### Implemented
| File | Line | Change |
|------|------|--------|
| `sidecar_service.py` (test_sidecar_scoring) | 997-1048 | Rewrote to call `summarize_and_score()`, `extract_tags()`, `extract_facts()` sequentially instead of a phantom combined prompt |

Test pass/fail now reflects actual production behavior. Removed bare-list edge case handling (each separate prompt has its own type guards). Field checks updated to `exchange_summary`/`exchange_outcome` per Item 12 alignment.

---

## ~~Item 18~~ ✅ Item 18 — Bump facts `max_tokens` from 2000 to 4000

**Status: COMPLETED.** Facts call now has same output budget as score call.

### Implemented
| File | Line | Change |
|------|------|--------|
| `roampal.ts` (facts call) | 1103 | `max_tokens: 2000` → `max_tokens: 4000` |

Reasoning models (qwen3, qwen3.6, glm) burn 500-2000 tokens on chain-of-thought before producing JSON. With only 2000 total budget, a 1500-token CoT left just 500 for output → truncated JSON → parse failure. Now matches score call budget.

---

---

## ~~Item 19~~ ✅ Item 19 — `scoreExchangeViaLLM` mutex → async queue + per-session deferred retry (v0.5.5 unfinished)

**Status: COMPLETED.** Async queue ships the code v0.5.5 notes promised; per-session deferred retry with profile refresh hardens multi-instance concurrency.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal.ts` L222-232 | Replace `scoringInFlight` mutex with `_ScoringQueueItem` type, `scoringQueueRunning`, `scoringQueue[]` | ✅ Done |
| `roampal.ts` L645-678 | New public `scoreExchangeViaLLM()` wrapper — queues concurrent callers instead of dropping them | ✅ Done |
| `roampal.ts` L680+ | Private `_scoreExchangeViaLLM()` worker — original scoring logic, no mutex gate | ✅ Done |
| `roampal.ts` L301-310 | Replace single-slot `pendingScoring` with `pendingScoringQueue: Map<sessionId, _PendingScoringEntry>` + retry budget | ✅ Done |
| `roampal.ts` L1864-1893 | Deferred retry block iterates per-session map with retry counter + `refreshProfile(pendingSid)` | ✅ Done |
| `roampal.ts` L1907-1915 | Failure path writes to per-session map instead of overwriting global slot | ✅ Done |
| `roampal.ts` L1960 | Session-end cleanup deletes from `pendingScoringQueue` as well | ✅ Done |

### What's missing

The v0.5.5 release notes describe a `scoringQueue` / `scoringQueueRunning` / `lastQueuedResult` async-queue refactor in `roampal.ts` (see `dev/docs/releases/v0.5.5/RELEASE_NOTES.md:226-273`). **That code never landed.**

Verification:

- `grep -n scoringQueue roampal/plugins/opencode/roampal.ts` → zero hits. Same in `.venv/Lib/site-packages/roampal/plugins/opencode/roampal.ts`.
- `git log --all -S "scoringQueue" -- roampal/plugins/opencode/roampal.ts` → zero commits.
- `git show 6bac65e -- roampal/plugins/opencode/roampal.ts` (the v0.5.5 commit) shows only the delimiter-fencing prompt change.
- Current shipped code at `roampal/plugins/opencode/roampal.ts:222-223` and `:640-643` is the original v0.3.2 hard-skip mutex:
  ```ts
  let scoringInFlight = false
  ...
  if (scoringInFlight) {
    debugLog(`scoreExchange SKIP — already in flight`)
    return false
  }
  ```

So the symptom the v0.5.5 notes claimed to fix — *"~50% of exchanges dropped with slow localhost models like qwen3:1.7b"* — is still reproducible in the shipped binary.

### Existing partial mitigation (and where it falls down)

When mutex skip returns `false`, the caller at `roampal.ts:1853-1856` stuffs the payload into a global `pendingScoring` slot, retried once on the next `session.idle`. That works for the simple case but has three robustness holes — the second and third matter especially for the multi-OpenCode-instance-against-one-sidecar setup:

1. **Single global slot.** `let pendingScoring: {...} | null = null` at `roampal.ts:294`. If session B's scoring fails before session A's deferred retry runs, **A's payload is overwritten and lost.**
2. **Single retry then drop.** `roampal.ts:1832, 1837` — on retry failure, `pendingScoring = null`. Two consecutive failures drops the data permanently.
3. **Global mutex serializes both sessions.** A slow ~28s qwen3:1.7b call from session A blocks scoring for session B; session B's exchange goes through the skip-and-defer path even though session B might be hitting a different (faster) target.

### Fix

Two changes in `roampal/plugins/opencode/roampal.ts`. (a) ships the queue refactor the v0.5.5 notes promised. (b) replaces the single-slot deferred queue with a per-session map and bumps the retry budget.

**(a) Replace `scoringInFlight` mutex with `scoringQueue` async queue.**

Split the public function from the work. Public `scoreExchangeViaLLM()` becomes a queue wrapper; private `_scoreExchangeViaLLM()` keeps the existing logic.

```ts
// Replace lines 222-223:
//   let scoringInFlight = false
// With:
type _ScoringQueueItem = {
  sessionId: string
  currentUserMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
  resolve: (ok: boolean) => void
}
let scoringQueueRunning = false
const scoringQueue: _ScoringQueueItem[] = []

// New public wrapper (replaces the existing scoreExchangeViaLLM signature at line 633).
// Move the existing function body into _scoreExchangeViaLLM (private).
async function scoreExchangeViaLLM(
  sessionId: string,
  currentUserMessage: string,
  exchange: { user: string; assistant: string },
  memories: Array<{ id: string; content: string }> | null
): Promise<boolean> {
  if (scoringQueueRunning) {
    debugLog(`scoreExchange QUEUED — ${scoringQueue.length + 1} waiting`)
    return new Promise<boolean>((resolve) => {
      scoringQueue.push({ sessionId, currentUserMessage, exchange, memories, resolve })
    })
  }

  scoringQueueRunning = true
  try {
    const result = await _scoreExchangeViaLLM(sessionId, currentUserMessage, exchange, memories)
    return result
  } finally {
    // Drain queue sequentially. Each waiter gets its own awaited result.
    while (scoringQueue.length > 0) {
      const next = scoringQueue.shift()!
      try {
        const ok = await _scoreExchangeViaLLM(
          next.sessionId, next.currentUserMessage, next.exchange, next.memories
        )
        next.resolve(ok)
      } catch (err) {
        debugLog(`scoreExchange queued-call error: ${err}`)
        next.resolve(false)
      }
    }
    scoringQueueRunning = false
  }
}
```

Notes:
- The earlier draft in v0.5.5 notes used a single `lastQueuedResult` shared across all waiters — that's wrong (waiter B would see waiter A's result). The version above gives each waiter its own resolved value.
- The drain loop runs **inside the `finally`** so the running flag is only flipped off after the queue is empty. Without that, a new request arriving during drain would race with the loop and start a parallel run.
- Drop the `// Mutex: only one scoring call at a time...` comment block at `roampal.ts:639` (the early-return at 640-643 disappears with this rewrite — `_scoreExchangeViaLLM` should not gate on the queue).

**(b) Per-session deferred retry, two attempts before drop.**

Replace the single-slot `pendingScoring` with a `Map<sessionId, payload>` and add a retry counter. Two consecutive failures still drop, but two *different sessions* no longer overwrite each other.

```ts
// Replace lines 294-299:
//   let pendingScoring: { sessionId; userMessage; exchange; memories } | null = null
// With:
type _PendingScoringEntry = {
  userMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
  retryAttempts: number
}
const pendingScoringQueue = new Map<string, _PendingScoringEntry>()
const PENDING_SCORING_MAX_RETRIES = 3   // initial attempt + 2 retries
```

Update the `session.idle` deferred-retry block at `roampal.ts:1820-1839` to iterate the map (drain a copy of entries; each retry runs through the queue at item (a) so concurrency is fine):

```ts
if (pendingScoringQueue.size > 0 && !SIDECAR_DISABLED) {
  const entries = Array.from(pendingScoringQueue.entries())
  for (const [pendingSid, payload] of entries) {
    payload.retryAttempts++
    debugLog(`session.idle: Retrying deferred scoring for ${pendingSid} (attempt ${payload.retryAttempts}/${PENDING_SCORING_MAX_RETRIES})`)
    try {
      const ok = await scoreExchangeViaLLM(pendingSid, payload.userMessage, payload.exchange, payload.memories)
      if (ok) {
        pendingScoringQueue.delete(pendingSid)
        debugLog(`session.idle: Deferred scoring succeeded for ${pendingSid}`)
      } else if (payload.retryAttempts >= PENDING_SCORING_MAX_RETRIES) {
        pendingScoringQueue.delete(pendingSid)
        consecutiveFailures++
        debugLog(`session.idle: Deferred scoring exhausted retries for ${pendingSid}, dropping payload`)
      } else {
        debugLog(`session.idle: Deferred scoring failed for ${pendingSid}, will retry next idle`)
      }
    } catch (err) {
      if (payload.retryAttempts >= PENDING_SCORING_MAX_RETRIES) {
        pendingScoringQueue.delete(pendingSid)
        consecutiveFailures++
        debugLog(`session.idle: Deferred scoring error for ${pendingSid}, exhausted: ${err}`)
      } else {
        debugLog(`session.idle: Deferred scoring error for ${pendingSid}, retry queued: ${err}`)
      }
    }
  }
}
```

And the failure path at `roampal.ts:1853-1862` becomes:

```ts
const ok = await scoreExchangeViaLLM(sid, scoringData.userMessage, scoringData.exchange, scoringData.memories)
if (!ok) {
  pendingScoringQueue.set(sid, { ...scoringData, retryAttempts: 0 })
  debugLog(`session.idle: Sidecar failed — queued for deferred retry (session ${sid})`)
}
```

Don't forget the cleanup hook at `roampal.ts:1905`: replace `pendingScoringData.delete(sid)` with both `pendingScoringData.delete(sid)` *and* `pendingScoringQueue.delete(sid)` so a destroyed session doesn't leave debris.

### Why this fixes Item 4's profile concern too

Item 4 (refresh profile in session.idle scope) becomes load-bearing once the deferred queue actually persists multiple sessions' payloads simultaneously. With the single-slot `pendingScoring`, only one session's profile mismatch was ever in play. With per-session retries, multiple sessions' deferred payloads coexist — Item 4's `await refreshProfile(pendingSid)` at the top of each retry attempt is the right place for that call.

In the retry block above, add at the top of the loop body (matches Item 4's option B):

```ts
await refreshProfile(pendingSid)   // v0.5.6: ensure deferred retry uses the right profile
```

### Validation

Manual reproduction with two OpenCode instances pointed at the same sidecar (qwen3:1.7b on Ollama):
1. Send a message in session A. Sidecar takes ~25-30s.
2. While A is scoring, send a message in session B.
3. **Pre-fix:** plugin debug log shows `scoreExchange SKIP — already in flight` for B. B's payload goes into the single `pendingScoring` slot. If A also fails (any reason), B's payload is overwritten on the next idle and silently lost.
4. **Post-fix:** plugin debug log shows `scoreExchange QUEUED — 1 waiting` for B. After A completes, B's call runs against the same sidecar and stores its summary correctly. No overwrite, no drop.

Plus a unit-style test on the queue logic itself (mock `_scoreExchangeViaLLM`, push N payloads, verify all N resolve in submission order with their own results).

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/plugins/opencode/roampal.ts` | 222-223 | Replace `scoringInFlight` boolean with `scoringQueue` + `scoringQueueRunning` |
| `roampal/plugins/opencode/roampal.ts` | 294-299 | Replace single-slot `pendingScoring` with `pendingScoringQueue: Map<sessionId, _PendingScoringEntry>` |
| `roampal/plugins/opencode/roampal.ts` | 633-1190 | Rename existing function to `_scoreExchangeViaLLM`; remove mutex gate at 640-643; new public `scoreExchangeViaLLM` wrapper as shown |
| `roampal/plugins/opencode/roampal.ts` | 1820-1839 | Iterate `pendingScoringQueue` with retry counter + per-session profile refresh |
| `roampal/plugins/opencode/roampal.ts` | 1853-1862 | Update failure path to write into the per-session map |
| `roampal/plugins/opencode/roampal.ts` | 1905 | Add `pendingScoringQueue.delete(sid)` to session-end cleanup |

---

---

# MCP tool definition quality (TDQS audit)

External evaluation against the Tool Definition Quality Score rubric (Purpose 25%, Usage Guidelines 20%, Behavioral Transparency 20%, Parameter Semantics 15%, Conciseness & Structure 10%, Contextual Completeness 10%) showed our descriptions score high on Purpose but drag on every other dimension. Items 20-25 rewrite each tool's `types.Tool(...)` block in `roampal/mcp/server.py` to lift the per-tool score into A range (≥3.5) so the server-level quality score (60% mean + 40% min) clears B comfortably.

Pattern of issues found in the first audit (`add_to_memory_bank`, scored 3.0/5):
- **BEHAVIOR section is half advice.** Bullets like "Keep facts concise (~300 chars). One concept per fact" are usage guidance, not behavior contracts. Real behavior questions (idempotency, sync vs eventual searchability, storage location, atomicity) go unanswered.
- **Cross-references to undefined concepts** ("working memory", "outcome-scored", "context injection", "books collection via CLI") leave a fresh agent unable to disambiguate.
- **Constraints in prose only.** `noun_tags` says "lowercase, 1-3 words, max 8" but the JSON Schema has no `maxItems`/`pattern`. `tags` lists six semantic categories but is not an `enum`.
- **Repetition across description and per-param descriptions** ("permanent" 4×; "~300 chars / one concept per fact" twice).
- **Optional params prescribed by required-use bullets** — the WHEN TO USE bullet says `tags=["identity"]` for the most common case but `tags` is not in `required[]`.

Items 21-25 audit `update_memory`, `delete_memory`, `search_memory`, `record_response`, and `score_memories` and apply the same rewrite pattern + style guide.

---

## ~~Item 20~~ ✅ Item 20 — Rewrite `add_to_memory_bank` tool description (TDQS 3.0 → ≥3.8 target)

**Status: COMPLETED.** Description rewritten with mechanical Behavior, self-contained Usage, no undefined jargon. Schema tightened with enum on tags, pattern on noun_tags, maxLength on content. Removed overly-prescriptive `always_inject`.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal/mcp/server.py` L546-615 | Rewritten description (Purpose/Usage/Behavior/Errors/Returns), tightened schema with enum/pattern/ranges, removed `always_inject` | ✅ Done |

### What's missing

`roampal/mcp/server.py:546-587` defines `add_to_memory_bank`. External TDQS scoring:

| Dimension | Score | Why |
|---|---|---|
| Purpose | 5/5 | Opening line is clear and complete. Keep as-is. |
| Usage Guidelines | 2/5 | References "working memory", "books collection via CLI", "auto-captured by the scoring system" without defining them. |
| Behavioral Transparency | 2/5 | Section is half usage advice. No mention of sync writes, dedup contract, embedding behavior, ChromaDB as backing store. |
| Parameter Semantics | 3/5 | `tags` lacks `enum`; `noun_tags` lacks `maxItems`/`pattern`; `importance` vs `confidence` not differentiated. |
| Conciseness & Structure | 3/5 | "permanent" 4×, "~300 chars" repeated in BEHAVIOR and `content` param. 27 lines for a write-one-fact tool. |
| Contextual Completeness | 2/5 | "Context injection" / "outcome-scored" never defined inline. doc_id format only shown by example. No scope statement (per-profile? global?). |

Server-level math: a single 2/5 in any dimension lowers the 40%-min component significantly. Lifting Behavior/Usage/Completeness from 2 to ≥4 alone moves the tool score from 3.0 to ~3.9.

### Fix

Rewrite the description and tighten the schema. The change is contained to `roampal/mcp/server.py:546-587`.

**(a) New description block.** Replace the existing description string with this — same five-section spine, but Behavior is now mechanical, Usage is self-contained, and undefined jargon is gone:

```python
description="""Permanent fact for cross-session memory: identity, preferences, goals, project context.

WHEN TO USE
• Identity (name, role) → tags=["identity"]
• Standing preference/rule → tags=["preference"]
• Persistent project fact → tags=["project"]
• Effectiveness tip for this user → tags=["system_mastery"]

WHEN NOT TO USE
• Session-only fact → skip; conversation history covers it.
• Last-response takeaway → record_response.
• Doc/research dump → out of scope (books collection).

BEHAVIOR
• Writes to memory_bank (current profile) after a tier-internal dedup check. If a near-duplicate already exists in memory_bank (cosine distance < 0.32, ~95% similar), the existing doc_id is returned and no new entry is written. Dedup scans memory_bank only — it never matches against working/history/patterns.
• Searchable via search_memory immediately on return.
• Persists until update_memory or delete_memory. Not modified by score_memories.

ERRORS
• Missing content or noun_tags → ValidationError.
• Backend unreachable → "Error: ..." with cause.

RETURNS: "Added to memory bank (ID: memory_bank_<8hex>)"."""
```

**(b) Tighten the input schema.**

```python
inputSchema={
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "Fact to store. ~300 char target, 600 hard cap. Example: \"Maya is a data scientist focused on AI memory systems\""
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]
            },
            "description": "Semantic category. Optional; recommended."
        },
        "noun_tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9 -]{0,30}$"
            },
            "description": "Topic nouns for retrieval. Names not pronouns. Example: [\"maya\", \"data science\", \"roampal\"]"
        },
        "importance": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.7,
            "description": "Ranking weight at retrieval. 0.9+ core identity, 0.5 low-priority. Default 0.7."
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.7,
            "description": "Trust level at retrieval. 0.9+ verified, 0.5 unconfirmed. Default 0.7."
        },
    },
    "required": ["content", "noun_tags"]
}
```

Notes:
- `tags` stays optional (a fact may be untagged), but the WHEN TO USE bullets now explicitly pair each common case with the right tag, so the optionality has a default behavior.
- `enum` on `tags` items prevents the agent from inventing tags that don't match the rest of the system's category space.
- `pattern` on `noun_tags` enforces the lowercase + dash/space + length constraints that prose-only docs couldn't.
- `maxLength: 600` on `content` gives a hard ceiling without preventing the legitimate ~300-char target.

### Validation

1. Re-score the rewritten description against the same rubric. Target: every dimension ≥4, overall ≥3.8 (A tier).
2. Behavior smoke test: call `add_to_memory_bank` with a sample fact, confirm `search_memory` returns it on the next call (verifies the "immediately searchable" claim).
3. Schema validation: send a `tags=["nonsense"]` request; verify it now ValidationErrors instead of being accepted.
4. Send `noun_tags=["UPPERCASE"]`; verify pattern rejects.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 546-587 | Rewrite `add_to_memory_bank` description + inputSchema per (a) and (b) above |

---

## ~~Item 21~~ ✅ Item 21 — Fix `update_memory` to require doc_id instead of semantic-match guesswork (TDQS 2.4 → ≥3.8)

**Status: COMPLETED.** All four layers shipped. `update_memory` now requires an exact `id` from search_memory — no semantic fallback.

### Original bug

The MCP tool `update_memory` described itself as doing a semantic match on `old_content` to find the memory to update. The code did no such thing. The call chain was:

| File:line | Call |
|---|---|
| `roampal/mcp/server.py:977` | `_api_call("POST", "/api/memory-bank/update", {"old_content", "new_content"})` |
| `roampal/server/main.py:1229` | `_memory.update_memory_bank(old_content=..., new_content=...)` |
| `unified_memory_system.py:1007` | `self._memory_bank_service.update(old_content, new_content)` |
| `memory_bank_service.py:130` | `update(self, doc_id: str, new_text: str, reason="llm_update")` |

The last hop passes `old_content` positionally into `doc_id`. `get_fragment(doc_id)` fails (content is never a valid doc_id), and the fallback silently calls `self.store(new_text, tags=["updated"])`. **Every `update_memory` call created a duplicate memory tagged `["updated"]` instead of updating anything.**

### Design decision: id-only path, no semantic fallback

Semantic-match update is wrong for `update_memory`. The tool is meant to correct a specific known fact — the agent already has the doc_id from `search_memory`. Semantic search introduces ambiguity: it can match the wrong memory, especially with short or generic corrections. The correct design is:

1. Agent calls `search_memory` → gets a list of results, each with an `[id:memory_bank_xxxx]`
2. Agent picks the exact ID → calls `update_memory` with `id=memory_bank_xxxx`
3. The update goes to that exact memory. No guessing.

This makes `update_memory` the precise scalpel it should be, and `delete_memory` already works this way (top-5 search + user-facing semantic, but still targets a specific result set).

### Shipped implementation

**Layer (a) — `memory_bank_service.py`: new `update_by_id()` method**

```python
async def update_by_id(
    self,
    doc_id: str,
    new_content: str,
    *,
    tags: Optional[List[str]] = None,
    noun_tags: Optional[List[str]] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
) -> Optional[str]:
    """Update a memory_bank entry by its exact doc_id.
    No semantic search — the caller must provide the ID from a prior
    search_memory call. Returns doc_id on success, None if ID not found."""
```

Direct lookup via `get_fragment(doc_id)`, re-embed `new_content`, patch metadata fields that were provided, call `upsert_vectors`. No threshold, no semantic search, no `UPDATE_MATCH_THRESHOLD`.

**Layer (b) — `unified_memory_system.py`: `update_memory_bank()` signature**

```python
async def update_memory_bank(
    self, doc_id: str, new_content: str,
    tags=None, noun_tags=None, importance=None, confidence=None,
) -> Optional[str]:
```

No `old_content` parameter. Routes directly to `_memory_bank_service.update_by_id()`.

**Layer (c) — `server/main.py`: `MemoryBankUpdateRequest`**

```python
class MemoryBankUpdateRequest(BaseModel):
    id: str                          # required — exact doc_id from search_memory
    new_content: str
    tags: Optional[List[str]] = None
    noun_tags: Optional[List[str]] = None
    importance: Optional[float] = None
    confidence: Optional[float] = None
```

No `old_content` field. `id` is required.

**Layer (d) — `mcp/server.py`: `update_memory` tool schema**

```python
"required": ["id", "new_content"]
```

Tool description, handler, and schema all reflect id-only update. `old_content` removed from everywhere.

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/memory_bank_service.py` | Replaced `update_by_content()` with `update_by_id()` — requires `doc_id`, no semantic search |
| `roampal/backend/modules/memory/unified_memory_system.py` | `update_memory_bank()` signature: `doc_id` required, `old_content` removed |
| `roampal/server/main.py` | `MemoryBankUpdateRequest`: `id` required, `old_content` removed |
| `roampal/server/main.py` | Endpoint handler passes `doc_id=request.id` instead of `old_content=request.old_content` |
| `roampal/mcp/server.py` | Tool description + schema rewritten: `id` required, `old_content` removed, TDQS-graded |
| `roampal/mcp/server.py` | Handler: reads `id` and `new_content` from arguments, forwards to API |

### Validation

1. Call `update_memory` with a valid `id` from `search_memory` → content updated, doc_id preserved, metadata preserved where not overridden.
2. Call `update_memory` with an invalid `id` → "Memory not found for update".
3. Call `update_memory` with `id` + `importance=0.9` → importance updated, other metadata unchanged.
4. Call `update_memory` without `id` → ValidationError from schema.

### Live verification (run on 2026-04-30 against v0.5.6 editable install, default profile)

Round-trip on `memory_bank_882b0815` ("User's name is Maya"):

| Step | Call | Result |
|------|------|--------|
| 1 | `search_memory(id="memory_bank_882b0815")` | Original: `User's name is Maya` (3mo, imp:1.0, conf:0.7, w:0.22, 4 uses, [Y]) |
| 2 | `update_memory(id="memory_bank_882b0815", new_content="User's name is Maya.")` | `Updated memory (ID: memory_bank_882b0815)` |
| 3 | `search_memory(id="memory_bank_882b0815")` | Now: `User's name is Maya.` — period appeared, all metadata preserved (age, imp, conf, wilson, uses, outcomes glyph) |
| 4 | `update_memory(id="memory_bank_deadbeef", new_content="...")` | `Memory not found for update` — no fallback to semantic match, no duplicate created |
| 5 | `update_memory(...)` to restore | Reverted to `User's name is Maya` |
| 6 | `search_memory(id=...)` | Confirmed restored |

Confirms end-to-end:
- id-only path works: content replaced, doc_id preserved, all metadata fields preserved when not overridden
- Invalid id is a hard failure, not a silent fallback (the original bug)
- Schema pattern `^memory_bank_[a-f0-9]{8}$` is enforced at the MCP edge

### Desktop port (v0.3.3)

`roampal-desktop` ships its own ChromaDB-backed update path. Once the v0.5.5 phantom-filter port lands in v0.3.3, this v0.5.6 id-only update change needs to mirror over: same bug (positional `old_content` into `doc_id`), same fix (require `id` from search).

---

## ~~Item 22~~ ✅ Item 22 — Rewrite `delete_memory` description to match actual archive() semantics (TDQS 2.4 → ≥3.8)

**Status: COMPLETED.** Description corrected from "permanently remove" / "irreversible" to accurately reflect soft-delete/archive behavior. Schema tightened with minLength/maxLength.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal/mcp/server.py` L651-684 | Rewritten description (Purpose/Usage/Behavior/Errors/Returns), tightened schema | ✅ Done |

### What's wrong

`roampal/mcp/server.py:623-656` describes `delete_memory` as *"Permanently remove a memory_bank fact. This action is irreversible."* The implementation at `server.py:993-1009` POSTs `/api/memory-bank/archive`, which routes through `unified_memory_system.delete_memory_bank` → `MemoryBankService.archive()` at `memory_bank_service.py:172`. `archive()` is **soft-delete**: sets `metadata.status="archived"`, the entry persists in ChromaDB, hidden from search via the existing `$ne: archived` filter. Hard-delete only happens later via `cleanup_archived()` (covered by Items 5, 8, 10).

The `archive()` docstring itself explains the design: *"HNSW index doesn't support true deletion — collection.delete() marks entries as deleted but they remain queryable in the vector graph, causing phantom matches during dedup. Soft delete is reliable, reversible…"* The soft-delete is the v0.5.5 phantom fix. Description just never got updated.

Every key behavior claim is false: "Permanently remove" → no; "irreversible" → no; "Deletes the best-matching memory permanently" → no. Honesty bug at the MCP boundary; behavior is correct.

### TDQS table

| Dimension | Score | Why |
|---|---|---|
| Purpose | 4/5 | Concise but factually wrong. |
| Usage Guidelines | 1/5 | "Permanently remove" framing routes agents to this tool with wrong expectations. |
| Behavioral Transparency | 1/5 | Every BEHAVIOR bullet describes hard-delete that does not happen. |
| Parameter Semantics | 2/5 | Single `content` string with no length cap or example. |
| Conciseness & Structure | 5/5 | Already short — keep. |
| Contextual Completeness | 2/5 | No mention of soft-delete, archived status, recoverability, eventual cleanup. |

### Fix

Description and schema rewrite. No backend change — soft-delete behavior is correct as-is.

```python
description="""Remove a memory_bank entry. Stops appearing in search and dedup. No undo via MCP — call only when you're sure.

WHEN TO USE
• Fact wrong, stale, or causing bad responses → delete_memory.
• Redundant entry superseded by a newer fact → delete_memory.

WHEN NOT TO USE
• Topic still relevant, only details changed → update_memory.
• Working/history/patterns are scoring-managed → score_memories.

BEHAVIOR
• Top-5 content-based search over memory_bank (current profile). Picks the first active result where the queried content is a substring of the entry's content (either direction); falls back to the top active result. "Memory not found for deletion" when no active entry exists in the top-5.
• On match: entry no longer returned by search_memory and not considered for dedup on future writes.
• No restore path is exposed via MCP.

ERRORS
• Missing content → ValidationError.
• No match → "Memory not found for deletion".
• Backend unreachable → "Error: ..." with cause.

RETURNS: "Memory deleted successfully" on success; "Memory not found for deletion" on no match."""
```

```python
inputSchema={
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "Fact to archive. Semantic match — paraphrase OK. Example: \"User prefers dark mode\""
        }
    },
    "required": ["content"]
}
```

### Naming question (for the dev reading this)

The MCP tool name `delete_memory` is the user-facing lie, even after the description is honest. Two options:
1. **Keep the name.** The new description is accurate; agents already know the name; renaming is a breaking change for any caller that has memorized the tool catalog.
2. **Rename to `archive_memory`.** Honest at the name level. Breaks every existing agent workflow that learned `delete_memory`.

Recommendation: keep the name (option 1). "Delete" is the user's mental verb for "make it stop appearing"; the description now tells the agent how that's actually implemented. Revisit if/when an `actually_hard_delete_memory` use case shows up.

### Validation
1. Re-score against rubric; target ≥3.8.
2. Smoke test: archive an entry, confirm `search_memory` no longer returns it.
3. Schema: `content=""` → ValidationError.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 623-656 | Replace description + schema per above |

---

## ~~Item 23~~ ✅ Item 23 — Tighten `search_memory` description (TDQS 3.2 → ≥3.8)

**Status: COMPLETED.** Description rewritten with full Behavior mechanics, collection details, output format, and error paths. Schema tightened for metadata and sort_by.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal/mcp/server.py` L469-545 | Rewritten description + tightened metadata/sort_by schemas | ✅ Done |

### What's missing

`roampal/mcp/server.py:469-545`. The tool already scores B (highest among the six). Lifts target Behavior and Usage Guidelines.

| Dimension | Score | Why |
|---|---|---|
| Purpose | 5/5 | Keep. |
| Usage Guidelines | 2/5 | WHEN bullets only mention `id` and `days_back`; no decision rule for `sort_by`, `type`, `metadata`. WHEN NOT bullets are obvious. |
| Behavioral Transparency | 2/5 | "Returns ranked results with scores, age, and usage metadata" describes the output appearance, not the mechanism. Threshold gating, auto-routing, cross-collection comparability, profile scope all unstated. |
| Parameter Semantics | 3/5 | `metadata` description is generic ("Optional metadata filters"). `sort_by` says "Auto-detected" but never defines the trigger. |
| Conciseness & Structure | 4/5 | Already good. |
| Contextual Completeness | 3/5 | Result-format glyphs `[YYN]`, `s:0.7`, `w:0.68` shown but not defined inline. |

### Fix

Description rewrite:

```python
description="""Search persistent memory (current profile) across collections. Returns ranked results with metadata.

WHEN TO USE
• User references past conversations ("remember", "I told you", "we discussed") → query=<their words>.
• Need detail beyond auto-injected context → query=<topic>.
• Verify or fetch a specific entry → id="<doc_id>".
• Browse recent memories → days_back=N (alone or with query).
• Filter to atomic facts vs exchange summaries → type="fact" / "summary".
• "Last thing we did" / temporal queries → sort_by="recency".
• Filter by stored metadata (e.g. tag) → metadata={"tags": "identity"}.

WHEN NOT TO USE
• Question answerable from training data or already-injected context.
• Storing something → add_to_memory_bank or record_response.

BEHAVIOR
• Embedding cosine search per collection (over-fetched), then merged. If the query matches known noun_tags, a tag-routed pass also runs and overlap-counts hits across tags. Date/metadata filters applied next.
• Final ranking is raw cross-encoder score over the candidate pool.
• Top-K returned. No distance threshold drop — top-K (set by `limit`) is what bounds results.
• Archived memory_bank entries filtered out at the source via `status != archived` predicate.
• Collections: working (recent exchanges), history (scored exchanges), patterns (proven recurring), memory_bank (permanent facts), books (ingested docs).
• Auto-routing (omit `collections`): `routing_service.route_query` picks collections by keyword/tag overlap; recommended default.
• Profile-scoped (current ROAMPAL_PROFILE).
• Output is a numbered list. Each item: `N. [collection] (meta) [id:doc_id] content`.
  – working/history/patterns meta: age, s:<score:.1f>, w:<wilson:.2f>, "<uses> uses", outcomes glyph.
  – memory_bank meta: age, imp:<importance:.1f>, conf:<confidence:.1f>; plus w:/uses/outcomes if the entry has been scored.
  – books meta: age, 📖 <title>.
  – Outcomes glyph is last 3 entries from outcome_history, e.g. `[Y~N]`. Y=worked, N=failed, ~=partial.

ERRORS
• None of query/days_back/id provided → "Provide at least one of: query, days_back, or id".
• No matches → "No results found for '...'".
• id lookup miss → "Memory '<id>' not found".
• Invalid collection name → silently ignored.

RETURNS: Numbered list of formatted result lines."""
```

Schema tightening — `metadata` and `sort_by` get real semantics:

```python
"metadata": {
    "type": "object",
    "description": "Equality filters on entry metadata fields. Example: {\"memory_type\": \"fact\", \"tags\": \"identity\"}",
    "additionalProperties": True
},
"sort_by": {
    "type": "string",
    "enum": ["relevance", "recency", "score"],
    "description": "relevance = vector distance (default). recency = updated_at desc. score = stored Wilson score desc. Auto-detected to recency for temporal queries (\"last\", \"recent\", \"today\")."
}
```

The other params already have decent descriptions; leave them.

### Validation
1. Re-score; target ≥3.8.
2. Smoke test: query without args → error string. query="x" with no matches → "No results found".
3. Verify the YYN/s/w glyph claims match the actual formatter (`unified_memory_system._format_results` or wherever) — if the formatter has drifted, fix the formatter or the description so they agree. **This is the kind of drift that turned Items 21/22 into bug fixes.**

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 469-545 | Replace description per above; tighten `metadata` and `sort_by` schemas |

---

## ~~Item 24~~ ✅ Item 24 — Schema-tighten `record_response` (TDQS 4.1 → ≥4.5)

**Status: COMPLETED.** Behavior section rewritten with promotion mechanics, raw-score deltas, and Wilson explanation. Schema tightened with minLength/maxLength on key_takeaway, minItems/maxItems/pattern on noun_tags. Removed obsolete error bullet.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal/mcp/server.py` L764-803 | Rewritten BEHAVIOR section, tightened inputSchema, removed obsolete error bullet | ✅ Done |

### What's missing

`roampal/mcp/server.py:725-764`. Already A-tier — Purpose 5, Usage 5, Conciseness 5, Completeness 4. Two targeted lifts.

| Dimension | Score | Why |
|---|---|---|
| Parameter Semantics | 2/5 | `noun_tags` description says "Lowercase, 1-3 words each, max 8" but the schema has no `maxItems`, no `pattern`. `key_takeaway` has no length cap. Same prose-only-constraint pattern that hit `add_to_memory_bank`. |
| Behavioral Transparency | 3/5 | Promotion mechanics (working → history → patterns) are stated, but sync/async, profile scope, and immediate-searchability are not. |
| Others | 4–5/5 | Keep as-is. |

### Fix

**(a) Schema tightening (Parameters 2 → 4):**

```python
inputSchema={
    "type": "object",
    "properties": {
        "key_takeaway": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "1-2 sentence learning. Specific — names, decisions, outcomes. Example: \"User prefers one bundled PR for refactors — splitting would be churn\""
        },
        "noun_tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "pattern": "^[a-z][a-z0-9 -]{0,30}$"},
            "description": "Topic nouns for retrieval. Names not pronouns. Example: [\"react\", \"auth flow\", \"logan\"]"
        }
    },
    "required": ["key_takeaway", "noun_tags"]
}
```

Note: `noun_tags` becomes effectively required by `minItems: 1` + presence in `required[]`. Today the description says missing it "reduces retrieval quality"; that bullet drops out — schema enforces it.

**(b) Description tweak (Behavior 3 → 4):**

Replace the BEHAVIOR section with:

```
BEHAVIOR
• Synchronous write to the working collection (current profile) with initial score 0.7. Searchable via search_memory immediately on return.
• When score_memories fires on later turns the entry's raw score moves by outcome: worked +0.2, partial +0.05, unknown -0.05, failed -0.3 (`outcome_service._calculate_score_update:237-284`). uses and success_count also accumulate.
• Quality-driven promotion uses the raw score: entries with score≥0.7 and uses≥2 move working → history → patterns. Below ~0.4 demoted, below ~0.2 removed (`config.py:28-34`). History items that never promote are cleaned up after ~30 days.
• Wilson lower-bound (computed from success_count/uses) is exposed as `wilson:N%` metadata on injected memories — a trust signal *for the agent to read*, not the substrate that drives demotion.
• conversation_id auto-attached for cross-session tracking.
```

ERROR HANDLING is fine; remove the obsolete "Missing noun_tags → accepted but reduces retrieval quality" bullet (schema now enforces).

**Verification note:** raw-score deltas are real (`outcome_service._calculate_score_update:237-284`): worked +0.2, partial +0.05, unknown -0.05, failed -0.3. Wilson is *not* the active scoring substrate — it's a derived display metric (lower bound of success_count/uses) shown to the agent in injected context as `wilson:N%`. Demotion/deletion thresholds operate on raw score, not Wilson.

### Validation
1. Re-score; target ≥4.5.
2. Schema: `noun_tags=[]` → ValidationError. `noun_tags=["UPPERCASE"]` → pattern reject. `key_takeaway=""` → ValidationError.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 725-764 | Replace BEHAVIOR section, drop obsolete error bullet, tighten inputSchema per (a) |

---

## ~~Item 25~~ ✅ Item 25 — Rewrite `score_memories` (TDQS 1.3 → ≥3.8) — **floor-dragger; critical for server score**

**Status: COMPLETED.** Description rewritten with hook context, multi-purpose surface explanation, promotion mechanics, and Wilson clarification. Schema tightened with enum/additionalProperties on memory_scores, minLength/maxLength on exchange_summary, minItems/maxItems/pattern on noun_tags, maxItems + per-item constraints on facts. This single rewrite lifts the server-level TDQS from ~2.7 (C) to B+.

### Verification
| File | Change | Status |
|------|--------|--------|
| `roampal/mcp/server.py` L703-762 | Rewritten description + tightened all 5 params in inputSchema | ✅ Done |

### Why this matters more than the others

Server-level quality = 60% mean(TDQS) + 40% min(TDQS). With `score_memories` at 1.3, the 40%-min term is 0.52 — even if the other five tools all hit 4.5, the server lands at ~2.7 (C). Lifting this single tool to ≥3.8 is what gates the server-level grade above B.

### What's wrong

`roampal/mcp/server.py:664-723`. The big miss: the description never explains that this tool fires in response to a *scoring hook* — a system-reminder injected at turn boundaries that lists which memory IDs to score. Without that frame:

- WHEN TO USE: "When prompted by the scoring hook" assumes the agent knows what a scoring hook is.
- The five-param surface (`memory_scores`, `exchange_summary`, `exchange_outcome`, `noun_tags`, `facts`) appears unrelated. Why does a "score memories" tool also store summaries and facts? Because the hook fires once per turn boundary and bundles all of it.
- The agent reading this description without hook context can't decide when to call the tool, what to put in `memory_scores`, or what the other fields are for.

| Dimension | Score | Why |
|---|---|---|
| Purpose | 2/5 | Opening line covers only the scoring action; the tool also stores summary, facts, outcome, tags. Underspecified for what the tool does. |
| Usage Guidelines | 1/5 | "Scoring hook" undefined; agent without hook context can't make a decision. |
| Behavioral Transparency | 1/5 | "Updates confidence scores" with no mechanism. "Demoted or removed" with no threshold. exchange_summary/facts storage relationship unclear. |
| Parameter Semantics | 1/5 | Five params with prose-only constraints (`noun_tags` "max 8, lowercase, 1-3 words", `facts` "max 150 chars" — none in schema). `exchange_summary` has no length cap. `facts` has no per-item or array cap. |
| Conciseness & Structure | 2/5 | Verbose for a tool that ships in every context. |
| Contextual Completeness | 1/5 | Hook flow unexplained, conversation_id auto-attachment unmentioned, two-lane summary-vs-fact architecture invisible. |

### Fix

Description rewrite — leads with the hook context, makes the multi-purpose surface explicit:

```python
description="""Record outcomes after a turn boundary: scores the memories that were injected, plus stores the turn's takeaway summary and any atomic facts.

Fires in response to a scoring hook — a system-reminder that lists doc_ids to score and asks for an exchange_summary + exchange_outcome. Don't call without that hook.

WHEN TO USE
• Scoring hook present this turn → call once. Score every doc_id the hook listed. Provide exchange_summary, exchange_outcome, noun_tags. Provide facts if the turn produced atomic specifics worth keeping.

WHEN NOT TO USE
• No scoring hook present → don't call. Use record_response for ad-hoc takeaway capture.
• Adding identity/preference/project facts → add_to_memory_bank.

BEHAVIOR
• Per memory_scores entry: raw score moves by outcome (worked +0.2, partial +0.05, unknown -0.05, failed -0.3 — `outcome_service._calculate_score_update:237-284`); uses and success_count also accumulate. Memories whose raw score drops under ~0.4 are demoted; under ~0.2 are removed (thresholds in `config.py:28-34`).
• Wilson lower-bound (from success_count/uses) is recomputed and surfaced as `wilson:N%` metadata on injected memories — a trust signal for the agent reading conflicting memories, not the active scoring substrate.
• exchange_summary stored as a new entry in the working collection (current profile), labeled with exchange_outcome and tagged with noun_tags. Searchable via search_memory immediately on return.
• Each fact stored as a separate working entry (atomic-fact lane, distinct from the summary lane).
• conversation_id auto-attached from the active MCP session.
• Unknown doc_ids in memory_scores silently skipped (not an error).

ERRORS
• Missing memory_scores → ValidationError.
• Empty memory_scores → accepted; nothing scored, but summary/facts still stored.
• Backend unreachable → "Error: ..." with cause; nothing recorded.

RETURNS: "Scored (N memories updated). Summary stored (M chars)"."""
```

Schema tightening:

```python
inputSchema={
    "type": "object",
    "properties": {
        "memory_scores": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": ["worked", "failed", "partial", "unknown"]},
            "description": "Map doc_id → outcome. worked=helped, partial=somewhat, unknown=present-but-unused, failed=misleading. Score every ID the hook listed. Example: {\"history_abc123\": \"worked\", \"patterns_def456\": \"unknown\"}"
        },
        "exchange_summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "1-3 sentences (~300 chars). What happened, what changed."
        },
        "exchange_outcome": {
            "type": "string",
            "enum": ["worked", "failed", "partial", "unknown"],
            "description": "worked = user confirmed/continued. failed = user corrected. partial = mixed. unknown = unclear."
        },
        "noun_tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "pattern": "^[a-z][a-z0-9 -]{0,30}$"},
            "description": "Topic nouns for the stored summary. Names not pronouns. Example: [\"react\", \"auth bug\", \"logan\"]"
        },
        "facts": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 150},
            "description": "Atomic facts (≤150 chars each). Include dates, names, decisions. Example: [\"User prefers snake_case\", \"v2.0 released 2026-04-01\"]"
        }
    },
    "required": ["memory_scores"]
}
```

### Verified mechanism

Verified at `roampal/backend/modules/memory/outcome_service.py:237-284` (`_calculate_score_update`):
- worked: raw score +0.2, success_count +1.0, uses +1
- partial: raw score +0.05, success_count +0.5, uses +1
- unknown: raw score -0.05, success_count +0.25, uses +1
- failed: raw score -0.3, success_count +0.0, uses +1

Demotion threshold 0.4, deletion threshold 0.2 — `roampal/backend/modules/memory/config.py:28-34`.

Wilson is **not** the active scoring substrate; it's a derived display metric (lower bound of `success_count`/`uses`) surfaced as `wilson:N%` in injected context. The CE rerank in `search_service._rerank_with_ce:132-162` explicitly removed Wilson blending in v0.4.5 because it hurt retrieval. Keep these in their lanes.

### Validation
1. Re-score; target ≥3.8.
2. Schema: `memory_scores={}` → accepted, returns "Scored (0 memories updated)". `noun_tags=["UPPERCASE"]` → pattern reject. `facts=["x" * 200]` → maxLength reject.
3. Smoke test: simulate a hook-driven call with 3 memory IDs (1 valid, 2 unknown) — confirm response says "Scored (1 memories updated)", unknown IDs silently dropped.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 664-723 | Replace description (leads with hook context, explains multi-purpose surface) + schema tighten per above |

---

## ~~Item 26~~ ✅ Item 26 — Remove `always_inject` (dead-code path, never reached the LLM)

### What's wrong

The `always_inject` flag on memory_bank entries was *meant* to mark a memory for unconditional injection into every conversation. It does not work end-to-end:

| Step | Verified at | Status |
|---|---|---|
| Storage of the flag | `memory_bank_service.py:65,76,114` | ✓ — flag persists in metadata |
| Reader function | `memory_bank_service.py:486-499` (`get_always_inject`) | ✓ — returns flagged entries |
| Call site | `unified_memory_system.py:1154-1161` | populates `result["user_facts"]` and tries to add doc_ids to scoring queue |
| **Scoring-queue append is wiped** | `unified_memory_system.py:1196` reassigns `result["doc_ids"]` to just lane-search results | ✗ — `always_inject` IDs never reach the queue |
| **Render to LLM context** | `_format_context_injection` (lines 1201-1319) iterates only `context["memories"]`, never `user_facts` | ✗ — content never displayed |
| Consumer of `GetContextResponse.user_facts` | grep across `roampal/` (Python + TS) | ✗ — zero readers |

So today: writing `always_inject=true` stores a flag, runs a function, populates an unread field. It has zero effect on what the LLM sees and zero effect on the scoring queue. The `<roampal-user-profile>` block at the top of every conversation does NOT come from this path — it comes from `_build_cold_start_profile` (`server/main.py:251-403`), which ranks memory_bank by `importance × confidence` (or Wilson once `uses ≥ 3`) and picks one entry per priority tag. That mechanism never consults `always_inject`.

### Design decision

**Decision:** nothing should bypass and inject every turn. The cold-start profile already produces "always-present" content via tagging + quality, which is the right knob. `always_inject` is redundant and confusing. Remove it.

### Fix — remove the flag from the surface and the dead code path

Touchpoints (all verified by grep on `roampal/` excluding tests):

**(a) MCP surface — already covered by Items 20 and 21** (drop the field from `add_to_memory_bank` schema and the `update_memory` metadata expansion). No additional change required here; the rewrites above already exclude `always_inject`.

**(b) MCP handler — drop the kwarg forwarding.**

```python
# roampal/mcp/server.py:950-967 (add_to_memory_bank handler)
# Before:
always_inject = arguments.get("always_inject", False)
result = await _api_call("POST", "/api/memory-bank/add", {
    "content": content,
    "noun_tags": noun_tags,
    "tags": tags,
    "importance": importance,
    "confidence": confidence,
    "always_inject": always_inject,
})

# After: drop the always_inject line in both reads and the payload.
```

**(c) FastAPI Pydantic model — remove the field.**

`roampal/server/main.py:496` — `MemoryBankAddRequest` (or whichever model holds the add payload) drops the `always_inject: bool = False` field. Line 1206 — endpoint stops passing `always_inject=request.always_inject` into `mem.store_memory_bank`.

**(d) UnifiedMemorySystem — drop the kwarg from `store_memory_bank`.**

`roampal/backend/modules/memory/unified_memory_system.py:944,955,976` — drop the `always_inject` parameter from `store_memory_bank`, drop its docstring line, drop the pass-through to `_memory_bank_service.store`.

**(e) MemoryBankService — drop the storage-layer kwarg.**

`roampal/backend/modules/memory/memory_bank_service.py:65,76,114` — `store()` removes the `always_inject` parameter and the metadata write. Line 486-499 — delete the `get_always_inject()` method entirely.

**(f) UnifiedMemorySystem.get_always_inject wrapper.**

`roampal/backend/modules/memory/unified_memory_system.py:1024-1036` — delete the wrapper method (it just delegates to `_memory_bank_service.get_always_inject`, which we just removed).

**(g) Drop the call site in `get_context_for_query`.**

`roampal/backend/modules/memory/unified_memory_system.py:1147-1161` — remove the `result["user_facts"]` initialization, remove the `# 0. Fetch always_inject memories` block entirely. After this change `get_context_for_query` starts directly at the two-lane retrieval.

**(h) Drop `user_facts` from the response model and assignment.**

`roampal/server/main.py:421` — remove `user_facts: List[Dict[str, Any]]` from `GetContextResponse`. Line 943 — remove the `user_facts=context.get("user_facts", [])` assignment in the response constructor.

**(i) Stale docstring fix — cold-start profile.**

`roampal/server/main.py:255-258` — the `_build_cold_start_profile` docstring claims "Always-inject memories (identity core)" as step 1, but the code below it (line 271+) just calls `list_all` and sorts by quality. Replace step 1 with: "Walk all memory_bank entries sorted by quality (Wilson once `uses ≥ 3`, else `importance × confidence`) and pick the top entry per priority tag."

### Migration / data hygiene

Existing memory_bank entries on user disks have `metadata.always_inject` set (`true` or `false`). After this change the field is ignored everywhere. Two options:

1. **Leave it.** ChromaDB doesn't care about extra metadata fields; they'll just sit there inert.
2. **One-shot strip on startup.** Add a migration in `_startup_cleanup` that scans memory_bank and removes the `always_inject` key from any entry's metadata. Cleanest, but adds a one-time write per existing entry on first v0.5.6 boot.

Recommendation: option 1. The field is harmless once nothing reads it, and avoiding write traffic on startup keeps boot fast. Document in the release notes that the flag is now ignored.

### Validation

1. After the cleanup, grep for `always_inject` and `get_always_inject` across `roampal/` — should match only test files (delete those references) and possibly historical changelog entries.
2. Run `add_to_memory_bank` with no `always_inject` field — confirm normal store path works.
3. Run `get_context_for_query` and confirm no `user_facts` field on the response (or, if response model still has it for backward compatibility, confirm it's always empty).
4. Diff `<roampal-user-profile>` block before/after on a profile that has `always_inject=true` entries: should be identical (because cold-start never used the flag anyway). No behavior change is expected for the LLM.

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 565, 583 | Drop `always_inject` mentions from `add_to_memory_bank` description + schema (covered by Item 20) |
| `roampal/mcp/server.py` | 956, 964 | Drop `always_inject` read and payload field in handler |
| `roampal/server/main.py` | 421, 496, 943, 1206 | Remove `user_facts` from `GetContextResponse`; remove `always_inject` from add request model + endpoint pass-through |
| `roampal/server/main.py` | 255-258 | Fix stale `_build_cold_start_profile` docstring |
| `roampal/backend/modules/memory/unified_memory_system.py` | 944, 955, 976 | Remove `always_inject` from `store_memory_bank` |
| `roampal/backend/modules/memory/unified_memory_system.py` | 1024-1036 | Delete `get_always_inject` wrapper method |
| `roampal/backend/modules/memory/unified_memory_system.py` | 1147-1161 | Remove `user_facts` init + always_inject fetch block |
| `roampal/backend/modules/memory/memory_bank_service.py` | 65, 76, 114 | Remove `always_inject` from `store()` signature, docstring, and metadata write |
| `roampal/backend/modules/memory/memory_bank_service.py` | 486-499 | Delete `get_always_inject()` method |
| any tests referencing `always_inject` | — | Update or delete |

### Desktop port (v0.3.3)

`roampal-desktop` likely has the same `always_inject` references on its memory_bank service. Strip it there too once the core change lands.

### Verification

| Check | Result |
|---|---|
| Grep for `always_inject`, `user_facts`, `get_always_inject` in source (excluding tests/docs) | Zero matches ✅ |
| MCP handler no longer forwards `always_inject` kwarg | Verified ✅ |
| FastAPI model drops `always_inject: bool = False` field | Verified ✅ |
| `store_memory_bank` signature clean of `always_inject` | Verified ✅ |
| `MemoryBankService.store()` clean of `always_inject` | Verified ✅ |
| `get_always_inject()` method deleted from both services | Verified ✅ |
| `user_facts` removed from GetContextResponse model + endpoint | Verified ✅ |
| `_build_cold_start_profile` docstring corrected (no "always-inject" reference) | Verified ✅ |
| All 4 modified Python files pass syntax check | Verified ✅ |
| Relevant tests updated and passing (3/3) | Verified ✅ |

---

## ~~Item 27~~ ✅ Item 27 — Clarify `[2] Use free Zen cloud models` wording for OpenCode Go subscribers

**Status: COMPLETED.** Note line added to both prompt branches.

### Verification
| Check | Result |
|---|---|
| Note line added below `[2]` in no-models-detected branch (`cli.py:3251-3253`) | ✅ |
| Note line added below `[{free_idx}]` in models-detected branch (`cli.py:3342-3344`) | ✅ |
| Uses only existing color constants (`YELLOW`, `RESET`) — no new imports needed | ✅ |
| No behavior change to option handler or config writes | ✅ |
| cli.py passes syntax check | ✅

---

## ~~Item 28~~ ✅ Item 28 — Add `Use OpenCode Go (detected)` option to sidecar setup wizard

**Status: COMPLETED.** Helpers + dual-branch prompt insertion with dynamic renumbering and model picker.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `roampal/cli.py` | Added `_DEFAULT_GO_MODELS`, `_detect_opencode_go()`, `_list_opencode_go_models()` helpers (searches `~/.local/share`, `~/.config`, `%APPDATA%`) | ✅ Done |
| `roampal/cli.py` | No-models-detected branch: Go option with dynamic renumbering + full handler (model picker, env write, defer_write) | ✅ Done |
| `roampal/cli.py` | Models-found branch: Go option inserted after local models with correct index math + identical handler | ✅ Done |

### Verification
| Check | Result |
|---|---|
| cli.py passes syntax check | ✅ |
| Helpers placed before `_detect_ollama_models()` for clean import order | ✅ |
| No-models branch: Go option shows when detected, menu renumbers to [1-4] with skip=4 | ✅ |
| Models-found branch: Go option inserted after local models, indices computed dynamically, menu printed in numerical order | ✅ |
| Handler writes `ROAMPAL_SIDECAR_URL`, `_KEY`, `_MODEL`; clears `_FALLBACK` and `_PRIORITY` | ✅ |
| defer_write path returns full env dict for deferred apply | ✅ |
| `_list_opencode_go_models` has 5s timeout, falls back to `_DEFAULT_GO_MODELS` on any error | ✅ |
| Windows `%APPDATA%` auth.json search path included | ✅ |

---

## ~~Item 3~~ ✅ Item 3 — Improve `_install_plugin_file()` error message to name actual likely causes

**Status: COMPLETED.** Error messages now list OpenCode Desktop lock, read-only attribute, Controlled Folder Access, and OneDrive as possible causes.

### Verification
| Check | Result |
|---|---|
| Exception handler lists 4 specific causes | ✅ |
| "File disappeared" branch lists 3 causes (AV, OneDrive, CFA) | ✅ |
| cli.py passes syntax check | ✅ |

---

## ~~Item 2~~ ✅ Item 2 — Surface `%APPDATA%` unset on Windows instead of silently skipping

**Status: COMPLETED.** Two-line warning added when `APPDATA` env var is empty.

### Verification
| Check | Result |
|---|---|
| Warning prints "Skipped %APPDATA% fallback install" when APPDATA unset | ✅ |
| Gated on `sys.platform == "win32"` — no effect on Linux/macOS | ✅ |
| cli.py passes syntax check | ✅ |

---

## ~~Item 1~~ ✅ Item 1 — Regression test for `_install_plugin_file()`

**Status: COMPLETED.** 6 tests covering all failure modes. All passing.

### Verification
| Check | Result |
|---|---|
| Happy path (shutil.copy works, sizes match) | ✅ Pass |
| Copy raises `PermissionError` — error message printed, no crash | ✅ Pass |
| Copy succeeds but destination missing — "file disappeared" message | ✅ Pass |
| Copy writes 0 bytes — fallback `write_bytes` corrects it | ✅ Pass |
| Copy writes wrong size — fallback corrects it | ✅ Pass |
| Both copy and fallback produce wrong size — error printed | ✅ Pass |

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` (new) | 6 unit tests, all passing |

---

## ~~Item 4~~ ✅ Item 4 — Refresh profile in `session.idle` scope (#10 follow-up)

**Status: COMPLETED.** Added `await refreshProfile(sid)` at top of debounced session.idle handler. Deferred retry already had per-session refresh from Item 19.

### What was missing

The v0.5.5.1 fix re-resolves the profile inside `chat.message` via `refreshProfile(sessionID)`. But `session.idle` makes HTTP calls (lifecycle stop hook at line 1845, current scoring at line 1908) using a **global** `_cachedProfile` that may have been shifted by another session's `chat.message`. This causes cross-profile contamination on multi-project Desktop users.

### Fix

Single `await refreshProfile(sid)` call at the top of the debounced idle handler (line 1823), before any HTTP call. Covers:
- Lifecycle stop hook (`/stop` with `roampalHeaders()`) — now uses correct profile for this session
- Current session scoring (`scoreExchangeViaLLM`) — now uses correct profile for this session
- Deferred retry loop — already had per-session `refreshProfile(pendingSid)` from Item 19

### Verification
| Check | Result |
|---|---|
| `await refreshProfile(sid)` added at line 1823, before all HTTP calls in idle scope | ✅ |
| Deferred retry still has per-session refresh (`await refreshProfile(pendingSid)`, line 1870) | ✅ (already present from Item 19) |
| TypeScript file parses without error | ✅ |

---

## ~~Item 28~~ ✅ Item 28 — Add `Use OpenCode Go (detected)` option to sidecar setup wizard

**Status: COMPLETED.** Helpers + dual-branch prompt insertion with dynamic renumbering and model picker.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `roampal/cli.py` | Added `_DEFAULT_GO_MODELS`, `_detect_opencode_go()`, `_list_opencode_go_models()` helpers | ✅ Done |
| `roampal/cli.py` | No-models-detected branch: Go option with dynamic renumbering + full handler (model picker, env write, defer_write) | ✅ Done |
| `roampal/cli.py` | Models-found branch: Go option inserted after local models with correct index math + identical handler | ✅ Done |

### Verification
| Check | Result |
|---|---|
| cli.py passes syntax check | ✅ |
| Helpers placed before `_detect_ollama_models()` for clean import order | ✅ |
| No-models branch: Go option shows when detected, menu renumbers to [1-4] with skip=4 | ✅ |
| Models-found branch: Go option inserted after local models, indices computed dynamically | ✅ |
| Handler writes `ROAMPAL_SIDECAR_URL`, `_KEY`, `_MODEL`; clears `_FALLBACK` and `_PRIORITY` | ✅ |
| defer_write path returns full env dict for deferred apply | ✅ |
| `_list_opencode_go_models` has 5s timeout, falls back to `_DEFAULT_GO_MODELS` on any error | ✅ |

### What's missing

OpenCode Go subscribers can technically configure a Go model as their sidecar today via `[1] Configure custom API endpoint`, but the UX is hostile:

- They have to know the Go base URL (`https://opencode.ai/zen/go/v1`) — undocumented in the wizard.
- They have to dig their Go API key out of `auth.json` (`opencode-go.key`) — not surfaced anywhere in roampal's UI.
- They have to know which Go-catalog model is small enough for scoring — no guidance.

A Go subscription is otherwise a great fit for sidecar scoring: reliable, sub-second latency, no rate-limit roulette. The wizard should detect Go credentials and offer a one-keystroke option that writes the right `ROAMPAL_SIDECAR_URL`/`_KEY`/`_MODEL` automatically.

### Fix

Add a new helper plus a new menu option in both prompt paths.

**1. New helper in `roampal/cli.py`** (place near `_detect_ollama_models` / `_detect_local_servers`):

```python
_DEFAULT_GO_MODELS = [
    # Fallback list if /models endpoint is unreachable. Pick small/cheap Go-catalog
    # models suitable for JSON-summary scoring. Ghost: CONFIRM exact IDs against
    # https://opencode.ai/docs/go/ before merge — these strings are illustrative,
    # not verified to match OpenCode's actual catalog identifiers.
    "glm-5.1",
    "qwen3.5-plus",
    "deepseek-v4",
]


def _detect_opencode_go() -> dict | None:
    """Detect an OpenCode Go subscription via OpenCode's auth.json.

    Returns a dict with {url, key, models} if detected, or None.

    auth.json location: verified empirically on a Windows install at
    `~/.local/share/opencode/auth.json` (XDG-style, NOT %APPDATA%). Ghost: confirm
    this path against `sst/opencode` source — search for how `auth.json` is
    located in `packages/opencode/src/auth/index.ts`. If OpenCode resolves the
    path differently per-platform, mirror their resolver here.
    """
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        auth = json.loads(auth_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    entry = auth.get("opencode-go")
    if not isinstance(entry, dict) or entry.get("type") != "api" or not entry.get("key"):
        return None
    url = "https://opencode.ai/zen/go/v1"
    models = _list_opencode_go_models(url, entry["key"]) or _DEFAULT_GO_MODELS
    return {"url": url, "key": entry["key"], "models": models}


def _list_opencode_go_models(url: str, key: str) -> list[str] | None:
    """Hit /models on the Go endpoint to list available models. Returns None on failure.

    Note: cli.py imports `urllib.request` (line 23) but does NOT define an
    `_ssl_context()` helper — that lives in `sidecar_service.py:72`. Either
    (a) use `ssl.create_default_context()` directly here (simplest), or
    (b) extract the helper from `sidecar_service.py` into a shared module.
    """
    import ssl
    try:
        req = urllib.request.Request(
            f"{url}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=5, context=ssl.create_default_context()) as resp:
            data = json.loads(resp.read().decode())
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            return ids or None
    except Exception:
        return None
```

**2. Insert a new option in `_sidecar_model_picker`** at both prompt paths (the no-models-detected branch around `cli.py:3247-3253` and the models-detected branch around `cli.py:3334-3341`).

The new option appears **only when `_detect_opencode_go()` returns a dict** — otherwise the menu stays as-is for non-Go users:

```python
go = _detect_opencode_go()
go_idx = None
if go:
    go_idx = len(options) + 1   # in the models-detected branch; for the
                                # no-models-detected branch, hardcode at [2]
    print(
        f"  {BOLD}[{go_idx}]{RESET} Use OpenCode Go (detected) {GREEN}— your subscription, your quota{RESET}"
    )
    print(
        f"      {YELLOW}Reliable scoring via Go's API. Each scored exchange consumes Go credits.{RESET}"
    )
```

The remaining `Configure custom API`, `Use free Zen cloud models`, and `Skip` options re-number around `go_idx` so the prompt input range stays contiguous.

**3. Branch handler when user picks `go_idx`:**

```python
elif choice_num == go_idx:
    # Let user pick which Go model to use as sidecar.
    print(f"\n{BOLD}Available OpenCode Go models for scoring:{RESET}")
    for i, m in enumerate(go["models"], 1):
        print(f"  {BOLD}[{i}]{RESET} {m}")
    try:
        m_choice = input(f"\nChoose [1-{len(go['models'])}]: ").strip()
        m_idx = int(m_choice) - 1
        if not (0 <= m_idx < len(go["models"])):
            raise ValueError
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"\n{YELLOW}Cancelled.{RESET}")
        return None

    chosen_model = go["models"][m_idx]
    print(f"\n{GREEN}Configuring: OpenCode Go ({chosen_model}){RESET}")
    print(f"  {YELLOW}Note: each scored exchange consumes Go credits.{RESET}")

    updates = {
        "ROAMPAL_SIDECAR_URL": go["url"],
        "ROAMPAL_SIDECAR_KEY": go["key"],
        "ROAMPAL_SIDECAR_MODEL": chosen_model,
        "ROAMPAL_SIDECAR_FALLBACK": None,
        "ROAMPAL_SIDECAR_PRIORITY": None,  # custom takes priority over zen automatically
    }
    # Always write env (even in defer_write mode) so the key lands on disk —
    # mirrors how `_prompt_custom_endpoint` handles its API key.
    _apply_sidecar_env_and_write(config_path, updates)

    if defer_write:
        return {"url": go["url"], "model": chosen_model}  # canonical shape; key already written

    return True
```

**4. `defer_write` contract — important.** Currently `_sidecar_model_picker` returns `dict | None` to its caller in defer_write mode. The existing dict shape is exactly `{"url": ..., "model": ...}` — see lines 3297, 3360, 3376-3378, 3400. Notably the existing custom-endpoint deferred path (3370-3382) reads `ROAMPAL_SIDECAR_URL` and `ROAMPAL_SIDECAR_MODEL` from the env file but **drops the key**, because the existing custom path always writes the key directly via `_apply_sidecar_env_and_write` before defer_write returns.

For Go, defer_write needs to carry the key through. Two options:
- **(a)** Always call `_apply_sidecar_env_and_write` immediately in the Go branch (even when `defer_write=True`) so the key lands on disk. This is consistent with `_prompt_custom_endpoint`'s behavior. Then return `{"url": go["url"], "model": chosen_model}` (existing shape).
- **(b)** Return `{"url": go["url"], "model": chosen_model, "key": go["key"]}` and update both deferred-write callers (`cli.py:3453-3456` and `cli.py:3652-3659`) to pass the key through to their `_apply_sidecar_env_and_write` calls.

Recommend (a) — smaller blast radius, no contract change to upstream callers.

### Edge cases ghost should handle

1. **`auth.json` exists but `opencode-go` entry is malformed.** `_detect_opencode_go` returns `None` and the option is hidden — user falls back to `[1]`/`[2]`.
2. **`/models` endpoint returns 401 / 403.** The captured Go key is invalid (expired or revoked). Catch the error in `_list_opencode_go_models`, fall through to `_DEFAULT_GO_MODELS`. If the user picks an option and the first scoring call also 401s, the existing circuit breaker handles it — no extra logic needed in the wizard.
3. **`/models` endpoint times out.** 5s timeout; fall back to `_DEFAULT_GO_MODELS`. Don't block the wizard.
4. **User has an OpenCode Go subscription but their main chat is configured to use a different (non-Go) provider.** The `auth.json` check is independent of which provider OpenCode is currently using — Go option still surfaces, which is fine (they can sidecar on Go even if they main-chat on Claude).

### Files affected
| File | Line | Change |
|------|------|--------|
| `roampal/cli.py` | new (top, near other detectors) | Add `_detect_opencode_go`, `_list_opencode_go_models`, `_DEFAULT_GO_MODELS` |
| `roampal/cli.py` | 3247-3253 | Insert Go option in no-models-detected prompt branch + handler |
| `roampal/cli.py` | 3334-3341 | Insert Go option in models-detected prompt branch + handler |
| `roampal/cli.py` | 3262-3277, 3367-3380 | Extend `defer_write` dict reader if Go path needs it |
| `roampal/backend/modules/memory/tests/unit/test_cli.py` | new test | Mock `auth.json` with/without `opencode-go` entry; assert option surfaces only when valid |

### Validation

1. **Detection on:** Place a fake `auth.json` with an `opencode-go: {type: "api", key: "fake"}` entry under `~/.local/share/opencode/`. Run `roampal sidecar setup`. Confirm the Go option appears.
2. **Detection off:** Remove the `opencode-go` entry. Re-run setup. Confirm the option is absent and the menu numbering is unchanged from current.
3. **`/models` failure:** Block network egress to `opencode.ai` (or stub the urlopen). Confirm fallback list is shown.
4. **End-to-end on a real Go account (maintainer-side):** Pick the Go option, choose a model, restart OpenCode, do one chat exchange, check OpenCode log for a `service=mcp` scoring call against `https://opencode.ai/zen/go/v1` (not `/zen/v1`). Confirm scoring succeeds (200 status, JSON returned).

### Desktop port (v0.3.3)

Same auto-detect option should land in the Desktop sidecar settings UI. Desktop already reads `auth.json` for its own provider list — wiring up a "Use OpenCode Go" toggle there is a small extension. Track separately in v0.3.3.

---

---

## ~~Item 29~~ ✅ Item 29 — Fix user name extraction in context injection to avoid grabbing assistant's name

**Status: COMPLETED.** Regex now only matches agent-stored facts about the user, not third-party references.

### What was wrong

`_format_context_injection()` scans identity-tagged memory_bank entries for a `name is (\w+)` pattern to populate the `User: <name>` label. But the agent stores summaries like `"User confirmed assistant's name is ghost"` and `"identified as Maya."`. The regex matched whichever fact came first in `list_all()`, so it could grab "ghost" (the assistant) instead of "Maya" (the user).

### Fix

Two-pattern approach, both gated on explicit user-side language:
1. `r"user(?:'s| )?\s*name\s*(?:is|=)\s+(\w+)"` — matches "User's name is Maya", "user name = Maya"
2. `r"identified\s+(?:as|=)\s+(\w+)"` — matches "identified as Maya." Agent summaries always use "identified" for the user, never the assistant

First pattern gates on `"user"` in content; second pattern falls through if first misses. Gated on `"user" in content_lower or "identified as" in content_lower`.

### Verification
| Check | Result |
|---|---|
| Only matches facts explicitly attributing a name to "user" | ✅ |
| `unified_memory_system.py` passes syntax check | ✅ |

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py:1234-1240` | Single regex match on user attribution only, no fallbacks |

---

## ~~Item 30~~ ✅ Item 30 — Cross-target verification on dual-path plugin install (issue #11 root cause)

**Status: COMPLETED.** Hash check after both writes; loud red warning + repair command when any destination diverges from source.

### What was wrong

v0.5.5.2 (commit `15b6e7f`) introduced two changes together: extracted `_install_plugin_file` (which catches `OSError`/`PermissionError`, prints in red, and returns without raising) and added a second write target (`%APPDATA%/opencode/plugins/`) as a Windows fallback. The intent was to make sure *at least one* destination got the new bytes when the other failed.

The unintended consequence: a partial failure now looks like full success. If `.config/opencode/plugins/roampal.ts` silently fails to write (transient file lock from OpenCode Desktop, antivirus scan, OneDrive sync, Controlled Folder Access) but `%APPDATA%/opencode/plugins/roampal.ts` succeeds, the install prints a green "Installed plugin" line for the AppData target. The user has no signal that the `.config` copy is stale. OpenCode Desktop on Windows preferentially loads from `.config/opencode/plugins/roampal.ts` — so the user keeps running the *old* plugin while the install reports success.

This is the actual symptom Marcus reported in issue #11 ("install says success but OpenCode keeps running old plugin"). v0.5.5.2's dual-path made it less catastrophic — at least one destination has the right bytes — but didn't add the verification step that would have surfaced the divergence loudly.

### How it was found

During v0.5.6 manual testing (2026-04-30), Item 4 + Item 19 verification appeared to pass against running OpenCode behavior. Hash comparison of installed plugins revealed:

- `.config/opencode/plugins/roampal.ts` → `00740d9b...` (v0.5.5.x, dated 2026-04-28)
- `%APPDATA%/opencode/plugins/roampal.ts` → `3ce9fb9a...` (v0.5.6 source, dated 2026-04-30)
- Source → `3ce9fb9a...`

OpenCode log confirmed it loaded from `.config` (`service=plugin path=file:///C:/Users/logte/.config/opencode/plugins/roampal.ts loading plugin`). Conclusion: every "live verification" run before discovering this was actually testing the v0.5.5.x plugin. The `.config` write had silently failed at install time and no signal surfaced.

### Fix

After both `_install_plugin_file` calls in `configure_opencode`, hash every install destination and compare against source. On drift:

- Print a `WARNING` line naming how many of N targets are stale
- List each `STALE:` and `FRESH:` path
- Suggest a concrete repair command using the fresh destination as donor (or source if all stale)
- Remind the user to fully restart OpenCode Desktop afterward

The verification helper lives next to `_install_plugin_file` for proximity:

```python
def _verify_plugin_install_targets(plugin_source: Path, targets: list[Path]) -> None:
    src_bytes = plugin_source.read_bytes()
    src_hash = hashlib.sha256(src_bytes).hexdigest()
    fresh, stale = [], []
    for target in targets:
        try:
            h = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            stale.append(target); continue
        (fresh if h == src_hash else stale).append(target)
    if not stale:
        return
    # ... loud red warning + repair command per stale target ...
```

The install caller tracks every destination it tries to write to in `install_targets: list[Path]`, then passes the list to the verifier. Source-unreadable case skips quietly (caller already failed earlier).

### Files affected
| File | Change |
|------|--------|
| `roampal/cli.py` | Added `import hashlib`; new `_verify_plugin_install_targets()` helper; `configure_opencode` collects `install_targets` list and calls the verifier after both `_install_plugin_file` calls |
| `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` | Added `TestVerifyPluginInstallTargets` class with 5 cases: all fresh (no warning), one stale + one fresh (repair from donor), all stale (repair from source), unreadable target (treated as stale), unreadable source (skips quietly) |

### Verification

`pytest roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` → 11 passed (6 prior + 5 new).

Manual walkthrough of the divergence case on the development machine on 2026-04-30: stale `.config` (v0.5.5.x) + fresh `%APPDATA%` (v0.5.6) reproduced; ran `_verify_plugin_install_targets` against the same paths post-fix and it produced the warning + repair command exactly as designed.

### Root-cause attribution

`git log -S "_install_plugin_file" -- roampal/cli.py` and `git log -S "alt_plugin_file" -- roampal/cli.py` both point at commit `15b6e7f` (v0.5.5.2 hotfix). Pre-v0.5.5.2 the install was a single inline `shutil.copy` to `.config` with the same swallow-and-continue exception handling — a single-target silent failure that the user would notice immediately because OpenCode would still load nothing new. The v0.5.5.2 dual-path was strictly better than that, but didn't close the verification gap.

### Implication for Marcus's report

Issue #11 may not have been fully resolved by v0.5.5.2. If Marcus's `.config` write failed silently (the most likely cause based on his "Desktop closed and no electron/node processes running" report), v0.5.5.2 would have shipped a fresh `%APPDATA%` copy but left the stale `.config` in place — and his OpenCode would still load the old plugin. With Item 30 in place, the install would print the warning and tell him exactly what to copy where. Worth pinging the issue thread once v0.5.6 ships.

### Timeline correlation

Issue #11 was opened **2026-04-28 17:18 UTC**, roughly 22 hours after v0.5.5 shipped (**2026-04-27 19:28 UTC**). The pre-v0.5.5.1 install code did not change in v0.5.5 — it was the same `shutil.copy` (no try/except) + `existing_content == source_content` skip-on-match logic that had been there since v0.4.2.1. So v0.5.5 itself did not introduce the underlying write-failure mechanism.

What v0.5.5 *did* do: ship the first substantive plugin change in several versions (`roampal/plugins/opencode/roampal.ts | 27 +++++++++++++++------------`, 15 insertions, 12 deletions). Prior releases (v0.5.1–v0.5.4) had little or no plugin change, so the install's skip-on-content-match path was functionally a no-op for most users — even if a write would have failed, there was nothing to install. v0.5.5 was the first release in that window where the install *had to actually copy bytes*, and that's when the latent transient install-time failure became user-visible.

So Marcus's report 22 hours after v0.5.5 isn't evidence that v0.5.5 caused the failure. It's evidence that v0.5.5 was the release where the failure had a chance to bite. Marcus stays current on releases, so the latent gap probably affected him on every release that brought a real plugin change — but most users on the same intervals never noticed because their `.config` writes happened to succeed.

---

## ~~Item 31~~ ✅ Item 31 — `/api/health` cold-boot 503 (v0.5.4 regression — narrow but real)

**Status: COMPLETED.** Health endpoint tests `_shared_embed_service` (initialized in `lifespan()` at startup) instead of requiring a populated `_memory_by_profile` dict. Falls back to per-profile path when the shared service is unavailable (test harnesses, edge cases).

### What was wrong

`server/main.py:health_check()` (pre-fix):

```python
embedding_ok = False
for mem in _memory_by_profile.values():     # ← empty on fresh boot
    if mem.initialized and mem._embedding_service:
        try:
            test_vector = await mem._embedding_service.embed_text("health check")
            if len(test_vector) > 0:
                embedding_ok = True
            break
        except Exception as e:
            embedding_error = str(e)

if not embedding_ok:
    raise HTTPException(status_code=503, detail=f"Embedding service unhealthy: ...")
```

`_memory_by_profile` is populated lazily — only when `get_memory_for_request` is called. On fresh boot it's empty. The loop runs 0 times. `embedding_ok` stays False. Endpoint returns 503 forever — until *some other endpoint* triggers a profile init.

The MCP server's `_ensure_server_running` gate at `mcp/server.py:885` rejects every tool call if `/api/health` doesn't return 200. So a freshly-spawned FastAPI process whose first inbound request is an MCP tool call is locked out: health needs a profile to exist; profile only exists after a real call; real calls are rejected because health says unhealthy.

### How it was found

During v0.5.6 manual testing (2026-04-30), after killing and restarting the FastAPI server to clear the unrelated server-flapping state I'd misdiagnosed, every subsequent `search_memory` MCP call returned `"Roampal server is restarting"` indefinitely. `score_memories` happened to keep working because the prior session had hooks fire that initialized profiles, briefly populating `_memory_by_profile`. Once the polling background task caught a momentary healthy response and was killed, the next call still returned 503 — proving the state wasn't sticky across restarts.

Spawning the server manually with `python -m roampal.server.main --port 27182` and watching stderr revealed the contradiction directly:

```
INFO:roampal.backend.modules.memory.embedding_service:Embedding model loaded (ONNX)
INFO:roampal.backend.modules.memory.embedding_service:Embedding model pre-warmed
INFO:     Application startup complete.
INFO:     127.0.0.1:63982 - "GET /api/health HTTP/1.1" 503 Service Unavailable
INFO:     127.0.0.1:63984 - "GET /api/health HTTP/1.1" 503 Service Unavailable
```

Embedding model loaded and pre-warmed at startup, then health returned 503 on every request. The `_shared_embed_service` was fine — the per-profile probe was the bug.

### Root-cause attribution

`git log -L 1954,1989:roampal/server/main.py` traces the change to commit `5e1e80a` (v0.5.4, 2026-04-24). Pre-v0.5.4 the health check tested a singleton `_memory` object that was always populated. v0.5.4's "per-request profile binding + cross-client header propagation" refactor replaced the singleton with `_memory_by_profile` (lazy dict), and the health check was updated to iterate it without realizing the iteration could yield an empty set during the boot-to-first-request window.

The bug has been live for ~6 days (v0.5.4 → v0.5.6) but was masked because hooks (Claude Code's `UserPromptSubmit`, OpenCode's `chat.message`) always fire `/api/hooks/get-context` before MCP tool calls happen. That endpoint goes through `get_memory_for_request` which lazily initializes the profile — closing the gap before users could notice. The bug only surfaces when an MCP client calls a tool before any profile-touching endpoint has been hit. We hit it today by manually restarting the FastAPI process mid-session with no hook firing in between.

### Fix

Test the shared embed service directly. It's created in `lifespan()` at startup (`server/main.py:627-634`), so it's available the moment the application is ready to accept requests. Fall back to the per-profile path when the shared service is None — keeps existing test fixtures (which skip lifespan) working unchanged.

```python
@app.get("/api/health")
async def health_check():
    embedding_ok = False
    embedding_error = None

    if _shared_embed_service is not None:
        try:
            test_vector = await _shared_embed_service.embed_text("health check")
            if len(test_vector) > 0:
                embedding_ok = True
        except Exception as e:
            embedding_error = str(e)
    else:
        # Fallback: per-profile check (legacy v0.5.4 behavior)
        for mem in _memory_by_profile.values():
            if mem.initialized and mem._embedding_service:
                try:
                    test_vector = await mem._embedding_service.embed_text("health check")
                    if len(test_vector) > 0:
                        embedding_ok = True
                    break
                except Exception as e:
                    embedding_error = str(e)

    if not embedding_ok:
        raise HTTPException(status_code=503, detail=f"Embedding service unhealthy: {embedding_error or 'not initialized'}")

    return {
        "status": "healthy",
        "memory_initialized": len(_memory_by_profile) > 0,
        "session_manager_ready": len(_session_manager_by_profile) > 0,
        "profiles_loaded": len(_memory_by_profile),
        "embedding_ok": embedding_ok,
        "timestamp": datetime.now().isoformat(),
    }
```

Why test the shared service rather than eager-initializing the default profile in `lifespan()`?

- Smaller blast radius — only the health endpoint changes
- Doesn't add ~5s startup latency for a profile that may never be used (e.g., users on non-default profiles)
- More accurate semantics — "is embedding working?" is what the health endpoint *should* be answering, not "has at least one profile been touched?"
- The per-profile loop was an artifact of v0.5.4 needing to verify the new per-profile binding worked, not of any genuine health signal that the shared service can't provide

### Files affected
| File | Change |
|------|--------|
| `roampal/server/main.py:1954-1995` | Health check prefers `_shared_embed_service`, falls back to per-profile loop when shared is None |
| `roampal/backend/modules/memory/tests/unit/test_fastapi_endpoints.py` | Updated `test_health_returns_503_when_not_initialized` docstring to clarify it covers the fallback path; added `test_health_passes_with_shared_service_when_profiles_empty` (the v0.5.6 cold-boot regression) and `test_health_returns_503_when_shared_service_throws` (v0.3.0 PyTorch-corruption signal still fires) |

### Verification

`pytest roampal/backend/modules/memory/tests/unit/test_fastapi_endpoints.py::TestHealthEndpoint` → 4 passed.

Live re-spawn of FastAPI on port 27182 with no profile-touching calls:

| State | Pre-fix `/api/health` | Post-fix `/api/health` |
|-------|----------------------|------------------------|
| Cold boot, no profiles, shared service initialized | `503 Embedding service unhealthy: not initialized` | `200 {"status":"healthy","embedding_ok":true,"profiles_loaded":0}` |
| Profile loaded, shared service initialized | 200 (worked already) | 200 (unchanged) |
| Shared service raises | 503 with raised error in detail (would only fire after first profile init under old code) | 503 with raised error in detail (fires immediately under new code — earlier signal) |

### Implication for v0.5.4–v0.5.5.x users

Anyone whose first FastAPI interaction was an MCP tool call (rather than a hook) saw "Roampal server is restarting" until they happened to send a chat message that fired a hook. Most users never noticed because the hook fires first in normal flows. Programmatic MCP clients without UserPromptSubmit hooks (e.g., automated test harnesses, IDE integrations that bypass the hook), or anyone running `roampal start` and then immediately searching, would see the symptom.

Once v0.5.6 ships, the cold-boot 503 disappears. No user action required.

---

## ~~Item 32~~ ✅ Item 32 — Extend startup phantom sweep to working/history/patterns (#8 follow-up — makes Marcus's restart workaround actually work)

**Status: COMPLETED.** `_startup_cleanup` now sweeps phantoms across all four dedup-affected collections, not just `memory_bank`. 3 unit tests cover the new tiers.

### Implemented
| File | Change | Status |
|------|--------|--------|
| `unified_memory_system.py:_startup_cleanup` | Added per-tier phantom sweep loop for `working`/`history`/`patterns` after the existing memory_bank sweep + status backfill | ✅ Done |
| `tests/unit/test_unified_memory_system.py::TestStartupSweepAllTiers` | 3 tests: phantoms swept across all three new tiers, missing/None adapter handled gracefully, sweep continues when one tier raises | ✅ Pass (57/57 + 3 skipped) |

### What's missing (pre-fix)

The v0.5.5/v0.5.6 phantom-sweep work fixed `MemoryBankService._sweep_phantoms()` and called it from `_startup_cleanup` and `cleanup_archived()` — but **only for the `memory_bank` collection.** The other three dedup-affected collections (`working`, `history`, `patterns`) get no sweep at all.

This matters because of the GitHub thread on issue #8. We told Marcus:

> "Delete memories in Desktop as normal, then restart the core server once. The startup migration will clean up all the phantom entries from those deletions and new memory generation works again."

That promise is only fully true if the user's bulk deletes were limited to `memory_bank`. In practice, Marcus's reported action ("keeping only Memory Bank and Books") triggers `roampal-desktop`'s "Delete data" tab, which calls `/clear/working`, `/clear/history`, and `/clear/patterns` — and each of those endpoints (pre-Desktop-v0.3.3-Section-9) uses `adapter.collection.delete(ids=batch)`, leaving phantoms in those tiers too. A core restart on v0.5.5 only swept `memory_bank`, leaving the other tiers' phantoms in place. The "restart fixes it" workaround silently underdelivered.

### Fix

Extend `_startup_cleanup` to inline the same phantom-sweep logic across the other three adapters:

```python
# v0.5.6 Item 32: Sweep phantoms across the other dedup-affected collections too.
for tier_name in ("working", "history", "patterns"):
    adapter = self.collections.get(tier_name)
    if adapter is None:
        continue
    try:
        tier_ids = adapter.list_all_ids()
        phantom_ids = [doc_id for doc_id in tier_ids if not adapter.get_fragment(doc_id)]
        if phantom_ids:
            adapter.delete_vectors(phantom_ids)
            logger.info(f"v0.5.6 startup sweep: removed {len(phantom_ids)} phantom entries from {tier_name}")
    except Exception as e:
        logger.warning(f"Phantom sweep error on {tier_name}: {e}")
```

The logic is inlined rather than factored through `MemoryBankService` because these adapters don't have a service wrapper — they're raw `ChromaDBAdapter` instances accessed directly via `self.collections[tier_name]`. Identical mechanics, no shared abstraction worth introducing for three short loop bodies.

### Why inline vs. extract a shared `_sweep_collection_phantoms()` helper

Considered and rejected: extracting a free function or instance method that takes an adapter and runs the sweep. Pros are obvious (one code path); con is that `MemoryBankService._sweep_phantoms()` already owns the canonical implementation and adding a second helper just to avoid 12 lines of inline code creates two things to keep in sync. If a third site needs the same logic, factor then. For now the inline version is shorter than the indirection.

### Coverage

| Test | What it verifies |
|------|------------------|
| `test_phantoms_swept_from_working_history_patterns` | All three tiers receive `delete_vectors([phantom_ids])` for the IDs that `get_fragment` returns None for. Tiers without phantoms aren't touched by sweep. |
| `test_sweep_handles_missing_adapter` | If a tier is missing from `self.collections` or set to `None`, the sweep loop's `continue` keeps the rest of startup running. No crash. |
| `test_sweep_continues_when_one_tier_errors` | If one adapter's `list_all_ids` raises (simulated ChromaDB error), the sweep logs a warning and proceeds to the next tier. The successful tiers still get cleaned. |

### Files affected
| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py` | New `for tier_name in ("working", "history", "patterns"):` sweep loop appended to `_startup_cleanup` after the memory_bank backfill block |
| `roampal/backend/modules/memory/tests/unit/test_unified_memory_system.py` | New `TestStartupSweepAllTiers` class with 3 unit tests |

### Why this matters for the issue #8 messaging

Once Desktop v0.3.3 Section 9 (bulk-clear nuke-and-recreate) ships, no new phantoms will be created in any tier on a clean install. But pre-existing phantoms from older bulk deletes will remain in users' ChromaDB until the next core startup. Item 32 ensures that startup actually cleans those up, completing the migration story:

- **Before v0.5.6:** restart core → memory_bank phantoms gone, other tiers still poisoned.
- **After v0.5.6 (this item):** restart core → all four collections cleaned. The GitHub reply on #8 becomes literally true.

Combined with Desktop v0.3.3 Section 9, this closes the full lifecycle: no new phantoms created (Section 9), and any leftover phantoms from prior deletes self-heal on restart (Item 32).

---

## Late additions (discovered during pre-release testing)

### Fix A — Add `User-Agent` header to all `opencode.ai` HTTP calls

**Problem:** Cloudflare (fronting `opencode.ai`) rejects requests without a `User-Agent` header with `HTTP 403 Forbidden` + `error code: 1010`. Python's `urllib.request` does not send one by default, and the TypeScript `fetch()` API in the plugin also omits it unless explicitly set.

**Impact:**
- `_list_opencode_go_models()` silently failed, so the wizard showed only the 3 hardcoded `_DEFAULT_GO_MODELS` instead of dynamically fetching the live catalog.
- `_call_custom()` (Python sidecar) would also 403 after the user picked a model.
- The TypeScript plugin's `_scoreExchangeViaLLM()` and fact-extraction fetch calls to Go/Zen endpoints were the **actual** scoring path — these 403'd continuously, tripping the circuit breaker and showing `[roampal scoring: failed (5 consecutive failures)]` in the system prompt.
- `_call_zen()` and the Zen probe in `get_backend_info()` were equally exposed.

**Fix:**
| File | Function | Change |
|------|----------|--------|
| `roampal/cli.py` | `_list_opencode_go_models()` | Added `User-Agent: Mozilla/5.0 ...` to the `/models` request |
| `roampal/sidecar_service.py` | `_call_custom()` | Added `User-Agent: roampal-sidecar/1.0` to the `/chat/completions` request |
| `roampal/sidecar_service.py` | `_call_zen()` | Added `User-Agent: roampal-sidecar/1.0` to the Zen `/chat/completions` request |
| `roampal/sidecar_service.py` | `get_backend_info()` Zen probe | Added `User-Agent: roampal-sidecar/1.0` to the probe request |
| `roampal/plugins/opencode/roampal.ts` | `_scoreExchangeViaLLM()` scoring fetch | Added `User-Agent: roampal-sidecar/1.0` to the `/chat/completions` request |
| `roampal/plugins/opencode/roampal.ts` | Fact-extraction fetch | Added `User-Agent: roampal-sidecar/1.0` to the `/chat/completions` request |
| `roampal/plugins/opencode/roampal.ts` | Zen `/models` discovery | Added `User-Agent: roampal-sidecar/1.0` to the `/models` request |

### Fix B — Filter out non-OpenAI-compatible MiniMax models from Go catalog

**Problem:** OpenCode Go serves MiniMax M2.5 and M2.7 through the Anthropic `/v1/messages` endpoint, not the OpenAI `/chat/completions` endpoint that `_call_custom()` uses. If a user selected one, the sidecar would receive a `400` or `403` on every scoring call.

**Fix:** In `_list_opencode_go_models()`, filter out any model ID starting with `minimax-`. If the filtered list is empty, fall back to `_DEFAULT_GO_MODELS` (which contains only OpenAI-compatible models).

### Fix C — Correct fallback model ID

**Problem:** The `_DEFAULT_GO_MODELS` fallback list contained `deepseek-v4`, which is not a valid Go catalog ID. The correct ID is `deepseek-v4-flash`.

**Fix:** Updated `_DEFAULT_GO_MODELS` to use `deepseek-v4-flash`.

### Fix D — Plugin syntax error in v0.5.6 `scoreExchangeViaLLM` refactor + CI parse guardrail

**Problem:** The v0.5.6 split of `scoreExchangeViaLLM` into a public queue wrapper + private `_scoreExchangeViaLLM` worker removed the outer `try { ... } finally { scoringInFlight = false }` block but left the `}` that closed `try` in place. Result: 1 unmatched closing brace at end of `_scoreExchangeViaLLM`. Bun's parser rejected the file with `Unexpected }`, the plugin's exported function never ran, OpenCode's renderer never received the plugin-ready event, and the desktop window stayed blank on every launch.

**Fix:** Removed the orphan `}` in `roampal/plugins/opencode/roampal.ts`. Verified with a string-/comment-/template-aware brace counter (`FINAL=0`, zero negative-depth events).

**Guardrail (so this can't ship again):** Added a `plugin-parse` job to `.github/workflows/tests.yml` that runs `bun build roampal/plugins/opencode/roampal.ts --target bun --packages external` on every push and PR. `--packages external` skips dependency resolution (no `bun install` needed); the step is pure parse/transpile and fails fast on any syntax error. Bun is the same parser OpenCode uses to load the plugin at runtime, so a green check here is a true "OpenCode will load this" signal.

### Fix E — Hardlink the AppData plugin path to the canonical `.config` copy on Windows

**Problem:** The dual-path install introduced in v0.5.5.2 made install say "✓" as long as one of the two writes succeeded. If `.config` write succeeded but `AppData` retained stale content (or vice versa), OpenCode could resolve whichever path was stale and silently load an old plugin. Item 30 detects that divergence after the fact, but doesn't prevent it.

**Fix:** In `cli.py`'s `setup_opencode()`, after the canonical write to `~/.config/opencode/plugins/roampal.ts`, expose the AppData path as a hardlink to the canonical file rather than a second copy. Both paths now point at the same on-disk inode, so:

- A single successful canonical write is automatically reflected at both paths — there's no second write to silently fail.
- A canonical write failure is loudly visible via `_install_plugin_file`'s existing error reporting, instead of being masked by a successful AppData copy.
- Divergence between the two paths becomes structurally impossible — they're literally the same bytes.

If `os.link()` raises (cross-volume install, OneDrive reparse-point quirk, antivirus, exotic filesystem), the code falls back to the v0.5.5.2 dual-copy behavior and Item 30's hash verification still runs. Net effect: the common path closes Marcus's class of failures at the structural level; the rare path is no worse than today.

**Tested:**

| Test | What it verifies |
|------|------------------|
| `test_hardlink_branch_produces_same_inode` | `configure_opencode(force=True)` produces two paths with **identical `st_ino`** and `nlink=2`. Confirms the hardlink branch fired. |
| `test_hardlink_reflects_canonical_changes_at_alt` | Writing bytes to canonical is **immediately readable at alt** with no second copy step. Confirms behavioral consistency, not just metadata coincidence. |
| `test_falls_back_to_copy_when_os_link_raises` | When `os.link` raises `OSError`, install completes successfully via copy fallback; both paths exist with matching content but distinct inodes. |
| `test_replaces_pre_existing_alt_file` | A stale pre-existing alt file is unlinked first, then replaced with the hardlink — handles the upgrade-from-v0.5.5.2-or-earlier case. |

All four added to `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py::TestOpenCodePluginHardlink`. Tests run on any OS by mocking `sys.platform` and `Path.home()`; `os.link` works on Linux/macOS too, so the happy path is exercisable in CI without a Windows runner.

Manual verification on Windows 11 (real install paths, not tmp):

```
=== running configure_opencode(force=True) ===
  Installed plugin: C:\Users\logte\.config\opencode\plugins\roampal.ts
  OpenCode configured!

=== inode check ===
  C:\Users\logte\.config\opencode\plugins\roampal.ts
    ino=22799473113670688  nlink=2  size=98429
  C:\Users\logte\AppData\Roaming\opencode\plugins\roampal.ts
    ino=22799473113670688  nlink=2  size=98429
  SAME INODE: True
  canonical-written marker visible at alt: '// hardlink test marker'
  hash MATCH: 0362a14e6c85 (both paths == source)
```

```python
# roampal/cli.py — inside setup_opencode(), after the canonical write
if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        alt_plugin_file = Path(appdata) / "opencode" / "plugins" / "roampal.ts"
        if alt_plugin_file != plugin_file:
            alt_plugin_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                if alt_plugin_file.exists() or alt_plugin_file.is_symlink():
                    alt_plugin_file.unlink()
                os.link(str(plugin_file), str(alt_plugin_file))
            except OSError:
                _install_plugin_file(plugin_source, alt_plugin_file)
            install_targets.append(alt_plugin_file)
```

### Fix F — Background drainer for `pendingScoringQueue` (Item 19 follow-up)

**Problem (caught during 2026-05-01 concurrent test):** Item 19's `pendingScoringQueue` correctly serialized overlapping calls and gave each failed exchange a 3-attempt retry budget. But the deferred-retry block only ran inside the `session.idle` handler — meaning a queued payload only got its next attempt when the user sent another message in that session. If the user went idle (or closed OpenCode), the queue stalled. Worse, the next user message added its own scoring task on top of the stuck ones, so the backlog grew faster than it drained.

Concretely, in the live test against `qwen3.6-27b` on LM Studio (sidecar timing out at ~30s on each call), 2 deferred-retry payloads sat in `pendingScoringQueue` for 16+ minutes with zero retry attempts because the user had stopped messaging.

**Fix:** Extract the deferred-retry loop from the `session.idle` handler into a shared `drainPendingScoringQueue(source)` function. Register a `setInterval` at plugin load (gated on `!SIDECAR_DISABLED`) that calls the drainer every 30s. The `session.idle` handler now also calls the same shared function — both paths share one body, so retry semantics, profile-refresh, max-retries-and-drop, and `consecutiveFailures` accounting stay identical. A `backgroundDrainRunning` flag prevents the interval and `session.idle` from racing each other.

| File | Line | Change |
|------|------|--------|
| `roampal/plugins/opencode/roampal.ts` | 312-356 | New `BACKGROUND_DRAIN_INTERVAL_MS` constant, `backgroundDrainTimer`, `backgroundDrainRunning` flag, and `drainPendingScoringQueue(source)` function |
| `roampal/plugins/opencode/roampal.ts` | 1303-1312 | Plugin init registers `setInterval` calling `drainPendingScoringQueue("background.drain")` every 30s |
| `roampal/plugins/opencode/roampal.ts` | 1937 | `session.idle` deferred-retry block replaced with `await drainPendingScoringQueue("session.idle")` |

**Trade-off considered.** The drain interval adds ~one wake-up every 30s for the lifetime of the plugin process, even when the queue is empty. Cost is negligible (the function returns immediately on `pendingScoringQueue.size === 0`), and the alternative (no background drain) leaves the in-memory queue as a silent data-loss vector when the user goes idle.

**Validation plan:** Re-run the 2026-05-01 concurrent test (two OpenCode windows, slow sidecar). Send a few messages in each window, stop messaging, watch the plugin debug log. After 30s of inactivity, expect `background.drain: Retrying deferred scoring for ses_…` lines firing without any new `chat.message` events.

**Live validation (2026-05-01, two OpenCode windows on `qwen3.6-27b` via LM Studio at `http://localhost:1234/v1`, profiles `test-a` / `test-b`):**

```
[2026-05-01T16:11:50.912Z] Background queue drainer started (interval=30000ms)
... user sends messages, sidecar succeeds, then a forced timeout fires:
[2026-05-01T16:20:42.857Z] scoreExchange error on sidecar(qwen3.6-27b) (attempt 0): TimeoutError: The operation timed out.
[2026-05-01T16:20:42.857Z] scoreExchange circuit breaker TRIPPED for qwen3.6-27b
[2026-05-01T16:20:42.858Z] session.idle: Sidecar failed — queued for deferred retry
[2026-05-01T16:20:50.987Z] background.drain: Retrying deferred scoring for ses_220a61512ffeK4FJLQsOXswJ1A (attempt 1/3)
[2026-05-01T16:21:31.237Z] scoreExchange SUCCESS: worked via sidecar(qwen3.6-27b), 2 memories
[2026-05-01T16:21:45.046Z] scoreExchange SUMMARY stored: 235 chars, doc_id=working_c2fef66b
```

The `background.drain` event fired exactly when the next 30-s interval tick (started at 16:11:50) landed after the failure — 8 seconds after the sidecar TimeoutError, with no user activity required. The retry succeeded and the summary persisted. End-to-end self-heal verified.

Same window also surfaced the gap that became Fix G (below): a `scoreExchange summary store error: TimeoutError` at 16:17:00 dropped a summary text to ChromaDB even though the sidecar score landed — that path is unchanged by Fix F.

### Fix G — Deferred retry for failed summary writes (Fix F sibling)

**Problem (caught during the same 2026-05-01 test):** When sidecar scoring succeeds but the FastAPI POST to `/api/hooks/stop` (the summary write) fails — `TimeoutError`, non-2xx status, network blip — the summary text was just dropped. Logged as `scoreExchange summary store error: TimeoutError` and the function moved on. Fix F doesn't catch this because Fix F only re-runs `scoreExchangeViaLLM`; the summary-write failure happens **inside** a successful `scoreExchangeViaLLM` call (it returns true once scoring lands, regardless of whether the summary store afterward succeeded).

Live evidence from the same session:
```
[2026-05-01T16:16:30.952Z] scoreExchange SUCCESS: worked via sidecar(qwen3.6-27b), 2 memories
[2026-05-01T16:17:00.957Z] scoreExchange summary store error: TimeoutError: The operation timed out.
```

The 16:16:30 score landed (memory rankings updated) but the ~230-char summary text never made it to ChromaDB. No retry fired. Counterfactual: this is exactly the lossy path Marcus reported — "the summary just isn't there even though the model talked."

**Fix:** Three TS-only changes in `roampal.ts`:

1. **New `pendingSummaryQueue: Map<sessionId, _PendingSummaryEntry>`** holding the exact bytes the original write attempted (exchange, summary, outcome, fingerprint, retry counter).
2. **Extract the summary-store call into `tryStoreSummary(sessionId, exchange, summary, outcome, fingerprint): Promise<boolean>`** — returns true on success or dedup-skip, false on any failure path. The fingerprint is computed by a shared `_summaryFingerprint(exchange)` helper so retry attempts produce the byte-identical hash the original used (lets a server-side already-stored summary short-circuit cleanly via the dedup query).
3. **Background drainer also drains the summary queue.** Same `setInterval` + 30s tick + `session.idle` co-drain as Fix F, calling `drainPendingSummaryQueue(source)`. Same 3-attempt budget (`PENDING_SUMMARY_MAX_RETRIES`), same drop-after-exhaustion, same per-session cleanup on session-end.

The dedup-by-fingerprint check in `tryStoreSummary` is what makes the retry safe: if the FastAPI handler completed the write server-side even though the client timed out (the v0.5.4 comment claim), the next retry hits the dedup query, sees the entry exists, and exits with `tryStoreSummary: SKIP — already exists` without double-storing. Worst case: one wasted retry; best case: data we would have lost is recovered.

| File | Lines | Change |
|------|-------|--------|
| `roampal/plugins/opencode/roampal.ts` | ~316-326 | New `_PendingSummaryEntry` type, `pendingSummaryQueue`, `PENDING_SUMMARY_MAX_RETRIES` |
| `roampal/plugins/opencode/roampal.ts` | ~374-518 | New `_summaryFingerprint`, `tryStoreSummary`, `drainPendingSummaryQueue` functions |
| `roampal/plugins/opencode/roampal.ts` | ~1200-1216 | Old inline summary-store block replaced with `tryStoreSummary` call + queue-on-failure |
| `roampal/plugins/opencode/roampal.ts` | ~1303-1318 | Background `setInterval` now calls both drainers |
| `roampal/plugins/opencode/roampal.ts` | ~1937-1942 | `session.idle` co-drain calls both drainers |
| `roampal/plugins/opencode/roampal.ts` | ~2090-2092 | Session-end cleanup also clears `pendingSummaryQueue` |

**Validation plan:** With both windows reopened on the same slow sidecar, force a `summary store error` (kill FastAPI mid-write, or wait for `qwen3.6-27b` server-side tag extraction to push the `/stop` write past 30s as it did at 16:17:00). Expect:
1. `scoreExchange: summary store failed — queued for deferred retry (session ses_…)` immediately on the failure.
2. `background.drain: Retrying deferred summary for ses_… (attempt 1/3)` within 30s, no user activity needed.
3. `tryStoreSummary: SKIP — already exists` if FastAPI did complete server-side, OR `tryStoreSummary: SUMMARY stored:` if the retry was the actual landing.

**Live validation (2026-05-01, fault-injected via temporary `ROAMPAL_TEST_FORCE_SUMMARY_FAIL=1` env hook in `tryStoreSummary`):**

```
[2026-05-01T16:46:51.366Z] scoreExchange SUCCESS: worked via sidecar(qwen3.6-27b), 4 memories
[2026-05-01T16:46:51.367Z] tryStoreSummary: TEST FAULT INJECTED (ROAMPAL_TEST_FORCE_SUMMARY_FAIL=1) — returning false
[2026-05-01T16:46:51.367Z] scoreExchange: summary store failed — queued for deferred retry (session ses_220a63db0ffezOTSFb4vyyAORE)
[2026-05-01T16:47:19.230Z] background.drain: Retrying deferred summary for ses_220a63db0ffezOTSFb4vyyAORE (attempt 1/3)
[2026-05-01T16:47:35.640Z] tryStoreSummary: SUMMARY stored: 304 chars, doc_id=working_14a0a79f
[2026-05-01T16:47:35.712Z] background.drain: Deferred summary stored for ses_220a63db0ffezOTSFb4vyyAORE
```

End-to-end self-heal: forced summary failure → push to `pendingSummaryQueue` → background interval (28s after the failure, no user activity) → fault-flag consumed → real `tryStoreSummary` succeeds → `working_14a0a79f` lands in ChromaDB. The fault-injection hook was a one-shot `_summaryFaultInjectionConsumed` flag gated on the env var, deliberately removed before commit; production plugin hash is `3f9384dc…` (verified clean: zero `TEST FAULT` / `ROAMPAL_TEST_FORCE_SUMMARY_FAIL` / `_summaryFaultInjectionConsumed` references in source).

**No-test caveat:** there is no Bun-based unit test infrastructure under `roampal/plugins/opencode/` (only `roampal.ts` and `__init__.py`). Adding one requires a `package.json` + `bun test` setup that's bigger than this fix warrants. The validation evidence is the live-log capture above (Fix F) and the fault-injected end-to-end run (Fix G). If/when a TS test runner gets added, candidate cases: (a) drainer no-ops when both queues empty, (b) drainer retries and re-queues on transient failures, (c) drainer drops + logs after `MAX_RETRIES`, (d) session-end cleanup removes both queue entries.

---

## Out of scope (recorded for posterity, not addressed here)

- **Long Windows paths (>260 chars).** No production users have hit this. If `roampal.ts` ends up under a deeply nested user profile, the install can fail with `OSError`. The post-copy verification surfaces it cleanly. Real fix would be `\\?\` long-path prefixing, but that's not warranted without a real report.
- **Portable / standalone OpenCode installs.** If a user runs OpenCode from a custom path that resolves plugins from neither `~/.config/opencode/plugins` nor `%APPDATA%/opencode/plugins`, the dual-path fix doesn't reach them. Out of scope unless someone reports it — adding a third path on speculation has more downside than upside.

---

## Verification plan

1. New unit tests pass on Linux/macOS CI (covers all 6 `_install_plugin_file` cases). ✅
2. Manual repro on Windows 11 (executed 2026-05-01):
   - **Item 2** — Set `$env:APPDATA=""`, call `configure_opencode(force=True)`. Output included:
     ```
     Skipped %APPDATA% fallback install — APPDATA env var is unset.
     If OpenCode can't find the plugin, set APPDATA and re-run, or copy manually.
     ```
     ✅ PASS
   - **Item 3** — `attrib +R` on existing `roampal.ts`, call `configure_opencode(force=True)`. Output included:
     ```
     Failed to install plugin: [Errno 13] Permission denied: '...\roampal.ts'
     Possible causes:
       - OpenCode Desktop is running and holds a file lock
       - Read-only attribute on existing roampal.ts
       - Antivirus / Controlled Folder Access blocking write
       - OneDrive sync quarantining the destination
     If those don't apply, copy manually:
       cp ... ...
     ```
     All 4 causes printed. ✅ PASS. Cleanup: `attrib -R` + reinstall verified hash match across source, `.config`, and AppData.
3. No behavior change on macOS/Linux (Item 2 and Item 3 are both gated on `sys.platform == "win32"` or only fire when copy fails — both unaffected). ✅

## Files changed
| File | Change |
|------|--------|
| `roampal/cli.py` | Expand error messages in `_install_plugin_file()`; add `APPDATA` unset warning in dual-path block; add OpenCode Go subscriber note to `[2]` wording (item 27); add `_detect_opencode_go` + `_list_opencode_go_models` helpers and surface a new `Use OpenCode Go` option in both `_sidecar_model_picker` prompt branches (item 28); User-Agent header on Go `/models` request + MiniMax filtering + corrected fallback model ID (late Fixes A–C) |
| `roampal/backend/modules/memory/tests/unit/test_install_plugin_file.py` | New unit-test file |
| `roampal/plugins/opencode/roampal.ts` | Field-name alignment (item 12), summary phrasing (14), facts inference rule (15), facts examples (16), facts max_tokens (18), mutex → async queue + per-session deferred retry (19) |
| `roampal/sidecar_service.py` | Temperature pinning across all backends (13), summary phrasing (14), facts inference rule (15), facts examples (16), test_sidecar_scoring rewrite (17), legacy field-name alignment (12), User-Agent header on custom + Zen requests (late Fix A) |
| `roampal/plugins/opencode/roampal.ts` | User-Agent header on scoring, fact-extraction, and Zen `/models` discovery fetches (late Fix A) |
| `roampal/mcp/server.py` | MCP tool description + schema rewrites for TDQS audit (item 20+) |
| `pyproject.toml` | version 0.5.5.2 → 0.5.6 |
| `roampal/__init__.py` | `__version__` 0.5.5.2 → 0.5.6 |
| `dev/docs/releases/v0.5.6/RELEASE_NOTES.md` | This file |
