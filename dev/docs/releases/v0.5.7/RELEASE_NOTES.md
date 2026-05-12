# Roampal Core v0.5.7

**Release Date:** TBD (target: ship paired with Roampal Desktop v0.3.3)
**Type:** Hardening / hygiene release (no new user-visible features)
**Triggered by:** Repro setup for issue #8 on 2026-05-02 surfaced an unbounded-growth defect in `_completion_state.json` with downstream correctness consequences.

## Summary

Core-side garbage collection for `mcp_sessions/_completion_state.json`. The file accumulates one entry per `conversation_id` ever seen, with no GC. Observed on a prod state file: ~140KB across **719 session entries**, of which 216 still had `scored_this_turn=True` from sessions that ended weeks or months earlier.

The accumulation creates three downstream problems — one of which is a credible candidate for the missing summary-side mechanism behind issue #8 (which v0.5.5 and v0.5.6 did not address).

---

## Item 1 — `_completion_state.json` accumulates unbounded — core-side GC missing

### Findings on prod state file (single user, several months of usage)

| Flag | Sessions stuck True | Notes |
|---|---|---|
| `completed` | 505 | Stop hook set; UserPromptSubmit only clears on a follow-up message from the *same* conversation_id. Sessions that never get revisited stay True forever. |
| `scoring_required` | 51 | get-context set; `set_scoring_required(False)` only fires if a follow-up turn comes through and skips scoring. Aborted/abandoned sessions stay True. |
| `scored_this_turn` | 216 | record_response set; never reset for sessions that never see another turn. |
| `first_message_seen` | 670 | Permanent by design — but holds the entry alive so the other flags don't get cleaned up either. |

### Problem

`roampal/hooks/session_manager.py:62-92` (`_cleanup_old_transcripts`) deletes JSONL transcripts older than its TTL at startup (bumped to 30 days in this release — see below). The corresponding `_completion_state.json` entries are never pruned. Result: monotonic growth, with three downstream consequences:

1. **I/O amplification per hook fire.** `_save_completion_state` writes the whole file via `write_json_atomic` (temp + rename) on every state mutation. A 200-byte logical update becomes a ~140KB rewrite. On a busy session this fires several times per turn (Stop hook, UserPromptSubmit, score_memories).

2. **Cross-session fallback correctness hazard.** `roampal/server/main.py:1087-1096` linearly scans every `mcp_*` session in the state file looking for `scored_this_turn=True`, takes the first match, and clears it — then treats the *current* turn as scored. With 216 entries permanently flagged True from sessions that ended weeks or months ago, the current turn can be falsely marked as scored on the basis of a leftover flag from a long-dead session, even when `score_memories` was never actually called this turn. The Stop hook then skips its soft-warn, and a turn that legitimately needed scoring goes silently unscored. This fallback was added for OpenCode/MCP id-mismatch resilience but degrades into a false-positive engine as the state file accumulates.

3. **Latent ID-collision risk.** If a stable client conversation_id ever recurs (which is allowed — IDs are client-supplied), `check_and_clear_completed`/`is_first_message` will return stale state from a session that ended weeks ago, yielding wrong scoring/cold-start decisions.

### Plausible relationship to issue #8

The reported workaround is *"delete `_completion_state.json` and restart"*. v0.5.5/v0.5.6 phantom-dedup fixes explain why **fact** writes silently stop after a Desktop bulk-delete (memory_bank dedup gating, HNSW phantoms in working/history/patterns). They do **not** explain why **summary** writes (no dedup gate) also stop, nor why deleting this specific file unwedges things.

Accumulated stuck flags + the cross-session fallback misrouting are a credible candidate for the missing summary-side mechanism, though unconfirmed without logs from the reporter. Independent of #8, the no-GC defect is shippable on its own.

### Fix — extend startup cleanup to prune state-file entries

In `roampal/hooks/session_manager.py`, alongside the existing `_cleanup_old_transcripts` and `_cleanup_exchange_cache` calls in `__init__`, add a `_cleanup_completion_state` pass that:

- Loads `_completion_state.json`.
- Drops every entry where:
  - `first_message_timestamp` (or `timestamp` if `first_message_timestamp` is absent) is older than `max_age_days` (default 30 — matched to the JSONL TTL bump in this release), **OR**
  - the corresponding `<conversation_id>.jsonl` no longer exists in `sessions_dir`.
- Saves once via `write_json_atomic`.
- Logs the count pruned.

Also cap the file's size with a hard ceiling — e.g. if the entry count exceeds 500 after age-based pruning, evict the oldest by `timestamp` until count ≤ 500. Mirrors the `_cleanup_exchange_cache(max_entries=50)` pattern at `session_manager.py:94-123`.

**TTL note:** The pre-existing JSONL transcript TTL is bumped from 7 days to 30 days in this same release, so the two cleanup passes stay consistent. Pruning the state-file entry before the JSONL is gone would otherwise make `is_first_message` return True on return to a paused conversation, triggering an unnecessary cold-start user-profile dump.

### Acceptance criteria

- Stale entries (older than 30 days OR with no matching JSONL) are pruned at startup.
- Post-prune entry count never exceeds 500.
- A single atomic write happens at the end of the prune pass (no partial states left if interrupted).
- A startup log line reports the number of entries pruned (zero is fine; silent is not).
- Existing hook behavior is unchanged for live sessions (entries newer than the cutoff and with active JSONL files are untouched).

### Files affected

| File | Change |
|---|---|
| `roampal/hooks/session_manager.py` | New `_cleanup_completion_state(max_age_days=30, max_entries=500)` method; called from `__init__` after `_cleanup_old_transcripts` and `_cleanup_exchange_cache`. Also bumps `_cleanup_old_transcripts` default from `max_age_days=7` to `max_age_days=30` to keep the two TTLs in lockstep. |
| `roampal/backend/modules/memory/tests/unit/test_session_manager.py` (new) | (a) entries with stale `first_message_timestamp` get pruned, (b) entries whose JSONL is missing get pruned, (c) `max_entries` ceiling enforced after age pass, (d) atomic write — file remains valid JSON if interrupted mid-prune. |

No dedicated `SessionManager` test file exists today; closest existing coverage is via `test_fastapi_endpoints.py`'s API-surface tests. New file is the right home.

### Coordination

- **Desktop:** ships the fix by bumping the bundled core version to v0.5.7 in Roampal Desktop v0.3.3. No Desktop-side touch points beyond the version bump.
- **No data migration required.** New `__init__` call runs idempotently; subsequent boots find nothing to prune.

### Why this matters even if Desktop v0.3.3 Sections 9 + 9.1 close #8

Desktop v0.3.3 Section 9 (bulk `/clear/*` nuke-and-recreate) stops the *cause* of phantoms going forward. Section 9.1 unlinks `_completion_state.json` at clear-time, automating the reported manual workaround. v0.5.6 Item 32 sweeps existing phantoms from working/history/patterns at startup. None of those addresses:

- accumulated stuck flags on installs that never hit the GUI clear button;
- the per-write I/O cost as the file grows;
- the cross-session fallback misrouting in `main.py:1087-1096` between user-triggered clears.

This item is independently load-bearing for write-path correctness on long-running installs.

---

## Implementation status

Implemented and verified 2026-05-11. Live shared-DB smoke-test passed 2026-05-12 (see Desktop v0.3.3 verification report — Section 9 + 9.1 cross-client repro confirms `_completion_state.json` lifecycle now operates without manual file-deletion).

- `SessionManager._cleanup_completion_state(max_age_days=30, max_entries=500)` added at `roampal/hooks/session_manager.py:128-219`, wired into `__init__` at line 60-61 after the existing transcript and exchange-cache passes.
- `SessionManager._cleanup_old_transcripts` default bumped from `max_age_days=7` to `max_age_days=30` (and the matching `__init__` call site updated). Pairs the JSONL TTL with the new state-file TTL so paused conversations don't get re-cold-started on return.
- Always logs a single info line per startup (`pruned N entries (was X, now Y)`), even when N=0, so the prune pass is never silent on installs that have a state file.
- Always uses a single `write_json_atomic` at the end (no partial state on crash). Skipped when `pruned == 0` to avoid pointless rewrites.
- Defensive on malformed input: non-dict state, non-dict entries, and entries with unparseable timestamps are handled without raising. Unparseable timestamps survive the age pass but sort to "oldest" in the max_entries pass, so they get evicted first under pressure.
- Version bumped 0.5.6 → 0.5.7 in `pyproject.toml` and `roampal/__init__.py`.

## Verification

1. **Unit tests.** 4 new tests in `test_session_manager.py` cover (a) stale-timestamp prune, (b) missing-JSONL prune, (c) 500-entry ceiling after age pass, (d) atomic-write contract preserves the original file on simulated `os.replace` failure. **All 4 pass.** Full `roampal/` test suite: **720/720 pass** on 2026-05-12 (624 memory-module unit + 29 memory-module integration + 67 other unit tests including `test_profile_manager` and `test_cache_eviction`).
2. **Prod-shaped repro.** `_completion_state.json` seeded with 800 synthetic entries (100 stuck `scored_this_turn=True` at 60–160d old, 600 fresh with JSONLs, 100 fresh orphans with no JSONL). `SessionManager.__init__` pruned to 500 entries: 0 stuck flags survived, 0 orphans survived, file size 119,800 → 68,501 bytes. Second init pruned 0 — idempotent.
3. **Live smoke test.** End-to-end shared-DB exchange (Desktop bulk-clear → OpenCode dev → new memories within seconds) confirmed via Desktop v0.3.3 verification report 2026-05-12. No regression in the live scoring path (`scored_this_turn` flips True at score time, False at next turn boundary).

## Files changed

| File | Change |
|---|---|
| `roampal/hooks/session_manager.py` | New `_cleanup_completion_state` method + call from `__init__` |
| `roampal/backend/modules/memory/tests/unit/test_session_manager.py` | New test file (4 tests) |
| `pyproject.toml` | version 0.5.6 → 0.5.7 |
| `roampal/__init__.py` | `__version__` 0.5.6 → 0.5.7 |
| `dev/docs/releases/v0.5.7/RELEASE_NOTES.md` | This file |
