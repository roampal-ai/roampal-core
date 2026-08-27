# v0.5.9 — Implementation Task List

Incremental build. Each task is implemented and smoke-tested on its own, then
reported back for review before the next task starts. The heavy acceptance
gates (RSS soak, 3× accuracy gate, latency benchmark) run once at the end.

Pre-flight facts (verified 2026-08-25):
- `roampal 0.5.8` installed; `onnxruntime 1.23.2`; Python 3.10.0.
- Both target INT8 models already in HF cache at the doc's stated sizes:
  e5-base `model_qint8_avx512_vnni.onnx` = 265.8 MB; mmMarco CE same = 113.1 MB.
  No network needed at build/install time.
- Real profile data exists at `~/.roampal/data` (the `main` profile) → the
  migration path will run against live vectors. **Back this up before Task 5.**

Known doc gaps to correct while implementing (from pre-impl review):
- G1: `unified_memory_system.py:1022` (memory_bank store/dedup embed) is a
  PASSAGE site omitted from the Item 2 call-site list; `:1525` misnumbered as
  `:1535`. Real passage count is 14, not "twelve"/13.
- G2: Item 2 rationale "both XLM-R base at INT8" is wrong for mpnet
  (mpnet ≠ XLM-R). Memory-neutral conclusion still holds; reword.
- G3: Item 3b CE INT8 file now confirmed cached (113.1 MB) — verification gap closed.

## Tasks (in dependency order)

| # | Task | Item(s) | Files | Verification |
|---|------|---------|-------|--------------|
| 1 | Disable ONNX CPU arena + mem-pattern on both loaders | 1 | `embedding_service.py`, `search_service.py` | py_compile + load both models, confirm no ratchet (RSS flat across varied shapes) |
| 2 | Cross-encoder → INT8 export (constant only) | 3b | `search_service.py` | CE loads `model_qint8_avx512_vnni.onnx`; rerank runs |
| 3 | Shared cross-encoder session (module-level + lock) | 3a | `search_service.py` | Two `SearchService` instances / profiles share one `InferenceSession` (id equality); concurrent load constructs once |
| 4 | Embedder → e5-base INT8 + `role`/prefix/cache/truncation 256 | 2 (incl. G1,G2) | `embedding_service.py` + all 18 call sites + `server/main.py` | e5 loads; `role="query"`/`"passage"` produce different vectors; `_embed_cache` keyed `(role,text)`; cap 256; 768-dim; prefix disabled for non-e5 repo |
| 5 | Background re-embed migrator + metadata + CLI | 2a | new `embedding_migrator.py`, `unified_memory_system.py`, `cli.py` | Per-collection `embeddings_meta.json`; auto-trigger on artifact mismatch; compare-and-swap; resume; lock; `roampal reembed` |
| 6 | Migration visibility (health/search state) | 2b | `routing_service.py`, `search_service.py`, `unified_memory_system.py` | `/api/health` answers during migration; `search_memory` returns progress msg; unflipped collections excluded |
| 7 | File logging + MemoryError handling + RSS heartbeat | 5 | `mcp/server.py`, `server/main.py`, `search_service.py` | Rotating + PID-scoped logs created; MemoryError logged before exit (incl. ExceptionGroup); 60s heartbeat lines |
| 8 | Degraded-state responses | 7 | `search_service.py`, `embedding_service.py`, `server/main.py`, `mcp/server.py` | Embedder-down → explicit message (never bare `[]`); reranker-down annotates; `/api/status` shape |
| 9 | Doc fixes (ARCHITECTURE/README) + version bump | 6 | `ARCHITECTURE.md`, `README.md`, `pyproject.toml`, `__init__.py` | Figures corrected; version 0.5.9 |
| 10 | Final gates | all | tests | RSS soak plateau ±5%; 3× accuracy gate (mpnet128 / e5-128 / e5-256) within 5pp; latency p95 improves |

## Status
- [x] 1 — DONE (verified: CE RSS flat 630→643 MB across varied shapes; arena/mem-pattern off asserted; py_compile clean). Awaiting review.
- [x] 2 — DONE (verified: CE loads model_qint8_avx512_vnni.onnx 113.1 MiB; ~160 MB lighter than FP16; 40x256 rerank 375 ms; ROAMPAL_CE_ONNX_FILE wired; py_compile clean). Awaiting review.
- [x] 3 — DONE (verified: mocked test shows exactly ONE InferenceSession across 2 concurrent loads; both instances share it via id() equality; py_compile clean). Awaiting review. NOTE: editable `pip install -e .` done during this task so imports use local source (env now points at C:\roampal-core; version still 0.5.8).
- [x] 4 — DONE (verified: e5 default repo+INT8 ONNX; prefix gating e5-only; query!=passage vectors; cache keyed (role,text); 768-dim; cap 256; all 9 production call sites carry role — 7 explicit + 2 health-check defaults; py_compile clean). Awaiting review. KNOWN FOLLOW-UP: existing unit tests using `def embed_fn():` mocks (e.g. test_search_service.py:62) will need role-aware mocks — handled in Task 10. NOTE: doc's "18 call sites" count was stale (pre-e5 reference count); real production count is 9.
- [x] 5 — DONE (verified: `embedding_migrator.py` + UMS `_maybe_schedule_reembed`/`_run_reembed` wiring + `roampal reembed` CLI all present; smoke test passes — re-embed changes vectors, text/documents preserved, per-collection resume, single-runner lock (live-pid holder skips; stale/dead-pid takes over), compare-and-swap skips records edited mid-batch; py_compile clean). Awaiting review. Live profile backed up to `~/.roampal/data_backup_20260825102213` before review. NOTE: module docstring "atomic per collection" corrected to "completed per collection" (flips batch-by-batch, resumable).
- [x] 6 — DONE (verified: `UMS.get_migration_state()` returns {active,onnx_file,migrated,pending}; `search()` excludes pending/un-flipped collections while a migration is active (short-circuits to [] when all requested are pending); `/api/health` and `/api/search` include `migration` in their JSON; `search_memory` MCP tool prepends a re-embed progress note when active; endpoint call sites guarded with `isinstance(dict)` so MagicMock-based tests stay serializable; py_compile clean; 7 existing health/search endpoint tests still pass). Awaiting review. NOTE: `routing_service.py` was assessed and intentionally NOT modified — it is stateless (always routes to all 5 collections) so the collection-exclusion logic belongs in `UMS.search`, which already owns collection selection.
- [x] 7 — DONE (verified: shared helpers added to `search_service.py` — `configure_logging` (console + rotating, PID-scoped `~/.roampal/logs/roampal-{component}-{pid}.log`, idempotent, never raises), `start_rss_heartbeat` (daemon thread logging RSS every 60s via psutil), `log_memory_error` + `_contains_memory_error` (detects MemoryError incl. ExceptionGroup-wrapped, structurally typed for 3.10). Wired into `server/main.py` `start_server` and `mcp/server.py` `run_mcp_server`: logging + heartbeat at startup, `uvicorn.run`/`asyncio.run(main())` wrapped in a BaseException guard that logs MemoryError before `sys.exit(1)`. Smoke test passed (file created + probe line + RSS heartbeat line; MemoryError/ExceptionGroup detection; idempotent logging); 42 existing endpoint tests still pass). Awaiting review.
- [x] 8 — DONE (verified: `search_service.py` now raises `EmbedderUnavailable` on embed failure (was bare `[]`); `_rerank_with_ce` annotates via `_rerank_skipped` flag + `get_status()`; `unified_memory_system.search` propagates `EmbedderUnavailable` (re-raises instead of falling back to a second failing embed) and wraps the inline embed path; `/api/search` returns an explicit 200 `degraded` message on embedder-down and includes `rerank_skipped`; new `/api/status` aggregates embedder/reranker/migration; `search_memory` MCP tool prepends degraded + rerank-skipped notices. Smoke test passed (embedder-down raises, reranker-down sets flag, get_status shape, available path clears flag); 42 endpoint tests still pass; `/api/status` model field hardened for serialization). Awaiting review. NOTE: real HTTP 200 degraded response + live `/api/status` boot not exercised against a running server (unit-level propagation + no-regression endpoint tests only).
- [x] 9 — DONE (verified: `ARCHITECTURE.md` lines 37/516 corrected to ~484MB steady state + shared-CE note (Item 3a); `README.md` line 263 RAM requirement corrected to ~500MB, plus a new v0.5.9 changelog entry added ahead of v0.5.8's; `pyproject.toml` and `roampal/__init__.py` bumped 0.5.8 → 0.5.9, confirmed via `python -c "import roampal; print(roampal.__version__)"` and `roampal --version`). Awaiting review.
- [~] 10 — RUN, GATE B FAILED, DECISION MADE (2026-08-26). RSS soak: PASS (1016→1020MB across 500 varied-shape ops, +0.4%, no ratchet). Latency: PASS (embed 20.7→8.1ms, CE rerank 1388→1190ms, both faster). Accuracy gate: FAIL — mpnet baseline 100%, e5-base at both cap128 and cap256 landed at 57.5% (synonym/typo/acronym/partial-match each down 60-75pp). Root cause: `FACT_DEDUP_DISTANCE_THRESHOLD=0.32` (`unified_memory_system.py:387`, a pre-existing v0.5.3 guard) was calibrated for mpnet's embedding geometry; under e5-base, genuinely distinct facts collapse to distance 0.06-0.09, so nearly every new fact write is silently rejected as a false duplicate. Verified live against the real, already-migrated production profile (not just the isolated test harness) — `data/main/embeddings_meta.json` showed the auto-migration had already completed in production. **Decision: held back Item 2.** `embedding_service.py`'s `HF_REPO`/`ONNX_FILE`/`DEFAULT_MODEL` reverted to `paraphrase-multilingual-mpnet-base-v2` INT8 (same quantization win as e5-base, zero dedup risk, zero migration risk — existing vectors stay valid, cosine agreement 0.990). e5-base is deferred to v0.6.0, gated on recalibrating `FACT_DEDUP_DISTANCE_THRESHOLD` for its embedding geometry (or making it model-aware) and re-running this gate clean. `roampal/backend/modules/memory/tests/unit` reconfirmed green (636 passed, 3 skipped) after the revert. Task 10 is not "done" in the sense of "ship e5-base" — it's done in the sense of "the gate did its job and caught a real regression before release."

## Post-gate correction #2: migration-detector blind spot (2026-08-26)

Executing the mpnet revert against the real live server (not just editing source) surfaced a second, independent bug: `collection_needs_migration()` (`embedding_migrator.py`) compared only the ONNX file's relative path against `embeddings_meta.json`, never the model repo. mpnet-INT8 and e5-base-INT8 both ship their export at the identical relative path `onnx/model_qint8_avx512_vnni.onnx` inside their own separate HF repos, so switching `HF_REPO` alone (onnx_file unchanged) was invisible to the mismatch check. Live consequence: after reverting `embedding_service.py` and restarting the real server, it came up correctly running mpnet, but the reverse-migration that should have followed never triggered — leaving the profile's stored vectors in e5-base's embedding space while new queries/writes were computed in mpnet's, inside the same collections. Exactly the cross-family mixing the whole Item 2a design exists to prevent.

Fix: `embedding_migrator.py` now keys migration state on `artifact_key(model, onnx_file)` — a composite `"<repo>::<onnx_file>"` string — everywhere a bare `onnx_file` was previously compared or stored (`collection_needs_migration`, `meta_has_mismatch`, the per-collection values written by `migrate_profile` and `_maybe_schedule_reembed`'s fresh-install path). `read_meta()` transparently upgrades pre-existing meta files written under the old bare-filename scheme to the composite key (using the file's top-level `model` field, which was always accurate for every collection under the old single-artifact-per-profile design), so no spurious re-embed fires for profiles already migrated correctly. Updated both call sites in `unified_memory_system.py` (`_maybe_schedule_reembed`, `get_migration_state`) to pass `model` through. `roampal/backend/modules/memory/tests/unit` reconfirmed green (636 passed, 3 skipped) after this fix. Verified live: `read_meta()` against the real (already-corrected-via-`roampal reembed --force`) profile now correctly reports `needs_migration=False` for all 5 collections with no further action needed.

Live remediation performed on the dev machine's actual profile (`main`, the developer's live profile): stopped the live server (`roampal stop`), which picked the reverted mpnet config back up on the MCP client's next auto-relaunch; then ran `roampal reembed --force` (221 records) to correct the vectors the blind-spot bug left un-migrated. Confirmed post-fix: `/api/status` embedder = mpnet, `embeddings_meta.json` fully mpnet with a fresh timestamp, and a real semantic query returns correct, relevant results.

## Post-review corrections (2026-08-26)

A full re-verification of tasks 1-8 against the actual diff (not just the
task-list narrative) found three regressions that `py_compile` (syntax-only)
did not catch, plus the known mock follow-up from task 4. All four are fixed;
`roampal/backend/modules/memory/tests/unit` is green (636 passed, 3 skipped)
and the CLI was smoke-tested end-to-end (`--help`, `--version`, `reembed --help`
all dispatch correctly).

- **`cli.py` — entire CLI was a no-op.** Task 5's `cmd_reembed` definition had
  been inserted at column 0 in the middle of `main()`, right after
  `subparsers = parser.add_subparsers(...)`. That silently ended `main()`'s
  body there; every subsequent subparser (`init`, `start`, `status`, `stats`,
  `doctor`, `profile`, ...), `parser.parse_args()`, and the whole dispatch
  chain became unreachable dead code nested inside `cmd_reembed`, after its
  own `return 0`. Fixed by moving `cmd_reembed` to its own top-level function
  (placed after `cmd_profile`, before `main()`), leaving `main()`'s body
  intact. Verified via AST (`main()` now has 108 top-level statements,
  `cmd_reembed` is a standalone module-level function) and by running the CLI.
- **`memory_bank_service.py:154` — `update()` crashed on the normal path.**
  The re-embed line (`new_embedding = await self.embed_fn(new_text,
  role="passage")`) was over-indented into the `if not old_doc:` branch
  (which already returns above it) instead of the main body, so
  `new_embedding` was never assigned on a real update — `NameError`. Fixed
  by dedenting the line back into `update()`'s main body.
- **`unified_memory_system.py` — `_migration_task` didn't exist until
  `initialize()` reached line ~648.** Anything that read it earlier (e.g.
  `search()`, `get_migration_state()`) raised `AttributeError`. Fixed by
  also initializing `self._migration_task = None` in `__init__`.
- **Task 4 mock follow-up, resolved now instead of deferring to task 10.**
  `test_unified_memory_system.py` had several `embed_text`/`embed_texts`
  mocks with bare `(text)`/`(texts)` signatures that didn't accept the new
  `role=` kwarg. Updated the mocks to accept `**kwargs` (and one
  `assert_called_with` to expect `role="passage"`) rather than silently
  passing on the wrong signature.

Scope note: `cli.py` is ~4,700 lines. Deliberately NOT refactoring it in this
release — bug fixes and a structural split shouldn't land in the same diff.
Worth a dedicated follow-up release.

## Post-review round 2 (2026-08-27)

External review verdict: hold on three blockers. All three confirmed in code and fixed, plus card/notes corrections.

1. **Zero new tests** — CONFIRMED: none of the coverage plan's test files existed; the "which claim each test proves" table asserted gates that were never written. Landed: 	est_onnx_memory_options.py (arena/mem-pattern off on both loaders, thread override, default INT8 artifacts), 	est_embedding_migrator.py (composite keying incl. same-filename model swap, legacy meta upgrade, CAS, lock semantics, dry-run guarantee), 	est_degraded_states.py (EmbedderUnavailable propagation, rerank-skip flag, status shape), 	est_memory_hardening.py (MemoryError/ExceptionGroup detection + fatal record + stderr fallback), TestSharedCrossEncoder in test_search_service.py (1 session across 4 concurrent loaders), TestPrefixGating in test_embedding_service.py, and 	est_rss_soak.py (integration, slow-marked, real INT8 embedder, 120 varied-shape ops, first-vs-last-fifth RSS drift < 5% - PASSED in 42 s on the dev machine). Deliberately not landed: TestQuantizedCE (needs both CE exports), 	est_reembed_live.py, 	est_benchmark_gates.py wrapper - recorded as manual/manual-pending in RELEASE_NOTES' coverage tables, which are now labeled LANDED / NOT LANDED / PARTIAL per row.
2. **--dry-run corrupted migration state** - migrate_profile wrote completed[coll_name] + write_meta unconditionally per collection (the existing if not dry_run guarded only a redundant second write). A dry run marked every needing-migration collection as done; later real reembeds skipped them. Fixed: meta write now guarded on 
ot dry_run; regression test asserts meta untouched and zero vector writes.
3. **oampal reembed self-blocked on its own lock** - cmd_reembed's initialize() fired the background scheduler, which raced the direct migrate_profile call for the PID lock (same process, same PID - PID-liveness can't arbitrate). Fixed two ways: cmd_reembed sets ROAMPAL_REEMBED_DISABLE=1 before constructing UMS (mechanism already existed), and _acquire_lock gained an in-process reentrancy guard so coroutines in one process can never fight over the lock file.

Non-blocking fixes also landed: chromadb_adapter.py temp-client close() is now guarded with hasattr (close() only exists on chromadb >= ~1.5.x; the supported pin allows 1.0+, and production uses HttpClient mode regardless). .gitignore now covers 	est_data_*/ so the quality-gate artifacts cannot be swept into a commit.

Card/notes corrections after the same review: website card now shows 735/3 full-suite tests (636 was the memory-unit subset), drops the soak percentage that was measured on the FP16 config, and uses the Task 10 gate-run latency figures (embed 20.7 -> 8.1 ms, rerank 1,388 -> 1,190 ms) instead of isolated A/B numbers; RELEASE_NOTES' latency table relabeled accordingly, Files-changed table corrected (routing_service.py removed - never modified; heartbeat location; MCP child capture marked as the known gap it is).

**Hook-path decision (2026-08-27, closed):** EmbedderUnavailable escaping /api/hooks/get-context previously surfaced as HTTP 500 (plugin swallows it — the model never knew, same net effect as the old silent empty injection). Decision: the hook endpoint now catches EmbedderUnavailable and returns HTTP 200 with an explicit degraded marker the plugin injects ("[Roampal: memory search unavailable - embedding model is down. Proceeding without memory context; do not claim memories were checked.]") — visible to the model on every path, consistent with Item 7's "never make degraded look like empty." Genuinely unexpected failures still hit the generic 500 handler. Regression test: test_fastapi_endpoints.py::TestGetContextEndpoint::test_embedder_down_returns_visible_marker_not_500.

## Post-review round 3 (2026-08-27)

External review probed concurrency paths directly. One new blocker, found and fixed:

1. **In-process reentrancy guard locked out OTHER PROFILES** (blocker, introduced by the round-2 lock fix). The guard was one global bool, but the lock it protects is per-profile (data_path): while profile A migrated, profile B was refused a completely different data_path until A finished — a fresh multi-profile v0.5.9 upgrade migrated only one profile per server lifetime. Reproduced live, fixed by keying the guard on a module-level set of resolved lock paths. Two-path regression test added (test_two_profiles_lock_independently); same-path reentrancy and the PID-file semantics unchanged. Also hardened _acquire_lock to mkdir the data_path parent (lock writes no longer fail on a not-yet-created path).
2. **Embedder lazy load had no lock** (the reviewer's degraded-path note, fixed rather than documented). _load_model was check-then-act with no serialization: concurrent first-touches could each build an InferenceSession (N x 285 MB), and the server was safe only because prewarm() happens to run first. Added a threading.Lock with double-checked loading to EmbeddingService._load_model — safety by construction. Regression test: test_onnx_memory_options.py::TestEmbedderLazyLoadRace (4 threads, barrier, exactly 1 session).
3. **Cross-encoder concurrency verified externally on real models**: 2 threads -> one shared session, concurrent 40-pair reranks interleave on ORT's intra-op pool (~29% latency penalty vs solo, not 2x; note: on low-core machines contention is higher — ROAMPAL_ORT_THREADS is the lever). Matches what TestSharedCrossEncoder asserts.

Status-only, documented: _rerank_skipped is per-SearchService while get_context_for_injection fires two search lanes concurrently (unified_memory_system.py:1358/1364), so a lane can overwrite the flag — rerank_skipped in API/status responses is best-effort, no data impact.

Website card: tile wording scoped to match the bullet ("runs wherever the INT8 model is cached") instead of implying CI enforcement. Full suite after fixes: 780 passed, 3 skipped.
