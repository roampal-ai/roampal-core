# Roampal Core v0.5.3 — 2026-04-23

**Release Date:** 2026-04-23
**Type:** Patch. Changes:

- Sidecar JSON-shape tolerance for small local models (mirrors Desktop v0.3.2 Section 0k)
- Port desktop's pre-store fact dedup into core with asymmetric per-tier scope
- Fix TagCascade `$contains` where-clause bug (mirrors Desktop v0.3.2 Section 0m)
- Fix memory_bank multi-key filter wrapping (mirrors Desktop v0.3.2 Section 0n)
- Fix tag extraction gap: summaries and facts stored without noun_tags because plugin never sent them and server-side extraction was never wired up
- Require explicit sidecar configuration: default scoring no longer silently cascades to Zen cloud or localhost Ollama/LM Studio. Users pick a backend via `roampal sidecar setup` (detected local models, custom endpoint, or explicit Zen opt-in) or scoring is disabled
- Fix `roampal sidecar setup --scope project` parse-error message: was "Cannot read {path}" (vague, same as I/O errors), now shows specific JSON error with repair guidance — matches `configure_opencode`'s format. Also removed dead `FileNotFoundError` catch from the block
- Fix `roampal sidecar status --scope <scope>` missing argument: parser didn't define `--scope` for the `status` subcommand even though `_cmd_sidecar_status()` expected it via `getattr(args, "scope", None)` — now added

## Summary

One class of bug in three places. Small quantized sidecar models (reported
on `qwen2.5:3b`; reproducible on comparable 3B-class checkpoints) sometimes
respond to structured-JSON prompts with a bare array (`["x", "y"]`) instead
of the schema-wrapped object (`{"tags": [...]}`, `{"facts": [...]}`) the
prompt explicitly asks for. Core's sidecar code assumes the wrapped shape
at three call sites and crashes with
`AttributeError: 'list' object has no attribute 'get'` when the wrapper
isn't there. Every subsequent call for that exchange fails the same way;
the retry queue eventually gives up. For users running a 3B-class sidecar,
the affected lanes (tags, facts) produce no data silently.

Surfaced 2026-04-21 during Desktop v0.3.2 laptop-testing with
`qwen2.5:3b` as the sidecar. Desktop ships its fix in v0.3.2 Section 0k;
core inherits the same three call sites from the ported sidecar code and
gets the same one-line-class fix here.

---

## Implementation order

Sections 1–7 and Section 11 are independent one-liners and small refactors —
land in any order (prefer Section 4 last because its test surface is the
largest).

Sections 8–10 share helpers and have a required order:

1. **Section 10 first.** Creates `roampal/utils/atomic_json.py` with
   `write_json_atomic()`. Nothing else depends on this being in place —
   land + test in isolation.
2. **Section 8 and Section 9 together.** Both introduce the shared
   `_safe_write_opencode_config()` helper (atomic + backup, layered on
   `write_json_atomic` from step 1). Both sidecar subcommands (Section 9)
   and `configure_opencode` (Section 8) must call it — no direct
   `path.write_text(json.dumps(...))` against `opencode.json` should
   remain in `cli.py` after this step.

Sections 13–14 are post-implementation audits: Section 13 fixes the one
remaining gap (parse-error message parity), Section 14 confirms no other
CLI write paths to `opencode.json` are missing parse guards or atomic writes.

Rule of thumb: `write_json_atomic` is for machine-managed state
(`profiles.json`, session state). `_safe_write_opencode_config`
is for human-authored config (`opencode.json`) and adds the timestamped
backup step. Don't conflate them.

---

## Scope

### 1. `extract_noun_tags` — bare-array tolerance

`sidecar_service.py:908`

```python
# Before
tags = result.get("tags")

# After
tags = result if isinstance(result, list) else result.get("tags")
```

The existing `if not isinstance(tags, list): return None` guard on the
next line already handles all other pathological shapes, so no downstream
logic changes.

**Impact if unfixed:** TagCascade retrieval runs on empty tag lists for
every memory stored by a 3B-class sidecar. Tag-first routing silently
degrades to cosine + CE only (no tag-prefilter), increasing retrieval
latency and hurting tag-conditioned recall.

### 2. `extract_facts` — bare-array tolerance

`sidecar_service.py:968`

```python
# Before
facts = result.get("facts")

# After
facts = result if isinstance(result, list) else result.get("facts")
```

**Impact if unfixed:** Fact extraction lane returns None for every
exchange scored by a 3B-class sidecar. User-level facts never land in
working memory; cold-start profile and `always_inject` paths see nothing.

### 3. Diagnostic validator — bare-array tolerance

`sidecar_service.py:test_sidecar_scoring()` (function backing
`roampal sidecar test`).

**Initial fix attempted** — copy the §1/§2 one-liner into the `facts`
field extraction at line 1038. **Did not work** because the function
extracts four fields (`summary`, `outcome`, `noun_tags`, `facts`) from
the same response, and the first three call `result.get(...)` BEFORE
reaching the facts line. A bare-list response crashes at line 1017
(`summary`) before the §3 fallback is reached.

**Final fix.** Early-return at the top of the function, after the
`if result is None` check:

```python
if isinstance(result, list):
    return {
        "passed": False,
        "fields": {},
        "error": "Sidecar returned a bare JSON array; expected an object "
                 "with summary/outcome/noun_tags/facts. Try a larger model.",
    }
```

The previous bare-list fallback at line 1038 is removed (dead code given
the early-return). User now gets a clear "try a larger model" message
instead of a crash trace from `roampal sidecar test`.

### 3.5. `summarize_only` — type guard *(found during v0.5.3 QA, same bug family)*

`sidecar_service.py:summarize_only()` lines 873 + 882.

Same family as §1–§3. `_call_llm` returns whatever `_extract_json`
parses — can be dict, list, string, or None despite the
`Optional[Dict[str, Any]]` annotation. `summarize_only` did
`result.get("summary")` with no type guard, so a 3B model returning a
bare string or list would crash mid-summarization.

```python
# Before
return result.get("summary") if result else None

# After
return result.get("summary") if isinstance(result, dict) else None
```

Same guard added on the `_call_anthropic_model` branch above it.
Less likely to fire than §1/§2 because the prompt asks for an object
not a list, but the failure shape is identical.

**General rule.** Anywhere `_call_llm` flows into `result.get(...)`,
guard with `isinstance(result, dict)` first. Audited the rest of the
file: `extract_tags`, `extract_facts`, `test_sidecar_scoring`, and
`summarize_only` are now the only consumers and all are protected.

### 4. Port desktop's pre-store fact dedup into core

**Problem.** Core has **no** pre-store deduplication in any store path —
`store_working`, `store_memory_bank`, and the generic `store` router all
just embed and write. Desktop v0.3.2 shipped a fact dedup guard
(`_find_duplicate_fact` in
`ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py:630`,
called from `store_memory` at :701). In the shared-ChromaDB deployment
(desktop app + core MCP server both writing to the same store), desktop's
guard is effectively half-on: every fact written through core's
`add_to_memory_bank` or `record_response` path skates right past desktop's
check and lands as a duplicate. A one-shot backfill cleaner is weaker than
closing the hole at the write path, because without the gate the backfill
keeps having to run.

**Port.** Mirror desktop's v0.3.2 final shape in
`roampal/backend/modules/memory/unified_memory_system.py`. Desktop's
implementation (verified 2026-04-21) uses a **per-tier filter dict**
because `memory_bank` rows don't carry `memory_type` metadata — a
unified `{"memory_type": "fact"}` filter would miss every existing
memory_bank entry:

```python
# Near the top of UnifiedMemorySystem class body:
# L2² = 2 − 2·cos_sim  ⇒  L2 = √(2 − 2·cos_sim)
# cos_sim > 0.95 ⇒ L2 < √0.1 ≈ 0.316. Using 0.32 matches desktop.
FACT_DEDUP_DISTANCE_THRESHOLD = 0.32
FACT_DEDUP_TIERS = ("working", "history", "patterns", "memory_bank")

# Per-tier filter. working/history/patterns hold mixed memory types
# (exchanges, summaries, facts) so scan only facts. memory_bank is
# fact-shaped by construction and doesn't tag rows with memory_type,
# so scan unfiltered.
FACT_DEDUP_FILTERS = {
    "working": {"memory_type": "fact"},
    "history": {"memory_type": "fact"},
    "patterns": {"memory_type": "fact"},
    "memory_bank": None,
}

async def _find_duplicate_fact(
    self,
    embedding: List[float],
    tiers: Optional[tuple] = None,
) -> Optional[str]:
    """Return existing doc_id if a near-dup fact is already stored.

    Scans the given tiers (default: all of FACT_DEDUP_TIERS) with
    per-tier filters (see FACT_DEDUP_FILTERS) and a cosine-distance
    threshold of 0.32 (~95% similarity). Returns the first match;
    None if no duplicate exists. Best-effort: exceptions are
    swallowed and logged — dedup must never block a store.

    Tier-scope is asymmetric by write persistence — see call sites.
    """
    scan_tiers = tiers if tiers is not None else self.FACT_DEDUP_TIERS
    for tier in scan_tiers:
        col = self.collections.get(tier)
        if col is None:
            continue
        try:
            hits = await col.query_vectors(
                query_vector=embedding,
                top_k=1,
                filters=self.FACT_DEDUP_FILTERS.get(tier),
            )
        except Exception as e:
            logger.debug(f"[DEDUP] query_vectors failed on tier={tier}: {e}")
            continue
        if hits and hits[0].get("distance", 2.0) < self.FACT_DEDUP_DISTANCE_THRESHOLD:
            return hits[0].get("id")
    return None
```

**Asymmetric scope — important.** Desktop v0.3.2's final shape (verified
2026-04-21) uses different `tiers` at each call site:

| Write surface | `tiers=` | Why |
|---|---|---|
| `store()` fact branch (ephemeral sidecar extraction) | default (all 4) | Cheap writes defer to any more-persistent copy that already exists anywhere. |
| `store_memory_bank` (permanent AI/user-curated) | `("memory_bank",)` | A permanent write must NEVER be blocked by an ephemeral working-tier copy. If it were, promoting a chat-extracted fact ("user's name is George") to permanent memory would get absorbed by the 24h-TTL working copy; when working rolls over, the user's identity is lost. Memory_bank scans only within its own tier. |

Age-out rather than cleanup: when `store_memory_bank` writes despite a
working-tier duplicate existing, the ephemeral copy is **not** deleted.
It ages out naturally at working's TTL. Memory_bank's `score: 1.0` row
dominates retrieval ranking in the overlap window, so the duplicate
doesn't poison context.

**Call sites.** Add the guard before the write in the paths that store
facts:

- `store_memory_bank` (`unified_memory_system.py:839`) — wraps
  `_memory_bank_service.store`. Compute the embedding once in this
  method, call `_find_duplicate_fact(embedding, tiers=("memory_bank",))`,
  and if a match is returned skip the write and return the existing
  `doc_id`. On cache miss, thread the embedding through to
  `_memory_bank_service.store` via a new optional `embedding` param
  (desktop did this — adds 0 extra embed calls).
- `store_working` (`unified_memory_system.py:1523`) and the `patterns`/
  `history` branch of `store` (:811–832) — gate **only** when
  `metadata.get("memory_type") == "fact"` (mirrors desktop's
  fact-type-only filter at desktop `unified_memory_system.py:701`).
  These sites call `_find_duplicate_fact(embedding)` with the default
  tier scope (all 4). Non-fact writes (exchanges, summaries, book
  chunks) bypass the check unchanged.

**Tiers list note.** Desktop's v0.3.2 ships with memory_bank in its
tiers list — this port is pure parity, not an expansion. The key detail
is that memory_bank is scanned **without** a `memory_type` filter
(see `FACT_DEDUP_FILTERS["memory_bank"] = None` above) because its rows
don't tag `memory_type`.

**Impact if unfixed.** In a shared-DB deployment, desktop's v0.3.2 dedup
is bypassed by every write that goes through core's MCP tools. Duplicate
facts accumulate — which is worse than no dedup, because the user thinks
the problem is solved. TagCascade retrieval then returns the same fact
twice at the top of the list, inflating its apparent confidence and
pushing genuinely different facts off the context budget.

**Not in scope.** No backfill. This PR closes the write-path hole only;
historical duplicates stay. If a cleanup pass is wanted later, it becomes
a separate `roampal dedup` one-shot (parallel to `roampal retag`) that
can run post-port without fighting fresh writes.

### 5. TagCascade `$contains` where-clause silently broken

`roampal/backend/modules/memory/search_service.py:353`

**Identical bug to desktop v0.3.2 Section 0m.** The TagCascade
tag-prefilter uses:

```python
tag_filter = {"noun_tags": {"$contains": f'"{tag}"'}}
# passed to ChromaDB's `where` clause
```

ChromaDB's `where` clause only supports value operators (`$gt / $gte /
$lt / $lte / $ne / $eq / $in / $nin`). `$contains` is a `where_document`
operator, not a `where` operator. Every tag query since whenever core's
TagCascade landed has thrown `Expected where operator to be one of ...`,
caught it in `try/except`, logged a warning, and moved on with empty
results — fall-through to unfiltered cosine on the next branch.

**Mock drift is the same root cause as desktop.** `test_search_service.py:134`
mocks `hybrid_query` with a side-effect that accepts `filters["noun_tags"]["$contains"]`
as a key lookup and returns pre-seeded results. The mock happily respected
an operator ChromaDB production rejects. Tests green, production broken.

**Fix.** Same approach as desktop v0.3.2 Section 0m:
- Drop the where-clause tag filter
- Over-fetch candidates by vector + text (`top_k=limit * 8`, vs `limit * 2`
  previously — compensates for lost pre-filter selectivity)
- Filter by tag membership in Python against `metadata.noun_tags`, tolerating
  both JSON-encoded string (sidecar write path) and already-parsed list
- Update `test_search_service.py` mock to stop pretending `$contains` works;
  rewrite or remove the tag-filter assertion since the filter no longer
  exists in the production path — tag match now happens post-fetch in
  Python and should be unit-tested as a pure-function helper (parse
  `noun_tags`, check membership).

**Impact.** Same as desktop: TagCascade's tag-scoping actually runs. No
schema change. No data migration. Tolerant of both storage shapes of
`noun_tags`.

### 6. TagCascade summary-lane multi-key filter on memory_bank

`roampal/backend/modules/memory/search_service.py` — same
`_tag_routed_search` function as Section 5, sibling bug at the
`memory_bank`-collection branch.

**Identical bug to desktop v0.3.2 Section 0n.** When combining the
incoming metadata filter (e.g., `{"memory_type": {"$ne": "fact"}}` from
the summary lane) with memory_bank's status filter, the code produces
a top-level multi-key dict that ChromaDB rejects:

```
[ChromaDB] Query failed: Expected where to have exactly one operator,
got {'memory_type': {'$ne': 'fact'}, 'status': {'$ne': 'archived'}} in query.
```

ChromaDB's `where` clause accepts either one single-key operator dict
or an explicit `$and`/`$or` logical wrapper. Multi-key top-level dicts
are rejected.

**Why it was hidden.** Section 5's `$contains` exception threw before
the filter shape ever reached `hybrid_query`. Fixing Section 5 exposes
this bug. Do not ship Section 5 without Section 6, or the bug simply
changes form.

**Fix.** Wrap in `$and` when combining filters — same pattern already
present in the cosine-fill branch of the same file (desktop's
`_search_collection` at `:547-553` has the correct form; copy it to
the tag-routed branch):

```python
mb_filters = metadata_filters
if coll_name == "memory_bank":
    status_filter = {"status": {"$ne": "archived"}}
    if metadata_filters:
        mb_filters = {
            "$and": [{k: v} for k, v in metadata_filters.items()]
                    + [status_filter]
        }
    else:
        mb_filters = status_filter
```

**Impact.** Summary lane's query against memory_bank returned zero
results before (query rejected, empty list). memory_bank holds facts
not summaries, so the retrieval-quality loss is near-zero; fix is about
log cleanliness and semantic correctness. One less error line per
retrieval query.

**Tests.** Add `TestMemoryBankFilterWrapping` class matching desktop's
4 cases:
- `test_no_metadata_filters_returns_bare_status_filter`
- `test_empty_metadata_filters_returns_bare_status_filter`
- `test_single_metadata_key_wraps_in_and`
- `test_multiple_metadata_keys_each_become_conditions`

---

## Files to touch

- `roampal/sidecar_service.py` — three one-line edits at lines 908, 968, 1035
- `roampal/backend/modules/memory/unified_memory_system.py` — add
  `FACT_DEDUP_DISTANCE_THRESHOLD`, `FACT_DEDUP_TIERS`,
  `FACT_DEDUP_FILTERS`, `_find_duplicate_fact` helper (with `tiers` param
  for asymmetric scope), and call sites in `store_memory_bank` (pass
  `tiers=("memory_bank",)`) and the fact-type branches of `store_working`
  / `store` (default scope, all 4 tiers)
- `roampal/backend/modules/memory/memory_bank_service.py` — add optional
  `embedding` param to `store()` so the caller can pass a pre-computed
  embedding through and avoid double-embedding
- `roampal/backend/modules/memory/search_service.py` — drop `$contains`
  tag filter (Section 5), over-fetch + Python-side tag match; also wrap
  memory_bank multi-key filter in `$and` (Section 6)
- `roampal/backend/modules/memory/tests/unit/test_search_service.py` —
  update mock that was pretending `$contains` worked; add Python-side
  tag-match tests matching desktop's `TestTagCascadePythonFilter` (7 cases)
  and memory_bank filter-wrapping tests matching desktop's
  `TestMemoryBankFilterWrapping` (4 cases)

---

## Tests to add

Mirror the two tests already landed on Desktop v0.3.2:

- `test_extract_noun_tags_bare_array` — LLM returns `'["calvin", "muscle car", "boston"]'`
  (no object wrapper); assert `extract_noun_tags` returns a non-empty list
  containing all three tags. Pre-fix: raises `AttributeError`. Post-fix:
  passes.
- `test_extract_facts_bare_array` — LLM returns `'["fact one is long enough", "fact two is also long enough"]'`;
  assert `extract_facts` returns a list of length 2. Pre-fix: raises
  `AttributeError`. Post-fix: passes.

The `test_non_list_*_returns_none` existing regression tests still apply
unchanged — the `isinstance(tags/facts, list)` guard after the fallback
continues to reject non-list, non-object payloads.

Mirror desktop's two test classes. Desktop ships 5 `TestV032FactDedup`
tests + 6 `TestV032MemoryBankDedup` tests (11 total, all green locally):

From `TestV032FactDedup` (fact-extraction surface / `store()`):
- `test_fact_stores_normally_when_no_duplicate` — cold path: empty DB,
  fact lands normally.
- `test_near_duplicate_fact_is_skipped` — cosine <0.32 in working →
  return existing id, no write.
- `test_distant_fact_is_not_deduped` — distance 0.4 → write proceeds.
- `test_summaries_are_not_deduped` — `memory_type != "fact"` bypasses
  the gate entirely.
- `test_dedup_checks_all_tiers` — dup lives in `history`, not `working`
  — still detected.

From `TestV032MemoryBankDedup` (memory_bank surface / `store_memory_bank`):
- `test_memory_bank_stores_normally_when_no_duplicate` — cold path.
- `test_memory_bank_near_duplicate_within_tier_is_skipped` — dup
  already in memory_bank, same surface → skip write.
- `test_memory_bank_write_not_blocked_by_working_fact` — **critical**:
  even with a high-similarity dup in working, memory_bank write must
  land. Verifies the asymmetric scope — memory_bank scanning only
  within its own tier. Without this, promoting a chat-extracted fact
  to permanent memory gets absorbed by the 24h-TTL working copy.
- `test_memory_bank_scans_only_memory_bank_tier` — regression guard
  for the asymmetric scope. Asserts `query_vectors` is called on
  memory_bank only, never on working/history/patterns. Catches any
  refactor that drops the `tiers` kwarg and leaks memory_bank writes
  back into the 4-tier scan.
- `test_working_fact_deduped_against_memory_bank` — reverse cross-tier:
  user saved via memory_bank first; subsequent sidecar fact write via
  `store()` fact branch IS blocked (default scope = all 4 tiers,
  including memory_bank). Confirms `store()`'s broad scan still sees
  memory_bank.
- `test_memory_bank_scan_uses_no_filter` — verifies
  `FACT_DEDUP_FILTERS["memory_bank"] is None`. A tempting refactor to
  unify filters would silently break memory_bank dedup (filter doesn't
  match → always cache miss → always writes).
- `test_distant_memory_bank_fact_is_not_deduped` — distance 0.4 →
  write proceeds.

From `TestConfigureOpencodeParseFailure` (Section 8 — parse-failure and
merge safety):
- `test_invalid_json_aborts_without_write` — pre-populate with invalid
  JSON; assert file byte-for-byte unchanged after call.
- `test_existing_mcps_preserved_on_merge` — pre-populate with another
  MCP entry, a `provider` block, and a `model` key; assert all survive.
- `test_existing_providers_preserved_on_merge` — pre-populate multiple
  providers; assert all survive.
- `test_atomic_write_leaves_no_tmp_file_on_success` — no `.tmp` left
  behind on success path.
- `test_backup_created_when_file_already_exists` — rewrite path creates
  a timestamped `.bak` of the pre-write contents.

Existing suite (534 backend + profile tests) must stay green.

---

## Release notes for users

```
v0.5.3: bug fixes, state-file safety, and retrieval quality repairs.

Retrieval and scoring
- TagCascade `$contains` filter was silently broken and falling back to
  cosine-only retrieval. Tag prefiltering now actually works; tag-conditioned
  recall improves correspondingly.
- memory_bank multi-key where-clause filters are now `$and`-wrapped; summary
  lane no longer logs ChromaDB query errors on every retrieval.
- Exchange summaries and extracted facts stored via the OpenCode plugin now
  carry `noun_tags`. A server-side extraction fallback was wired into both
  `/record-outcome` and `/api/hooks/stop`; previously tags were silently
  dropped because neither the plugin nor the server extracted them.

Sidecar robustness
- Sidecar tolerance for small local models: 3B-class checkpoints
  (qwen2.5:3b etc.) that return bare JSON arrays instead of the
  schema-wrapped shape no longer crash tag + fact extraction (mirrors
  Desktop v0.3.2 Section 0k).
- Pre-store fact dedup ported from Desktop v0.3.2. Core's write paths
  (`add_to_memory_bank`, `record_response`) now skip writes that would
  create a near-duplicate (cosine > 0.95) of a fact already in working /
  history / patterns / memory_bank. Closes the shared-ChromaDB hole where
  core writes bypassed desktop's existing dedup.

CLI and state-file safety
- `roampal init --opencode` no longer clobbers an existing `opencode.json`
  when the file has a JSON syntax error. Parse errors now abort with a
  clear message; writes are atomic (temp + rename) with a timestamped
  backup of the pre-write contents.
- `roampal sidecar {status,setup,disable}` is now scope-aware. `status`
  reports both user-global and project-local configs and flags when a
  project-local config shadows user-global. `setup` and `disable` accept
  `--scope {user|project|both}` and prompt interactively when a shadow
  is detected.
- Machine-managed JSON state files (`profiles.json`, session completion
  state) now write atomically. Crashes or power loss mid-save no longer
  corrupt the file.

Defense-in-depth
- `TagService.extract_tags()` guards against async `llm_extract_fn`
  callers. Previously silent empty results; now logs WARNING and returns
  `[]` explicitly.

Privacy and user trust
- No default cascade. Previous versions silently tried Zen cloud
  (opencode.ai), then localhost Ollama, then localhost LM Studio if no
  sidecar was configured. Now: if you haven't configured a sidecar,
  scoring is disabled. No exchange data leaves your machine
  automatically. Retrieval from existing memories still works.
- To enable scoring, run `roampal sidecar setup`. The picker shows
  any detected local models as first-class options. You can also pick
  "free Zen cloud models" explicitly (clearly labeled as rate-limited
  with data sent to opencode.ai), configure a custom API endpoint, or
  skip setup entirely (scoring stays off until you rerun setup).
- `roampal sidecar disable` says "scoring, summaries, and fact
  extraction are now disabled" instead of the previous misleading
  "reverted to free community models." It also clears the
  `ROAMPAL_SIDECAR_PRIORITY` env var so an earlier Zen opt-in doesn't
  persist after disable.

No config change. No data migration. Memories already stored with empty
`noun_tags` stay that way until re-scored; `roampal retag` (v0.4.9+) can
backfill tags on historical memories if desired. Historical duplicate
facts also stay — a future `roampal dedup` one-shot will clean those.
```

---

## Coordination

- Desktop v0.3.2 Section 0k Bug 1 already shipped the bare-array fix at
  two sites (`extract_facts` and `extract_noun_tags`). Core has a third
  site in its diagnostic validator that Desktop doesn't have — that's the
  only extra scope for Section 3.
- Desktop v0.3.2 already ships pre-store fact dedup at
  `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py:630/701`.
  Section 4 is a straight port into core so the shared ChromaDB is
  guarded from both write surfaces.
- No API or schema change. `sidecar_config.json` format is unchanged. The
  dedup guard is transparent to MCP clients — `add_to_memory_bank` returns
  a doc_id either way (fresh write or existing duplicate's id).
- No data migration. Memories stored with empty tag lists because of the
  bare-array bug will remain empty-tagged until re-scored — v0.5.3 simply
  prevents new memories from hitting the same path. If a user wants to
  repair historical memories, `roampal retag` (v0.4.9+) re-runs tag
  extraction with the current sidecar on already-stored memories.
- Historical duplicate facts are **not** cleaned up by the Section 4
  port. They stay until a future `roampal dedup` one-shot (tracked
  separately; not in v0.5.3). The port's job is to stop the bleed; the
  cleaner's job (later) is to wash the wound.

---

## Section 7 — Defense-in-depth: guard sync `TagService.extract_tags()` against async `llm_extract_fn`

**File:** `roampal/backend/modules/memory/tag_service.py:516-539`

**Not a shipping bug — latent dead code.** Core already removed every
production caller of the sync `extract_tags()` in v0.4.9.1 (the three
sites: `store_working`, `store_memory_bank`, direct `store`). The
method exists but nothing in the live path invokes it.

Desktop's v0.3.2 Bug 5 audit caught that the sync method silently
fails when `_llm_extract_fn` is async — it calls the async fn without
`await`, gets a coroutine, passes it to `_normalize_llm_tags`, which
crashes on `for tag in tags`, and the blanket `except` returns `[]`
with a DEBUG-level log (suppressed at INFO). Desktop was hit because
`store_memory_bank` still called sync `extract_tags` there.

Core isn't hit today — but the method is still a booby trap for the
next person who adds a caller.

**Fix (add when touching `tag_service.py` for any reason):**

```python
# Add at the top of extract_tags() body, before the current
# if self._llm_extract_fn: block
if self._llm_extract_fn:
    import inspect
    if inspect.iscoroutinefunction(self._llm_extract_fn):
        logger.warning(
            "TagService.extract_tags() called with an async llm_extract_fn — "
            "sync path can't await it. Use extract_tags_async() instead. "
            "Returning []."
        )
        return []
    # existing try/except below, with DEBUG → WARNING on the exception log
```

Also bump the exception log level in the `except Exception` block from
DEBUG to WARNING — silent empties should not be the default any more
than they should have been in desktop.

**Alternative:** delete the sync `extract_tags()` entirely. `extract_tags_async`
handles both sync and async `_llm_extract_fn` cleanly via
`inspect.iscoroutinefunction`; there's no reason to keep a sync entry
point once all callers use the async one. Tests that exercise the sync
method (`test_tag_service.py`) would need updating.

**Priority.** P3 / defense-in-depth. No ship block, no user impact
today. Do it next time the file is touched for any reason; not worth
cutting a release for on its own.

**Coordination with desktop.** Desktop ships the full fix in v0.3.2
(caller-side `await extract_tags_async` in `store_memory_bank` +
service-side `iscoroutinefunction` guard). Core only needs the
service-side guard since its callers are already gone.

---

## Section 8 — `configure_opencode` silently clobbers user's opencode.json on JSON parse failure

**File:** `roampal/cli.py:893-974` (`configure_opencode`)

**Discovered during:** investigation of GitHub issue #6 (Marcus Young,
re-opened 2026-04-22). Original v0.5.1 report included a third symptom
that couldn't be reproduced at the time: *"after `roampal init --force`,
none of my MCP servers were present when I started the CLI."* Marcus's
follow-up confirmed the file is **not a symlink** and he lost the
before/after hashes because he restored from backup. Code audit now
identifies the clobber path.

**Merge logic is correct when it runs.** Lines 963–973 only mutate
`config["mcp"]["roampal-core"]`; other providers, other MCP entries,
`model`, and every other top-level key are preserved:

```python
config["mcp"]["roampal-core"] = roampal_mcp_config
config_file.write_text(json.dumps(config, indent=2))
```

**Root cause — silent JSON-parse-failure fallback (`cli.py:937-961`).**

```python
config = {}                                                     # default
mcp_needs_write = True

if config_file.exists():
    try:
        config = json.loads(config_file.read_text())            # can raise
        ...
    except Exception as e:
        logger.warning(f"Failed to parse existing opencode.json: {e}")
        # config STAYS {} — failure is silent, logged only
```

If the existing `opencode.json` fails to parse, the exception is caught,
a WARNING is logged (usually not surfaced to the user), and **`config`
remains the empty dict initialized on line 937**. Execution then falls
through to the write path at :963–973 with `config == {}`:

```python
if mcp_needs_write:
    if "mcp" not in config: config["mcp"] = {}
    config["mcp"]["roampal-core"] = roampal_mcp_config
    config_file.write_text(json.dumps(config, indent=2))
```

The file is overwritten with **only** `{"mcp": {"roampal-core": {...}}}`.
Every provider, every other MCP, `model`, and any other top-level key
is destroyed. No backup. No error surfaced.

**Trigger scenarios (any of these causes the clobber):**

- Trailing comma (very common after hand-editing)
- Stray `//` or `/* */` comment (JSON doesn't allow comments, but several
  OpenCode examples in the wild show them; an editor inserting them
  during format is possible)
- BOM / UTF-16 encoding (Windows Notepad occasionally saves UTF-16 with
  BOM if the file was touched by another tool)
- Truncated file from a crashed editor / power loss during a save
- Any other manual JSON syntax error

This exactly matches Marcus's symptom: multi-MCP file existed before
init; after init, only `roampal-core` remained. His file was almost
certainly unparseable at the moment `init` ran.

**The `force` parameter is dead code.** Lines 893, 898 accept `force`
and document it as *"If True, overwrite existing config even if
different"* — but the parameter is **never referenced in the function
body**. `init` and `init --force` behave identically for the OpenCode
path. `configure_claude_code` (line 520) does use `force`; `configure_opencode`
does not. Either wire it to mean "overwrite even on parse failure" (an
explicit destructive opt-in) or remove it from the signature.

**Fix.**

1. On JSON parse failure, abort with a clear error. **Never write.**

    ```python
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            ...
        except json.JSONDecodeError as e:
            print(
                f"  {RED}[ERROR] Cannot parse existing opencode.json:{RESET}\n"
                f"    {config_file}\n"
                f"    {e}\n"
                f"    Fix the JSON or back up + delete to regenerate.\n"
            )
            return  # do not touch the file
    ```

2. Atomic write via temp + rename. Eliminates partial-write corruption
   and makes the "the exact moment init ran" window shrink to near-zero:

    ```python
    tmp = config_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(config_file)  # atomic on same filesystem
    ```

3. Create a timestamped backup before any write, so users have a recovery
   path if a future bug slips through:

    ```python
    if config_file.exists():
        backup = config_file.with_suffix(
            f".json.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(config_file, backup)
    ```

4. Remove `force` from `configure_opencode`'s signature (and from any
   call sites that pass it positionally) — it's dead code and its
   presence implies behavior that doesn't exist. If a future "yes,
   clobber even on parse error" flag is wanted, add it as a new named
   param then.

**Impact if unfixed.** Any user who hand-edits `opencode.json` —
which is the documented way to register custom providers, MCPs, model
aliases — is one `roampal init --force` away from losing their entire
OpenCode config if their edit introduced any JSON quirk. Footgun severity
is high because the user-global `opencode.json` often holds API keys
(DeepSeek, OpenAI, Anthropic) that the user may not have copies of, and
other MCP entries that represent hours of setup.

**Not in scope.**

- `roampal sidecar` CLI (and any other CLI command that mutates
  `opencode.json`) should get the same three-part treatment: abort on
  parse error, atomic write, timestamped backup. Audit pass is a separate
  item — tracked but not in v0.5.3. For now, Section 8 only hardens
  `configure_opencode`.
- `configure_claude_code` uses a different code path (lines 520-760);
  audit it separately before assuming the same bug exists there.

**Test list.** See the `TestConfigureOpencodeParseFailure` block in the
**Tests to add** section below.

---

## Section 9 — `roampal sidecar` CLI: project-local detection + scope-aware writes

**Files:** `roampal/cli.py` — `cmd_sidecar` dispatcher (line 3133), subcommand
handlers (`_cmd_sidecar_status` :3154, `_cmd_sidecar_setup` :3182,
`_cmd_sidecar_disable` :3204), and helpers (`_get_opencode_config_path` :2654,
`_check_sidecar_configured` :2664). Argument parser registration at
`:3572-on`.

**Problem statement.** Sidecar configuration lives in `opencode.json`'s
MCP environment block. OpenCode's merge rule is **project-local
overrides user-global** when a session is started from within a
project directory. The current `roampal sidecar` subcommands
(`status`, `setup`, `disable`) always read and write the **user-global**
file only (`~/.config/opencode/opencode.json`). Consequence: any
project-local `opencode.json` with sidecar env vars set silently
shadows the user-global value whenever the user runs `opencode` from
that directory — and `roampal sidecar` never warns, never reports the
shadow, and never offers to update the project-local file.

**Concrete footgun, observed 2026-04-22.** User ran `roampal sidecar
setup` and picked a new model — user-global updated correctly. User
then launched `opencode` from a project directory that had a stale
sidecar configuration in its own `opencode.json`. The project-local
stale value won; the "new" sidecar never actually ran. No error, no
warning. Took direct file inspection to diagnose.

**Scope for v0.5.3.** Fix the three symptoms:

1. `roampal sidecar status` reports only user-global; cannot see project-local shadows.
2. `roampal sidecar setup` writes only user-global; cannot target project-local.
3. `roampal sidecar disable` removes env vars only from user-global; project-local leftovers are invisible.

### Implementation

#### 9.1 New helper: `_find_project_opencode_config(start: Path | None = None) -> Path | None`

Walks up the directory tree from `start` (default `Path.cwd()`) looking
for an `opencode.json` that has a `mcp.roampal-core` block. Stops at
home directory or filesystem root. Returns the first match found, or
`None` if no project-local config exists in the cwd ancestry.

Placement: near `_get_opencode_config_path` at `:2654`.

```python
def _find_project_opencode_config(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a project-local opencode.json
    that has a roampal-core MCP block. Returns the first match or None.

    A project-local config is only "project-local" if it's DISTINCT from
    the user-global path — otherwise we'd double-report the same file.
    """
    start = start or Path.cwd()
    user_global = _get_opencode_config_path().resolve()
    home = Path.home().resolve()

    current = start.resolve()
    while True:
        candidate = current / "opencode.json"
        if (
            candidate.exists()
            and candidate.resolve() != user_global
            and candidate.is_file()
        ):
            try:
                cfg = json.loads(candidate.read_text())
                if "roampal-core" in cfg.get("mcp", {}):
                    return candidate
            except json.JSONDecodeError:
                # Surface the parse error to the caller — don't silently skip.
                # Downstream code will decide whether to abort or report.
                return candidate  # return it; caller re-parses and sees the error
        # stop conditions
        if current == home or current.parent == current:
            return None
        current = current.parent
```

#### 9.2 New shared helper: `_safe_write_opencode_config(path: Path, config: dict) -> None`

Atomic-write + timestamped backup. Reused by Section 8
(`configure_opencode`), all three sidecar subcommands, and any future
code that mutates `opencode.json`. Single code path, single test
surface.

```python
def _safe_write_opencode_config(path: Path, config: dict) -> None:
    """Write `config` to `path` atomically, with a timestamped backup
    of the pre-write contents if the file already exists.

    Must NEVER be called with a `config` that was synthesized from an
    empty dict after a parse failure — callers are responsible for
    aborting on parse errors BEFORE calling this. See Section 8.
    """
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(f".json.bak-{timestamp}")
        shutil.copy2(path, backup)

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(path)  # atomic on same filesystem
```

Section 8's `configure_opencode` rewrite should call this helper
instead of inlining the same logic. Update Section 8's code block in
the final implementation to reference `_safe_write_opencode_config`.

#### 9.3 `_cmd_sidecar_status` — report both locations + effective resolution

Replace current behavior (user-global only) with:

```python
def _cmd_sidecar_status(args):
    """Show current sidecar configuration at all scopes."""
    user_path = _get_opencode_config_path()
    project_path = _find_project_opencode_config()

    def _read_sidecar(path: Path | None) -> dict | None:
        if not path or not path.exists():
            return None
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return {"_parse_error": str(e)}
        env = cfg.get("mcp", {}).get("roampal-core", {}).get("environment", {})
        return {
            "url": env.get("ROAMPAL_SIDECAR_URL", ""),
            "model": env.get("ROAMPAL_SIDECAR_MODEL", ""),
            "has_key": bool(env.get("ROAMPAL_SIDECAR_KEY")),
        }

    user_sc = _read_sidecar(user_path)
    project_sc = _read_sidecar(project_path)

    print(f"{BOLD}Sidecar scoring configuration:{RESET}")

    # User-global block
    if user_sc is None:
        print(f"  User-global: {YELLOW}not found ({user_path}){RESET}")
    elif "_parse_error" in user_sc:
        print(f"  User-global: {RED}JSON parse error — {user_sc['_parse_error']}{RESET}")
        print(f"               {user_path}")
    elif user_sc["url"]:
        print(f"  User-global: {GREEN}{user_sc['model'] or '(no model)'} @ {user_sc['url']}{RESET}")
        print(f"               {user_path}")
    else:
        print(f"  User-global: {YELLOW}no sidecar configured{RESET}  ({user_path})")

    # Project-local block
    if project_sc is None:
        print(f"  Project-local: {GREEN}none found in cwd ancestry{RESET}")
    elif "_parse_error" in project_sc:
        print(f"  Project-local: {RED}JSON parse error — {project_sc['_parse_error']}{RESET}")
        print(f"                 {project_path}")
    elif project_sc["url"]:
        label = "OVERRIDES user-global here" if user_sc and user_sc.get("url") else "sets sidecar here"
        print(f"  Project-local: {YELLOW}{project_sc['model'] or '(no model)'} @ {project_sc['url']}{RESET}  ⚠ {label}")
        print(f"                 {project_path}")
    else:
        print(f"  Project-local: {GREEN}found, no sidecar override{RESET}  ({project_path})")

    # Effective resolution summary
    effective = project_sc if project_sc and project_sc.get("url") else user_sc
    if effective and effective.get("url"):
        print()
        print(f"  {BOLD}Effective in cwd:{RESET} {effective['model']} @ {effective['url']}")
```

No argument changes. `status` remains read-only. Works when no
project-local config exists (the common case) — "Project-local: none
found" is the normal path.

#### 9.4 `_cmd_sidecar_setup` — scope-aware prompt + non-interactive `--scope`

Add `--scope` to the `sidecar setup` argparse subparser
(`choices=["user", "project", "both"]`, no default — see interactive
fallback below):

```python
# in cmd_sidecar argument registration (~line 3572+)
setup_parser.add_argument(
    "--scope",
    choices=["user", "project", "both"],
    default=None,
    help="Where to write the new sidecar config. Required in non-interactive mode.",
)
```

New `_cmd_sidecar_setup` logic:

```python
def _cmd_sidecar_setup(args):
    """Configure sidecar scorer with scope-aware write.

    If a project-local opencode.json with sidecar vars exists in cwd
    ancestry, prompt the user for scope (or honor --scope). Otherwise
    write user-global only.
    """
    user_path = _get_opencode_config_path()
    project_path = _find_project_opencode_config()

    if not user_path or not user_path.exists():
        print(f"{RED}No sidecar config found.{RESET}")
        print(f"  Run {BLUE}roampal init --opencode{RESET} to create the config,")
        print(f"  then run {BLUE}roampal sidecar setup{RESET} again.")
        return

    # Determine if project-local has existing sidecar vars that shadow user-global
    project_has_shadow = False
    if project_path and project_path.exists():
        try:
            cfg = json.loads(project_path.read_text())
            env = cfg.get("mcp", {}).get("roampal-core", {}).get("environment", {})
            if env.get("ROAMPAL_SIDECAR_URL") or env.get("ROAMPAL_SIDECAR_MODEL"):
                project_has_shadow = True
        except json.JSONDecodeError as e:
            print(f"{RED}Project-local opencode.json has a JSON parse error:{RESET}")
            print(f"  {project_path}")
            print(f"  {e}")
            print(f"  Fix the file or delete it to proceed.")
            return

    # Pick scope
    if args.scope:
        scope = args.scope
    elif project_path and project_has_shadow:
        if not sys.stdin.isatty():
            print(f"{RED}Project-local opencode.json has sidecar settings that shadow user-global:{RESET}")
            print(f"  {project_path}")
            print(f"Pass --scope {{user|project|both}} to resolve (non-interactive).")
            return
        # Interactive: show status, prompt
        _cmd_sidecar_status(args)
        print()
        choice = input(
            f"Write new sidecar settings to: [u]ser-global / [p]roject-local / [b]oth / [s]kip? "
        ).strip().lower()
        if choice in ("u", "user", ""):   # default = user on bare enter
            scope = "user"
        elif choice in ("p", "project"):
            scope = "project"
        elif choice in ("b", "both"):
            scope = "both"
        else:
            print(f"{YELLOW}Cancelled.{RESET}")
            return
    else:
        scope = "user"

    # Run the existing model picker and write to the chosen scope(s)
    targets: list[Path] = []
    if scope in ("user", "both"):
        targets.append(user_path)
    if scope in ("project", "both"):
        if not project_path:
            print(f"{YELLOW}Skipping project scope — no project-local opencode.json found.{RESET}")
        else:
            targets.append(project_path)

    if not targets:
        print(f"{YELLOW}Nothing to write.{RESET}")
        return

    # Load the primary config (user-global) for the picker UI
    config = json.loads(user_path.read_text())

    if "roampal-core" not in config.get("mcp", {}):
        print(f"{RED}roampal-core not configured yet.{RESET}")
        print(f"  Run {BLUE}roampal init --opencode{RESET} first.")
        return

    # Delegate to the existing picker, which mutates `config` in-place
    # and returns the new env vars to apply across all target scopes.
    chosen_env = _sidecar_model_picker(config, user_path, defer_write=True)

    if chosen_env is None:
        return  # picker cancelled

    # Apply the chosen env vars to every target and write with parse-guard
    for target in targets:
        _apply_sidecar_env_and_write(target, chosen_env)

    if scope == "both":
        print(f"{GREEN}Sidecar updated in user-global AND project-local.{RESET}")
    elif scope == "project":
        print(f"{GREEN}Sidecar updated in project-local only — user-global unchanged.{RESET}")
    else:
        print(f"{GREEN}Sidecar updated in user-global only.{RESET}")
    print(f"{YELLOW}Restart OpenCode for changes to take effect.{RESET}")
```

New helper `_apply_sidecar_env_and_write(path, env_updates)`:

```python
def _apply_sidecar_env_and_write(path: Path, env_updates: dict[str, str]) -> None:
    """Apply sidecar env var updates to opencode.json at `path`,
    preserving all other config and writing atomically with backup.

    Aborts (prints error, does not write) if `path` has a JSON parse error.
    """
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"{RED}[ERROR] Cannot parse {path}:{RESET} {e}")
        print(f"  Fix the JSON or back up + delete to regenerate.")
        return

    mcp = config.setdefault("mcp", {})
    roampal = mcp.setdefault("roampal-core", {})
    env = roampal.setdefault("environment", {})

    # Clear any existing sidecar keys, then apply new ones
    for k in ("ROAMPAL_SIDECAR_FALLBACK", "ROAMPAL_SIDECAR_URL",
              "ROAMPAL_SIDECAR_KEY", "ROAMPAL_SIDECAR_MODEL"):
        env.pop(k, None)
    env.update(env_updates)

    _safe_write_opencode_config(path, config)
```

This helper is what `_cmd_sidecar_disable` also calls (with
`env_updates = {}` after clearing). Replaces the current inline
`config_path.write_text(json.dumps(config, indent=2))` at :3227.

**Existing `_sidecar_model_picker`:** gets an optional `defer_write=True`
param. When True, it returns the computed `chosen_env` dict instead of
writing the config itself — lets the caller apply to multiple scopes.
When False (default), existing behavior preserved.

#### 9.5 `_cmd_sidecar_disable` — scope-aware + parse-guard

Mirror `_cmd_sidecar_setup`'s scope logic. Same `--scope` flag on the
subparser, same interactive fallback when project-local exists with
sidecar vars. On execute, call `_apply_sidecar_env_and_write(path, {})`
against each target path (empty updates = just clear the keys).

Replace current inline write at :3227 with the helper.

### Files to touch (Section 9 only)

- `roampal/cli.py` —
  - Add `_find_project_opencode_config` helper
  - Add `_safe_write_opencode_config` helper (shared with Section 8)
  - Add `_apply_sidecar_env_and_write` helper
  - Rewrite `_cmd_sidecar_status`, `_cmd_sidecar_setup`, `_cmd_sidecar_disable`
  - Modify `_sidecar_model_picker` to accept `defer_write=True`
  - Add `--scope` to the `setup` and `disable` subparsers

### Tests to add (Section 9 only)

New class `TestSidecarScopeAwareness` in `test_cli.py`:

- `test_find_project_config_returns_none_when_none_in_ancestry` —
  cwd has no project `opencode.json` → helper returns `None`.
- `test_find_project_config_skips_user_global` — cwd equals home
  (so `~/.config/opencode/opencode.json` is the only candidate) →
  returns `None`, not the user-global path.
- `test_find_project_config_walks_up_to_first_match` — nested
  directories: only the grandparent has `opencode.json` → returns that
  path.
- `test_status_reports_both_scopes` — user-global and project-local
  both exist with different sidecar vars → stdout contains both blocks
  and the effective-resolution line.
- `test_status_flags_project_shadow` — user-global and project-local
  both have sidecar URLs set → stdout contains "OVERRIDES" for the
  project-local block.
- `test_setup_writes_user_only_when_no_project_found` — no
  project-local in cwd ancestry → only user-global is modified.
- `test_setup_scope_flag_user_writes_only_user` — `--scope user` →
  user-global modified, project-local byte-for-byte unchanged.
- `test_setup_scope_flag_project_writes_only_project` — `--scope project`
  → project-local modified, user-global byte-for-byte unchanged.
- `test_setup_scope_flag_both_writes_both` — `--scope both` → both
  modified identically.
- `test_setup_non_interactive_no_scope_flag_errors_when_shadow_present` —
  shadow exists and stdin is not a TTY and no `--scope` → exits with
  error, no writes.
- `test_setup_parse_error_on_project_aborts_with_message` — project-local
  has invalid JSON, run `setup --scope both` → prints parse error,
  neither file is modified.
- `test_disable_scope_flag_both_clears_both` — both files have sidecar
  env vars → `--scope both` clears from both.
- `test_disable_creates_backup_before_clearing` — disable → a
  `.bak-<timestamp>` file appears next to the written config.
- `test_safe_write_is_atomic` — monkeypatch `Path.replace` to raise →
  original file byte-for-byte unchanged; `.tmp` file may exist
  (cleanup is separate concern).

### Impact if unfixed

Users who run `roampal sidecar setup` from inside a project directory
with a stale project-local sidecar config will continue to experience
the "my changes don't take effect" footgun. The symptom is silent — no
error, no warning, just the wrong model scoring memories. High
confusion cost, very low detection rate because users assume the CLI
wrote where it should have.

### Coordination

- Section 8's `configure_opencode` rewrite and Section 9's sidecar
  commands **share `_safe_write_opencode_config`**. Land the helper
  first; both sections depend on it. Code review should verify no
  remaining direct `.write_text(json.dumps(...))` against `opencode.json`
  anywhere in `cli.py` — all writes route through the helper.
- `roampal init --opencode` (Section 8 surface) and `roampal sidecar
  setup / disable` (Section 9 surface) are the only CLI surfaces that
  write `opencode.json` today. The silent-clobber audit (tracked in
  Section 10 below) verifies this claim across the rest of the
  codebase.

---

## Section 10 — State-file mutation hardening (silent-clobber audit)

**Discovered during.** Silent-clobber audit run 2026-04-22 across
`roampal-core` and `roampal-desktop` (read-only, production code only).
Scope: find every write to a user-state file that lacks (1) silent
parse-failure protection, (2) atomic write via temp + rename, or (3)
backup before overwrite.

**Audit results — core (this release).** One non-atomic-write site
outside the `opencode.json` surfaces already covered by Sections 8
and 9. No silent-parse-failure sites outside Section 8's
`configure_opencode`. No missing-backup sites (non-atomic write is the
more urgent defect).

| File | Line | File mutated | What's lost on crash | Severity |
|---|---|---|---|---|
| `roampal/profile_manager.py` | 237-238 | `profiles.json` (profile registry) | All registered named profiles; user must re-register every v0.5.1 profile | HIGH |

The site uses this pattern:

```python
with open(path, "w") as f:
    json.dump(data, f, ...)
```

If the process is killed (crash, Ctrl-C, power loss, OOM) between
`open(...)` truncating the file and `json.dump` completing, the file
is left partially written and corrupt. Next read raises; silent
fallback logic elsewhere may then overwrite with defaults.

**Existing helper to reuse.** `roampal/hooks/session_manager.py:585-611`
(`_save_completion_state`) already implements the correct pattern:

```python
fd, tmp_path = tempfile.mkstemp(
    dir=str(self._state_file.parent), suffix=".tmp"
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, str(self._state_file))  # atomic on Linux and Windows
except Exception:
    os.unlink(tmp_path)
    raise
```

### Fix

Extract `_save_completion_state`'s write logic into a module-level
utility, then call it from all three sites (the two audit findings plus
session_manager itself).

**New file.** `roampal/utils/atomic_json.py`:

```python
"""Atomic JSON write with temp-file + rename dance. Crash-safe
for any filesystem that supports os.replace() atomicity (Linux and
Windows NTFS/ReFS both qualify).
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes to a sibling .tmp file first, then os.replace()s into place.
    If any exception is raised during the write, the temp file is
    removed and the original `path` is left untouched.

    Args:
        path: destination path
        data: JSON-serializable object
        indent: passed through to json.dump (default 2 for readability)
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)  # atomic on POSIX and NTFS
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
```

### Call-site changes

**`roampal/profile_manager.py:237-238`**

```python
# Before
with open(self._registry_path, "w", encoding="utf-8") as f:
    json.dump(registry_data, f, indent=2)

# After
from roampal.utils.atomic_json import write_json_atomic
write_json_atomic(self._registry_path, registry_data)
```

**`roampal/hooks/session_manager.py:585-611`** (housekeeping — not a
bug fix, but avoids two copies of the same pattern):

```python
# Replace the inline tempfile.mkstemp + os.replace block with:
from roampal.utils.atomic_json import write_json_atomic
write_json_atomic(self._state_file, state, indent=None)
```

`session_manager` passes `indent=None` because completion-state is
machine-written and machine-read; `profile_manager` accepts the default
`indent=2` for human-readable config files.

### Sections 8 & 9 coordination

Sections 8 and 9 each introduced `_safe_write_opencode_config` for
atomic + backup writes specifically of `opencode.json`. That helper
has a backup-before-write step that this site doesn't need
(`profiles.json` doesn't carry hand-authored content that users would
miss on reset — it's rebuilt by the app's registration / detection
flows). Two related helpers is correct:

- `write_json_atomic` (Section 10) — atomic write, no backup. For
  machine-managed state.
- `_safe_write_opencode_config` (Sections 8 & 9) — atomic write +
  timestamped backup. For human-authored config.

### Tests to add

New file `roampal/backend/modules/memory/tests/unit/test_atomic_json.py`:

- `test_write_json_atomic_creates_file_on_fresh_path` — path doesn't
  exist → after call, file exists and round-trips through `json.load`.
- `test_write_json_atomic_replaces_existing_file` — path exists with
  old content → after call, file contains new content.
- `test_write_json_atomic_leaves_no_tmp_files_on_success` — after a
  successful call, no `*.tmp` sibling remains in the parent dir.
- `test_write_json_atomic_preserves_original_on_exception` —
  monkeypatch `os.replace` to raise `OSError` → original file
  byte-for-byte unchanged; no `*.tmp` file left behind (unlink path in
  the `except` runs).
- `test_write_json_atomic_creates_parent_dirs` — path's parent doesn't
  exist → after call, parents created and file written.
- `test_write_json_atomic_indent_none_produces_compact` — pass
  `indent=None`; output has no newlines or extra whitespace.

Plus one integration test per consumer (update existing tests, don't
create new classes):

- `test_profile_manager.py` — assert the write goes through a tmp-file
  intermediate by monkeypatching `os.replace` and verifying it's called
  exactly once with a `.tmp`-suffix source.
- `test_session_manager.py` — existing `_save_completion_state` tests
  must continue to pass after the refactor to
  `write_json_atomic(indent=None)`.

### Files to touch

- `roampal/utils/atomic_json.py` — new file, one public function
- `roampal/profile_manager.py` — replace direct write at 237-238
- `roampal/hooks/session_manager.py` — refactor `_save_completion_state`
  body (585-611) to call the shared helper
- `roampal/backend/modules/memory/tests/unit/test_atomic_json.py` — new
  test module (6 tests)
- `roampal/backend/modules/memory/tests/unit/test_profile_manager.py` —
  one additional tmp-file monkeypatch test

### Impact if unfixed

A crash mid-save of `profiles.json` — which every `roampal profile
create / delete / register / rename` command writes — corrupts the
profile registry. Next `roampal start` that depends on profile
resolution fails, and the user's recourse is to hand-reconstruct the
JSON.

### Not in scope

- Three additional non-atomic-write sites exist in `roampal-desktop`
  (not this repo) — `feature_flags.py:126-132`, `model_contexts.py:78-79
  + 103-104`, `model_limits.py:500-501`. Those are desktop-side state
  files and are tracked independently on desktop's release stream, not
  here.

---

## Section 11 — Fix tag extraction gap: summaries and facts stored without noun_tags

**Files:** `roampal/server/main.py` — `/record-outcome` endpoint (lines 1493-1605)
and `/api/hooks/stop` OpenCode branch (lines 905-936).

**Symptom before fix.** All exchange summaries and facts written through the
OpenCode plugin landed with empty `noun_tags` metadata. TagCascade retrieval
ran on empty tag lists for these memories, silently degrading to cosine + CE
only (no tag-prefilter). Increased retrieval latency and hurt tag-conditioned
recall for every memory stored since the plugin v0.4.8 change that stopped
sending `noun_tags` on the wire.

**Root cause.** Comments in both the plugin (`roampal.ts:760, 798, 863, 958`)
and server claimed "server extracts tags at store time" — but that extraction
was never wired up. Both store paths checked `if request.noun_tags:`, which
was always false because the plugin stopped sending the field. Tags were
dropped silently on every exchange.

**The fix.** When `request.noun_tags` is empty, the server now extracts tags
itself using the sidecar LLM (`extract_tags()` from `sidecar_service.py`) at
each store surface:

1. **Exchange summaries via `/record-outcome`** (Claude Code path and OpenCode
   summary-replace path). Server extracts tags from the summary text before
   upserting the existing exchange document or creating a new working memory.

2. **Exchange summaries via `/api/hooks/stop`** (OpenCode full-storage path).
   When the request's metadata flags it as `memory_type == "exchange_summary"`
   and `noun_tags` is empty, the server extracts tags from the assistant
   response before `store_working`. This closes the gap that the original
   Section 11 patch missed — the `/record-outcome` path alone didn't cover
   OpenCode's summary storage, which goes through `/api/hooks/stop`.

3. **Facts via `/record-outcome`.** Per-fact tag extraction when no
   exchange-level tags are provided. Each fact gets its own `extract_tags`
   call for more precise per-fact tagging than a single exchange-level
   extraction would produce.

```python
# v0.5.3: Exchange summary path — extract from content if plugin didn't provide
extracted_tags = None
if not request.noun_tags:
    try:
        from roampal.sidecar_service import extract_tags as sidecar_extract_tags
        extracted_tags = sidecar_extract_tags(request.exchange_summary)
    except Exception as tag_err:
        logger.warning(f"Failed to extract tags from summary (non-fatal): {tag_err}")

effective_summary_tags = request.noun_tags or extracted_tags

# v0.5.3: Facts path — per-fact extraction when no exchange-level tags
fact_tags = None
if not request.noun_tags:
    try:
        from roampal.sidecar_service import extract_tags as sidecar_extract_tags
        fact_tags = sidecar_extract_tags(fact_text)
    except Exception:
        pass  # silently skip tag extraction per-fact

await _memory.store_working(
    ...
    noun_tags=request.noun_tags or fact_tags,
)
```

**Tag extraction is non-critical.** If the sidecar LLM fails to extract tags
(or returns empty), memories are still stored — just without tags. This matches
the existing pattern where tag extraction failures don't block memory storage.

**Files touched:** `roampal/server/main.py` (record-outcome endpoint, lines 1493-1572)
- No silent-parse-failure or missing-backup bugs were found outside
  Section 8's `configure_opencode`. Audit is clean on those two
  red-flag categories for this release.

---

## Section 12 — Explicit sidecar configuration: no silent defaults

**Files:** `roampal/sidecar_service.py` (default priority),
`roampal/cli.py` (picker options, sidecar disable message,
`_apply_sidecar_env_and_write` sidecar_keys list).

**Symptom before fix.** When a user had no `ROAMPAL_SIDECAR_URL` and no
`ROAMPAL_SIDECAR_PRIORITY` configured, the scoring cascade defaulted to
`["zen", "ollama", "lmstudio"]`. On every scored exchange, the sidecar
would try Zen first; if Zen failed (rate-limit, network), it would
silently probe `localhost:11434` (Ollama) and `localhost:1234` (LM Studio)
and send the full exchange text to whichever local service happened to
be running. Users who had an unrelated local LLM running had their
Roampal exchanges silently processed through it without opting in.

Separately, even users who reached Zen successfully never explicitly
consented to sending data to `opencode.ai/zen`. The init picker's
"Use free community models" option wrote no config and relied on the
silent cascade default, so users picking it didn't know where their
exchange data was being sent.

Compounding both: `roampal sidecar disable` printed *"Reverted to free
community models"* — implying benign cloud fallback — when the actual
behavior included the localhost probing and silent cloud dispatch above.

**Design change.** No default cascade. If the user has not explicitly
chosen a backend (via `roampal sidecar setup`, custom endpoint env vars,
or `ROAMPAL_SIDECAR_PRIORITY`), scoring is disabled. Retrieval from
existing memories still works. Summaries, fact extraction, noun-tag
extraction, and per-memory outcome scoring require an explicit opt-in.

**Fixes.**

1. **Default priority is empty.** `sidecar_service.py:536-544` — with no
   `ROAMPAL_SIDECAR_URL` and no `ROAMPAL_SIDECAR_PRIORITY`,
   `priority_order = []`. No backend is tried. Exchange text never leaves
   the user's machine automatically. To enable scoring, the user picks a
   model via `roampal sidecar setup` (which writes
   `ROAMPAL_SIDECAR_URL/MODEL`) or sets `ROAMPAL_SIDECAR_PRIORITY`
   explicitly (e.g. `zen` or `zen,ollama,lmstudio`).

2. **Init picker options are explicit and consequences are spelled out.**
   `cli.py` — `_sidecar_model_picker` in both the no-models-detected
   branch and the models-detected branch:

   - Detected local models are listed as first-class choices (unchanged).
   - "Configure custom API endpoint" (unchanged).
   - "Use free Zen cloud models (rate-limited, may be flaky — data sent to
     opencode.ai)" — picking this now writes
     `ROAMPAL_SIDECAR_PRIORITY=zen` explicitly, so the user's opt-in is
     recorded in `opencode.json` rather than being implicit.
   - "Skip — no scoring, no summaries, no fact extraction (retrieval
     still works)" — replaces the ambiguous "Cancel" option. Picking
     this leaves all sidecar env vars unset; scoring is off until the
     user runs `roampal sidecar setup`.

   Invalid or empty input no longer defaults to "free community models"
   — it prints a Cancel message and returns without writing.

3. **`roampal sidecar disable` is accurate.** `cli.py:3500+` — replaces
   *"Reverted to free community models"* with a plain statement:
   scoring/summaries/fact extraction are disabled, retrieval still
   works, run `roampal sidecar setup` to re-enable.

4. **Disable clears `ROAMPAL_SIDECAR_PRIORITY` too.** `cli.py:2818-2824`
   — `_apply_sidecar_env_and_write`'s `sidecar_keys` list now includes
   `ROAMPAL_SIDECAR_PRIORITY` so a stale Zen-opt-in doesn't persist
   across a disable/re-enable cycle.

**Impact if unfixed.** Exchange text leaves the user's control without
an explicit opt-in, in two forms: silent localhost probing of any LLM
service they have running for unrelated work, and silent cloud dispatch
to `opencode.ai/zen`. Users reasonably expect a memory tool to be
local-first by default.

**Not in scope for v0.5.3.**

- A terminal-level alert when the scoring banner reads "failed N times."
  Today the banner is injected into the OpenCode system prompt and the
  assistant relays it to the user. A direct terminal alert would
  require plugin-side changes.
- An interactive upgrade nudge for existing users whose configs still
  carry stale `ROAMPAL_SIDECAR_PRIORITY` values set before this release.
  Users who previously picked "free community models" will keep the
  Zen opt-in that was recorded post-upgrade; users on older configs

---

## Section 13 — `roampal sidecar setup`: parse-error message parity with `configure_opencode`

**Files:** `roampal/cli.py:3577-3581` (`_cmd_sidecar_setup`).

**Problem.** When a project-local `opencode.json` has invalid JSON,
`roampal sidecar setup --scope project` printed:

```
Cannot read <path>
```

This is the same message used for I/O errors (file not found, permission denied).
It gives the user no indication that the problem is a **JSON parse error** vs.
a missing file — and `FileNotFoundError` in the except clause was dead code
because line 3572 already checks `.exists()` before reaching this block.

By contrast, `configure_opencode` (Section 8) prints:

```
[ERROR] Cannot parse <path>: <specific error>
    Fix the JSON or back up + delete to regenerate.
```

**Fix.** Replaced the vague "Cannot read" message with a parse-specific
error that includes the JSON parser's detail, matching `configure_opencode`'s
format exactly:

```python
# Before
except (json.JSONDecodeError, FileNotFoundError):
    print(f"{RED}Cannot read {primary}{RESET}")
    return

# After
except json.JSONDecodeError as e:
    print(
        f"  {RED}[ERROR] Cannot parse {primary}:{RESET}\n"
        f"    {e}\n"
        f"    Fix the JSON or back up + delete to regenerate.\n"
    )
    return
```

Also removed `FileNotFoundError` from the except clause — it was unreachable
since `.exists()` is checked at line 3572.

**Why this matters.** Users with corrupted project-local configs (e.g.,
from an interrupted write, editor crash, or merge conflict) need to know
the file needs repair vs. regeneration. A vague "Cannot read" message
leads them to run `roampal init --opencode` which would overwrite their
existing config — the opposite of what they want.

**Not in scope.** The other sidecar commands (`status`, `disable`) already
have proper parse-error handling via `_apply_sidecar_env_and_write` and
`_read_sidecar()` inside `_cmd_sidecar_status`. No changes needed there.

---

## Section 14 — Audit: no other CLI write-to-opencode.json paths missing parse guards

**Files:** `roampal/cli.py` (all functions that read/write `opencode.json`).

**Audit scope.** Every code path in `cli.py` that reads or writes
`opencode.json` was checked for two properties:

1. **Abort on JSON parse error** — does the code print a clear message and
   return without modifying the file?
2. **Atomic write + backup** — if it modifies the file, does it use
   `_safe_write_opencode_config()` (not direct `path.write_text`)?

**Findings.** All remaining write paths go through `_apply_sidecar_env_and_write`,
which has both properties (parse-error abort at line 2837-2840, atomic write
at line 2869-2870). The only gap was the error message in `sidecar setup`
(Section 13 above), now fixed.

**Conclusion.** No other CLI commands mutating `opencode.json` have missing
parse guards or non-atomic writes as of v0.5.3. Future additions should use
`_safe_write_opencode_config()` and follow the abort-on-parse-error pattern.

---

## Section 15 — `roampal sidecar status --scope`: parser argument was missing

**File:** `cli.py:4168-4178` (argparse setup for `sidecar status`).

**Problem.** The `_cmd_sidecar_status()` handler reads scope via
`getattr(args, "scope", None)` and uses it to filter which scopes to show.
But the argparse parser for the `status` subcommand never defined `--scope`,
so passing it produced:

```
roampal: error: unrecognized arguments: --scope user
```

**Fix.** Added `--scope` argument to the status subparser, matching setup
and disable:

```python
sidecar_status_cmd = sidecar_sub.add_parser(
    "status", help="Show current sidecar configuration"
)
sidecar_status_cmd.add_argument(
    "--scope",
    choices=["user", "project", "both"],
    default=None,
    help="Config scope: user (global), project (local), or both. Default: auto-detect.",
)
```

`test` does not need `--scope` — it reads env vars directly via
`get_backend_info()` and doesn't touch opencode.json files.

**Not in scope.** No other subcommand parsers are missing arguments that
their handlers expect. This was the only one caught by the Section 14 audit.
