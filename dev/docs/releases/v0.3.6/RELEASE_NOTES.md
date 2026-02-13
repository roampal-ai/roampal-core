# v0.3.6 Release Notes

**Status:** All 18 changes implemented. Changes 9 + 12 use platform-split architecture (main LLM for Claude Code, sidecar for OpenCode). Dead sidecar code removed.
**Platforms:** All clients (Claude Code, Cursor, OpenCode)
**Theme:** Retrieval fairness & scoring accuracy

---

## Overview

v0.3.6 addresses a systemic retrieval imbalance where memory_bank entries overpower working/history memories through compounding boosts, and fixes promotion lifecycle issues that reset Wilson scores on collection transitions.

### Problem Statement

Memory retrieval has a fairness problem:

1. **memory_bank gets 5 compounding boosts** — importance×confidence, Wilson blending, entity boost, doc effectiveness, metadata multiplier. The boosts compound to give memory_bank a 40-65% distance advantage in the 2 open slots.
2. **Wilson resets on promotion** — A working memory that proves itself (high score, multiple uses) gets promoted to history with `success_count=0` and `uses=0`. It immediately becomes the weakest competitor.
3. **memory_bank 80/20 quality/Wilson blend** — Quality (importance×confidence) dominates even when outcome data says the memory isn't helpful. A memory with importance=0.9, confidence=0.9 but terrible Wilson (0.3) still gets a high blended score of 0.708.
4. **KG entity extraction is dead code** — `add_entities_from_text()` is defined but never called. The entity boost feature in retrieval references empty data.

### Impact on 4-Slot Context Injection

The 4-slot allocation (1 reserved working + 1 reserved history + 2 open) means the 2 open slots are where the imbalance matters. Reserved slots already handle recency — working and history are guaranteed representation. The open slots should be pure best-match competition, but memory_bank's compounding boosts dominate them.

---

## Changes

### Change 1: Memory_bank Wilson Blend — 80/20 → 50/50

**File:** `scoring_service.py` → `calculate_final_score()` (line ~285)
**Risk:** Low
**Impact:** High

**Current:**
```python
# 80% quality + 20% Wilson (cold start protection: quality only below 3 uses)
learned_score = 0.8 * quality + 0.2 * wilson_score
```

**Proposed:**
```python
# 50% quality + 50% Wilson (cold start protection: quality only below 3 uses)
learned_score = 0.5 * quality + 0.5 * wilson_score
```

**Effect:**
| Scenario | importance | confidence | Wilson | Current (80/20) | Proposed (50/50) |
|---|---|---|---|---|---|
| High quality, bad outcomes | 0.9 | 0.9 | 0.3 | 0.708 | 0.555 |
| High quality, good outcomes | 0.9 | 0.9 | 0.9 | 0.828 | 0.855 |
| Low quality, great outcomes | 0.5 | 0.5 | 0.95 | 0.39 | 0.6 |
| Default everything | 0.7 | 0.7 | 0.5 | 0.492 | 0.495 |

Cold start protection (uses < 3 → quality only) is unchanged.

---

### Change 2: Reduce Memory_bank Base Multiplier — 0.8 → 0.4

**File:** `search_service.py` → `_apply_collection_boost()` (line ~442)
**Risk:** Medium
**Impact:** High

**Current:**
```python
metadata_boost = 1.0 - blended_score * 0.8  # Perfect score → 0.2 distance (80% reduction)
```

**Proposed:**
```python
metadata_boost = 1.0 - blended_score * 0.4  # Perfect score → 0.6 distance (40% reduction)
```

**Effect:** A perfect memory_bank entry (blended_score=1.0) currently gets distance reduced to 20% of original. At 0.4 multiplier, it gets reduced to 60%. Still meaningful, but doesn't obliterate other collections.

| blended_score | Current distance multiplier | Proposed distance multiplier |
|---|---|---|
| 1.0 | 0.20 (80% boost) | 0.60 (40% boost) |
| 0.8 | 0.36 (64% boost) | 0.68 (32% boost) |
| 0.5 | 0.60 (40% boost) | 0.80 (20% boost) |
| 0.3 | 0.76 (24% boost) | 0.88 (12% boost) |

---

### Change 3: Carry Wilson Forward on Promotion

**File:** `promotion_service.py` → `_promote_working_to_history()` (lines 149-151)
**Risk:** Medium
**Impact:** High

**Current (v0.2.9):**
```python
# v0.2.9: Reset counters on history entry - memory must prove itself fresh
metadata["success_count"] = 0.0
metadata["uses"] = 0
```

**Proposed:**
```python
# v0.3.6: Carry Wilson forward — memory already proved itself via reserved slot scoring
# No reset. success_count and uses carry through from working → history → patterns.
metadata["promoted_to_history_at"] = datetime.now().isoformat()
# Remove the success_count and uses reset lines
```

**Patterns gate update** (`promotion_service.py` line ~111):

Current requires `success_count >= 5` (which was post-promotion since counters reset). Since we no longer reset, change to total lifetime successes:

```python
# v0.3.6: 5 total lifetime successes (no reset on promotion)
if score >= self.config.high_value_threshold and uses >= 3 and success_count >= 5:
```

This is functionally the same threshold but now counts the full lifecycle. A memory that earned 3 "worked" in working and 2 more in history qualifies (previously it needed 5 after reset).

**Rationale:** Reserved slots (1 working, 1 history) guarantee these memories get surfaced and scored. The Wilson data from working is real signal — resetting it throws away proven track records.

---

### Change 4: Fix Batch Promotion Inconsistency

**File:** `promotion_service.py` → `_do_batch_promotion()` (lines 346-358)
**Risk:** Low
**Impact:** Low-Medium

**Problem:** `_do_batch_promotion()` does NOT reset counters when promoting working → history, while the outcome-triggered `_promote_working_to_history()` does. Memories promoted via different code paths get different treatment.

**Fix:** Since Change 3 removes the reset from `_promote_working_to_history()`, both paths are now consistent — neither resets. Add `promoted_to_history_at` timestamp to batch promotion for parity:

```python
await self.collections["history"].upsert_vectors(
    ids=[new_id],
    vectors=[await self.embed_fn(text)],
    metadatas=[{
        **metadata,
        "promoted_from": "working",
        "promotion_time": datetime.now().isoformat(),
        "promotion_reason": "batch_promotion",
        "promoted_to_history_at": datetime.now().isoformat()  # v0.3.6: parity with outcome path
    }]
)
```

---

### Change 5: Wire KG Entity Extraction

**Files:**
- `unified_memory_system.py` → `store_working()` / `store_memory_bank()` / `store()` — call `kg_service.add_entities_from_text()` after storing
- `knowledge_graph_service.py` → `add_entities_from_text()` (line 1012) — already defined, just never called

**Current:** `add_entities_from_text()` exists but is dead code. The Content KG's `_doc_entities` map is always empty, so `_calculate_entity_boost()` always returns 1.0.

**Proposed:** After every memory storage call, extract entities:

```python
# After successful storage:
if self.kg_service:
    try:
        self.kg_service.add_entities_from_text(
            doc_id=doc_id,
            text=content,
            collection=collection,
            quality=metadata.get("importance", 0.7)
        )
    except Exception as e:
        logger.warning(f"Entity extraction failed for {doc_id}: {e}")
```

**Impact:** Enables the entity boost to actually function. Currently it's a no-op.

---

### Change 6: Extend Entity Boost to All Collections

**File:** `search_service.py` → `_apply_collection_boost()` (line ~443)

**Current:** Entity boost only applied inside the `memory_bank` branch.

**Proposed:** Move entity boost calculation to apply after all collection-specific boosts:

```python
def _apply_collection_boost(self, result: Dict, coll_name: str, query: str):
    # ... existing collection-specific boosts ...

    # v0.3.6: Entity boost for ALL collections (was memory_bank only)
    entity_boost = self._calculate_entity_boost(query, result.get("id", ""))
    if entity_boost > 1.0:
        result["distance"] = result.get("distance", 1.0) / entity_boost
```

**Depends on:** Change 5 (entity extraction must be wired first, otherwise entity data is empty for all collections)

---

### Change 7: Track All Outcomes in Problem-Solution KG

**File:** `outcome_service.py` → `record_outcome()` (line ~342)

**Current:** `_track_problem_solution()` only called for `"worked"` outcomes. `update_success_rate()` called for worked, failed, and partial.

```python
if outcome == "worked":
    self.kg_service.update_success_rate(doc_id, outcome)
    await self._track_problem_solution(doc_id, metadata, context)
elif outcome == "failed":
    self.kg_service.update_success_rate(doc_id, outcome)
```

**Proposed:**
```python
# v0.3.6: Track ALL outcomes in KG
self.kg_service.update_success_rate(doc_id, outcome)
if outcome in ("worked", "partial", "failed", "unknown"):
    await self._track_problem_solution(doc_id, metadata, context)
```

**Impact:** Builds richer problem-solution patterns. Failed solutions are as informative as successful ones — knowing what DIDN'T work prevents re-surfacing bad advice. Unknown patterns reveal which memories keep surfacing irrelevantly.

---

### Change 8: Fix Unknown Outcome Score — 0.0 → -0.05

**File:** `outcome_service.py` → `_calculate_score_update()` (line ~288)
**Risk:** Low
**Impact:** Medium

**Problem:** Unknown currently has `score_delta = 0.0` — the raw outcome score never changes when a memory surfaces without being used. A memory can accumulate unlimited unknown scores without any penalty to its raw score, meaning it never drifts toward demotion/deletion thresholds.

**Current:**
```python
elif outcome == "unknown":
    score_delta = 0.0      # raw score unchanged
    new_score = current_score
    uses += 1
    success_delta = 0.25   # Wilson
```

**Proposed:**
```python
elif outcome == "unknown":
    # v0.3.6: unknown = opposite of partial on raw score (-0.05 vs +0.05)
    # Surfaces without being used → slight raw score penalty
    # Wilson success_delta unchanged — this only affects the raw score path
    score_delta = -0.05
    new_score = max(0.0, current_score + score_delta)
    uses += 1
    success_delta = 0.25   # Wilson stays the same
```

**Effect:** Unknown is now the opposite of partial on the raw score axis:
- Partial: `score_delta = +0.05` (slight reward)
- Unknown: `score_delta = -0.05` (slight penalty)

A memory starting at score=0.7 that gets scored `unknown` 10 times drops to 0.2, approaching the demotion threshold. Wilson tracking (`success_delta = 0.25`) is unaffected.

| Score after N unknowns | Current | Proposed |
|---|---|---|
| Start (score=0.7) | 0.70 | 0.70 |
| After 5 unknowns | 0.70 | 0.45 |
| After 10 unknowns | 0.70 | 0.20 |
| After 14 unknowns | 0.70 | 0.00 |

This means a memory that keeps surfacing without being used will eventually hit deletion. But one "worked" (`score_delta = +0.2`) recovers 4 unknowns worth of damage.

---

### Change 9: Exchange Summarization + Scoring (Platform-Split Architecture)

**Files:** `mcp/server.py` (tool schema), `session_manager.py` (scoring prompt), `server/main.py` (handler), OpenCode plugin (`roampal.ts`)
**Risk:** Low (extends existing scoring flow)
**Impact:** High (token cost reduction + scoring coverage + kills broken sidecar)

**Problem:** Working/history memories store full conversation exchanges (avg 1,638 chars, ~467 tokens each). The first 300 chars is typically just the user's question — the useful answer/solution is buried deeper. This bloats context injection (4 slots × 467 tokens = 1,868 tokens/turn, 56K over 30 turns = 175% of a 32K context window — literally overflowing). Truncation destroys value. Raw storage wastes tokens.

The previous sidecar design (two-tier: Tier 1 direct API, Tier 2 `claude -p`) is **dead** — no API key means Tier 1 skips, and Tier 2 times out after 30s. Both tiers fail. No summaries get produced.

**New sidecar backend (`sidecar_service.py`):** Auto-detects best available backend with fallback chain:
0. `ROAMPAL_SIDECAR_URL` + `ROAMPAL_SIDECAR_MODEL` set → **custom endpoint** (any OpenAI-compatible API: Groq, Together, OpenRouter, etc.)
1. `ANTHROPIC_API_KEY` set → **direct Haiku API** (~3s, ~$0.001/call)
2. Zen free models → **Zen** (~5s, $0) — only available inside OpenCode environments
3. Ollama or LM Studio running locally → **local model** (~14s, $0, tested 100% success rate)
4. Nothing works → **`claude -p --model haiku`** fallback (~30-60s, uses existing auth)

**Configurable via env vars (both Python sidecar and OpenCode plugin):**
- `ROAMPAL_SIDECAR_URL` — base URL for any OpenAI-compatible endpoint (e.g., `http://localhost:1234/v1`)
- `ROAMPAL_SIDECAR_KEY` — API key (optional for local servers)
- `ROAMPAL_SIDECAR_MODEL` — model ID (e.g., `llama3.2:3b`, `mixtral-8x7b-32768`)
- `ROAMPAL_OLLAMA_URL` — override Ollama URL (default: `http://localhost:11434`)
- `ROAMPAL_LMSTUDIO_URL` — override LM Studio URL (default: `http://localhost:1234`)

The command auto-detects and uses the best option. Users run `roampal summarize` and it just works — no config needed. Custom endpoint takes priority over everything when configured.

**Solution:** Platform-split architecture — each platform uses what actually works.

**Scoring architecture:**

| | Claude Code | OpenCode |
|---|---|---|
| Per-memory scoring | Main LLM via `score_memories` (full schema) | Main LLM via `score_memories` (lean schema, no exchange fields) |
| Exchange summary | Main LLM via `score_memories` | Sidecar (free model on session.idle) |
| Exchange outcome | Main LLM via `score_memories` | Sidecar (free model on session.idle) |
| Exchange storage | JSONL only (lifecycle_only=True), summary in ChromaDB via score_memories | Stop hook stores full in ChromaDB → sidecar replaces with summary |
| Fallback if LLM skips | None needed (prompt forces it) | Sidecar covers everything |
| Extra infra | None | Sidecar call via Zen free models |

---

**Claude Code — Main LLM Does Everything:**

The main LLM already runs every turn. The scoring prompt already fires every turn. Just add two fields to `score_memories`:

```python
# score_memories tool schema (updated)
{
    "memory_scores": {"mem_1": "worked", "mem_2": "failed", ...},  # per-memory (existing)
    "exchange_summary": "~300 char summary of the previous exchange",  # NEW
    "exchange_outcome": "worked"  # NEW
}
```

The scoring prompt asks for all three in one call:

```
Score the previous exchange before responding.

1. Score each cached memory (worked/partial/unknown/failed)
2. Summarize the previous exchange in ~300 chars
3. Rate the exchange outcome (worked/failed/partial/unknown)

Call score_memories(
    memory_scores={...},
    exchange_summary="...",
    exchange_outcome="worked|failed|partial|unknown"
)
```

**What dies for Claude Code:**
- `_summarize_exchange_background()` — removed from `server/main.py`
- `_call_anthropic_direct()` — removed from `server/main.py`
- `_summarized_fingerprints` set — removed
- `_api_key_valid` flag — removed
- **Stop hook ChromaDB storage** — stop hook no longer stores exchanges in ChromaDB working memory
- Stop hook still reads transcripts and stores in session JSONL (needed for scoring prompt lifecycle)
- Uses `lifecycle_only=True` flag: JSONL exchange tracking for `get_previous_exchange()`, no ChromaDB
- No API key needed. No CLI subprocess. No timeout. No two-tier fallback. No background task.

**Server-side handler:**
When `score_memories` is called with `exchange_summary` + `exchange_outcome`:
1. Apply per-memory scores (existing behavior)
2. Store summary directly as NEW working memory with `memory_type: "exchange_summary"` metadata (no replacement — there's nothing to replace)
3. Record exchange outcome via `record_outcome` if outcome != "unknown"

**Flow:**
```
User sends message
  → UserPromptSubmit hook fires
    → Server injects scoring prompt (includes summary + outcome request)
  → Main LLM responds:
    → Calls score_memories({
        memory_scores: { ... },
        exchange_summary: "User asked about stop hook diagnostics. Added stderr logging...",
        exchange_outcome: "worked"
      })
  → Server:
    → Applies per-memory scores (existing)
    → Stores summary as NEW working memory (memory_type: "exchange_summary")
    → Records exchange outcome
  → Stop hook fires:
    → Reads transcript, sends exchange data with lifecycle_only=True
    → Server stores exchange in session JSONL (for scoring prompt generation)
    → Server skips ChromaDB storage (main LLM already stored summary above)
    → Server marks turn complete, checks scoring compliance
  → Done. No sidecar. No background task. No ChromaDB exchange storage in stop hook.
```

**Why the stop hook still reads transcripts:**
The stop hook sends exchange data to the server with `lifecycle_only=True`. The server stores it in session JSONL (NOT ChromaDB) so `get_previous_exchange()` can find it and generate scoring prompts on the next `UserPromptSubmit`. Without this, the scoring lifecycle breaks — no exchange in JSONL means no scoring prompt injected, which means `score_memories` never gets called. The stop hook also handles turn lifecycle state management (marking turns complete, checking scoring compliance).

**Cost:** Zero additional — the main LLM is already running and already responding to the scoring prompt. Adding ~50 tokens for summary + outcome to the response is negligible.

---

**OpenCode — Sidecar Stays:**

OpenCode keeps the sidecar because it actually works there (free models via Zen providers).

- Per-turn scoring via `session.idle` handler (fires after each agent response)
- Uses configured Zen free models (trinity-large-preview-free, glm-4.7-free, etc.) — truly free, zero cost
- Sidecar handles: exchange summary + exchange outcome
- Main LLM handles: per-memory scoring via `score_memories` (same tool, same prompt)
- Double-scoring prevention: checks both client-side flag AND server-side `/api/hooks/check-scored` endpoint

**Fallback logic (OpenCode only):**
- If main LLM called `score_memories` → use its per-memory scores, sidecar's exchange score is informational only
- If main LLM did NOT call `score_memories` → apply sidecar's exchange outcome uniformly to all cached memories from that turn

**Sidecar prompt (OpenCode only):**
```
You are part of a memory system for an AI assistant.

The user said: "{user message}"
The assistant responded: "{assistant response}"
The user then followed up with: "{followup message}"

1. Summarize the exchange in under 300 characters.
2. Based on the follow-up, was the response effective?

Return ONLY valid JSON:
{"summary": "...", "outcome": "worked|failed|partial|unknown"}
```

---

**Hook config (Claude Code):**

```json
// ~/.claude/settings.json (auto-configured by roampal init)
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "python -m roampal.hooks.user_prompt_submit_hook"
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python -m roampal.hooks.stop_hook"
      }]
    }]
  }
}
```

Stop hook handles turn lifecycle + JSONL exchange tracking (with `lifecycle_only=True` — no ChromaDB storage). UserPromptSubmit injects scoring prompt + context. The main LLM stores exchange summaries directly via `score_memories`.

---

**The `roampal score` CLI command** (kept for manual use and testing):

```bash
roampal score --transcript /path/to/transcript.jsonl
roampal score --last
roampal score --session-id <session-uuid>
```

CLI command uses `sidecar_service.py` with auto-detected backend (Haiku API > Ollama > `claude -p`). For manual/testing only — not part of the automated flow.

---

**Storage architecture: Summary is the only content**

**Claude Code:** The main LLM stores the ~300 char summary directly via `score_memories`. No full exchange is ever stored in memory. The full conversation exists in Claude Code's transcript logs if needed.

**OpenCode:** The stop hook stores the full exchange first, then the sidecar replaces it with a summary on `session.idle`. End state is the same — only the summary exists in memory.

- **Content** is the <300 char summary (what gets embedded and injected)
- A good summary captures the same semantic concepts the embedding needs ("auth token expiration fix in session_manager.py" matches queries about auth tokens just as well as the 1500-char raw exchange)
- Users can run `roampal summarize` to retroactively clean up any full-size exchanges that slipped through

**Token savings (verified against live data, Feb 11 2026):**

| Config | Per Slot | Per Turn (4 slots) | 30 Turns | % of 32K | % of 128K |
|---|---|---|---|---|---|
| Current (full content, avg 1638 chars) | 467 tok | 1,868 | 56,040 | **175%** (overflow!) | 43.8% |
| Summarized (<300 chars) | 102 tok | 408 | 12,240 | **38%** | 9.6% |

**78% token reduction.** Current setup literally overflows a 32K context window over 30 turns. After summarization it fits comfortably.

**Why summarization, not truncation:** Many memories have the answer starting AFTER char 300. Dumb truncation would cut the answer — the useful part. Summarization reads the full exchange and picks what matters.

---

### Change 10: Retroactive Memory Summarization (Three Paths)

**Files:** `server/main.py`, `sidecar_service.py`, `cli.py`, OpenCode plugin (`roampal.ts`)
**Risk:** Low (read-modify on existing memories, no schema changes)
**Impact:** High (fixes existing long memories for ALL user types)

**Problem:** Existing working/history memories store full exchanges (avg 1,638 chars). Even with Change 9, those old memories won't benefit until they're re-summarized. The CLI `roampal summarize` uses `sidecar_service.py`, but:
- Zen free models return HTTP 403 from CLI (only accessible inside OpenCode's proxy)
- `claude -p` fails with "nested session" error when run from inside Claude Code
- OpenCode-only users without Ollama or an API key have NO working backend

**Solution:** Three complementary paths — automatic server-side, automatic plugin-side, and manual CLI.

**Design principles:**
- **Sidecar-first:** Both platforms default to user's configured sidecar (Custom > Haiku > Ollama)
- **Zen is fallback only:** Plugin falls back to Zen if sidecar has no backend — never the default
- **Main model opt-in:** Users can set `ROAMPAL_SUMMARIZE_MODEL` to use their expensive model
- **One memory per exchange:** Lightweight, non-blocking, gradual cleanup

---

**New endpoint: `POST /api/memory/auto-summarize-one`**

Central endpoint called by BOTH the OpenCode plugin and the Claude Code background task:

1. Finds one candidate: search working+history+patterns, fetch 10, client-side filter `len(content) > 400` AND no `summarized_at`
2. Calls `summarize_only(content)` from `sidecar_service.py` (uses full backend chain: Custom > Haiku > Zen > Ollama > `claude -p`)
3. On success: updates memory (sets `summarized_at`, `original_length`)
4. Returns `{summarized: true, doc_id, old_len, new_len}` or `{summarized: false, reason: "no_candidates"|"backend_failed"}`
5. On `backend_failed`: includes `doc_id`, `collection`, `content` so the plugin can Zen-fallback

---

**Path 1: Claude Code Background Task (Automatic)**

Background task `_auto_summarize_one_memory()` spawned via `asyncio.create_task()` at end of stop_hook handler. Non-blocking — doesn't delay stop hook response. Follows existing pattern from `_deferred_action_kg_updates()`.

---

**Path 2: OpenCode Plugin Auto-Summarize (Automatic, Zen Fallback)**

`autoSummarizeOldMemory()` called in `session.idle` handler after exchange scoring:

1. Calls `POST /api/memory/auto-summarize-one`
2. If `{summarized: true}` → done
3. If `{summarized: false, reason: "backend_failed"}` → falls back to Zen directly from plugin (same model rotation as scoring: `ZEN_SCORING_MODELS`)
4. If `{summarized: false, reason: "no_candidates"}` → silently returns
5. On any error: logs warning, doesn't retry (next idle picks it up)

This ensures users who configured Custom/Haiku/Ollama → data stays on their chosen backend. Users with nothing → Zen free models (graceful fallback). 4s timeout per Zen attempt.

---

**Path 3: `roampal summarize` CLI (Manual, All Platforms)**

For users who want to clean up old memories in one shot.

```bash
roampal summarize              # Auto-detects best backend
roampal summarize --dry-run    # Preview without changes
roampal summarize --limit 20   # Batch processing
roampal summarize --max-chars 500
roampal summarize --collection working
```

**Improved error messaging when no backend is available:**

```
No summarization backend available.

  What was checked:
    ROAMPAL_SUMMARIZE_MODEL  not set  (opt-in to main model)
    ROAMPAL_SIDECAR_URL      not set  (custom OpenAI-compatible API)
    ANTHROPIC_API_KEY         not set  (Haiku direct)
    Zen free models           unavailable (CLI only works inside OpenCode)
    Ollama                    not running (http://localhost:11434)
    LM Studio                 not running (http://localhost:1234)
    claude CLI                not found

  Options (pick one):
    1. Install Ollama (recommended, free, local, ~14s/memory)
    2. Set ANTHROPIC_API_KEY (~$0.001/memory via Haiku)
    3. Set ROAMPAL_SUMMARIZE_MODEL (use your main model)
    4. Set ROAMPAL_SIDECAR_URL (any OpenAI-compatible endpoint)

  Note: OpenCode users don't need this — memories auto-summarize
  during normal use via the plugin (1 per exchange, zero config).
```

---

**Main model config: `ROAMPAL_SUMMARIZE_MODEL` env var**

When set (e.g. `ROAMPAL_SUMMARIZE_MODEL=claude-sonnet-4-5-20250929`), `summarize_only()` uses this model via the Anthropic API instead of the sidecar chain. Requires `ANTHROPIC_API_KEY`. Not the default — users must explicitly opt in.

**Backend availability by user type:**

| User Type | Backend | Works from CLI? | Speed |
|---|---|---|---|
| ROAMPAL_SUMMARIZE_MODEL set | Specified Anthropic model | Yes | ~3-10s/memory |
| Custom endpoint configured | Custom | Yes | Varies |
| Has ANTHROPIC_API_KEY | Haiku API | Yes | ~3s/memory |
| Ollama/LM Studio installed | Local model | Yes | ~14s/memory |
| Claude Code subscriber | `claude -p` | Yes (regular terminal only*) | ~30-60s/memory |
| OpenCode only (no Ollama) | None from CLI | **No** — use Path 2 | — |

*`claude -p` fails with "nested session" error when run from inside Claude Code. Must run from a separate terminal.

---

**All paths together:**
- **Claude Code:** Background task auto-summarizes 1 old memory per stop hook (sidecar-first)
- **OpenCode:** Plugin auto-summarizes 1 old memory per session.idle (sidecar-first, Zen fallback)
- **Impatient user:** Runs `roampal summarize` with Ollama/API key, done in minutes
- **New user with no old data:** No action needed — Change 9 handles new exchanges automatically

**Safeguards:**
- `--dry-run` mode shows count + sample summaries before touching anything
- Skips memory_bank (already concise facts), books (reference chunks), patterns (earned their length)
- Skips memories already under the threshold or with `summarized_at` metadata
- Enforces summary length (truncates to 400 chars) to prevent re-summarization loops
- Auto-summarize: 1 memory per idle/stop cycle, best-effort (never blocks)

**Expected behavior:**
- Path 1+2: Cleans up 1-2 old memories per session automatically
- Path 3: Bulk cleanup for impatient users
- Token savings: ~78% reduction in context injection per session
- CLI cost: ~$0.001 per memory via Haiku, $0 via Ollama/Zen
- `--dry-run` shows exactly what would be summarized before committing

---

### Change 11: Compaction Recovery — Recent Exchange Buffer

**Files:** `cli.py` (`roampal context` command), `sidecar_service.py`, `stop_hook.py`, Claude Code hook config, OpenCode plugin
**Risk:** Low (additive, read-only temporal query)
**Impact:** High (context recovery after compaction)

**Problem:** When Claude Code or OpenCode compacts the conversation (to fit context limits), the model loses track of what just happened. Current KNOWN CONTEXT uses semantic search — it finds *relevant* memories, not *recent* ones. After compaction, the model might surface a memory from 3 days ago instead of what was discussed 2 minutes ago.

**Additional problem (discovered in testing, now resolved):** The Stop hook previously captured compaction summaries as regular exchanges, creating recursive "Previous exchange: Previous exchange:..." garbage. This is now a non-issue for Claude Code since the stop hook uses `lifecycle_only=True` (JSONL tracking only, no ChromaDB storage). For OpenCode, the compaction artifact filter should be in the plugin's exchange capture logic.

**Solution:**
1. Inject the last 4 exchange summaries automatically via platform hooks on cold start and after compaction. No model tool calls. No relying on the model to follow instructions. The system handles it.
2. **Claude Code:** No artifact filter needed — stop hook uses `lifecycle_only=True` (JSONL only, no ChromaDB).
3. **OpenCode:** Plugin should filter compaction artifacts before sending to `/api/hooks/stop`.

**Depends on:** Change 9 (exchange summaries must be stored with `memory_type: "exchange_summary"` metadata tag — main LLM for Claude Code, sidecar for OpenCode)

**How it works:**

1. Change 9 creates exchange summaries (<300 chars) stored in working memory with `memory_type: "exchange_summary"` tag (Claude Code: via `score_memories` tool; OpenCode: via sidecar on `session.idle`)
2. Platform hooks detect cold start and compaction events automatically
3. Hook calls `roampal context --recent-exchanges` CLI command
4. CLI queries server for last 4 exchange summaries, sorted by `created_at` desc
5. Returns formatted RECENT EXCHANGES block to stdout
6. Platform injects this as context — model sees it immediately, no tool call needed

**Output format (injected by hook, not by model):**
```
RECENT EXCHANGES (last 4):
1. [2min ago] User asked about auth token expiry. Fix was TTL clamping in session_manager.py.
2. [8min ago] Discussed sidecar scoring for Claude Code. Decided on Haiku via Stop hook.
3. [15min ago] Dropped dual storage — just store summary, embed summary.
4. [22min ago] Verified token savings: 78% reduction, 175% overflow -> 38%.
```

**Platform mechanics:**

**Claude Code — fully automatic via SessionStart hook:**

Claude Code fires `SessionStart` with different `source` values:
- `source: "startup"` — new session (cold start)
- `source: "compact"` — after compaction
- `source: "resume"` — after `/resume`

```json
// ~/.claude/settings.json (auto-configured by roampal init)
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [{
          "type": "command",
          "command": "roampal context --recent-exchanges"
        }]
      },
      {
        "matcher": "startup",
        "hooks": [{
          "type": "command",
          "command": "roampal context --recent-exchanges"
        }]
      }
    ],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python -m roampal.hooks.stop_hook"
      }]
    }]
  }
}
```

Flow:
1. Compaction happens -> `SessionStart(compact)` fires automatically
2. Hook runs `roampal context --recent-exchanges`
3. CLI queries server for last 4 exchange summaries
4. Outputs formatted RECENT EXCHANGES to stdout
5. Claude Code injects this as context — model sees it immediately
6. Same for cold start via `SessionStart(startup)`

No `include_recent_exchanges` parameter on `get_context_insights`. No model cooperation. The hook system handles everything.

**OpenCode — two-layer automatic:**

OpenCode has two compaction-related plugin events:

1. **`experimental.session.compacting`** (fires BEFORE compaction) — can inject context INTO the compaction prompt so the model's own summary includes recent exchanges:

```typescript
"experimental.session.compacting": async (input, output) => {
  const recent = await fetchRecentExchanges()
  if (recent) {
    output.context.push(`<recent-exchanges>\n${recent}\n</recent-exchanges>`)
  }
}
```

2. **`session.compacted`** (fires AFTER compaction) — sets flag to inject RECENT EXCHANGES on the next turn's context injection:

```typescript
"session.compacted": async (event) => {
  includeRecentOnNextTurn = true  // next system.transform includes recent exchanges
}
```

OpenCode is actually stronger here — it can inject recent context INTO the compaction itself, so the summary naturally includes what happened recently. Claude Code's `PreCompact` hook can't do this (it can only write to stdout/stderr, not inject into the compaction prompt).

**Token cost:**
- 4 summaries x ~300 chars = ~1,200 chars = ~324 tokens
- Only injected on cold start + post-compaction (2-3 times per session)
- Normal turns: zero extra tokens
- 30-turn session with 1 compaction: 28 x 408 + 2 x 732 = **12,888 tokens** (vs 56K current)

**No changes to `get_context_insights` tool.** The recent exchanges are injected by the platform hooks, not by the MCP tool. The tool stays exactly as-is.

### Change 12: Rename `score_response` → `score_memories` + Add Summary/Outcome Fields

**Files:** `mcp/server.py` (tool registration + schema), `session_manager.py` (scoring hook prompt), OpenCode plugin (`roampal.ts` — tool call detection)
**Risk:** Low (tool rename + additive fields)
**Impact:** High (enables Change 9's platform-split architecture)

**Problem:** The tool is named `score_response` and says "Score the previous exchange." It needs to be renamed to reflect its actual purpose (per-memory scoring) and extended with `exchange_summary` + `exchange_outcome` fields to support the main-LLM-as-summarizer architecture (Change 9).

**Changes:**

1. **Rename tool:** `score_response` → `score_memories` in MCP server registration
2. **Add `exchange_summary` field** (optional, string, ~300 chars) — the main LLM's summary of the previous exchange. Server stores this as a working memory with `memory_type: "exchange_summary"` metadata.
3. **Add `exchange_outcome` field** (optional, enum: worked/failed/partial/unknown) — the main LLM's judgment of whether its previous response was effective. Server records this via `record_outcome`.
4. **Keep `memory_scores` field** (required, dict) — per-memory scoring, unchanged.
5. **Drop old `outcome` parameter** — replaced by `exchange_outcome` (clearer naming). Handler still accepts `outcome` silently for backward compat.

**Updated tool schema:**
```python
inputSchema = {
    "type": "object",
    "properties": {
        "memory_scores": {
            "type": "object",
            "description": "Score for each memory: doc_id -> outcome (worked/partial/unknown/failed)",
            "additionalProperties": {"type": "string", "enum": ["worked", "failed", "partial", "unknown"]}
        },
        "exchange_summary": {
            "type": "string",
            "description": "~300 char summary of the previous exchange"
        },
        "exchange_outcome": {
            "type": "string",
            "enum": ["worked", "failed", "partial", "unknown"],
            "description": "Was the previous response effective?"
        }
    },
    "required": ["memory_scores"]
}
```

6. **Update description opening:**

**Current:**
```
Score the previous exchange. ONLY use when the <roampal-score-required> hook prompt appears.
```

**Proposed (Claude Code):**
```
Score individual cached memories from your previous context. ONLY use when the <roampal-score-required> hook prompt appears.

Your job here is to:
1. Score each cached memory individually — was it helpful, misleading, or unused?
2. Summarize the previous exchange in ~300 chars
3. Rate the exchange outcome (worked/failed/partial/unknown)
```

**Proposed (OpenCode — platform-aware via `ROAMPAL_PLATFORM=opencode` env var):**

The MCP server detects `ROAMPAL_PLATFORM=opencode` (set in OpenCode's MCP environment config) and returns a **different tool description AND schema** — no mention of exchange summary or outcome, no `exchange_summary`/`exchange_outcome` fields in the schema. This prevents any LLM (even weak ones) from trying to generate summaries, since the tool itself doesn't ask for them.

```
Score cached memories from your previous context. ONLY call when <roampal-score-required> appears.

Score each memory ID listed in the scoring prompt:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING

⚠️ "failed" means MISLEADING, not just unused. If you didn't use it, mark "unknown".

Memory IDs correspond to [id:...] tags in KNOWN CONTEXT from the previous turn.
```

**OpenCode schema** (no exchange fields):
```python
inputSchema = {
    "type": "object",
    "properties": {
        "memory_scores": {
            "type": "object",
            "description": "Score for each memory: doc_id -> outcome.",
            "additionalProperties": {"type": "string", "enum": ["worked", "failed", "partial", "unknown"]}
        }
    },
    "required": ["memory_scores"]
}
```

**How it works:** `roampal init --opencode` sets `"ROAMPAL_PLATFORM": "opencode"` in the MCP environment block of `opencode.json`. When OpenCode launches the MCP server, the env var is present. The server checks `os.environ.get("ROAMPAL_PLATFORM")` at startup and returns the lean tool definition. Claude Code doesn't set this env var, so it gets the full tool with all three fields.

7. **Update scoring hook prompt** (`session_manager.py`):
   - Claude Code (`build_scoring_prompt`): Change `Call score_response(...)` → `Call score_memories(...)`. Add `exchange_summary` and `exchange_outcome` to the prompt template.
   - OpenCode (`build_scoring_prompt_simple`): Change `Call score_response(...)` → `Call score_memories(...)`. Per-memory only — no summary/outcome fields (sidecar handles those).

8. **Update OpenCode plugin** (`roampal.ts`): Change tool call detection from `score_response` → `score_memories`

**Fallback behavior (OpenCode only):** If the main LLM does NOT call `score_memories`, the sidecar's exchange outcome applies uniformly to all cached memories from that turn. If the main LLM DOES call `score_memories`, its per-memory scores are used instead.

**Claude Code:** No fallback needed — the scoring prompt forces the main LLM to call `score_memories` with all three fields every turn.

**Migration:** No backwards compatibility needed. The hook prompt and tool name update together in the same release. No external consumers of this tool name.

---

### Change 13: `roampal --help` CLI Command

**File:** `cli.py`
**Risk:** Low
**Impact:** Low (UX polish)

**Problem:** No unified help command. Users have to guess available commands.

**Proposed:** Add `roampal --help` (and `roampal help`) that lists all commands with one-line descriptions:

```
roampal - Persistent memory for AI coding assistants

Commands:
  init          Configure roampal for Claude Code, Cursor, or OpenCode
  start         Start the memory server
  stop          Stop the memory server
  status        Check server status
  stats         Show memory statistics
  score         Score the last exchange (sidecar)           [v0.3.6]
  summarize     Summarize existing long memories             [v0.3.6]
  context       Output recent exchange context               [v0.3.6]
  ingest        Ingest a document into books collection
  remove        Remove a book from memory
  books         List all books in memory
  doctor        Diagnose installation issues

Options:
  --help, -h    Show this help message
  --version     Show version number

Examples:
  roampal init --claude-code    Set up for Claude Code
  roampal start                 Start the server
  roampal stats                 See memory statistics
  roampal summarize --dry-run   Preview what would be summarized
```

---

## Change Summary

| # | Change | File | Status |
|---|---|---|---|
| 1 | 50/50 Wilson blend | `scoring_service.py` | Implemented |
| 2 | 0.4 base multiplier | `search_service.py` | Implemented |
| 3 | Carry Wilson on promotion | `promotion_service.py` | Implemented |
| 4 | Batch promotion parity | `promotion_service.py` | Implemented |
| 5 | Wire KG entity extraction | `unified_memory_system.py` | Implemented |
| 6 | Entity boost all collections | `search_service.py` | Implemented |
| 7 | Track all outcomes in KG | `outcome_service.py` | Implemented |
| 8 | Unknown outcome score: 0.0 → -0.05 | `outcome_service.py` | Implemented |
| 9 | Exchange summarization (platform-split) | `mcp/server.py`, `session_manager.py`, `server/main.py`, `roampal.ts` | Implemented |
| 10 | `roampal summarize` retroactive cleanup | `cli.py`, `sidecar_service.py` | Implemented |
| 11 | Compaction recovery — recent exchange buffer | `cli.py`, hook config, OpenCode plugin | Implemented |
| 12 | Rename `score_response` → `score_memories` + add summary/outcome fields | `mcp/server.py`, `session_manager.py`, `roampal.ts` | Implemented |
| 13 | `roampal --help` CLI command | `cli.py` | Implemented |
| 14 | Remove `get_context_insights` tool | `mcp/server.py`, tests | Implemented |
| 15 | Platform-aware `score_memories` tool | `mcp/server.py`, `cli.py`, `opencode.json` | Implemented |
| 16 | Move scoring prompt to system prompt (OpenCode) | `roampal.ts` | Implemented |

---

### Change 14: Remove `get_context_insights` MCP Tool

**Files:** `mcp/server.py`, `test_mcp_server.py`
**Risk:** Low (removes redundant tool)
**Impact:** High (fixes tool-call loops on weaker models)

**Problem:** The `get_context_insights` tool was redundant. Hooks (Claude Code) and plugin (OpenCode) already call `/api/hooks/get-context` and inject KNOWN CONTEXT before the model responds. The tool was a leftover from before hooks existed. Its description said "WORKFLOW: step 1 = call this tool", causing:

- **Claude Code:** Redundant double-fetch (works, just wasteful)
- **OpenCode:** Qwen called it 4+ times in a loop, alternating correct/invalid tool names, wasting 9 turns before responding

**Fix:** Remove the tool entirely from both platforms. Context injection is fully automatic via hooks/plugin.

**What was unique to `get_context_insights` (moved elsewhere):**
- Self-audit reminder (every 4th call) → moved to hook context injection
- Doc_id caching for scoring → already handled by `/api/hooks/get-context` (line 592)
- Memory system explanation → stays in other tool descriptions

**MCP tools after removal (6 total):**
`search_memory`, `add_to_memory_bank`, `update_memory`, `delete_memory`, `score_memories`, `record_response`

---

### Change 15: Platform-Aware `score_memories` Tool (OpenCode)

**Files:** `mcp/server.py`, `cli.py` (`configure_opencode`), `opencode.json`
**Risk:** Low
**Impact:** High (prevents models from generating exchange summaries)

**Problem:** The `score_memories` tool description tells models to "summarize the exchange in ~300 chars" and "rate the exchange outcome." Even with a lean prompt, any LLM reading the tool schema will try to fill `exchange_summary` and `exchange_outcome` fields — defeating the sidecar architecture.

**Fix:** `ROAMPAL_PLATFORM=opencode` env var (set in OpenCode's MCP config):
- **Tool description:** Only mentions per-memory scoring (worked/failed/partial/unknown)
- **Tool schema:** Only exposes `memory_scores` field — no `exchange_summary`, no `exchange_outcome`
- **Init:** `roampal init --opencode` auto-sets the env var

Claude Code gets the full tool (all 3 fields). OpenCode gets the lean version.

---

### Change 16: Move Scoring Prompt to System Prompt (OpenCode)

**Files:** `roampal.ts` (OpenCode plugin)
**Risk:** Low
**Impact:** High (fixes models responding about scoring instead of to user)

**Problem:** The scoring prompt was injected INTO the user's message via deep clone (`messages.transform`). From qwen's perspective, the USER asked about scoring — so qwen responded about scoring instead of answering the actual question. Claude Code injects scoring prompts as `<system-reminder>` (system-level), which models treat as silent directives.

**Fix:** Moved scoring prompt from `messages.transform` (user message clone) to `system.transform` (system prompt injection). Same mechanism already used for KNOWN CONTEXT and SCORING REFERENCE blocks. Models treat system prompt content as directives to follow, not conversation to respond to.

**If model ignores it:** Sidecar handles exchange scoring anyway (see Change 17 for per-memory sidecar scoring).

---

### Change 17: Sidecar Per-Memory Scoring (OpenCode)

**Files:** `roampal.ts` (OpenCode plugin)
**Risk:** Low (falls back to blanket scoring if model doesn't return memory_scores)
**Impact:** Medium (irrelevant memories stop inheriting unearned scores)

**Problem:** When the main LLM doesn't call `score_memories`, the sidecar scores the exchange (summary + outcome) and applies that outcome uniformly to ALL cached memories. If the exchange "worked" but 3 of 4 injected memories were irrelevant, all 4 get "worked" — inflating scores for memories that contributed nothing.

**Fix:** The `memories` array (already passed to `scoreExchangeViaLLM()`) is now included in the sidecar prompt with content snippets (first 200 chars each, ~400 extra tokens total). The sidecar scores each memory individually using a relevance heuristic:

1. Memory NOT about the topic discussed → `unknown`
2. Memory IS about the topic AND outcome is `worked` → `worked`
3. Memory IS about the topic AND outcome is `failed`:
   - Response echoed/relied on info from this memory → `failed`
   - Failure unrelated to this memory → `unknown`
4. Memory IS about the topic AND outcome is `partial` → `partial`
5. Response contradicts the memory and exchange worked → `unknown`
6. When in doubt → `unknown`

**Design rationale:** The sidecar can judge topic relevance but NOT actual usage (only the main LLM knows if it used a memory). So scoring is conservative — irrelevant memories get `unknown` instead of inheriting the exchange outcome, while relevant memories inherit it. This is a heuristic, not ground truth, but it's strictly better than blanket scoring.

**Fallback:** If the model doesn't return `memory_scores` (bad JSON, missing field, invalid values), each memory falls back to the blanket exchange outcome. Per-memory is best-effort on top of the existing system.

---

### Change 18: Remove Time-Weighted Score Deltas

**Files:** `outcome_service.py`
**Risk:** Low
**Impact:** Medium (all score deltas are now flat values)

**Problem:** Score deltas for `worked`, `failed`, and `partial` were multiplied by a `time_weight` factor: `1.0 / (1 + age_days / 30)`. A memory last used 30 days ago got half the score delta. This was never requested — it was in the initial commit, added by the AI tool that generated the codebase. The `unknown` outcome already used a flat `-0.05` without time_weight, creating an undocumented asymmetry.

**Fix:** Removed `_calculate_time_weight()` method and `time_weight` parameter from `_calculate_score_update()`. All score deltas are now flat:
- worked: `+0.2`
- failed: `-0.3`
- partial: `+0.05`
- unknown: `-0.05`

A score is a score regardless of memory age.

---

## Test Plan

### Test 1: Side-by-Side Retrieval Comparison

**Script:** `dev/benchmarks/test_retrieval_fairness.py`
**Type:** Read-only benchmark against live data

Snapshots real memories from all collections, runs test queries through both current and proposed scoring pipelines, outputs a comparison table showing rank changes.

**Test queries:**
1. Query matching a high-importance memory_bank entry AND a proven history entry — verify history is competitive
2. Query matching a frequently-failed memory_bank entry — verify it drops in ranking
3. Query matching a freshly promoted history entry (low uses) — verify reserved slot still works
4. Query matching a working memory with high Wilson — verify it competes in open slots after promotion
5. Generic query that hits multiple collections — verify distribution is balanced

**Metrics:**
- MRR (Mean Reciprocal Rank) per collection — should be more balanced
- Rank position delta for same memories under current vs proposed
- Collection distribution in top-4 results

### Test 2: Promotion Lifecycle Simulation

**Script:** `dev/benchmarks/test_promotion_lifecycle.py`
**Type:** Unit test with mock data

Simulates the full lifecycle:
1. Create working memory → score it 3x "worked" → verify Wilson = 1.0, uses = 3
2. Trigger promotion to history → verify success_count and uses carried forward
3. Score it 2 more times in history → verify success_count = 5 total
4. Trigger promotion to patterns → verify it qualifies (score >= 0.9, uses >= 3, success_count >= 5)

Compare against current behavior where step 2 resets to zero and step 4 would fail (only 2 successes post-reset).

### Test 3: Memory_bank Dominance Check

**Script:** `dev/benchmarks/test_membank_dominance.py`
**Type:** Statistical benchmark

Create controlled scenarios:
- 5 memory_bank entries (varying importance/confidence/Wilson)
- 5 history entries (varying Wilson/uses)
- 5 working entries (varying recency/Wilson)

Run 20 diverse queries, measure what % of top-4 open slots go to each collection.

**Current expected:** memory_bank gets 70-90% of open slots
**Target after changes:** memory_bank gets 40-60% of open slots (still strong, not dominant)

### Test 4: Wilson Blend Impact

**Script:** `dev/benchmarks/test_wilson_blend.py`
**Type:** Mathematical unit test

For each blend ratio (80/20, 70/30, 60/40, 50/50), compute `learned_score` across a matrix of:
- importance: [0.5, 0.7, 0.9]
- confidence: [0.5, 0.7, 0.9]
- wilson: [0.2, 0.5, 0.7, 0.9]

Verify 50/50 produces expected behavior:
- High quality + bad Wilson → noticeable penalty
- Low quality + great Wilson → meaningful reward
- Cold start (uses < 3) → quality only, no Wilson contamination

### Test 5: Entity Boost Wiring

**Script:** `dev/benchmarks/test_entity_boost.py`
**Type:** Integration test

1. Store a memory via `store()` / `store_memory_bank()` → verify `add_entities_from_text()` was called
2. Query for concepts in that memory → verify `_calculate_entity_boost()` returns > 1.0
3. Store memories across collections → verify entity boost applies to all, not just memory_bank

### Test 6: Unknown Outcome Score Penalty

**Script:** `dev/benchmarks/test_unknown_scoring.py`
**Type:** Unit test

**6a — Raw score drift toward demotion:**
1. Start with score=0.7 (default)
2. Apply 5 unknowns → verify score=0.45 (5 × -0.05)
3. Apply 5 more unknowns → verify score=0.20 (approaching demotion threshold)
4. Verify score never goes below 0.0 (clamped)

**6b — Recovery from unknowns:**
1. Start with score=0.7, apply 10 unknowns → score=0.20
2. Apply 1 "worked" (score_delta=+0.2) → score=0.40 (recovers 4 unknowns worth)
3. Apply 2 more "worked" → score=0.80 (fully recovered past original)

**6c — Wilson unaffected:**
Verify that unknown outcome does NOT change Wilson's success_delta (still +0.25):
1. Apply 10 unknowns → success_count=2.5, uses=10
2. Verify Wilson calculation uses p_hat=0.25 (same as current behavior)
3. Confirm raw score dropped to 0.20 while Wilson is unchanged

**6d — Demotion/deletion threshold interaction:**
1. Start with score=0.5 (middling memory)
2. Apply unknowns until score drops below demotion threshold
3. Verify the memory would trigger demotion on next evaluation
4. Confirm that a timely "worked" score can prevent demotion

### Test 7: Summarization Impact (COMPLETED)

**Script:** `dev/benchmarks/test_summarization_impact.py`
**Type:** Read-only analysis against live ChromaDB
**Status:** Run and verified Feb 11 2026

Full census of all working+history memories measuring content length distribution, token savings, quality risks, and retrieval impact. Results documented in Benchmark Data section below.

**Key verified findings:**
- 489/564 (86.7%) working+history memories exceed 300 chars
- Avg content: 1,638 chars (previous estimate of 868 was from a 102-sample — this is the full census)
- Token savings: 78% reduction (56K -> 12K per 30-turn session)
- Quality risk: 86/489 (17.6%) have the answer after char 300 — truncation would lose it, summarization preserves it
- Retrieval: Summary captures semantic concepts well enough for embedding (no dual storage needed)

### Test 8: Exchange Scoring + Summarization End-to-End

**Script:** `dev/benchmarks/test_exchange_scoring.py`
**Type:** Integration test

**8a — Claude Code: score_memories with summary + outcome:**
1. Call `score_memories` with `memory_scores`, `exchange_summary`, and `exchange_outcome`
2. Verify per-memory scores applied (existing behavior)
3. Verify summary stored as working memory with `memory_type: "exchange_summary"` metadata
4. Verify exchange outcome recorded via `record_outcome`
5. Verify summary content is the provided `exchange_summary` string (not full exchange)

**8b — Claude Code: score_memories without summary (backward compat):**
1. Call `score_memories` with only `memory_scores` (no summary, no outcome)
2. Verify per-memory scores applied
3. Verify NO summary stored (field was omitted)
4. Verify no crash or error

**8c — OpenCode: sidecar fallback when LLM doesn't score:**
1. Simulate: main LLM did NOT call `score_memories`
2. Simulate: sidecar runs and returns `{"summary": "...", "outcome": "worked"}`
3. Verify sidecar's exchange outcome applies uniformly to all cached memories from that turn
4. Verify summary stored with correct metadata

**8d — OpenCode: main LLM scores + sidecar runs:**
1. Simulate: main LLM called `score_memories` with per-memory scores
2. Simulate: sidecar runs and returns exchange outcome
3. Verify main LLM's per-memory scores are used (not overridden by sidecar)
4. Verify sidecar's summary is still stored (summarization always happens)
5. Verify no double-scoring

### Test 9: `roampal summarize` CLI Command

**Script:** `dev/benchmarks/test_summarize_cli.py`
**Type:** Integration test with mock data

**9a — Dry run accuracy:**
1. Create test memories: 3 over 300 chars, 2 under, 1 memory_bank (exempt), 1 pattern (exempt)
2. Run `roampal summarize --dry-run`
3. Verify: reports exactly 3 candidates, shows sample summaries, does NOT modify any memory

**9b — Content replacement correctness:**
1. Create a test memory with 800-char content
2. Run `roampal summarize` (with mock Haiku returning a 250-char summary)
3. Verify: content field is now the 250-char summary (replaces original)
4. Verify: embedding is re-computed from summary content
5. Verify: all metadata (score, uses, Wilson, etc.) preserved

**9c — Context injection token reduction:**
1. Store a summarized memory (250 chars)
2. Trigger context injection via `_format_context_injection()`
3. Verify: injected text is the 250-char summary
4. Verify: token count matches expected (~95 tok, not ~467 tok)

**9d — Collection exemptions:**
1. Create memories in memory_bank, books, patterns, working, history
2. Run `roampal summarize`
3. Verify: only working and history memories are summarized
4. Verify: memory_bank, books, patterns are untouched

### Test 10: Compaction Recovery — Recent Exchange Buffer

**Script:** `dev/benchmarks/test_compaction_recovery.py`
**Type:** Integration test

**10a — Exchange summary tagging:**
1. Simulate sidecar creating an exchange summary
2. Verify: stored in working with `memory_type: "exchange_summary"` metadata
3. Verify: content is <300 chars
4. Verify: `created_at` timestamp is present

**10b — CLI recent exchanges query:**
1. Store 6 exchange summaries with varying timestamps (oldest to newest)
2. Run `roampal context --recent-exchanges`
3. Verify: output contains exactly 4 entries
4. Verify: entries are in reverse chronological order (newest first)
5. Verify: the 2 oldest summaries are NOT in the output

**10c — Deduplication with semantic results:**
1. Store 4 exchange summaries
2. Ensure one of them also appears in the 4-slot semantic search results (high semantic relevance)
3. Run `roampal context --recent-exchanges` and compare against `get_context_insights` KNOWN CONTEXT
4. Verify: if a summary appears in both, hook output should still include it (deduplication is display-side, not query-side)

**10d — Compaction simulation (Claude Code):**
1. Store 4 exchange summaries simulating a multi-turn session
2. Simulate `SessionStart(compact)` hook firing → runs `roampal context --recent-exchanges`
3. Verify: output contains RECENT EXCHANGES block with last 4 exchanges
4. Verify: output is valid for injection (model sees it as context)

**10e — Both platforms:**
1. Verify Claude Code path: UserPromptSubmit → inject scoring prompt → main LLM calls `score_memories(exchange_summary=...)` → server stores tagged summary. Stop hook stores exchange in JSONL (lifecycle_only=True, no ChromaDB). `SessionStart(compact)` → `roampal context --recent-exchanges` → outputs recent block.
2. Verify OpenCode path: plugin `session.idle` → stores tagged summary. `experimental.session.compacting` → injects recent exchanges into compaction prompt. `session.compacted` → flags next turn for injection.
3. Both use the same server-side temporal query (filter by `memory_type: "exchange_summary"`, sort by `created_at` desc, limit 4)

---

## Benchmark Data (from live DB, Feb 11 2026)

### Collection Stats (excluding books)

| Collection | Total | Scored | Never Scored | Never Scored % | Avg Uses | Avg Score |
|---|---|---|---|---|---|---|
| working | 224 | 156 | 68 | 30% | 1.1 | 0.59 |
| history | 379 | 186 | 193 | **51%** | 2.1 | 0.95 |
| patterns | 12 | 12 | 0 | 0% | 17.1 | 0.98 |
| memory_bank | 37 | 21 | 16 | 43% | 24.6 | 0.94 |
| **Total** | **652** | **375** | **277** | **42.5%** | — | — |

Books (1056 chunks) have no scoring fields — pure reference, unaffected by scoring changes.

### History Never-Scored Analysis

193 history memories (51%) have uses=0 — never surfaced after promotion. All entered via working→history promotion (0 from batch). All have success_count=0 (v0.2.9 reset confirmed).

**Age distribution (never-scored):**
| Age | Count |
|---|---|
| <1 day | 37 |
| 1-3 days | 82 |
| 3-7 days | 40 |
| 7-14 days | 21 |
| 14-30 days | 13 |

**Entry score distribution (never-scored vs scored):**
| Score Range | Never Scored | Scored |
|---|---|---|
| 0.0-0.5 | 0 | 5 |
| 0.5-0.7 | 0 | 2 |
| 0.7-0.8 | 56 | 8 |
| 0.8-0.9 | 131 | 13 |
| 0.9-1.0 | 6 | 158 |

Never-scored memories entered history with decent scores (0.7-0.9). They passed the promotion threshold from working. They're not bad memories — they just never surfaced in history because the 4-slot bottleneck means only ~1 history memory surfaces per turn.

### Unknown Penalty Replay (-0.05)

| Metric | Value |
|---|---|
| Total scored memories | 374 |
| Impacted by -0.05 penalty | 9 (2.4%) |
| Status changes | 1 (working → DEMOTE) |
| Deletions | 0 |
| Verdict | Safe. Minimal impact on current data. |

The -0.05 penalty is conservative. Most memories don't accumulate enough unknowns to matter. The real cleanup happens via "failed" scores (score_delta = -0.30).

### Context Injection Token Cost (CORRECTED — full census, not sampled)

**Previous estimate (102 sampled) was wrong.** Full census of 564 working+history memories:

Memory content stats: avg **1,638 chars**, median **916 chars**, max **23,339 chars**.

| Config | Per Slot | Per Turn | 30 Turns | % of 32K | % of 128K |
|---|---|---|---|---|---|
| **4 slots, full (current)** | **467 tok** | **1,868** | **56,040** | **175%** | **43.8%** |
| 4 slots, summarized (<300 chars) | 102 tok | 408 | 12,240 | 38% | 9.6% |
| 4 slots + recent exchanges (cold start/compaction only) | — | 732 | 12,888* | 40.3% | 10.1% |

*Assumes 2 turns with recent exchanges (cold start + 1 compaction), 28 normal turns.

Previous estimate said 90% of 32K — actual is **175%**, literally overflowing small model context windows.
Summarization brings this to 38% on normal turns. Cold start / post-compaction turns cost 732 tokens but only happen 2-3 times per session — negligible overall impact.

**Script:** `dev/benchmarks/test_summarization_impact.py` (read-only, verified against live ChromaDB)

---

## Rollback Plan

Changes 1-8: scoring/ranking math — no schema changes, no data migrations. Rollback = revert the code. Existing memories are unaffected.

Change 9-10 (summarization): Summaries replace original content. If reverted, new memories would store full exchanges again (current behavior). Already-summarized memories stay summarized — no way to restore original content (it's in transcript logs only). This is acceptable since summaries are strictly better for injection.

Change 11 (compaction recovery): The `memory_type: "exchange_summary"` tag and temporal query are additive. If reverted, `get_context_insights` just stops including the RECENT EXCHANGES section — semantic search continues unchanged. Tagged memories are harmless.

The Wilson carry-forward (Change 3) means newly promoted memories will have non-zero counters. If reverted, those counters would be reset on next promotion anyway. No data corruption risk.

---

Change 12 (prompt update): Pure text change to MCP tool description. Rollback = revert the description string. No behavior impact.

Change 13 (`--help`): Additive CLI output. Rollback = remove the help handler.

---

## Dependencies

- **Claude Code:** No external dependencies for scoring/summarization — main LLM handles everything via `score_memories` tool
- **OpenCode:** Existing Zen free models for sidecar (unchanged)
- **CLI (`roampal score`, `roampal summarize`):** Uses `sidecar_service.py` with auto-detected backend: Custom endpoint (if `ROAMPAL_SIDECAR_URL` set) → Haiku API (if `ANTHROPIC_API_KEY` set) → Zen (if available) → Ollama/LM Studio (if running) → `claude -p` fallback
- No schema/migration changes required
- No new dependencies
- New `memory_type` metadata field on exchange summaries (additive, backwards compatible)
- New `exchange_summary` and `exchange_outcome` fields on `score_memories` tool (optional, backwards compatible)
- New CLI commands: `roampal score`, `roampal summarize`, `roampal context --recent-exchanges`, `roampal --help`
- **Dead code to remove:** `_summarize_exchange_background()`, `_call_anthropic_direct()`, `_summarized_fingerprints`, `_api_key_valid` from `server/main.py`
