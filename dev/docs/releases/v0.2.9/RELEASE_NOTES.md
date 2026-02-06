# Roampal Core v0.2.9 Release Notes

## Overview
Memory cleanup release - removes archive-on-update behavior and adds cleanup for existing archived memories.

## Changes

### 1. Remove Archive-on-Update
**Problem:** `update_memory` was creating archived copies of old content, leading to:
- Ghost memories with `_archived_` suffix accumulating in ChromaDB
- Archived memories leaking into scoring hooks
- Unnecessary storage bloat

**Fix:** Update now overwrites in place without creating archive copies. No more audit trail, but cleaner database.

**File:** `roampal/backend/modules/memory/memory_bank_service.py`

### 2. Add Archived Memory Cleanup
**New method:** `cleanup_archived()` - deletes all memories with `status: "archived"` or `_archived_` in their ID.

Called automatically on server startup to purge existing archived memories.

### 3. Wilson Scoring for memory_bank
**Problem:** memory_bank items surface based on semantic similarity + quality (importance × confidence) only. No feedback loop exists - irrelevant facts keep surfacing forever because "unknown" scores don't penalize them.

**Observation:** In testing, ~70% of surfaced memory_bank items are "Roampal adjacent" but not useful for the current task. These get scored "unknown" repeatedly but never demote.

**Fix:** Add Wilson score influence to memory_bank ranking.

**Scoring formula:**
```
final_score = 0.8 * (importance × confidence) + 0.2 * wilson_score
```

**Unknown outcome change (all collections/Wilson Scoring):**
```
unknown = +1 use, +0.25 success (was +0, +0)
```

This means:
- `worked` = 1.0 success (memory was helpful)
- `partial` = 0.5 success (somewhat helpful)
- `unknown` = 0.25 success (surfaced but not used - weak negative signal)
- `failed` = 0.0 success (memory was misleading)

**Threshold protection:**
- Only apply Wilson influence if `uses >= 3`
- Below threshold, use quality score only (protects new facts from cold start)

**Expected outcome:**
- Noise memories gradually demote through accumulated "unknown" scores
- Useful memories maintain/improve scores through "worked"/"partial"
- Natural selection for relevant context over time

**How natural selection works:**

Consider a memory_bank with 100 facts. On any given query, ~5 facts surface in KNOWN CONTEXT. Of those:
- 1-2 are directly useful → scored "worked" (1.0 success)
- 3-4 are "Roampal adjacent" but not used → scored "unknown" (0.25 success)

Over 10 exchanges:
```
Useful fact:    8 worked, 2 unknown → Wilson: 8.5/10 = 0.85
Noise fact:     0 worked, 10 unknown → Wilson: 2.5/10 = 0.25
```

With the 80/20 blend:
```
Useful:  0.8 * quality + 0.2 * 0.85 = quality + 0.17
Noise:   0.8 * quality + 0.2 * 0.25 = quality + 0.05
```

The 0.12 gap compounds over time. Useful facts maintain ranking. Noise facts drift down, eventually dropping below the retrieval threshold.

**Key insight:** We don't need to explicitly demote noise - we just need to create a weak negative signal for "surfaced but unused" memories. The Wilson math handles the rest. Facts that keep proving useful stay. Facts that keep getting ignored fade away.

This is why `unknown = 0.25` (not 0.0 or 0.5):
- 0.0 would be too harsh (unused ≠ bad, just not relevant to THIS query)
- 0.5 would be neutral (no selection pressure)
- 0.25 creates gradual drift without punishing contextually-irrelevant facts

**Implementation details:**

1. **search_service.py lines 391-401** - `_apply_collection_boost()`:
   - Current: `quality_score = importance * confidence`, `metadata_boost = 1.0 - quality_score * 0.8`
   - Change: Add Wilson lookup, apply `final = 0.8 * quality + 0.2 * wilson` if uses >= 3

2. **search_service.py lines 588-596** - cross-encoder reranking:
   - Current: `quality_boost = 1.0 + quality * 0.3`
   - Change: Include Wilson in quality_boost calculation

3. **outcome_service.py lines 288-291** - `_calculate_score_update()`:
   - Current: unknown/else returns early with `(0.0, current_score, uses, 0.0)`
   - Change: For ALL collections, unknown returns `(0.0, current_score, uses + 1, 0.25)`

4. **memory_bank_service.py** - metadata fields:
   - Add `uses`, `success_count` fields to memory_bank items on store/update

### 4. Stricter History → Patterns Promotion
**Problem:** Memories can coast from working → history → patterns on initial good scores without proving long-term usefulness.

**Current flow:**
```
working (24h) → history (30d) → patterns (permanent)
Promotion based on: score threshold + uses
```

**Fix:** Add "probation period" when entering history.

**Changes:**
1. **Reset success count on history promotion** - When a memory moves from working → history, reset `success_count` to 0. It must prove itself fresh in history.

2. **Require 5 worked outcomes for patterns eligibility** - A memory in history cannot be promoted to patterns until it has accumulated 5 "worked" outcomes (not just high score).

**Logic:**
```python
# On working → history promotion
metadata["success_count"] = 0  # Reset
metadata["uses"] = 0           # Reset
metadata["promoted_to_history_at"] = timestamp

# On history → patterns eligibility check
if success_count >= 5 and score >= threshold:
    promote_to_patterns()
```

**Why this helps:**
- Prevents one-hit wonders from becoming permanent
- Forces memories to prove usefulness over time
- 5 worked outcomes = meaningful sample size
- Score alone isn't enough - need concrete positive outcomes

**Implementation details:**

1. **promotion_service.py lines 102-105** - working → history check:
   - Current: `if score >= threshold and uses >= 2`
   - No change to eligibility check

2. **promotion_service.py lines 141-172** - `_promote_working_to_history()`:
   - Current: preserves metadata as-is
   - Change: Reset `success_count = 0`, `uses = 0`, add `promoted_to_history_at = timestamp`

3. **promotion_service.py lines 108-110** - history → patterns check:
   - Current: `if score >= high_value_threshold and uses >= 3`
   - Change: `if score >= threshold and uses >= 3 and success_count >= 5`

4. **promotion_service.py lines 174-220** - `_promote_history_to_patterns()`:
   - No changes needed (eligibility already checked)

### 5. Fix Retrieval/Display/Scoring Mismatch
**Problem:** v0.2.7 changed display to show 3 memories but retrieval still fetched 5. This caused:
- Claude asked to score 5 memories but only seeing 3 in context
- Scoring felt "off" because 2 memories were invisible

**Fix:** Aligned retrieval to 3 memories, matching display.

**File:** `roampal/backend/modules/memory/unified_memory_system.py`
```python
# Before: top_memories = scored_results[:5]
# After:  top_memories = valid_results[:3]
```

### 6. Filter Empty Memories from Context
**Problem:** Empty/blank memories could surface in context injection.

**Fix:** Filter out memories with no content before slicing.

**File:** `roampal/backend/modules/memory/unified_memory_system.py`
```python
valid_results = [m for m in scored_results if m.get("content") or m.get("text")]
top_memories = valid_results[:3]
```

### 7. Skip Empty Exchange Storage
**Problem:** Stop hook was storing exchanges without validating content, leading to empty working memories.

**Fix:** Validate user_message and assistant_response before storing.

**File:** `roampal/server/main.py`
```python
if not user_msg or not assistant_msg:
    logger.warning(f"Skipping empty exchange storage")
    return StopHookResponse(status="skipped", message="Empty exchange not stored")
```

## Migration
Automatic - existing archived memories are cleaned up on first server start after upgrade.

## Files Changed
- `roampal/backend/modules/memory/memory_bank_service.py` - remove archive logic, add cleanup, add uses/success_count fields
- `roampal/backend/modules/memory/outcome_service.py` - Wilson scoring for all collections, unknown = +1 use, +0.25 success
- `roampal/backend/modules/memory/search_service.py` - Wilson influence in collection boost and cross-encoder reranking
- `roampal/backend/modules/memory/promotion_service.py` - reset counters on history entry, require success_count >= 5 for patterns
- `roampal/backend/modules/memory/unified_memory_system.py` - fix retrieval/display mismatch (5→3), filter empty memories
- `roampal/server/main.py` - call cleanup on startup, skip empty exchange storage
- `roampal/__init__.py` - version bump
- `pyproject.toml` - version bump