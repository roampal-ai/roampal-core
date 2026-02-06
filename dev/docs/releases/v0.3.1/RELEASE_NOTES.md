# Roampal Core v0.3.1 Release Notes

## Overview
Context injection quality fix - guarantees working memory always surfaces in automatic context.

## Changes

### 1. Reserved Working Memory Slot
**Problem:** Working memories (recent session context) were being crowded out by memory_bank items with high importance scores and BM25 keyword matches on stale patterns. With 264+ working memories available, none were surfacing in the 3-slot injection.

**Root cause:** `get_context_for_injection()` searched all collections equally and took the top 3 by combined Wilson score. High-importance memory_bank items (importance=0.9) and keyword matches on patterns outranked semantically-relevant working memories.

**Fix implemented:**
1. **Reserved slot** - Always fetch 1 working memory separately
2. **Remaining slots** - Search patterns/history/memory_bank for top 2
3. **Combine** - 1 reserved working + 2 from other collections = 3 total

**Files changed:**
- `roampal/backend/modules/memory/unified_memory_system.py:791-818` - Refactored context injection to reserve 1 slot for working collection

**How it works now:**
1. Search working collection for best match → guaranteed 1 slot
2. Search other collections (patterns, history, memory_bank) for top 4
3. Take top 2 from other collections, excluding duplicates
4. Final result: 1 working + 2 other = 3 memories injected

**Why this matters:**
Working memories contain recent session context - what you just did, what worked, what failed. Without a reserved slot, this context was being displaced by:
- memory_bank items with high importance
- Pattern matches on keywords (e.g., "memory" matching unrelated patterns)

Now you'll always see recent session context in the injected memories.

## Testing
1. Start a new session
2. Work on several tasks (building up working memories)
3. Verify that KNOWN CONTEXT includes at least one `(working)` memory
4. Previously: working memories rarely appeared; Now: always 1 slot reserved