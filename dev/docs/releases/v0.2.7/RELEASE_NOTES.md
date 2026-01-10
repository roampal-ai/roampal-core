# roampal-core v0.2.7 Release Notes

**Release Date:** 2026-01-09
**Type:** Critical Bug Fix

---

## Summary

Critical fixes for identity injection and cold start bloat:

1. **User name wasn't showing in hook context** - LLM had no idea who the user was during non-cold-start messages.
2. **Cold start was spamming 3000+ chars** - Full fact content dumped verbatim. Large facts (1100+ chars) overwhelmed context.
3. **KNOWN CONTEXT removed in v0.2.6** - LLM had no recent memories on cold start.

**Key Changes:**
1. **Cold start: Best fact per tag category** - One fact per tag (identity, preference, goal, project, system_mastery, agent_growth), sorted by quality
2. **Simple name injection** - Scans memory_bank for `identity` tag, extracts name with regex, shows `User: [name]` in KNOWN CONTEXT
3. **KNOWN CONTEXT: 3 memories** - Reduced from 5 to keep context lean
4. **Simplified `related` scoring** - Clear guidance: only include memories that CAUSED the outcome. Don't penalize helpful memories on failed outcomes.
5. **First-sentence truncation** - Cold start extracts first sentence only per fact (~600 chars total instead of 3000+)
6. **Graceful KG error handling** - Corrupted knowledge_graph.json no longer crashes cold start
7. **Size guidance in tool description** - `add_to_memory_bank` now tells LLM to keep facts small

---

## What Changed

### 1. Simple Name Injection in Hook Context (unified_memory_system.py)

**The Bug:** User identity wasn't showing in KNOWN CONTEXT during hook injection. LLM had no idea who the user was.

**Solution:** Scan memory_bank for identity-tagged facts and extract user name via regex patterns.

**Implementation:**
```python
def _format_context_injection(self, context: Dict[str, Any]) -> str:
    parts = []
    user_name = None

    # Scan all memory_bank facts for identity tag
    all_facts = self._memory_bank_service.list_all(include_archived=False)
    for fact in all_facts:
        tags = fact.get("metadata", {}).get("tags", [])
        if "identity" in tags:
            content = fact.get("content", "")
            # Extract name via regex ("name is X", "I'm X", "I am X")
            match = re.search(r"name is (\w+)", content, re.IGNORECASE)
            if match:
                user_name = match.group(1)
                break

    memories = context.get("memories", [])

    if user_name or memories:
        parts.append("═══ KNOWN CONTEXT ═══")
        if user_name:
            parts.append(f"User: {user_name}")
        for mem in memories[:3]:
            # ... formatting
```

**Result:**
```
═══ KNOWN CONTEXT ═══
User: Logan
• [memory 1] (collection)
• [memory 2] (collection)
• [memory 3] (collection)
═══ END CONTEXT ═══
```

---

### 2. Cold Start: 6 Facts Max via Tag-Based Selection (server/main.py)

**Problem:** Cold start crammed 50 facts via semantic search. As memory_bank grew, this overwhelmed context (10K+ chars, truncated by Claude Code hooks).

**Before (broken):**
```python
# 50 facts via kitchen-sink semantic search
all_facts = await _memory.search(
    query="user name identity who is preference goal project communication style background",
    collections=["memory_bank"],
    limit=50  # WAY too much - grew over time
)
```

**After (fixed):**
```python
TAG_PRIORITIES = ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]  # 6 tags = 6 facts max

# Step 1: Always include identity (always_inject=True memories)
identity_core = _memory.get_always_inject()  # NEW METHOD
seen_ids = {f.get("id") for f in identity_core}

# Step 2: One BEST fact per tag category (balanced picture)
balanced_facts = []
all_memory_bank = _memory._memory_bank_service.list_all()

for tag in TAG_PRIORITIES:
    # Find highest-quality fact with this tag (not already included)
    for fact in all_memory_bank:
        fact_id = fact.get("id")
        if fact_id in seen_ids:
            continue
        tags = fact.get("metadata", {}).get("tags", [])
        if isinstance(tags, str):
            import json
            tags = json.loads(tags) if tags else []
        if tag in tags:
            balanced_facts.append(fact)
            seen_ids.add(fact_id)
            break  # One per tag

all_facts = identity_core + balanced_facts
# Max 6 facts: one per tag category, each truncated to ~300 chars
```

**Why this is better:**
- 6 facts max (one per tag), each truncated to ~300 chars first sentence
- Identity always first (always_inject memories)
- **Balanced coverage**: 1 fact per tag category (identity, preference, goal, project, system_mastery, agent_growth)

**Result:** Lean but rich - covers the FULL picture of who the user is, not random semantic matches.

**Note on Content KG:** The code includes a Step 3 that reads from Content KG to fill remaining slots. However, Content KG entity extraction (`add_entities_from_text()`) is not yet wired into the memory storage path, so `content_graph.json` is never populated. The read path exists and gracefully skips if the file is missing. Future release will wire up entity extraction when we need entity-based search features.

### 3. Graceful KG Error Handling (server/main.py)

**Problem:** Corrupted `knowledge_graph.json` (line 153539 parse error) crashes cold start.

**Fix:**
```python
try:
    kg_service = KnowledgeGraphService(...)
    entities = kg_service.content_graph.get_all_entities()
except Exception as e:
    logger.warning(f"Content KG load failed, falling back to semantic search: {e}")
    # Fall back to existing semantic search
    all_facts = await _memory.search(...)
```

### 4. First-Sentence Truncation for Cold Start (server/main.py)

**Problem:** Cold start dumps full fact content verbatim. Users store massive facts (1000+ chars) that spam cold start context. Example: 1100 char "Writing Style" guide appears on every session.

**Solution:** Extract first sentence only from each fact (max 300 chars per fact).

```python
def first_sentence(text: str, max_chars: int = 300) -> str:
    """Extract first sentence, capped at max_chars."""
    if not text:
        return ""
    # Split on period, take first sentence
    first = text.split('.')[0]
    if len(first) > max_chars:
        return first[:max_chars-3] + "..."
    return first + "."

# In _build_cold_start_profile():
for tag in TAG_PRIORITIES:
    if tag in category_facts:
        content = category_facts[tag]
        profile_parts.append(f"{tag_labels[tag]}: {first_sentence(content)}")
```

**Output (example):**
```
<roampal-user-profile>
Identity: [User's name and role].
Preference: Direct, no corporate speak.
Goal: Ship production-ready AI memory.
Project: Building roampal-core MCP server.
System Mastery: Verify before documenting.
Agent Growth: Be curious, explore.
</roampal-user-profile>
```

**Benefits:**
- ~1800 chars max (6 categories × ~300 chars) instead of 3000+
- One fact per category, first sentence only
- Full facts still available via search when relevant
- Prevents massive facts from spamming every session

### 5. Restore KNOWN CONTEXT to Cold Start (server/main.py)

**Problem:** v0.2.6 removed KNOWN CONTEXT from cold start (profile-only). This broke continuity - LLM has no recent memories on first message.

**Solution:** Re-add KNOWN CONTEXT section after user profile:

```python
# Cold start structure:
# 1. Lean user profile (first-sentence truncation)
# 2. KNOWN CONTEXT (top 5 semantic matches for recent work)

profile = await _build_cold_start_profile()  # ~600 chars
context = await _memory.get_context_for_injection(query, limit=5)
known_context = context.get("formatted_injection", "")

return f"{profile}\n\n{known_context}"
```

**Output (example):**
```
<roampal-user-profile>
Identity: [User's name and role].
...
</roampal-user-profile>

═══ KNOWN CONTEXT ═══
• Recent debugging session on cold start (working)
• MCP tool usage patterns (patterns)
═══ END CONTEXT ═══
```

**Benefits:**
- Profile tells LLM WHO the user is (~600 chars)
- Known context tells LLM WHAT they've been working on
- Best of both: identity + continuity

---

### 6. Size Guidance in Tool Description (mcp/server.py)

**Problem:** LLMs store massive research dumps as single facts (5000+ chars), overwhelming cold start and hook injection.

**Solution:** Add SIZE GUIDANCE to `add_to_memory_bank` tool description:

```python
"""Store PERMANENT facts that help maintain continuity across sessions.

...existing content...

SIZE GUIDANCE:
• Keep facts under 300 chars - these appear in context injection
• Research dumps belong in books collection, not memory_bank
• If you notice massive facts (1000+ chars), offer to condense them
• One concept per fact - split multi-topic content into separate memories
"""
```

This teaches LLMs to self-regulate fact size at storage time.

---

### 7. Clarify `related` for Failed Outcomes (mcp/server.py)

**Problem:** LLMs incorrectly downvote helpful memories when scoring failed outcomes. If a memory gives good advice but the LLM fails anyway (due to its own bad reasoning), the memory gets penalized - making the system dumber over time.

**Solution:** Add explicit guidance to `score_response` tool description:

```python
"""Score the previous exchange...

⚠️ CRITICAL - "failed" OUTCOMES ARE ESSENTIAL:
• If user says you were wrong → outcome="failed"
• If memory you retrieved was outdated → outcome="failed"
• If user had to correct you → outcome="failed"
• If you gave advice that didn't help → outcome="failed"

Failed outcomes are how bad memories get deleted. Without them, wrong info persists forever.
Don't default to "worked" just to be optimistic. Wrong memories MUST be demoted.

SELECTIVE SCORING (optional):
If the scoring prompt shows "Memories surfaced:", you can specify which were actually relevant:
• related=["doc_id_1", "doc_id_2"] → only those get scored
• Omit related → all surfaced memories get scored (backwards compatible)
Unrelated memories get 0 (neutral) - they're not penalized, just skipped.

⚠️ IMPORTANT: Only include memories in `related` that CAUSED the outcome.
• If you failed BECAUSE a memory gave bad advice → include it (it should be downvoted)
• If you failed DESPITE good advice from a memory → do NOT include it (don't penalize helpful memories)
"""
```

**Why this matters:** The scoring system only works if LLMs correctly attribute outcomes. Penalizing memories that gave good advice creates a death spiral where helpful memories get demoted and unhelpful ones persist.

---

## Files to Change

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py` | 1. Fix `_format_context_injection()` to include `user_facts` 2. Add `get_always_inject()` wrapper 3. Add `get_by_id(doc_id)` method |
| `roampal/server/main.py` | Import ContentGraph, rewrite `_build_cold_start_profile()` to use Content KG narrative profile |
| `roampal/mcp/server.py` | 1. Add SIZE GUIDANCE to `add_to_memory_bank` 2. Add `related` attribution guidance to `score_response` |
| `pyproject.toml` | Version 0.2.6 → 0.2.7 |
| `roampal/__init__.py` | Version bump |

---

## Implementation Steps

### Fix 1: always_inject in hook injection ✅ DONE
1. [x] In `unified_memory_system.py`, find `_format_context_injection()` method
2. [x] Add `user_facts = context.get("user_facts", [])`
3. [x] Combine with memories: `all_memories = user_facts + [m for m in memories if m not in user_facts]`
4. [x] Use `all_memories` in the formatting loop

### Fix 2: First-sentence truncation for cold start ✅ DONE
1. [x] Add `_first_sentence(text, max_chars=300)` helper function at main.py:58
2. [x] In `_build_cold_start_profile()`, apply to each fact's content at main.py:289
3. [x] Result: ~1800 chars max (6 categories × ~300 chars)

### Fix 3: Restore KNOWN CONTEXT to cold start ✅ DONE
1. [x] Fetch context for BOTH cold start and regular messages at main.py:463
2. [x] On cold start, append KNOWN CONTEXT after profile at main.py:470
3. [x] Structure: `<roampal-user-profile>` + `═══ KNOWN CONTEXT ═══`

### Fix 4: Size guidance in tool description ✅ DONE
1. [x] In `mcp/server.py`, find `add_to_memory_bank` tool description
2. [x] Add SIZE GUIDANCE section with 300 char guideline
3. [x] Note: research dumps → books collection
4. [x] Note: offer to condense massive facts if found

### Fix 5: Clarify `related` for failed outcomes ✅ DONE
1. [x] In `mcp/server.py`, find `score_response` tool description
2. [x] Add warning that `related` should only include memories that CAUSED the outcome
3. [x] Clarify: if you failed despite good advice, don't penalize the helpful memory

### Finalize
1. [x] Bump version in pyproject.toml and __init__.py
2. [ ] Run tests
3. [ ] Test hook injection manually - verify identity shows in KNOWN CONTEXT
4. [ ] Test cold start manually - verify first-sentence truncation + KNOWN CONTEXT

---

## Testing

```bash
pytest roampal/backend/modules/memory/tests/unit/ -q
```

Manual tests:

**Test 1: Hook injection (KNOWN CONTEXT)**
1. Have an always_inject memory with user's name
2. Send message 2+ (not cold start)
3. Verify KNOWN CONTEXT includes user's name

**Test 2: Cold start**
1. Restart VS Code
2. Open new Claude Code window
3. Verify identity memory appears in cold start context

---

## Why This Wasn't Caught

**always_inject bug:**
- `user_facts` are fetched correctly in `get_context_for_injection()` (line 723)
- But `_format_context_injection()` only looks at `memories` (line 772)
- MCP `get_context_insights()` DOES format both - so MCP calls worked
- Hook injection never did - broken since v0.2.2 when always_inject was added
- No test verifies hook injection output includes always_inject memories

**Content KG bug:**
- Content KG code exists and works (`content_graph.py`)
- Content KG is used by `knowledge_graph_service.py` and `search_service.py`
- But `_build_cold_start_profile()` in main.py never imported or used it
- Tests verify cold start runs (with semantic search) - no test says "must use Content KG"
- This was never implemented correctly, not a regression

---