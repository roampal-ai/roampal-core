# roampal-core v0.2.6 Release Notes

**Release Date:** 2026-01-09
**Type:** Feature + Code Quality

---

## Summary

Adds new user identity detection with onboarding prompts, cleaner cold start, and code quality improvements.

**Key Changes:**
1. **Identity prompt** - Asks for user's name when no identity-tagged facts exist
2. **Profile-only cold start** - Cold start now just shows user profile (LLM can search if context needed)
3. **Recency sort fix** - `sort_by="recency"` now actually works in search()
4. **Books excluded from auto-injection** - Books are now opt-in only
5. **--force flag fix for Cursor** - Flag now works correctly for MCP config
6. **Code cleanup** - Removed dead config fields, fixed silent exceptions
7. **Tool description improvements** - TAG MEANINGS, PROACTIVE MEMORY, READING RESULTS added to help LLMs use Roampal properly

---

## What Changed

### 1. New User Identity Detection (server/main.py)

When a user has NO identity-tagged facts in memory_bank, the system now prompts the LLM to ask for their name.

**Two modes:**
- **Truly new user** (no history/patterns): Full onboarding prompt
- **Existing user without identity** (has history): Softer nudge

```python
# Check if user has NO identity-tagged facts (name, role, background)
if not identity_facts:
    has_history = await _memory.search(query="", collections=["history", "patterns"], limit=1)

    if has_history:
        return "<roampal-identity-missing>..."  # Softer prompt
    else:
        return "<roampal-new-user>..."  # Full onboarding
```

### 2. Profile-Only Cold Start (server/main.py)

Cold start now shows ONLY the user profile (memory_bank facts). No recent exchanges are auto-injected.

**Rationale:**
- New window = intentional fresh start
- LLM can use `search_memory(sort_by="recency")` if context needed
- Avoids truncated snippets that provide little value
- Cleaner, lighter cold start

```python
# Cold start profile structure:
# - IDENTITY (name, role, background)
# - PREFERENCES (communication style, tools)
# - GOALS
# - PROJECTS
# - OTHER FACTS
# That's it - no recent context injection
```

### 3. Skip Query-Based Context on Cold Start (server/main.py)

Previously, cold start would run BOTH the profile dump AND a query-based context search. This caused irrelevant results (e.g., searching for "yo" might return Bhagavad Gita chunks from books).

**Fix:** On cold start, ONLY inject the profile. Query-based context injection starts on message 2.

```python
# Skip query-based context injection on cold start - profile is enough
context = {}
if not is_cold_start:
    context = await _memory.get_context_for_injection(...)
```

### 4. Exclude Books from Auto-Injection (unified_memory_system.py)

Books were being auto-injected into context alongside patterns/history/working, often producing irrelevant noise (e.g., Bhagavad Gita appearing when debugging code).

**Fix:** Books excluded from `get_context_for_injection()`. Users can still access books via explicit `search_memory(collections=["books"])`.

```python
# Before: books included in auto-injection
all_collections = ["working", "patterns", "history", "books", "memory_bank"]

# After: books are opt-in only
all_collections = ["working", "patterns", "history", "memory_bank"]
```

### 5. Recency Sort Support in Search (unified_memory_system.py)

The `search()` method now accepts a `sort_by` parameter for controlling result ordering.

**Before:** `sort_by="recency"` was passed but silently ignored - results always sorted by semantic relevance.

**After:** Supports three sort modes:
- `sort_by="relevance"` (default) - Quality-weighted semantic similarity (returns limit*2 for cross-collection coverage)
- `sort_by="recency"` - Most recent first by timestamp (returns exactly limit)
- `sort_by="score"` - Highest outcome score first (returns exactly limit)

```python
# Used by cold start to get recent context
recent_context = await mem.search(
    query="",
    collections=["working", "history", "patterns"],
    limit=4,
    sort_by="recency"  # Now actually works!
)
```

### 6. --force Flag Fix for Cursor (cli.py)

The `--force` flag was accepted but never used for Cursor MCP config. Now works correctly.

**Before (broken):**
```python
def configure_cursor(args):
    force = args.force  # Accepted but never checked
    # Always wrote config regardless of existing
```

**After (fixed):**
```python
if existing_config and not force:
    print("Config differs - use --force to overwrite")
    return
```

### 7. Silent Exceptions → Logging (cli.py)

Replaced 3 silent `except: pass` blocks with proper logging.

**Before:**
```python
except:
    pass  # Silent failure
```

**After:**
```python
except Exception as e:
    logger.warning(f"Failed to parse existing settings.json: {e}")
```

### 8. Dead Config Fields Removed (config.py)

Removed 8 config fields that were defined but never used anywhere in the codebase:

- `cross_encoder_blend_ratio`
- `default_importance`
- `default_confidence`
- `promotion_use_threshold`
- `promotion_age_days`
- `working_memory_retention_hours`
- `history_retention_days`
- `cross_encoder_top_k`

### 9. Dependencies Updated (ARCHITECTURE.md)

Fixed outdated dependency versions in documentation:

| Dependency | Old | New |
|------------|-----|-----|
| chromadb | >=0.4.0 | >=1.0.0,<2.0.0 |
| mcp | >=0.1.0 | >=1.0.0 |
| sentence-transformers | >=2.0.0 | >=2.2.0 |
| rank-bm25 | (missing) | >=0.2.0 |
| nltk | (missing) | >=3.8.0 |

### 10. Tool Description Improvements (mcp/server.py)

Enhanced MCP tool descriptions so LLMs understand how to use Roampal properly:

**add_to_memory_bank:**
- Added TAG MEANINGS section explaining each tag (identity, preference, goal, project, system_mastery, agent_growth)
- Added ALWAYS_INJECT guidance (use sparingly, only for core identity)

**get_context_insights:**
- Added PROACTIVE MEMORY reminder: store new facts about user when learned

**search_memory:**
- Added READING RESULTS section explaining output format:
  - `[YYN]` = outcome history (Y=worked, ~=partial, N=failed)
  - `s:0.7` = outcome score (0-1, statistically weighted)
  - `5d` = age of memory
  - `[id:...]` = memory ID for selective scoring

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/server/main.py` | Identity prompt logic, recent context in cold start, skip context on cold start |
| `roampal/mcp/server.py` | Tool description improvements (TAG MEANINGS, PROACTIVE MEMORY, READING RESULTS) |
| `roampal/backend/modules/memory/unified_memory_system.py` | Exclude books from auto-injection, add sort_by parameter |
| `roampal/cli.py` | --force flag fix, silent exception logging |
| `roampal/backend/modules/memory/config.py` | Removed 8 dead fields |
| `ARCHITECTURE.md` | Updated dependencies section |
| `pyproject.toml` | Version 0.2.5 → 0.2.6 |
| `roampal/__init__.py` | Version bump |

---

## Testing

All 240 unit tests pass.

```bash
$ pytest roampal/backend/modules/memory/tests/unit/ -q
240 passed
```

---

## Upgrade Instructions

```bash
pip install --upgrade roampal
# Restart Claude Code to pick up changes
```

---

## For New Users

The identity prompt will fire on first session if you don't have any identity-tagged facts. The LLM will naturally ask for your name and store it with `add_to_memory_bank(content="User's name is X", tags=["identity"])`.

---
