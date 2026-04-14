# v0.4.9.2 Release Notes

**Date:** 2026-04-15
**Type:** Quality improvement
**Platforms:** Claude Code, OpenCode, Cursor

## Summary

Improve MCP tool descriptions to meet Glama TDQS (Tool Definition Quality Score) standards. Currently scoring Quality C (2.12/5.0) — target B (3.0+) or A (3.5+).

## Current Glama Scores

| Tool | Grade | Score | Behavior | Conciseness | Completeness | Parameters | Purpose | Usage Guidelines |
|------|-------|-------|----------|-------------|-------------|------------|---------|-----------------|
| score_memories | D | 1.5 | 1 | 3 | 1 | 1 | 2 | 1 |
| update_memory | D | 1.9 | 2 | 5 | 1 | 1 | 2 | 1 |
| search_memory | C | 2.6 | 2 | 4 | 2 | 3 | 3 | 2 |
| delete_memory | C | 2.8 | 2 | 5 | 2 | 2 | 4 | 2 |
| add_to_memory_bank | C | 2.9 | 2 | 5 | 2 | 3 | 4 | 2 |
| record_response | A | 3.5 | 2 | 5 | 3 | 2 | 4 | 5 |

Server score = 60% mean (2.53) + 40% min (1.5) = **2.12 → C**

## Root Cause Analysis

### Consistent weaknesses (1-2/5 across most tools):

**Behavior (Behavioral Transparency):** Descriptions don't clearly state what changes in the system when the tool runs. Need explicit: "This tool creates/modifies/deletes X. On success returns Y. On failure returns Z."

**Completeness (Contextual Completeness):** Edge cases not documented. What happens with empty input? Duplicate content? Invalid IDs? No match found?

**Usage Guidelines:** Most tools say WHEN TO USE but not clearly enough for an automated evaluator. record_response scores 5/5 because it says "OPTIONAL" upfront and has clear "NOT for X" guidance.

**Parameters:** Missing explicit defaults, format examples, and constraints in the descriptions. The JSON schema has them but the description text doesn't reinforce.

### What record_response (A, 3.5) does right:
- Short, focused description
- "OPTIONAL" immediately signals when NOT to use
- Explicit "NOT for X — use Y instead" 
- Mentions scoring behavior (side effect)
- Concise parameter descriptions

## Dimension Definitions (from Glama)

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Purpose Clarity | 25% | What does the tool do |
| Usage Guidelines | 20% | When to use / not use |
| Behavioral Transparency | 20% | Side effects, state changes, errors |
| Parameter Semantics | 15% | What each param means, constraints, examples |
| Conciseness & Structure | 10% | Not bloated, well-organized |
| Contextual Completeness | 10% | Edge cases, gaps covered |

---

## Proposed Fixes Per Tool

### 1. score_memories (D → target B+)

**Current problems:**
- Purpose 2/5: Opens with jargon ("outcome-based retrieval system", "Wilson confidence scoring")
- Usage Guidelines 1/5: References internal `<roampal-score-required>` hook — meaningless to external evaluator
- Behavior 1/5: SIDE EFFECTS section exists but doesn't describe error states
- Parameters 1/5: `memory_scores` additionalProperties pattern not explained clearly
- Completeness 1/5: No edge case coverage
- Conciseness 3/5: Way too long (~1500 chars)

**Proposed description:**
```
Score memories that were retrieved in your previous context. Each memory gets an outcome rating that adjusts its future retrieval priority.

WHEN TO USE:
- When prompted by the scoring hook (appears as <roampal-score-required> in context)
- Score every memory ID listed in the prompt

WHEN NOT TO USE:
- To store new information (use record_response or add_to_memory_bank)
- Without a scoring prompt — this tool evaluates existing memories, not new ones

BEHAVIOR:
- Updates confidence scores on each scored memory
- Memories repeatedly scored "failed" are automatically demoted or removed
- exchange_summary is stored as a new working memory for future retrieval
- facts (if provided) are stored as separate working memories
- Returns JSON with count of memories scored

ERROR HANDLING:
- Unknown memory IDs are silently skipped
- Missing required field memory_scores returns an error
- Empty memory_scores map is accepted but scores nothing
```

**Proposed parameter descriptions:**
- memory_scores: "Object mapping memory IDs to outcomes. Keys are doc_ids (e.g. 'history_abc123'), values are one of: 'worked' (helped your response), 'partial' (somewhat relevant), 'unknown' (present but unused), 'failed' (misleading — caused incorrect response). Example: {\"history_abc123\": \"worked\", \"patterns_def456\": \"unknown\"}"
- exchange_summary: "1-3 sentence summary of the previous exchange (~300 chars). Captures what happened and the outcome. Stored as a working memory for future retrieval."
- exchange_outcome: "Overall result of the previous exchange. 'worked' = user confirmed or continued. 'failed' = user corrected you. 'partial' = mixed. 'unknown' = unclear."
- noun_tags: "Topic nouns from the exchange for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"react\", \"auth bug\", \"logan\"]"
- facts: "Atomic facts extracted from this exchange. One fact per string, max 150 chars. Include specifics (dates, names, decisions). Example: [\"User prefers snake_case\", \"v2.0 released 2026-04-01\"]"

### 2. update_memory (D → target B+)

**Current problems:**
- Purpose 2/5: Opens okay but MEMORY LIFECYCLE section is off-topic — describes all 5 collections when this tool only touches memory_bank
- Usage Guidelines 1/5: WHEN TO USE section exists but competing with UPDATE vs DELETE section
- Behavior 1/5: MATCHING BEHAVIOR exists but no error state description
- Parameters 1/5: Only 2 params but descriptions lack examples
- Completeness 1/5: No edge cases (what if both old and new match the same thing?)

**Proposed description:**
```
Replace an existing memory_bank fact with updated content. Use when information changes but the topic is still relevant.

WHEN TO USE:
- A stored fact is outdated (e.g., version number changed, project status updated)
- A fact needs correction or more detail
- You gave wrong info because of a stale memory — fix it immediately

WHEN NOT TO USE:
- The fact is completely wrong or irrelevant — use delete_memory instead
- You want to update working/history/patterns — those are managed automatically by scoring
- You want to add a new fact — use add_to_memory_bank instead

BEHAVIOR:
- Finds the closest semantic match to old_content in memory_bank
- Replaces the matched memory's content with new_content, preserving its doc_id and metadata
- If no match is found (cosine similarity too low), nothing is changed
- Returns JSON with status ('updated' or 'not_found') and the matched doc_id

ERROR HANDLING:
- No semantic match found → returns {"status": "not_found"}, no changes made
- old_content does not need to be exact — a close paraphrase works
- Only searches memory_bank collection, not working/history/patterns
```

**Proposed parameter descriptions:**
- old_content: "The existing fact to find and replace. Matched by semantic similarity — a close paraphrase works, exact text is not required. Example: \"User prefers dark mode\""
- new_content: "The corrected or updated fact. Keep concise (~300 chars max). One concept per fact. Example: \"User switched to light mode in April 2026\""

### 3. search_memory (C → target B+)

**Current problems:**
- Purpose 3/5: Good but wordy opening
- Usage Guidelines 2/5: WHEN TO SEARCH / WHEN NOT TO sections exist but too brief
- Behavior 2/5: RETURNS section describes format but not error states
- Parameters 3/5: Good detail but some params lack examples
- Completeness 2/5: READING RESULTS section is internal jargon
- Conciseness 4/5: Long but structured

**Proposed description:**
```
Search persistent memory across all collections. Use when you need details beyond what was automatically provided in context.

WHEN TO USE:
- User references past conversations ("remember", "I told you", "we discussed")
- You need more detail than the injected KNOWN CONTEXT provided
- You want to verify a memory by ID before acting on it
- You want to browse recent memories by time (days_back parameter)

WHEN NOT TO USE:
- General knowledge questions — use your training data
- The injected KNOWN CONTEXT already answers the question
- You want to store something — use add_to_memory_bank or record_response

BEHAVIOR:
- Searches across 5 collections: working (recent), history (scored), patterns (proven), memory_bank (permanent facts), books (documents)
- Omit 'collections' for automatic routing (recommended)
- Returns ranked results with scores, age, and usage metadata
- Empty results mean no relevant memories found — this is normal, not an error

ERROR HANDLING:
- No query, days_back, or id provided → returns validation error
- No matches found → returns empty result list
- Invalid collection name → collection is ignored
```

**Proposed parameter descriptions:**
- query: "Natural language search query. Use the user's exact words — do not simplify or extract keywords. Example: \"auth bug we fixed last week\""
- days_back: "Return memories from the last N days (1-365). Can be used alone or combined with a query for time-filtered search."
- id: "Look up a specific memory by doc_id (e.g. 'patterns_abc123'). Returns full memory with all metadata. Bypasses semantic search."
- collections: "Which collections to search. Omit for automatic routing (recommended). Options: working, history, patterns, memory_bank, books."
- limit: "Number of results to return (1-20, default: 5)."
- metadata: "Optional metadata filters. Example: {\"memory_type\": \"fact\"}"
- sort_by: "Sort order: 'relevance' (default), 'recency' (for temporal queries), 'score' (by confidence)."
- type: "Filter by memory type: 'fact' (atomic facts) or 'summary' (exchange summaries). Omit to search all."

### 4. delete_memory (C → target B+)

**Current problems:**
- Purpose 4/5: Good
- Usage Guidelines 2/5: WHEN TO USE exists but could be tighter
- Behavior 2/5: MATCHING BEHAVIOR section but no error examples
- Parameters 2/5: Single param, minimal description
- Completeness 2/5: Doesn't cover multiple matches scenario clearly

**Proposed description:**
```
Permanently remove a memory_bank fact. This action is irreversible.

WHEN TO USE:
- A fact is completely wrong, stale, or misleading
- A fact is redundant (superseded by a newer memory)
- A memory_bank fact keeps causing incorrect responses — remove it directly

WHEN NOT TO USE:
- The fact's topic is still relevant but details changed — use update_memory instead
- You want to remove working/history/patterns memories — those are managed by the scoring system automatically (score "failed" to demote them)

BEHAVIOR:
- Searches memory_bank by semantic similarity to find the closest match
- Deletes the best-matching memory permanently
- Only operates on memory_bank — cannot delete from other collections
- Returns JSON with status ('deleted' or 'not_found'), matched doc_id, and content preview

ERROR HANDLING:
- No semantic match found → returns {"status": "not_found"}, nothing deleted
- Content parameter is matched by meaning, not exact text — a paraphrase works
- If multiple memories are similar, only the closest match is deleted
```

**Proposed parameter descriptions:**
- content: "Describe the memory to delete in natural language. Matched by semantic similarity — a close paraphrase works. Example: \"User's old email address\""

### 5. add_to_memory_bank (C → target A)

**Current problems:**
- Purpose 4/5: Good
- Conciseness 5/5: Well-structured
- Usage Guidelines 2/5: Good WHAT BELONGS / WHAT DOES NOT sections but verbose
- Behavior 2/5: No explicit state change description
- Completeness 2/5: Edge cases not covered
- Parameters 3/5: Decent but noun_tags description is a wall of text

**Proposed description:**
```
Store a permanent fact for cross-session continuity. Use for identity, preferences, goals, and project context.

WHEN TO USE:
- User shares identity information (name, role, background)
- User states a preference or standing rule
- Important project context that should persist across all sessions
- Knowledge that helps you be more effective for this user

WHEN NOT TO USE:
- Session-specific details or temporary context — let working memory handle it
- Raw conversation content — auto-captured by the scoring system
- Exchange takeaways — use record_response instead (those get outcome-scored)

BEHAVIOR:
- Stores the fact in memory_bank collection with a generated doc_id
- Memory_bank facts are permanent and NOT outcome-scored — they persist until you update or delete them
- If always_inject=true, the fact appears in every context injection (use sparingly — only for core identity)
- Duplicate content is allowed — check with search_memory first to avoid redundancy
- Returns JSON with status ('stored'), assigned doc_id, and collection ('memory_bank')

ERROR HANDLING:
- Empty content → stored but not useful (avoid)
- Missing noun_tags → returns validation error (required field)
- Invalid tag names in tags array are accepted but have no special behavior
```

**Proposed parameter descriptions:**
- content: "The fact to store. Keep concise (~300 chars). One concept per fact. Put the most important info first. Example: \"Logan is a data scientist focused on AI memory systems\""
- tags: "Semantic categories: 'identity' (name, role), 'preference' (workflow, style), 'goal' (objectives), 'project' (codebases, repos), 'system_mastery' (effectiveness tips), 'agent_growth' (meta-learning). Use 'identity' for user profile facts."
- noun_tags: "Topic nouns for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"logan\", \"data science\", \"roampal\"]"
- importance: "How critical this fact is (0.0-1.0, default: 0.7). Use 0.9+ for core identity facts."
- confidence: "How certain you are (0.0-1.0, default: 0.7). Use 0.9+ only for verified facts. Use 0.5 for unconfirmed claims."
- always_inject: "If true, this fact appears in every context injection. Use only for core identity. Default: false."

### 6. record_response (A — maintain)

Already scoring 3.5/5. Only weak on Behavior (2/5) and Parameters (2/5). Minor improvements:

**Add to description:**
```
BEHAVIOR:
- Stores the takeaway as a new working memory with initial score 0.7
- Working memories are automatically scored on subsequent turns: +0.2 worked, +0.05 partial, -0.3 failed
- Over time, useful takeaways promote to history (30d) then patterns (permanent)
- Returns JSON with status ('stored') and the assigned doc_id
```

**Improved parameter descriptions:**
- key_takeaway: "1-2 sentence summary of the important learning. Be specific — include names, decisions, and outcomes. Example: \"User prefers one bundled PR for refactors — splitting would be churn\""
- noun_tags: "Topic nouns for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"react\", \"auth flow\", \"logan\"]"

---

## Projected Impact

If all tools improve to match record_response's pattern (targeting 3.5/5 each):
- Mean: 3.5
- Min: 3.5
- Server: 60%(3.5) + 40%(3.5) = **3.5 → A**

Conservative estimate (3.0/5 average, 2.5 min):
- Server: 60%(3.0) + 40%(2.5) = **2.8 → C** (still C but close to B)

Realistic target (3.2 average, 2.8 min):
- Server: 60%(3.2) + 40%(2.8) = **3.04 → B**

## Files Changed

| File | Change |
|------|--------|
| `roampal/__init__.py` | Version bump 0.4.9.1 -> 0.4.9.2 |
| `pyproject.toml` | Version bump 0.4.9.1 -> 0.4.9.2 |
| `roampal/mcp/server.py` | Rewrite all 6 tool descriptions for TDQS compliance |

## Key Principles for All Descriptions

1. **Lead with one-sentence purpose** (Purpose Clarity)
2. **WHEN TO USE / WHEN NOT TO USE** with specific examples (Usage Guidelines)
3. **BEHAVIOR section** describing state changes explicitly (Behavioral Transparency)
4. **ERROR HANDLING section** with specific failure modes (Completeness)
5. **Parameter descriptions** with types, defaults, constraints, and examples (Parameter Semantics)
6. **Keep total description under 800 chars** where possible (Conciseness)
