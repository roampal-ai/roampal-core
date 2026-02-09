# v0.3.5 Release Notes (PLANNED)

**Status:** Design only — not yet implemented
**Platforms:** Claude Code, Cursor, OpenCode (all clients)
**Features:**
1. Scoring prompt token optimization (Claude Code/Cursor only — OpenCode exempt by design)
2. Standardized memory metadata + search improvements (all clients)
3. Tool description rewrites + system mastery prompt (required — LLM-facing, all clients)
4. Memory hygiene & active memory management prompting (all clients)
5. Security hardening & storage hygiene (all clients)

---

## Feature 1: Scoring Prompt Token Optimization

**Platforms:** Claude Code, Cursor only (NOT OpenCode — see note below)
**Feature:** Eliminate duplicate memory content from scoring prompts

### Problem

Every turn that requires scoring, the scoring prompt re-includes content that's already in the conversation context from the previous turn. This creates compound bloat:

**Turn N (KNOWN CONTEXT injection):**
```
═══ KNOWN CONTEXT ═══
• User prefers direct communication style, concise answers. (2d, patterns)
• Roampal v0.3.4 shipped with OpenCode garbled UI fix. (5h, 92% proven, history)
• Wilson scoring uses success_count and uses from outcome_service. (1d, working)
═══ END CONTEXT ═══
```

**Turn N+1 (scoring prompt injection) — duplicates the same content:**
```
<roampal-score-required>
Score the previous exchange before responding.

Previous:
- User asked: "how does wilson scoring work?"
- You answered: "Wilson scoring uses the success_count and uses fields..."

Memories surfaced:
• [patterns_abc123] "User prefers direct communication style, concise answers..."
• [history_def456] "Roampal v0.3.4 shipped with OpenCode garbled UI fix..."
• [working_ghi789] "Wilson scoring uses success_count and uses from outcome_serv..."

Current user message: "ok cool thanks"

Based on the user's current message, evaluate if your previous answer helped:
...
</roampal-score-required>
```

**What's duplicated:**
1. **Previous exchange** (~200-500 tokens): `user_asked` and `assistant_said` are already in the conversation history from the actual previous turn
2. **Memory content** (~150-300 tokens): 3 memories × 100 chars truncated — content was already in KNOWN CONTEXT from the previous turn
3. **Current user message** (~50-100 tokens): the actual user message follows immediately after the scoring prompt

Both the KNOWN CONTEXT block and the scoring prompt persist in the conversation history. Over a 30-turn session (~20 scored turns), the LLM's context window accumulates ~8K-18K tokens of pure duplication, compounding each turn as all previous injections remain visible.

### Prerequisite: Add doc_ids to KNOWN CONTEXT

**Current format** (`unified_memory_system.py:1005`):
```
• {content} ({age}, {score}, {collection})
```

The LLM sees full content but **no doc_ids**. Without IDs in KNOWN CONTEXT, the scoring prompt can't reference memories by ID only — the LLM wouldn't know which memory is which.

**New format:**
```
• {content} [id:{doc_id}] ({age}, {score}, {collection})
```

Example:
```
═══ KNOWN CONTEXT ═══
• User prefers direct communication style, concise answers. [id:patterns_abc123] (2d, patterns)
• Roampal v0.3.4 shipped with OpenCode garbled UI fix. [id:history_def456] (5h, 92% proven, history)
═══ END CONTEXT ═══
```

This is a ~20 token increase per turn (3 IDs × ~7 tokens each) but enables the scoring prompt to drop ~300-500 tokens of content.

### Fix: Lean Scoring Prompt (Claude Code)

**File:** `roampal/hooks/session_manager.py` — `build_scoring_prompt()` (line 333)

**Current prompt (~600-800 tokens):**
```
<roampal-score-required>
Score the previous exchange before responding.

Previous:
- User asked: "{user_asked}"
- You answered: "{assistant_said}"

Memories surfaced:
• [{doc_id}] "{content[:100]}..."
• [{doc_id}] "{content[:100]}..."

Current user message: "{current_user_message}"

Based on the user's current message, evaluate if your previous answer helped:
- "worked" = user satisfied, says thanks, moves on to new topic
- "failed" = user corrects you, says no/wrong, repeats question
- "partial" = lukewarm response, "kind of", "I guess"
- "unknown" = no clear signal

Score each cached memory individually:
...template with doc_ids...
</roampal-score-required>
```

**New prompt (~200-250 tokens):**
```
<roampal-score-required>
Score the previous exchange before responding.

Look at your previous response and the user's follow-up below.
The memories injected last turn had IDs shown in [id:...] tags in KNOWN CONTEXT.

Score each memory:

Call score_response(
    outcome="worked|failed|partial|unknown",
    memory_scores={
        "{doc_id_1}": "___",
        "{doc_id_2}": "___",
        "{doc_id_3}": "___"
    }
)

SCORING GUIDE:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING

You MUST score every memory listed above.
</roampal-score-required>
```

**What's removed:**
- `user_asked` / `assistant_said` — already visible in conversation history
- `current_user_message` — follows immediately after this prompt
- Memory content — already in KNOWN CONTEXT from previous turn (now with `[id:...]` tags for mapping)
- Outcome definitions for exchange — kept only for memory scoring guide

**What's kept:**
- The `<roampal-score-required>` wrapper — needed for hook detection and tool description matching
- Memory doc_ids — essential for per-memory scoring
- `score_response()` call template — needed to trigger the MCP tool
- Scoring guide — kept concise

### OpenCode: Scoring Reference in System Prompt

OpenCode doesn't have the accumulation problem (system prompt replaced each turn, scoring prompt in cloned user message). But it still has a per-turn duplication problem: `build_scoring_prompt_simple()` includes full memory content (~200 tokens) even though the LLM just saw those memories on the previous turn.

The challenge: the lean scoring prompt can't say "look at KNOWN CONTEXT from previous turn" because OpenCode's system prompt replacement means that context is gone. The LLM would be scoring blind.

**Solution: Scoring Reference block in system prompt.**

On scoring turns, `system.transform` injects both:

```
═══ SCORING REFERENCE (previous turn) ═══
• [id:patterns_abc123] "User prefers direct..."
• [id:history_def456] "v0.3.4 shipped with..."
═══ END SCORING REFERENCE ═══

═══ KNOWN CONTEXT ═══
• Current turn memories here... [id:memory_bank_ghi789]
═══ END CONTEXT ═══
```

The scoring reference is the previous turn's memories — just enough content (30-char hint) for the LLM to score them. The scoring prompt in the cloned user message becomes lean:

```
<roampal-score-required>
Score the memories listed in SCORING REFERENCE above.

Call score_response(
    outcome="worked|failed|partial|unknown",
    memory_scores={
        "patterns_abc123": "___",
        "history_def456": "___"
    }
)
</roampal-score-required>
```

**Why this works:**
- System prompt gets replaced next turn — no accumulation
- Content is in system prompt, not user message — no UI leak (GitHub issue #1)
- Scoring prompt is lean (~80 tokens instead of ~400)
- LLM can see both the scoring reference AND current KNOWN CONTEXT simultaneously

**Server-side requirement:** The `/api/hooks/get-context` response already includes `scoring_memories` data. The plugin's `system.transform` hook needs to format that data as a SCORING REFERENCE block and prepend it to the system prompt on scoring turns. The plugin already knows if it's a scoring turn (it has `scoringRequired` flag from `chat.message`).

**Note:** Moving KNOWN CONTEXT back to user messages is NOT an option — it was moved to system prompt to fix GitHub issue #1 (injected text showing up in the input box due to OpenCode holding direct references to message objects).

### Token Savings Estimate

**Claude Code / Cursor (compound savings — injections persist in conversation):**

| Component | Current (per scored turn) | After fix | Saved |
|-----------|--------------------------|-----------|-------|
| Previous exchange in scoring prompt | ~300 tokens | 0 | ~300 |
| Memory content in scoring prompt | ~200 tokens | 0 | ~200 |
| Current user message in scoring prompt | ~75 tokens | 0 | ~75 |
| Doc_ids added to KNOWN CONTEXT | 0 | +20 tokens | -20 |
| Scoring instructions (simplified) | ~200 tokens | ~100 tokens | ~100 |
| **Net per scored turn** | **~775** | **~120** | **~655** |

Over a 30-turn session with ~20 scored turns:
- Direct savings: ~13K tokens
- Context window savings (compound — all past injections persist): ~30K-40K tokens

**OpenCode (per-turn savings — system prompt replaced, no compounding):**

| Component | Current (per scored turn) | After fix | Saved |
|-----------|--------------------------|-----------|-------|
| Memory content in scoring prompt (user msg) | ~200 tokens | 0 | ~200 |
| Scoring reference in system prompt | 0 | ~80 tokens | -80 |
| Scoring instructions (simplified) | ~200 tokens | ~80 tokens | ~120 |
| **Net per scored turn** | **~400** | **~160** | **~240** |

Over a 30-turn session with ~20 scored turns:
- Direct savings: ~4.8K tokens (no compounding since system prompt resets)
- Leaner than current approach but less dramatic than Claude Code savings

### Files to Change

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py` | Add `[id:{doc_id}]` to KNOWN CONTEXT format at line 1005 |
| `roampal/hooks/session_manager.py` | Rewrite `build_scoring_prompt()` (line 333) — remove exchange text and memory content, keep IDs only |
| `roampal/hooks/session_manager.py` | Rewrite `build_scoring_prompt_simple()` (line 424) — lean version, IDs only (no longer exempt) |
| `roampal/plugins/opencode/roampal.ts` | `system.transform`: inject SCORING REFERENCE block from `scoring_memories` data on scoring turns |
| `roampal/server/main.py` | Ensure `scoring_memories` in `/api/hooks/get-context` response includes 30-char content hints alongside IDs |

### Testing Plan

- Unit test: verify KNOWN CONTEXT output includes `[id:doc_id]` tags
- Unit test: verify new `build_scoring_prompt()` does NOT contain `user_asked` or `assistant_said`
- Unit test: verify new `build_scoring_prompt()` contains only doc_ids, no content
- Unit test: verify new `build_scoring_prompt_simple()` contains only doc_ids, no content
- Live test (Claude Code): run a 5-turn session, confirm the LLM can still score memories correctly using IDs from KNOWN CONTEXT
- Live test (OpenCode): run a 5-turn session, confirm the LLM can score using SCORING REFERENCE block in system prompt
- Token count comparison: log token counts before/after for a 10-turn session on both Claude Code and OpenCode

### Risk: LLM Fails to Map IDs

The biggest risk: the LLM might not map `[id:patterns_abc123]` from KNOWN CONTEXT (Claude Code) or SCORING REFERENCE (OpenCode) to the doc_id in the scoring prompt. If the LLM can't make this connection, it will either skip scoring or hallucinate scores.

**Mitigation:**
1. The `[id:doc_id]` tag format is explicit and machine-readable
2. Claude Code: scoring prompt says "The memories injected last turn had IDs shown in [id:...] tags in KNOWN CONTEXT"
3. OpenCode: SCORING REFERENCE block is in the same system prompt as the scoring turn — LLM sees content and IDs simultaneously, no cross-turn mapping needed
4. If testing shows LLMs struggle, we can increase content hint from 30 to 60 chars — still much leaner than full content

---

## Feature 2: Standardized Memory Metadata + Search Improvements

**Platforms:** All clients
**Feature:** Every memory retrieval returns a consistent shape with full scoring data; search gains `days_back` and `id` filter

### Problem

Memory retrieval is inconsistent across paths:

1. **Content stored in 4 places:** `result["content"]`, `result["text"]`, `metadata["content"]`, `metadata["text"]` — display code uses a 4-way fallback chain
2. **Timestamp field varies by collection:** working uses `created_at`, history uses `timestamp`, memory_bank uses `created_at` — sorting breaks when the wrong field is checked
3. **`_search_all()` returns bare-bones data:** When searching without a semantic query (empty query), results have no Wilson score, no learned_score, no ranking data — `apply_scoring_to_results()` is never called
4. **No way to search purely by time:** `search_memory` requires a text query (`server.py:473`). You can't do "show me everything from the last 3 days" without also providing semantic text
5. **No way to look up a memory by ID:** If the LLM sees `[id:patterns_abc123]` in KNOWN CONTEXT and wants full details, there's no direct lookup — it has to do a semantic search and hope the memory appears

### Standardized Memory Shape

A new `normalize_memory(result, collection)` function runs on every retrieval path. After normalization, every memory has this shape:

```python
{
    # === Identity ===
    "id": "patterns_abc123",           # Always present, always at root
    "collection": "patterns",          # Always present, always at root
    "content": "Full memory text...",   # ONE canonical location — no duplicates

    # === Time ===
    "created_at": "2025-02-07T15:42:00",  # Standardized — always this field name
    "age": "2d",                           # Relative age, always computed

    # === Outcome Scoring (working, history, patterns) ===
    "score": 0.72,                     # Raw outcome score (0-1)
    "wilson_score": 0.68,              # Statistical confidence (0-1), default 0.5 if unscored
    "uses": 8,                         # Times surfaced and scored
    "success_count": 6.0,             # Cumulative weighted successes
    "learned_score": 0.71,            # Outcome-based learning score
    "last_outcome": "worked",          # Most recent outcome
    "outcome_history": "[YY~N]",       # Last 3 outcomes formatted

    # === Memory Bank Only ===
    "tags": ["preference", "goal"],    # Parsed list (empty [] for non-memory_bank)
    "importance": 0.9,                 # memory_bank only (absent on other collections)
    "confidence": 0.8,                 # memory_bank only (absent on other collections)

    # === Search Context (only when from semantic search) ===
    "embedding_similarity": 0.82,      # Present when semantic search was used
    "final_rank_score": 0.76,          # Present when full ranking pipeline ran
    "effectiveness": 0.85,             # Doc effectiveness if tracked

    # === Raw metadata preserved ===
    "metadata": { ... }                # Original ChromaDB metadata, untouched
}
```

**Key decisions based on feedback:**
- **No `created_at_human`** — the ISO timestamp + `age` field is enough. The LLM can interpret ISO dates; this data is for the LLM, not for human display.
- **`importance` and `confidence` only on memory_bank** — other collections use Wilson scoring for quality signals. No fake defaults on collections where they don't apply.
- **`tags` always present** — empty list `[]` for non-memory_bank collections. Prevents KeyError without adding fake data.

### Normalization Implementation

**File:** `roampal/backend/modules/memory/unified_memory_system.py`

New function called at every retrieval exit point:

```python
def normalize_memory(result: Dict, collection: str = None) -> Dict:
    """
    Standardize memory shape across all retrieval paths.
    Called after every search, get, and injection retrieval.
    """
    metadata = result.get("metadata", {})

    # === Content: ONE canonical location ===
    content = (
        result.get("content")
        or metadata.get("content")
        or metadata.get("text")
        or result.get("text", "")
    )
    result["content"] = content
    # Remove duplicates
    result.pop("text", None)

    # === Collection ===
    if not collection:
        collection = result.get("collection") or ""
        # Infer from ID prefix if missing
        if not collection and result.get("id"):
            for prefix in ["working", "history", "patterns", "memory_bank", "books"]:
                if result["id"].startswith(prefix):
                    collection = prefix
                    break
    result["collection"] = collection

    # === Timestamps: standardize to created_at ===
    created_at = (
        metadata.get("created_at")
        or metadata.get("timestamp")
        or ""
    )
    result["created_at"] = created_at
    metadata["created_at"] = created_at  # Write back so _apply_date_filters() sees it (Desktop direct-store path only has 'timestamp')
    result["age"] = _humanize_age(created_at)

    # === Outcome scoring (always present, defaults for unscored) ===
    result["score"] = float(metadata.get("score", 0.5))
    result["uses"] = int(metadata.get("uses", 0))
    result["success_count"] = float(metadata.get("success_count", 0.0))
    result["last_outcome"] = metadata.get("last_outcome", "")
    result["outcome_history"] = _format_outcomes(metadata.get("outcome_history", ""))

    # Wilson + learned scores: compute if not already present
    if "wilson_score" not in result:
        # Use scoring_service.calculate_learned_score if available,
        # otherwise simple fallback
        uses = result["uses"]
        success_count = result["success_count"]
        if uses > 0:
            result["wilson_score"] = wilson_score_lower(success_count, uses)
        else:
            result["wilson_score"] = 0.5  # Untested default

    if "learned_score" not in result:
        result["learned_score"] = result.get("score", 0.5)

    # === Memory bank specific ===
    tags_raw = metadata.get("tags", "[]")
    if isinstance(tags_raw, str):
        try:
            result["tags"] = json.loads(tags_raw)
        except json.JSONDecodeError:
            result["tags"] = []
    elif isinstance(tags_raw, list):
        result["tags"] = tags_raw
    else:
        result["tags"] = []

    # importance/confidence: only on memory_bank
    if collection == "memory_bank":
        result["importance"] = float(metadata.get("importance", 0.7))
        result["confidence"] = float(metadata.get("confidence", 0.7))

    # Preserve raw metadata
    result["metadata"] = metadata

    return result
```

**Where it gets called:**
- `SearchService._search_all()` — after building results, before returning
- `SearchService._search_collections()` — after collection boost, before returning
- `UnifiedMemorySystem.get_context_for_injection()` — on each memory before building KNOWN CONTEXT
- `UnifiedMemorySystem.get_by_id()` — single doc lookup (line 707)
- MCP `search_memory` handler — already gets normalized results from above
- MCP `get_context_insights` handler — already gets normalized results from above

### `_search_all()` Gets Full Scoring Pipeline + Date Filters

**File:** `roampal/backend/modules/memory/search_service.py` — `_search_all()` (line 276)

**Current problems (3 bugs):**

1. **No scoring:** `apply_scoring_to_results()` is never called — results lack Wilson/learned_score
2. **Sort uses wrong field:** Line 311 sorts by `metadata.timestamp`, but working/memory_bank/books only have `created_at` — those memories sort with empty strings, pushed to the end
3. **Date filters bypassed:** `search()` at line 206-209 returns early from `_search_all()` BEFORE date filter extraction (line 234) and application (line 245-246). So `days_back` with no query would return unfiltered results.

```python
# Current — line 311: only checks 'timestamp', misses 'created_at'
all_results.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
```

**Fix requires two changes:**

**Change 1:** `_search_all()` accepts new params and handles them:
```python
async def _search_all(
    self,
    collections: List[str],
    limit: int,
    offset: int,
    return_metadata: bool,
    metadata_filters: Optional[Dict] = None,  # NEW
    sort_by: Optional[str] = None              # NEW
) -> Union[List[Dict], Dict]:
    # ... fetch all items (existing code) ...

    # Normalize all results
    all_results = [normalize_memory(r, r.get("collection", "")) for r in all_results]

    # Apply date filters (Python post-filtering — same as normal search path)
    if metadata_filters:
        _, date_filters = self._extract_date_filters(metadata_filters)
        if date_filters:
            all_results = self._apply_date_filters(all_results, date_filters)

    # Apply scoring pipeline (Wilson, learned_score, final_rank_score)
    all_results = self.scoring_service.apply_scoring_to_results(all_results, sort=False)

    # Sort by requested order (default: recency for no-query searches)
    if sort_by == "recency" or sort_by is None:
        all_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "score":
        all_results.sort(key=lambda x: x.get("score", 0.5), reverse=True)
    elif sort_by == "relevance":
        all_results.sort(key=lambda x: x.get("final_rank_score", 0.0), reverse=True)
```

**Note on date filter field:** `_apply_date_filters()` reads from `r.get('metadata', {}).get(field, '')`. After normalization, `created_at` is at root level AND still in metadata (normalize doesn't remove from metadata). The filter on `created_at` field works because all collections have `created_at` in their raw metadata:
- working: set in `store_working()` (line 1411)
- history/patterns: inherited from working through promotion
- memory_bank: set in `store_memory_bank()`
- books: set in `store_book()`

**Change 2:** `search()` passes filters and sort_by through to `_search_all()`:
```python
# Line 206-209 — pass metadata_filters and sort_by through
if not query or query.strip() == "":
    return await self._search_all(
        collections, limit, offset, return_metadata,
        metadata_filters=metadata_filters,  # NEW
        sort_by=sort_by                      # NEW (threaded from UnifiedMemorySystem)
    )
```

**Threading `sort_by`:** Currently, `sort_by` is handled in `UnifiedMemorySystem.search()` (line 482-495) AFTER `SearchService.search()` returns. Neither `SearchService.search()` nor `_search_all()` accepts `sort_by`. The fix threads it through:
- `UnifiedMemorySystem.search()` → passes `sort_by` to `SearchService.search()`
- `SearchService.search()` → passes `sort_by` to `_search_all()` (for empty-query path)
- For non-empty queries, `sort_by` can remain in `UnifiedMemorySystem.search()` since scoring and results flow back normally

### New: `days_back` Parameter on `search_memory`

**File:** `roampal/mcp/server.py` — search_memory tool schema (line 441)

Add to inputSchema:
```python
"days_back": {
    "type": "integer",
    "description": "Only return memories from the last N days. Can be used alone (no query needed) or combined with a semantic query.",
    "minimum": 1,
    "maximum": 365
}
```

Change `required` from `["query"]` to `[]` (empty — both query and days_back are optional, but at least one must be provided).

**Server handling** (`main.py` — `/api/search` endpoint):

```python
days_back = request.days_back  # New field on SearchRequest model
if days_back:
    # Compute cutoff date
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
    # Merge into metadata_filters
    if not metadata_filters:
        metadata_filters = {}
    metadata_filters["created_at"] = {"$gte": cutoff}
```

This piggybacks on the existing date filter infrastructure in `SearchService._extract_date_filters()` (line 84) and `_apply_date_filters()` (line 111). `_apply_date_filters()` does Python-side string comparison on ISO dates — works correctly since ISO format sorts lexicographically.

**`SearchRequest` model change** (`main.py` line 337):

```python
class SearchRequest(BaseModel):
    query: Optional[str] = ""   # Was: query: str (required) — now optional for days_back/id searches
    days_back: Optional[int] = None  # NEW
    id: Optional[str] = None         # NEW
    conversation_id: Optional[str] = None
    collections: Optional[List[str]] = None
    limit: int = 10
    metadata_filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
```

**Validation:** If neither `query` nor `days_back` is provided, return an error: `"Provide at least one of: query, days_back"`.

**Examples:**
```
search_memory(days_back=7)                           → all memories from last 7 days, sorted by recency
search_memory(days_back=3, sort_by="score")           → last 3 days, best-scored first
search_memory(query="wilson", days_back=14)           → semantic search filtered to last 14 days
search_memory(days_back=1, collections=["working"])   → today's working memory only
```

### New: `id` Filter on `search_memory`

Instead of a separate MCP tool, add `id` as a filter parameter on the existing `search_memory`:

**File:** `roampal/mcp/server.py` — search_memory tool schema

Add to inputSchema:
```python
"id": {
    "type": "string",
    "description": "Look up a specific memory by its doc_id (e.g., 'patterns_abc123'). Returns the full memory with all metadata. Bypasses semantic search."
}
```

**Server handling:**

When `id` is provided, skip search entirely:
```python
if request.id:
    doc = _memory.get_by_id(request.id)
    if doc:
        normalized = normalize_memory(doc)
        return [normalized]
    return []  # Not found
```

`UnifiedMemorySystem.get_by_id()` already exists (line 707) — it iterates collections and calls `adapter.get_fragment(doc_id)`. Just needs to run `normalize_memory()` on the result.

**Validation:** When `id` is provided, `query` and `days_back` are ignored (direct lookup takes priority).

### Timestamp Standardization

**The problem:** Different collections use different field names for creation time.

| Collection | Current field | Actual behavior |
|-----------|--------------|-----------------|
| working | `created_at` | Set in `store_working()` |
| history | `created_at` (promoted) or `timestamp` (direct store) | Promoted from working: inherits `created_at`. Direct Desktop store (line 610): uses `timestamp`. |
| patterns | `created_at` (promoted) or `timestamp` (direct store) | Same as history — depends on storage path. |
| memory_bank | `created_at` | Set in `add_to_memory_bank()` |
| books | `created_at` | Set in `store_book()` line 1198 |

**Fix:** `normalize_memory()` handles this with the fallback chain:
```python
created_at = metadata.get("created_at") or metadata.get("timestamp") or ""
```

No migration needed — existing data stays as-is. The normalization layer absorbs the inconsistency. All sorting and filtering uses the normalized `created_at` field.

### Content Field Cleanup

**The problem:** Content is stored redundantly in up to 4 locations:
- `result["content"]` — from ChromaDB document field
- `result["text"]` — alias added in some paths
- `metadata["content"]` — stored in ChromaDB metadata
- `metadata["text"]` — also stored in metadata

**Fix:** `normalize_memory()` resolves content with one fallback chain and stores it ONLY in `result["content"]`. The `text` key at root level is removed. Metadata fields are left as-is (ChromaDB owns those), but consumers only need to check `result["content"]`.

### Display Format Update

**File:** `roampal/mcp/server.py` — search_memory display (line 698-729)

**Current format:**
```
1. [patterns] (2d, s:0.7, [YYN]) [id:patterns_abc123] Memory content here
```

**New format (richer, using normalized fields):**
```
1. [patterns] (2d, s:0.72, w:0.68, 8 uses, [YY~N]) [id:patterns_abc123] Memory content here
```

Added: Wilson score (`w:0.68`), use count (`8 uses`). These were already computed but not displayed.

For memory_bank items:
```
1. [memory_bank] (5d, imp:0.9, conf:0.8) [id:memory_bank_abc123] Memory content here
```

Shows importance/confidence as the primary ranking signals. memory_bank is NOT outcome-scored (score stays at 1.0), but the ranking algorithm blends in Wilson as a secondary signal after 3 uses (scoring_service.py:280-287: 80% quality + 20% Wilson).

### Files to Change

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/unified_memory_system.py` | Add `normalize_memory()` function. Call it in `get_context_for_injection()`, `get_by_id()` (line 707), and `search()`. Add `[id:{doc_id}]` to KNOWN CONTEXT format (line 1005). Pass `sort_by` through to `SearchService.search()`. |
| `roampal/backend/modules/memory/search_service.py` | `_search_all()`: accept `metadata_filters` and `sort_by` params, call `normalize_memory()` + `_apply_date_filters()` + `apply_scoring_to_results()`, fix sort to use `created_at` instead of `timestamp`. `search()`: pass `metadata_filters` and `sort_by` through to `_search_all()` at line 207-209. |
| `roampal/mcp/server.py` | Make `query` optional, add `days_back` param, add `id` param. Update display format. Update `required` from `["query"]` to `[]`. |
| `roampal/server/main.py` | Change `SearchRequest.query` from `str` to `Optional[str] = ""`. Add `days_back: Optional[int]` and `id: Optional[str]` to `SearchRequest`. Handle `days_back` → date filter conversion. Handle `id` → `get_by_id()` direct lookup. |
| `roampal/__init__.py` | Version bump → 0.3.5 |
| `pyproject.toml` | Version bump → 0.3.5 |

### Testing Plan

- Unit test: `normalize_memory()` with working memory input → verify all fields present, `text` key removed
- Unit test: `normalize_memory()` with memory_bank input → verify `importance`/`confidence` present, `tags` is parsed list
- Unit test: `normalize_memory()` with bare `_search_all` result → verify Wilson score computed
- Unit test: `_search_all()` with `sort_by="score"` → verify results sorted by score not timestamp
- Unit test: `search_memory(days_back=3)` with no query → verify returns results, no error
- Unit test: `search_memory(id="patterns_abc123")` → verify direct lookup, full normalized shape
- Unit test: `search_memory()` with neither query nor days_back → verify error message
- Unit test: timestamp fallback — memory with only `timestamp` field → verify `created_at` populated
- Live test: call `search_memory(days_back=7)` → verify timeline of recent work with full scoring data
- Live test: see `[id:xxx]` in KNOWN CONTEXT → call `search_memory(id="xxx")` → verify full memory returned

---

## Feature 3: Tool Description Rewrites (LLM-Facing)

**Platforms:** All clients (Claude Code, Cursor, OpenCode)
**Feature:** Rewrite MCP tool descriptions so LLMs understand and use the new search capabilities and scoring workflow

### Why This Is Required (Not Optional)

Tool descriptions are the **only persistent instruction surface** for an LLM. When a new session starts, the LLM has zero conversation history — the only thing it knows about Roampal's capabilities comes from the tool descriptions registered via MCP. If `days_back`, `id` lookup, and the new scoring workflow aren't documented in tool descriptions, no model will ever use them.

The KNOWN CONTEXT injection is only visible during an active session after the first hook fires. A fresh session starts cold — tool descriptions are all the LLM has.

### System Mastery Prompt

**Problem:** The current tool descriptions are functional checklists — "search_memory does X, params are Y." There's no unified mental model of the memory system. A fresh LLM sees 7 disconnected tools and a one-liner preamble ("You have persistent memory"). It doesn't understand the lifecycle, the scoring philosophy, or how to use the system with mastery.

**Solution:** Two changes:

#### 1. `get_context_insights` Description — Add System Overview

This is the entry point to the workflow, registered once via MCP at session start. Best place for the orientation since it's free (no per-turn cost). The description currently opens with "Search your memory before responding." Replace with a full system overview:

```
You have outcome-based persistent memory that learns across sessions.

HOW IT WORKS:
Your memories are scored every time they surface. Scores drive the lifecycle:
• working (new, 24h) → score ≥0.7 + 2 uses → history (30d) → score ≥0.9 + 5 proven successes → patterns (permanent)
• Score drops below 0.4 → demoted. Below 0.2 → deleted.
• memory_bank: permanent facts you store manually (not outcome-scored — ranked by importance × confidence, with Wilson weighting blended into ranking after 3 uses)
• books: reference documents (not scored, permanent)

READING RESULTS:
• w:0.68 = Wilson confidence (statistical lower bound — higher = more proven useful)
• s:0.7 = outcome score (raw success rate, 0-1)
• 8 uses = times surfaced and scored (more = more battle-tested)
• [YYN] = last 3 outcomes (Y=worked, ~=partial, N=failed)
• 5d = age
• [id:patterns_abc123] = memory ID — look up with search_memory(id=...)

SEARCHING:
• Semantic: search_memory(query="auth bug") — find by meaning
• Temporal: search_memory(days_back=7) — recent memories, no query needed
• Direct: search_memory(id="patterns_abc") — look up by ID
• Combined: search_memory(query="auth", days_back=14) — meaning + time filter

YOUR ROLE:
Score honestly — your scores determine which memories survive and improve.
Use add_to_memory_bank() SELECTIVELY for permanent facts only.
Use record_response() for significant learnings worth preserving.

WORKFLOW:
1. get_context_insights(query) ← YOU ARE HERE
2. Read context — memories include [id:...] tags for reference
3. search_memory() if you need more details (use id= for direct lookup)
4. Respond to user
5. record_response() for significant learnings

PROACTIVE MEMORY: If you learn something NEW about the user during the conversation
(name, preference, goal, project context), use add_to_memory_bank() to store it.
Don't wait to be asked - good assistants remember what matters.

SUPPORT INFO (only mention if user asks about support/pricing/contributing):
- Roampal Core is completely free and open source (Apache 2.0)
- Support development: https://roampal.gumroad.com/l/aagzxv
- Feature ideas/feedback: Discord at https://discord.com/invite/F87za86R3v
```

**Verification against code:**

| Claim | Source | Verified |
|-------|--------|----------|
| working → history: score ≥0.7, uses ≥2 | promotion_service.py:104, config.py:28 | ✓ |
| history → patterns: score ≥0.9, uses ≥3, success_count ≥5 | promotion_service.py:111, config.py:25 | ✓ |
| Demotion: score < 0.4 | promotion_service.py:116, config.py:31 | ✓ |
| Deletion: score < 0.2 | promotion_service.py:120, config.py:34 | ✓ |
| memory_bank: not outcome-scored, ranking uses importance × confidence + 20% Wilson after 3 uses | scoring_service.py:280-287, memory_bank_service.py:107 ("score: 1.0, Never decays") | ✓ |
| books: not scored | outcome_service.py:104 (explicit skip) | ✓ |
| Wilson = statistical lower bound | scoring_service.py:163-164 | ✓ |
| Scores drive promotion/demotion/deletion | promotion_service.py:69-123 | ✓ |

#### 2. KNOWN CONTEXT Preamble — Expand Slightly

**File:** `roampal/backend/modules/memory/unified_memory_system.py` — line 943

**Current (~30 tokens):**
```
You have persistent memory about this user via Roampal. The context below was retrieved from past conversations. If the user references past interactions or asks if you remember them, use this context — you DO remember.
```

**New (~55 tokens):**
```
You have persistent memory about this user via Roampal. Scored memories include Wilson confidence and outcome history. [id:...] tags can be looked up with search_memory(id=...). The context below was retrieved from past conversations. If the user references past interactions or asks if you remember them, use this context — you DO remember.
```

Added ~25 tokens per turn. Reinforces the system model (outcome-scored, Wilson, ID lookup) without bloating. The tool description has the full explanation; this is just the per-turn reminder.

### Tool 1: `search_memory` Description Rewrite

**File:** `roampal/mcp/server.py` — line 420-474

**Current description** (~340 chars):
```
Search your persistent memory. Use when you need details beyond what get_context_insights returned.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• get_context_insights recommended a collection → search that collection
• You need more detail than the context provided

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• get_context_insights already gave you the answer

Collections: working (24h then auto-promotes), history (30d scored), patterns (permanent scored), memory_bank (permanent), books (permanent docs)
Omit 'collections' parameter for auto-routing (recommended).

READING RESULTS:
• [YYN] = outcome history (last 3: Y=worked, ~=partial, N=failed)
• s:0.7 = outcome score (0-1, higher = more successful outcomes, statistically weighted)
• 5d = age of memory
• [id:patterns_abc123] = memory ID for selective scoring (use with related=["id1","id2"])
```

**Problems with current:**
1. No mention of `days_back` — LLM can't discover temporal search
2. No mention of `id` lookup — LLM can't look up memories by ID
3. `READING RESULTS` is stale — references `related=["id1","id2"]` which doesn't exist as a parameter
4. Missing Wilson score (`w:`) and use count from new display format
5. `query` is still marked `required` in schema — needs to become optional
6. `metadata` description references `timestamp=` — should use `created_at=` after normalization

**New description:**
```
Search your persistent memory. Use when you need details beyond what get_context_insights returned.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• get_context_insights recommended a collection → search that collection
• You need more detail than the context provided
• You see [id:...] in KNOWN CONTEXT and want full details → use id= parameter

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• get_context_insights already gave you the answer

Collections: working (24h then auto-promotes), history (30d scored), patterns (permanent scored), memory_bank (permanent), books (permanent docs)
Omit 'collections' parameter for auto-routing (recommended).

TEMPORAL SEARCH:
Use days_back=N to search by time without a semantic query.
Examples: days_back=7 → last 7 days. days_back=1, collections=["working"] → today's working memory.
Combine with query for time-filtered semantic search: query="auth", days_back=14.

ID LOOKUP:
Use id="patterns_abc123" to look up a specific memory directly (bypasses search).
Useful when you see [id:...] tags in KNOWN CONTEXT and want full metadata.

READING RESULTS:
• [YYN] = outcome history (last 3: Y=worked, ~=partial, N=failed)
• s:0.7 = outcome score (0-1, higher = more successful outcomes)
• w:0.68 = Wilson confidence score (statistical lower bound, 0-1)
• 8 uses = times this memory has been surfaced and scored
• 5d = age of memory
• [id:patterns_abc123] = memory ID for lookup or scoring
```

**Schema changes:**
- `required`: `["query"]` → `[]` (empty — at least one of query, days_back, or id must be provided)
- Add `days_back` parameter: `{"type": "integer", "description": "Only return memories from the last N days. Can be used alone or with query.", "minimum": 1, "maximum": 365}`
- Add `id` parameter: `{"type": "string", "description": "Look up a specific memory by doc_id (e.g. 'patterns_abc123'). Bypasses search."}`
- Update `metadata` description: `timestamp=` → `created_at=`

### Tool 2: `score_response` Description Rewrite

**File:** `roampal/mcp/server.py` — line 551-593

**Current description** (~530 chars):
```
Score the previous exchange. ONLY use when the <roampal-score-required> hook prompt appears.

⚠️ IMPORTANT: This tool is ONLY for scoring when prompted by the hook. Do NOT call it at other times.
For storing important learnings at any time, use record_response(key_takeaway="...") instead.

EXCHANGE OUTCOME (read user's reaction):
✓ worked = user satisfied, says thanks, moves on
✗ failed = user corrects you, says "no", "that's wrong", provides the right answer
~ partial = user says "kind of" or takes some but not all of your answer
? unknown = no clear signal from user

PER-MEMORY SCORING (v0.2.8):
You MUST score each cached memory individually in memory_scores:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING

You MAY add scores for any other memory in your context (KNOWN CONTEXT, earlier conversation, searched memories).

⚠️ CRITICAL - "failed" for memories means MISLEADING, not just unused.
Only mark a memory as "failed" if it gave bad advice that led you astray.
If you didn't use a memory, mark it "unknown" not "failed".
```

**Problems with current:**
1. Doesn't explain where to find memory IDs — the new lean scoring prompt only includes IDs, not content
2. Doesn't reference KNOWN CONTEXT `[id:...]` tags — the LLM needs to know to look there for the content-to-ID mapping
3. Still references "cached memories" without explaining what that means in the new flow

**New description:**
```
Score the previous exchange. ONLY use when the <roampal-score-required> hook prompt appears.

⚠️ IMPORTANT: This tool is ONLY for scoring when prompted by the hook. Do NOT call it at other times.
For storing important learnings at any time, use record_response(key_takeaway="...") instead.

FINDING MEMORY IDs:
The scoring prompt lists memory IDs to score. These IDs correspond to [id:...] tags
in the KNOWN CONTEXT block from the previous turn. Look at KNOWN CONTEXT to see what
each memory contained, then score based on whether it helped your response.

EXCHANGE OUTCOME (read user's reaction):
✓ worked = user satisfied, says thanks, moves on
✗ failed = user corrects you, says "no", "that's wrong", provides the right answer
~ partial = user says "kind of" or takes some but not all of your answer
? unknown = no clear signal from user

PER-MEMORY SCORING:
You MUST score each memory ID listed in the scoring prompt:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING (gave bad advice that led you astray)

You MAY also score any other memory visible in KNOWN CONTEXT or earlier conversation.

⚠️ CRITICAL - "failed" means MISLEADING, not just unused.
If you didn't use a memory, mark it "unknown" not "failed".
```

**Key changes:**
- Added `FINDING MEMORY IDs` section — explains the KNOWN CONTEXT → ID mapping
- Removed "cached memories" phrasing — replaced with "memory IDs listed in the scoring prompt"
- Added explicit note: "Look at KNOWN CONTEXT to see what each memory contained"

### Tool 3: `get_context_insights` Description Update

**File:** `roampal/mcp/server.py` — line 620-651

**Current description** (~400 chars):
```
Search your memory before responding. Returns what you know about this user/topic.

WORKFLOW (follow these steps):
1. get_context_insights(query) ← YOU ARE HERE
2. Read the context returned
3. search_memory() if you need more details
4. Respond to user
5. record_response() to complete

Returns: Known facts, past solutions, recommended collections, tool stats.
Uses semantic search across your memory collections.

PROACTIVE MEMORY: If you learn something NEW about the user during the conversation
(name, preference, goal, project context), use add_to_memory_bank() to store it.
Don't wait to be asked - good assistants remember what matters.

SUPPORT INFO (only mention if user asks about support/pricing/contributing):
...
```

**Problems with current:**
1. Doesn't mention that returned context includes `[id:doc_id]` tags
2. Workflow step 3 doesn't mention `id=` lookup for direct memory access

**New description:**
```
Search your memory before responding. Returns what you know about this user/topic.

WORKFLOW (follow these steps):
1. get_context_insights(query) ← YOU ARE HERE
2. Read the context returned — memories include [id:...] tags for reference
3. search_memory() if you need more details (use id= to look up a specific memory)
4. Respond to user
5. record_response() to complete

Returns: Known facts, past solutions, recommended collections, tool stats.
Uses semantic search across your memory collections.

PROACTIVE MEMORY: If you learn something NEW about the user during the conversation
(name, preference, goal, project context), use add_to_memory_bank() to store it.
Don't wait to be asked - good assistants remember what matters.

SUPPORT INFO (only mention if user asks about support/pricing/contributing):
- Roampal Core is completely free and open source (Apache 2.0)
- Support development: https://roampal.gumroad.com/l/aagzxv
- Feature ideas/feedback: Discord at https://discord.com/invite/F87za86R3v
```

**Key changes:**
- Step 2: added note about `[id:...]` tags
- Step 3: added `id=` lookup hint

### Bug Fix: `_humanize_age` Timestamp Fallback

**File:** `roampal/mcp/server.py` — line 709

**Current:**
```python
age = metadata.get("created_at", "")
```

Only checks `created_at`. History and patterns collections use `timestamp`, so their age displays as empty in search results.

**Fix:** After `normalize_memory()` is in place, this becomes a non-issue — all results will have `created_at` at the root level. The display code should read from `result.get("created_at")` instead of digging into metadata. But as a belt-and-suspenders fix, the `_humanize_age` helper (line 63) should also accept the normalized field:

```python
# In display formatting, after normalize_memory:
age = result.get("age", "")  # Already computed by normalize_memory
```

This is technically part of Feature 2 (normalization), but noted here because the display code in `server.py` is the consumer.

### Files to Change

| File | Change |
|------|--------|
| `roampal/mcp/server.py` | Rewrite `get_context_insights` description (line 622-641) — full system mastery prompt with lifecycle, scoring guide, search modes. |
| `roampal/mcp/server.py` | Rewrite `search_memory` description (line 422-440). Add `days_back` and `id` to schema. Change `required` to `[]`. Update `metadata` description. |
| `roampal/mcp/server.py` | Rewrite `score_response` description (line 552-574). Add FINDING MEMORY IDs section. |
| `roampal/mcp/server.py` | Fix display format code (line 698-729) to use normalized `result.get("age")` and show Wilson/uses. |
| `roampal/backend/modules/memory/unified_memory_system.py` | Expand KNOWN CONTEXT preamble (line 943) — add outcome-scoring and `[id:...]` lookup note. |

### Testing Plan

- Verify tool descriptions are under MCP character limits (if any)
- Live test (Claude Code): fresh session with no conversation history — confirm the LLM can discover and use `days_back` on its own when asked "what have I been working on this week?"
- Live test (Claude Code): confirm the LLM uses `id=` lookup when it sees an `[id:...]` tag and wants details
- Live test (OpenCode): same tests — confirm tool descriptions surface correctly
- Live test (fresh session): confirm a model with 0 prior Roampal knowledge understands the lifecycle, scores memories properly, and uses search filters appropriately based on the mastery prompt alone
- Live test: confirm the LLM can score using ID-only scoring prompts by referencing KNOWN CONTEXT

---

## Feature 4: Memory Hygiene & Active Memory Management Prompting

**Platforms:** All clients (Claude Code, Cursor, OpenCode)
**Feature:** Prompting additions that shift the LLM from passive memory consumer to active memory maintainer

### Problem

Features 1-3 give the LLM better tools and better data — but the LLM is still a passive consumer of its own memory. Memories arrive, it uses them, it scores the exchange, it moves on. The system does all lifecycle management. The LLM is along for the ride.

This creates five systematic cognitive failure modes:

1. **Hallucination amplified by false authority:** A memory exists, so the LLM treats it as fact. If the memory is wrong (stale path, outdated version, hallucinated detail that got stored), the LLM states the wrong information with the confidence of "I checked my memory."
2. **Plan/done ambiguity:** Memories about planned features and shipped features look identical in storage. "Cross-encoder bypass skips CE for scored memories" could be a plan or a shipped feature — the LLM can't tell.
3. **Temporal credibility:** Older high-scored memories outrank newer corrections. A wrong memory with 8 uses and score 0.9 outranks a correction with 1 use and score 0.7.
4. **Fact staleness:** memory_bank has no expiration or validation mechanism. Facts stored once persist forever unless manually cleaned.
5. **Associative contamination:** Related-but-distinct concepts merge when retrieved together. "Wilson scoring" and "promotion criteria" both involve scores but use different thresholds — memories about one contaminate reasoning about the other.

### Solution: Six Prompting Additions

These additions don't change code logic — they change LLM behavior by adding explicit instructions to existing tool descriptions and the KNOWN CONTEXT preamble.

#### 1. Memory Hygiene Instructions → `get_context_insights` description

**Where:** Append to the system mastery prompt in `get_context_insights` description, after the `YOUR ROLE` section.

**Add:**
```
MEMORY HYGIENE (your responsibility):
• When you give a wrong answer and get corrected, actively score the
  misleading memory as "failed" — don't just score the exchange
• When you notice a memory_bank fact that's outdated or wrong, use
  update_memory or delete_memory immediately — don't wait to be told
• When you store a new fact, verify it first. If you can't verify it,
  store with confidence=0.5, not 0.9
• When KNOWN CONTEXT contains a memory you know is wrong from THIS
  session's work, call it out: "Note: [id:xxx] appears outdated because..."
• Never state specific details (ports, paths, versions, numbers) from
  memory without checking — if it's not in KNOWN CONTEXT verbatim or
  confirmed by a tool call, say "I believe X but let me verify"
```

**Why:** The LLM is told HOW the system works but never told to MAINTAIN it. This makes memory hygiene an explicit responsibility, not an emergent behavior.

#### 2. Active Failure Scoring → `score_response` description

**Where:** Append to the `score_response` tool description, after the `PER-MEMORY SCORING` section.

**Add:**
```
ACTIVE MEMORY MANAGEMENT:
• "failed" is not punishment — it's pruning. A memory scored "failed"
  3 times will be demoted or deleted. This is GOOD. Use it.
• If a memory was MISLEADING in your response (you gave wrong info
  because of it), score it "failed" even if the exchange went okay
• If you notice a pattern memory surfacing repeatedly with "unknown"
  scores, consider: is this memory actually useful? If not, scoring
  "failed" once helps the system stop wasting context on it
• You are the gardener. Pull the weeds.
```

**Why:** The scoring prompt tells the LLM to score each memory but doesn't emphasize that "failed" is a powerful tool for self-improvement. LLMs default to being polite — they avoid marking things as "failed" unless explicitly told it's constructive.

#### 3. Verification Workflow → `search_memory` description

**Where:** Append to the `search_memory` tool description, after the `READING RESULTS` section.

**Add:**
```
VERIFICATION USE:
• Before stating a fact from memory as definitive, search for it
• If you find conflicting memories on the same topic, prefer the
  newer one and score the older one "failed" on the next scoring prompt
• Use id= lookup to verify specific memories referenced in KNOWN CONTEXT
  before building conclusions on them
```

**Why:** LLMs use search_memory to FIND things but not to VERIFY things. This reframes search as a verification tool, not just a retrieval tool. Combined with `id=` lookup, the LLM can check whether a specific memory it's about to rely on is still accurate.

#### 4. Storage Discipline → `add_to_memory_bank` description

**Where:** Append to the `add_to_memory_bank` tool description, after the existing `ALWAYS_INJECT` and `SIZE GUIDANCE` sections.

**Add:**
```
STORAGE DISCIPLINE:
• confidence=0.9+ is reserved for facts you VERIFIED against source
  (code, docs, tool output). Not for things you "know"
• confidence=0.7 is the default — use it unless you have evidence
• confidence=0.5 for unverified claims, approximate numbers, or
  things the user told you that you couldn't confirm
• Before storing, ask: "Will this still be true next week?" If not,
  it doesn't belong in memory_bank — let working/history handle it
• One fact per memory. Don't combine unrelated information
```

**Why:** LLMs store things at high confidence without verification. A hallucinated port number stored at confidence=0.9 will outrank a real port number stored at confidence=0.7. This creates a confidence calibration framework.

#### 5. Epistemic Humility Line → KNOWN CONTEXT preamble

**Where:** Add one sentence to the KNOWN CONTEXT preamble in `unified_memory_system.py`.

**Add:**
```
Memories may be outdated or wrong. Verify before treating as ground truth.
```

**Why:** Six words of overhead per turn. Every turn the LLM is reminded that memories are suggestions, not facts. This directly counters failure mode #1 (hallucination amplified by false authority).

**Updated full preamble:**
```
You have persistent memory about this user via Roampal. Scored memories include Wilson confidence and outcome history. [id:...] tags can be looked up with search_memory(id=...). Memories may be outdated or wrong. Verify before treating as ground truth. The context below was retrieved from past conversations. If the user references past interactions or asks if you remember them, use this context — you DO remember.
```

#### 6. Periodic Self-Audit Prompt → `get_context_insights` handler

**Where:** In the `get_context_insights` handler logic (not in tool description — in the response returned by the tool). When `get_context_insights` is called and returns memory_bank items in the context, occasionally append a self-audit reminder.

**Concept:**
```
SELF-AUDIT: Review any memory_bank items in KNOWN CONTEXT.
If any look outdated, wrong, or redundant, use update_memory or
delete_memory to clean them up. Good assistants maintain their own memory.
```

**Implementation note:** This is the most complex of the six additions because it's not just a description change — it requires logic in the handler to decide WHEN to append the audit prompt. Options:
- Every N calls (e.g., every 5th call to get_context_insights)
- When memory_bank items exceed a threshold count in the context
- Randomly with a probability (e.g., 20% of calls)

**Decision deferred:** The exact trigger mechanism should be decided during implementation. The text content is finalized.

### What These Changes Do Together

| Change | Failure Mode Addressed | Mechanism |
|--------|----------------------|-----------|
| Memory Hygiene Instructions | #1 (false authority), #4 (staleness) | LLM actively flags and corrects wrong memories |
| Active Failure Scoring | #3 (temporal credibility) | LLM prunes misleading memories instead of politely marking "unknown" |
| Verification Workflow | #1 (false authority), #5 (associative contamination) | LLM verifies before asserting; resolves conflicts between memories |
| Storage Discipline | #1 (false authority), #2 (plan/done ambiguity) | Calibrated confidence prevents false-authority storage |
| Epistemic Humility Line | #1 (false authority) | Per-turn reminder that memories are suggestions, not facts |
| Periodic Self-Audit | #4 (staleness) | Active maintenance turns passive retrieval into gardening |

**Token cost:**
- Changes 1-4: Zero per-turn cost (tool descriptions, loaded once at session start)
- Change 5: ~15 tokens per turn (one sentence in preamble)
- Change 6: ~40 tokens when triggered (not every turn)

### Files to Change

| File | Change |
|------|--------|
| `roampal/mcp/server.py` | Append Memory Hygiene Instructions to `get_context_insights` description (after YOUR ROLE section) |
| `roampal/mcp/server.py` | Append Active Failure Scoring to `score_response` description (after PER-MEMORY SCORING section) |
| `roampal/mcp/server.py` | Append Verification Workflow to `search_memory` description (after READING RESULTS section) |
| `roampal/mcp/server.py` | Append Storage Discipline to `add_to_memory_bank` description (after SIZE GUIDANCE section) |
| `roampal/backend/modules/memory/unified_memory_system.py` | Add epistemic humility sentence to KNOWN CONTEXT preamble |
| `roampal/mcp/server.py` or handler logic | Implement periodic self-audit prompt in `get_context_insights` response (trigger mechanism TBD) |

### Testing Plan

- Live test (Claude Code): Store a fact with wrong info → verify LLM flags it on next turn when it contradicts tool output
- Live test (Claude Code): Surface a known-wrong memory → verify LLM scores it "failed" instead of "unknown"
- Live test (Claude Code): Ask LLM to state a port number from memory → verify it says "let me verify" instead of asserting
- Live test (Claude Code): Store something at confidence=0.9 → verify LLM only does this for verified facts
- Live test (OpenCode): Same tests — confirm tool descriptions surface correctly in system prompt transform
- Token count: verify preamble change adds ≤20 tokens per turn
- Verify no regressions: scoring still works, memory retrieval still works, search still works

---

## Feature 5: Security Hardening & Storage Hygiene

**Platforms:** All clients
**Trigger:** Post-implementation security audit identified defense-in-depth gaps

### Changes

#### 1. CORS Tightened (main.py)

**Before:** `allow_origins=["*"]` — any browser tab on localhost could reach the API.
**After:** `allow_origins=[]`, `allow_credentials=False`, methods restricted to GET/POST.

Since all clients (hooks, plugin, MCP) communicate via direct HTTP on localhost, CORS is not needed at all. Removing it blocks browser-based CSRF attacks against the memory API.

#### 2. PID Validation on Process Kill (all 4 files)

**Before:** PID extracted from netstat/lsof output and passed directly to `taskkill`/`kill -9` without validation.
**After:** Numeric-only check (`/^\d+$/.test(pid)` in TypeScript, `pid.isdigit()` in Python) before executing kill command.

Defense-in-depth — PIDs from OS tools are always numeric, but validating prevents any edge case where malformed output could execute unintended commands.

**Files changed:**
- `roampal/plugins/opencode/roampal.ts` — Windows + Unix paths
- `roampal/hooks/stop_hook.py` — Windows + Unix paths
- `roampal/hooks/user_prompt_submit_hook.py` — Windows + Unix paths

#### 3. API Input Validation (main.py)

**Before:** `SearchRequest` had no field constraints — `limit` could be 1M, `days_back` could be negative, `query` unbounded.
**After:** Pydantic `Field` constraints:
- `query`: max 2000 chars
- `days_back`: 1-365
- `limit`: 1-100
- `sort_by`: regex-validated enum (relevance|recency|score)
- `id`, `conversation_id`: max 200 chars

#### 4. Ingest Size Limit at API Layer (main.py)

**Before:** `/api/ingest` accepted arbitrarily large payloads (the backend had a 10MB check in `store_book()`, but the endpoint didn't fail-fast).
**After:** 10MB check at the endpoint level with HTTP 413 response. Fails before allocating ChromaDB resources.

#### 5. Session Transcript Cleanup — 7-Day TTL (session_manager.py)

**Before:** Session JSONL transcripts in `mcp_sessions/` grew unbounded. Long-running users would accumulate megabytes of transcript data.
**After:** `_cleanup_old_transcripts(max_age_days=7)` runs at `SessionManager.__init__()` (every server startup). Deletes `.jsonl` files with mtime older than 7 days. Transcripts are redundant once content is captured in memory (working → history → patterns).

#### 6. Debug Log Rotation (roampal.ts)

**Before:** `roampal_plugin_debug.log` grew unbounded.
**After:** When log exceeds 1MB, truncates to last 256KB with rotation marker. Checked on every `debugLog()` call.

#### 7. .gitignore Updated

Added patterns for debug/temp files that could contain PII or memory content:
```
debug_*.json, debug_*.txt, *_providers.json, *search_out.json, *verify_fix.json
```

#### 8. Bug Fixes Found During Audit

| Bug | Fix | File |
|-----|-----|------|
| `_search_all()` discarded non-date metadata filters | Now applies chromadb_filters in Python post-fetch | search_service.py |
| System mastery prompt omitted `uses ≥3` from promotion criteria | Added to lifecycle description | server.py |
| `r.get("metadata", {})` returned `None` for explicit null | Changed to `r.get("metadata") or {}` | server.py |
| `float(score)` crashed on non-numeric strings | Wrapped in try/except with safe defaults | server.py |
| FastAPI app version stuck at "0.3.2" | Bumped to "0.3.5" | main.py |

### Files Changed

| File | Changes |
|------|---------|
| `roampal/server/main.py` | CORS tightened, Field constraints, ingest size limit, version bump, Field import |
| `roampal/hooks/stop_hook.py` | PID numeric validation |
| `roampal/hooks/user_prompt_submit_hook.py` | PID numeric validation |
| `roampal/hooks/session_manager.py` | Transcript cleanup (7-day TTL), `time` import |
| `roampal/plugins/opencode/roampal.ts` | PID validation, debug log rotation, fs imports |
| `roampal/backend/modules/memory/search_service.py` | Metadata filter fix in `_search_all()` |
| `roampal/mcp/server.py` | Null-safe metadata, float safety, promotion criteria fix |
| `.gitignore` | Debug file patterns added |

### What We Explicitly Did NOT Fix (and Why)

| Audit Finding | Decision | Rationale |
|---------------|----------|-----------|
| Missing API authentication | Won't fix | Server binds to 127.0.0.1 only — auth adds complexity for zero security gain on localhost |
| Rate limiting | Won't fix | Localhost tool, not a public API. Resource exhaustion requires self-attack. |
| Transcript encryption at rest | Won't fix | Local files on user's own machine. OS-level disk encryption is the right layer. |
| Prompt injection via memory content | Won't fix | Inherent to any memory system — content was already in KNOWN CONTEXT before self-audit. |
| Dependency version pinning | Won't fix | Broad ranges are intentional for compatibility across user environments. |

---

## Migration Compatibility

**Zero migration required.** All changes are additive, application-layer only. Existing user data in ChromaDB works as-is.

### Verified: Existing Data Has Required Fields

| Collection | `created_at` | `score` | `uses` | `success_count` | `tags` |
|-----------|-------------|---------|--------|-----------------|--------|
| working | Set in `store_working()` line 1411 | 0.5 default | 0 | NOT present until first scoring event | N/A |
| history | Inherited from working | Yes | Reset to 0 on promotion | Reset to 0.0 on promotion | N/A |
| patterns | Inherited through promotions | Yes | Yes | Yes (≥5 required for promotion) | N/A |
| memory_bank | Set in `store_memory_bank()` | Always 1.0 | 0 (v0.2.9) | 0.0 (v0.2.9) | JSON string `'["identity"]'` |
| books | Set in `store_book()` | NOT present | NOT present | NOT present | N/A |

### Key Compatibility Details

1. **`normalize_memory()` fallback chains handle all cases:**
   - Content: tries `result["content"]` → `metadata["content"]` → `metadata["text"]` → `result["text"]`
   - Timestamp: tries `metadata["created_at"]` → `metadata["timestamp"]` → empty string
   - Scoring: defaults `score=0.5`, `uses=0`, `success_count=0.0`, `wilson_score=0.5` when absent

2. **`success_count` edge case:** Working memories start WITHOUT `success_count` in metadata. After the first scoring event, `outcome_service.py` (line 116) reads it with `metadata.get("success_count", 0.0)`, adds the delta, and writes it back (line 151). So:
   - Before any scoring: `uses=0`, `success_count` absent → normalize defaults both to 0, Wilson = 0.5 (untested). **Correct.**
   - After first scoring: both `uses` and `success_count` written. **Correct.**
   - Edge case: if `success_count` is absent but `uses > 0` (shouldn't happen, but defensive): normalize_memory should check `success_count is None` vs `0.0` and default Wilson to 0.5 rather than computing from 0 successes.

3. **Books have no scoring fields at all** — normalize defaults everything to neutral values (score=0.5, uses=0, wilson=0.5). Books are reference material, not scored. **Correct.**

4. **`days_back` filtering uses existing `created_at` metadata** — verified all collections write `created_at` to ChromaDB metadata. `_apply_date_filters()` reads from raw metadata (`r.get('metadata', {}).get(field, '')`), which still has the field after normalization. **Works on existing data.**

5. **ChromaDB `collection.get()` returns same shape as `collection.query()`** except no `distances` field. `apply_scoring_to_results()` defaults `distance=1.0` when absent (line 326 of scoring_service.py). **Compatible.**

6. **Tags parsed from JSON strings** — memory_bank stores tags as `'["identity", "preference"]'`. `normalize_memory()` handles both string (parses JSON) and list types. Other collections don't have tags → defaults to `[]`. **Compatible.**

7. **No ChromaDB schema changes.** No new indexes. No field renames in storage. All standardization happens in the application layer at read time.

### Pre-Existing Bugs Fixed by This Release

1. **`_search_all()` sorts by `timestamp` only** (line 311) — working/memory_bank/books only have `created_at`, so they get empty-string timestamps and sort incorrectly (pushed to end). Fixed by normalize_memory + sort by `created_at`.

2. **`_humanize_age` in display only checks `created_at`** (line 709) — if a memory only had `timestamp`, age wouldn't display. Less severe than initially thought (all collections carry `created_at` from storage), but normalize_memory's `age` field makes this robust.

3. **`_search_all()` bypasses metadata filters entirely** — empty query at line 206-209 returns early before date filters run. Fixed by passing `metadata_filters` through to `_search_all()`.
