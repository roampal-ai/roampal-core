# v0.4.5 Release Notes

**Platforms:** Claude Code, OpenCode
**Theme:** TagCascade retrieval — benchmark-validated tags-first architecture, KG removed

---

## Summary

Replace knowledge graph with TagCascade retrieval, validated on LoCoMo benchmark (paper: "Beyond Ingestion", Section 5.2.3). Tag-cascade+cosine won both clean (27.3% Hit@1) and poison (29.0% Hit@1) conditions, beating all alternatives including Wilson-blended configs (p<0.0001).

**Key architecture decisions (all benchmark-validated):**
- Tags-first cascade entry point beats cosine-first (p=0.0003)
- Cosine distance tiebreaker within overlap tiers beats Wilson tiebreaker (p<0.0001)
- Raw CE score as final ranking beats Wilson+CE blend (p=0.0001)
- Wilson removed from retrieval entirely — kept as metadata for display/outcome-tracking/promotion

---

## Changes

### 1. TagCascade retrieval pipeline
**File:** `roampal/backend/modules/memory/search_service.py`

Rewrote search pipeline to match benchmark `tag_cascade_cosine` algorithm:

```
1. Match query nouns against known tag index
2. Per-tag ChromaDB queries, count overlaps per memory
3. Fill candidate pool (40) from highest overlap tier down:
   - Tier N (most tags matched) → sort by cosine distance → fill slots
   - Tier N-1 → sort by cosine distance → fill remaining slots
   - ... down to Tier 1
4. Cosine-fill remaining slots from unfiltered search
5. If no tags matched: pure cosine candidates
6. CE reranks pool — raw CE score as final ranking (no Wilson blend)
7. Take top_k
```

**Removed:**
- Wilson+CE blend in `_rerank_with_ce()` (was 60%CE/40%Wilson for proven, 90%CE/10%Wilson for new)
- Post-CE re-sort by tag overlap count
- KG entity boost, graph expansion, routing patterns

### 2. Noun tag extraction at store time
**File:** `roampal/backend/modules/memory/unified_memory_system.py`

Enabled tag extraction on all store paths:
- `store()` — patterns/history (post-upsert metadata update)
- `store_memory_bank()` — memory bank (post-upsert, alongside existing category tags)
- `store_working()` — working memory (inline before upsert)

LLM extracts noun_tags and passes via MCP tool `noun_tags` param. No sidecar LLM calls. Regex fallback only when tags not provided (promotion, internal stores).

Memory bank retains both category `tags` (identity/preference/etc.) and content `noun_tags` (people, places, topics).

### 3. noun_tags on MCP tools + OpenCode sidecar
**Files:** `mcp/server.py`, `server/main.py`, `unified_memory_system.py`, `plugins/opencode/roampal.ts`

- `add_to_memory_bank`: required `noun_tags` param
- `record_response`: required `noun_tags` param
- `score_memories`: `noun_tags` param for exchange summary
- OpenCode sidecar: scoring prompt extracts `noun_tags` in JSON, passes to store calls
- Server accepts and stores directly — no server-side LLM extraction
- Summary prompts match benchmark (first-person, detailed guidance with good/bad examples)

### 4. Tag infrastructure
**Files:** `tag_service.py` (new), `tag_migration.py` (new)

- `TagService`: regex extraction (migration only), known-tag index, word-boundary query matching
- `TagMigration`: resumable regex backfill for existing memories (one-time on first startup)

### 5. KG removal
**Deleted:** `knowledge_graph_service.py`, `content_graph.py`, `test_knowledge_graph_service.py`, `test_action_kg_sync.py`

**Modified:** All services that referenced KG now use `**kwargs` for backward compat (context_service, outcome_service, search_service accept and ignore `kg_service`).

### 6. Cross-encoder via ONNX (no PyTorch)
**File:** `search_service.py`

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, 14+ languages)
- Format: ONNX `model_O4.onnx` (235MB, CPU-only via onnxruntime)
- Lazy-loaded on first search, falls back to cosine-only if unavailable
- `pytorch` optional dependency removed from `pyproject.toml`

### 7. Scoring cleanup
**File:** `scoring_service.py`

- Memory bank uses same 5-tier scoring system as all collections (removed special-case `(0.0, 1.0)` weight override)
- Wilson scores still computed as metadata for display and promotion thresholds
- `kg_debounce_seconds` removed from config

### 8. Peripheral improvements
- **`cli.py`**: `roampal doctor` warns about unnecessary sentence-transformers install
- **`session_manager.py`**: Improved scoring prompt with 7-rule guide + fact extraction instructions
- **`README.md`**: Added system requirements (RAM, disk, CPU, GPU)

### 9. Atomic fact extraction + two-lane retrieval
**Files:** `mcp/server.py`, `server/main.py`, `unified_memory_system.py`, `plugins/opencode/roampal.ts`, `hooks/session_manager.py`

Matches benchmark architecture (runner.py:173-218, 801-818):

**Storage:** `score_memories` tool accepts `facts` array. Each fact stored as separate working memory with `memory_type: "fact"`. Claude Code: main LLM extracts facts during scoring. OpenCode: sidecar extracts facts in same scoring call.

**Retrieval:** `get_context_for_injection()` does two-lane retrieval:
- Lane 1 (4 slots): summaries from working/history/patterns/memory_bank (`memory_type != "fact"`)
- Lane 2 (4 slots): facts from working/history/patterns (`memory_type == "fact"`)
- 8 total per query, both lanes through full TagCascade pipeline

**search_memory tool:** Optional `type` filter (`fact`/`summary`) for manual searches.

**Book ingestion:** `store_book()` now extracts regex noun_tags per chunk.

**Stale code removed:** `_extract_concepts()`, `get_tier_recommendations()`, nursery slot — all KG remnants.

**Bugs fixed during implementation:**
- ChromaDB compound filters: `_merge_filters()` now uses `$and` wrapper (ChromaDB rejects multiple top-level keys)
- Two-lane results: removed `[:4]` slice that was cutting merged 4+4 back to 4

### 10. `roampal summarize` extracts facts + noun_tags
**Files:** `cli.py`, `sidecar_service.py`, `server/main.py`

- `roampal summarize` now extracts noun_tags from summaries and atomic facts from original content
- Each fact stored as separate working memory with `memory_type: "fact"` via `/api/record-response`
- `/api/memory/update-content` accepts `noun_tags` for summarized memories
- New `extract_facts()` function in `sidecar_service.py` matching benchmark prompt

### 11. `roampal sidecar test` command
**File:** `cli.py`, `sidecar_service.py`

- Tests sidecar with a sample exchange and validates response format
- Checks for all v0.4.5 required fields: summary, outcome, noun_tags, facts
- Reports pass/fail per field with values
- New `test_sidecar_scoring()` function in `sidecar_service.py`

---

## Files Modified

| File | Change |
|------|--------|
| `search_service.py` | REWRITE — TagCascade pipeline, CE via ONNX, no Wilson blend |
| `unified_memory_system.py` | Enable tag extraction at store time, remove KG wiring |
| `tag_service.py` | NEW — Regex + LLM tag extraction, query matching, known-tag index |
| `tag_migration.py` | NEW — Resumable regex backfill for existing memories |
| `sidecar_service.py` | NEW functions: `extract_facts()`, `test_sidecar_scoring()`. Haiku removed from auto-fallback. |
| `scoring_service.py` | Remove memory_bank special cases |
| `routing_service.py` | Remove KG dependency, always all 5 collections |
| `outcome_service.py` | Remove KG calls |
| `context_service.py` | Remove KG, simplify |
| `config.py` | Remove `kg_debounce_seconds` |
| `session_manager.py` | Improved OpenCode scoring prompt |
| `cli.py` | sentence-transformers deprecation warning |
| `server/main.py` | Remove ContentGraph import |
| `ARCHITECTURE.md` | Updated for TagCascade |
| `README.md` | System requirements |
| `pyproject.toml` | Version 0.4.5, remove pytorch optional dep |
| `__init__.py` | Version 0.4.5 |

## Files Deleted

| File | Reason |
|------|--------|
| `knowledge_graph_service.py` | KG replaced by TagCascade |
| `content_graph.py` | Entity tracking replaced by noun tags |
| `tests/unit/test_knowledge_graph_service.py` | Tests for deleted code |
| `tests/unit/test_search_service.py` | Old file heavily KG-dependent — replaced with new TagCascade tests |
| `tests/integration/test_action_kg_sync.py` | Action KG removed |

## Tests

| File | Status |
|------|--------|
| `test_tag_service.py` | NEW — 29 tests (extraction, matching, index rebuild) |
| `test_search_service.py` | NEW — 14 tests (tier filling, CE raw score, cosine fallback, boosts) |
| `test_routing_service.py` | Updated — no KG, always-all-collections + acronym expansion |
| `test_scoring_service.py` | Updated — memory_bank uses 5-tier system |
| `test_context_service.py` | Updated — no KG |
| `test_outcome_service.py` | Updated — no KG assertions |
| `test_unified_memory_system.py` | Updated — no knowledge_graph attr |
| `test_v036_changes.py` | Updated — no entity boost |
| **Total: 466 passed** | |

---

## Verification

```bash
# 1. Run tests
pytest roampal/backend/modules/memory/tests/unit/ -v

# 2. No KG imports in production code
grep -r "knowledge_graph_service\|KnowledgeGraphService\|ContentGraph" roampal/ --include="*.py" | grep -v test | grep -v __pycache__

# 3. Verify tag extraction
python -c "
from roampal.backend.modules.memory.tag_service import extract_tags_regex
tags = extract_tags_regex('Calvin restored a 1967 Ford Mustang in Tokyo')
print('Tags:', tags)  # Expected: ['ford mustang', 'calvin', 'tokyo']
"

# 4. Version check
python -c "import roampal; print(roampal.__version__)"  # 0.4.5
```

---

## Known Limitation: Claude Code context accumulation

Hook-injected KNOWN CONTEXT and scoring prompts accumulate in Claude Code's conversation history every turn. Investigated extensively — no mechanism exists for ephemeral/transient injection:

- **Hooks**: `UserPromptSubmit` stdout and `additionalContext` both persist in conversation history
- **MCP**: Tools, resources, and prompts all accumulate identically
- **Skills DCI**: `!`command`` syntax doesn't accumulate but only runs on explicit invocation, not automatic per-turn
- **System prompt**: Not accessible via hooks or MCP — only CLAUDE.md (static) feeds into it
- **Compaction**: Claude Code auto-compresses old tool outputs, which helps over long sessions

**OpenCode** injects memory context via `system.transform` (fresh per turn for that injection). However, OpenCode also uses compaction for the conversation itself — neither platform has a true sliding window.

**Feature requests filed**:
- anthropics/claude-code#45849 — ephemeral hook output or system prompt access for dynamic per-turn context injection without accumulation
- anthropics/claude-code#45851 — configurable sliding window for conversation context (last N exchanges instead of full history + compaction). Cites roampal-labs benchmark: 4-exchange window + 8 retrieved memories achieves 76.6% on LoCoMo.

---

## Carried Forward

### From earlier v0.4.5 planning (not shipped)
- Test sidecar model during setup — still useful, deferred
- Tag cleanup via Wilson averages (`roampal cleanup`) — deferred pending real-world data
- Nursery slot — benchmark showed no benefit (p=1.0), removed

---

## Benchmark Reference

Results from cascade isolation test (1,537 non-adversarial LoCoMo questions, two-lane retrieval):

| Config | Clean Hit@1 | Poison Hit@1 |
|--------|-------------|--------------|
| **tag_cascade_cosine** | **27.3%** | **29.0%** |
| overlap_cosine | 25.8% | 28.0% |
| pure_ce | 25.4% | 28.4% |
| tag_cascade_wilson | 23.0% | 25.0% |
| overlap_wilson | 22.8% | 25.8% |

Paper: "Beyond Ingestion: What Conversational Memory Learning Reveals on a Corrected LoCoMo Benchmark" (Logan Teague, April 2026)
