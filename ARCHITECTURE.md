# Roampal Core - Architecture

## Overview

**One command install. AI coding tools get persistent memory.**

Roampal Core provides hook-based memory injection and outcome learning for AI coding tools like Claude Code and Cursor. The external LLM (Claude, GPT, etc.) sits in the driver seat - using the same memory system as Roampal's internal LLM, but without local model limitations.

```bash
pip install roampal
roampal init
# That's it! MCP server auto-starts FastAPI hook server when Claude Code launches
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MCP SERVER PROCESS                         │
│  (launched by Claude Code when it starts)                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  MCP Server (stdio)          │  FastAPI Server      │    │
│  │  - 7 memory tools            │  - Hook endpoints    │    │
│  │  - search, add, score, etc.  │  - Port 27182        │    │
│  │                              │  (background thread) │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
   MCP tool calls       Hook HTTP calls      Future: Dashboard
   from Claude          from hooks
```

**Key change:** The MCP server now auto-starts the FastAPI hook server in a background thread.
No separate `roampal start` command needed - when Claude Code launches the MCP server, hooks work automatically.

---

## Hook-Based Outcome Scoring

The key innovation: **hooks prompt the LLM to call score_response()** via soft enforcement - no blocking, just prompt injection.

### Completion-Aware Scoring

Scoring is only required when the user responds to a **completed** assistant response:
- ✅ User sends message → Assistant completes response → User sends new message → **Score required**
- ❌ User interrupts mid-work → Assistant still processing → **No scoring** (not annoying)

This distinction prevents forcing the LLM to score when the user is just providing follow-up context or interrupting.

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER SENDS MESSAGE                                        │
│    ↓                                                         │
│ 2. HOOK: UserPromptSubmit (calls /api/hooks/get-context)    │
│    - Checks if assistant COMPLETED last response             │
│    - If completed AND previous exchange unscored:            │
│      → Injects scoring prompt with previous exchange         │
│      → Sets scoring_required=True                            │
│    - If interrupted mid-work: skip scoring                   │
│    - KG-ROUTED UNIFIED SEARCH:                               │
│      → Extract concepts from user message                    │
│      → Search ALL 5 collections                              │
│      → Wilson-rank results                                   │
│      → Return top 5 regardless of collection                 │
│      → Cache doc_ids for scoring                             │
│    ↓                                                         │
│ 3. LLM SEES (only if scoring required):                      │
│    <roampal-score-required>                                  │
│    Previous: User asked "..." You answered "..."             │
│    Current user message: "..."                               │
│    Call score_response(outcome="...") FIRST                  │
│    </roampal-score-required>                                 │
│                                                              │
│    ═══ KNOWN CONTEXT ═══                                     │
│    • User preference: Python                                 │
│    ═══ END CONTEXT ═══                                       │
│                                                              │
│    Original user message                                     │
│    ↓                                                         │
│ 4. LLM calls score_response(outcome) then responds           │
│    ↓                                                         │
│ 5. HOOK: Stop (calls /api/hooks/stop)                       │
│    - Stores exchange with doc_id                             │
│    - Sets assistant_completed=True for next turn             │
│    - SOFT ENFORCE: logs warning if score_response not called │
│    ↓                                                         │
│ 6. OUTCOME RECORDED → Score updates applied                  │
│    - Previous exchange scored                                │
│    - Cached search results scored                            │
│    - Pattern promotion evaluated                             │
└─────────────────────────────────────────────────────────────┘
```

### Completion State Tracking

The `_completion_state.json` file tracks:
- `completed`: True when Stop hook fires (assistant finished responding)
- `scoring_required`: True when get-context injected a scoring prompt

This allows the system to distinguish between:
1. User responding to completed work → scoring required
2. User interrupting mid-work → no scoring needed

### What The User Sees vs What The LLM Sees

**User sees:** `Help me fix this auth bug`

**LLM sees:**
```
<roampal-score-required>
Score the previous exchange before responding.

Previous:
- User asked: "How do I configure TypeScript?"
- You answered: "You can configure TypeScript using tsconfig.json..."

Memories surfaced:
• [mem_abc123] "User prefers TypeScript, detailed explanations"
• [mem_def456] "JWT refresh token pattern worked for auth issues"
• [mem_ghi789] "Random marketing note about features"

Current user message: "Help me fix this auth bug"

Based on the user's current message, evaluate if your previous answer helped:
- "worked" = user satisfied, says thanks, moves on to new topic
- "failed" = user corrects you, says no/wrong, repeats question
- "partial" = lukewarm response, "kind of", "I guess"
- "unknown" = no clear signal

Call score_response(outcome="...", related=["doc_ids that were relevant"]) FIRST.
- related is optional: omit to score all, or list only the memories you actually used

Separately, record_response(key_takeaway="...") is OPTIONAL - only for significant learnings.
</roampal-score-required>

═══ KNOWN CONTEXT ═══
• User: backend engineer
• Preference: TypeScript, detailed explanations

[Past Solutions]
• JWT refresh token pattern worked for auth issues (92% effective, from patterns)
═══ END CONTEXT ═══

Help me fix this auth bug
```

---

## File Structure

```
roampal-core/
├── ARCHITECTURE.md           # This file
├── README.md                 # Quick start guide
├── pyproject.toml            # Package config
│
├── roampal/
│   ├── __init__.py           # Exports UnifiedMemorySystem, MemoryConfig
│   ├── cli.py                # CLI: init, start, status, stats
│   │
│   ├── backend/
│   │   ├── __init__.py
│   │   └── modules/
│   │       └── memory/
│   │           ├── __init__.py              # Exports all memory types
│   │           ├── unified_memory_system.py # Main orchestrator
│   │           ├── chromadb_adapter.py      # Vector storage
│   │           ├── embedding_service.py     # sentence-transformers
│   │           ├── scoring_service.py       # Wilson score calculation
│   │           ├── outcome_service.py       # Score updates from outcomes
│   │           ├── promotion_service.py     # Promotion/demotion/cleanup
│   │           ├── memory_bank_service.py   # Permanent user facts
│   │           ├── context_service.py       # Context analysis
│   │           ├── config.py                # MemoryConfig
│   │           └── types.py                 # TypedDicts, enums
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI server (port 27182)
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py         # MCP server (7 tools)
│   │
│   └── hooks/
│       ├── __init__.py
│       ├── session_manager.py          # Exchange tracking (JSONL)
│       ├── user_prompt_submit_hook.py  # Context injection
│       └── stop_hook.py                # Enforcement + storage
```

---

## Memory Collections

| Collection | Purpose | Scorable | Decay |
|------------|---------|----------|-------|
| `books` | Uploaded reference docs | No | Never |
| `working` | Current session context | Yes | Session |
| `history` | Past conversations | Yes | Score-based |
| `patterns` | Proven solutions (promoted from history) | Yes | Never |
| `memory_bank` | Permanent user facts (LLM-controlled) | No | Never |

### Memory Bank Guidelines

**What belongs in memory_bank:**
- User identity (name, role, background)
- Preferences (communication style, tools, workflows)
- Goals and projects (what they're working on, priorities)
- Progress tracking (what worked, what failed, strategy iterations)
- Useful context that would be lost between sessions

**What does NOT belong:**
- Raw conversation exchanges (auto-captured in working/history)
- Temporary session facts (current task details)
- Every fact heard - be SELECTIVE, this is for PERMANENT knowledge

**Rule of thumb:** If it helps maintain continuity across sessions OR enables learning/improvement, store it. If it's session-specific, don't.

**Note:** memory_bank is NOT outcome-scored (unlike working/history/patterns). Facts persist until archived.

### Promotion Flow

```
working → history → patterns
   ↓         ↓         ↓
 score     score     score ≥ 0.8
 < 0.2:    ≥ 0.7     AND uses ≥ 3
 deleted   AND       = PATTERN
           uses ≥ 2
           = promote

 Age > 24h without promotion = deleted
```

---

## API Endpoints

### Hook Endpoints (FastAPI - Port 27182)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hooks/get-context` | POST | KG-routed unified search → top 5 memories across all collections |
| `/api/hooks/stop` | POST | Stores exchange, logs if score_response not called |
| `/api/record-outcome` | POST | Records outcome, updates scores |

### Memory API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Search across collections |
| `/api/memory-bank/add` | POST | Add to memory bank |
| `/api/memory-bank/update` | POST | Update memory bank entry |
| `/api/memory-bank/archive` | POST | Archive memory bank entry |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Memory statistics |

---

## MCP Tools

The MCP server provides 7 tools for deep memory access:

| Tool | Description |
|------|-------------|
| `get_context_insights` | Get user profile + relevant memories (caches doc_ids for scoring) |
| `search_memory` | Search across memory collections (for detailed lookups) |
| `add_to_memory_bank` | Store permanent user facts |
| `update_memory` | Update existing memories |
| `archive_memory` | Archive outdated memories |
| `score_response` | **SOFT ENFORCED** - Score the previous exchange (worked/failed/partial/unknown) |
| `record_response` | **OPTIONAL** - Store key takeaways when transcript won't capture learning |

### Recommended Workflow

```
1. Hook injects context + previous exchange for scoring
2. score_response(outcome) → Score the previous exchange
3. Respond to user
4. record_response(key_takeaway) → OPTIONAL, only for significant learnings
```

### score_response (Soft Enforced)

```json
{
  "outcome": "worked" | "failed" | "partial" | "unknown",
  "related": ["doc_id_1", "doc_id_3"]  // optional
}
```

The hook presents the previous exchange and current user message. Based on the user's reaction, score whether your previous answer helped.

**Outcome Detection:**
- `worked` = user satisfied, says thanks, moves on to new topic
- `failed` = user corrects you, says "no", "that's wrong"
- `partial` = lukewarm response, "kind of", "I guess"
- `unknown` = no clear signal

**Selective Scoring (optional `related` param):**
- If `related` omitted → all cached memories get scored (current behavior)
- If `related` provided → only those doc_ids get scored, others get 0
- Enables surgical scoring: relevant memories inherit outcome, noise stays neutral

Example: 6 memories surfaced, only 2 were actually used:
```
score_response(outcome="worked", related=["mem_abc123", "mem_def456"])
→ mem_abc123, mem_def456 get +0.20
→ other 4 memories get 0 (skipped, not demoted)
```

**Critical:** Failed outcomes are how bad memories get deleted. Without them, wrong info persists forever.

### record_response (Optional Key Takeaways)

```json
{
  "key_takeaway": "1-2 sentence summary of important learning",
  "initial_score": "worked"  // optional: "worked", "failed", or "partial"
}
```

Only use when the transcript alone won't capture something important:
- Major decisions made
- Complex solutions that worked
- User corrections (what you got wrong and why)
- Important context that would be lost

**Initial Scoring (optional):** Score the takeaway at creation time based on the current exchange:
- `initial_score="worked"` → starts at 0.7 (boosted)
- `initial_score="failed"` → starts at 0.2 (demoted, but stored as "what not to do")
- `initial_score="partial"` → starts at 0.55
- Omit → starts at 0.5 (neutral default)

Most routine exchanges don't need this - the transcript is enough.

### Doc ID Caching

Both hook injection and `search_memory` cache doc_ids by session_id. When `score_response` is called, it:
1. Scores the most recent unscored exchange (from any session)
2. Uses `related` doc_ids directly if provided (bypasses cache)
3. Falls back to cached doc_ids if `related` omitted
4. Applies outcome to memories

**How `related` parameter works:**

| `related` param | What happens |
|-----------------|--------------|
| `related=["id1", "id2"]` | Score those specific memories (+ exchange) |
| `related=[]` | Score NO context memories, just the exchange |
| `related` omitted | Score ALL cached memories (+ exchange) |

This ensures that:
- Hook-injected memories get scored (cached under Claude Code's session_id)
- MCP search results get scored (cached under "default")
- Cold start profile dumps get scored
- Session ID mismatches are handled gracefully
- Noise can be filtered out without being penalized
- **Direct doc_id scoring avoids cache timing issues** (Dec 2025 fix)

#### Cache Timing Fix (Dec 2025)

**Problem:** When the LLM searches for additional context (via `search_memory`) before scoring, the cache gets overwritten with new search results. By the time `score_response` is called with `related=["prev_turn_id"]`, those doc_ids no longer exist in the cache.

**Solution:** When `related` is provided, use those doc_ids directly instead of filtering against the (potentially stale) cache:

```python
# Before (broken): filter related against stale cache
if request.related is not None:
    filtered_doc_ids = [d for d in cached_doc_ids if d in related_set]

# After (fixed): use related directly
if request.related is not None and len(request.related) > 0:
    doc_ids_scored.extend(request.related)
```

This makes selective scoring reliable regardless of what searches happen between context injection and scoring.

---

## Score Updates

When `score_response(outcome)` is called:

| Outcome | Score Delta | Effect |
|---------|-------------|--------|
| `worked` | +0.20 | Promotes toward patterns |
| `partial` | +0.05 | Slight boost |
| `failed` | -0.30 | Demotes, may delete |

Score range: 0.0 to 1.0

### Time Weighting

Recent memories get stronger score updates:
```python
time_weight = 1.0 / (1 + age_days / 30)  # Decay over month
score_delta = base_delta * time_weight
```

---

## Data Storage

### ChromaDB (Vector Store)
- Location: `%APPDATA%/Roampal/data/chromadb` (Windows prod)
- Location: `%APPDATA%/Roampal_DEV/data/chromadb` (Windows dev, `ROAMPAL_DEV=1`)
- Embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim)
- Collections: `roampal_books`, `roampal_working`, `roampal_history`, `roampal_patterns`, `roampal_memory_bank` (matches Desktop)

### Session Files (JSONL)
- Location: `%APPDATA%/Roampal/data/mcp_sessions/`
- Format: One exchange per pair of lines
- Purpose: Track scored/unscored state for enforcement

---

## Knowledge Graphs

roampal-core shares the same Knowledge Graph data as Roampal Desktop, enabling intelligent routing and learning.

### Knowledge Graph Structure

```
knowledge_graph.json
├── routing_patterns      # concept -> best_collection
├── success_rates         # collection -> success_rate
├── failure_patterns      # concept -> failure_reasons
├── problem_categories    # problem_type -> preferred_collections
├── problem_solutions     # problem_signature -> [solution_ids]
├── solution_patterns     # pattern_hash -> {problem, solution, success_rate}
└── context_action_effectiveness  # (context, action, collection) -> stats
```

### KG Methods

| Method | Description |
|--------|-------------|
| `get_tier_recommendations(concepts)` | Query Routing KG for best collections |
| `get_action_effectiveness(context, action, collection)` | Get Action KG effectiveness stats |
| `get_facts_for_entities(entities)` | Query Content KG for memory_bank facts |
| `detect_context_type(messages)` | Classify query context (coding, fitness, etc.) |
| `analyze_conversation_context(message)` | Find patterns, failures, insights |
| `record_action_outcome(action)` | **Write** to Action KG - track tool effectiveness |
| `_update_kg_routing(query, collection, outcome)` | **Write** to Routing KG - learn query→collection patterns |

### How KGs Improve Search

1. **Routing KG**: Directs searches to collections that worked for similar concepts
2. **Action KG**: Tracks which tools work in which contexts (e.g., search_memory on patterns: 92% success in coding)
3. **Content KG**: Links entities to memory_bank facts for fast lookup

### memory_bank and books KG Benefits

While `memory_bank` and `books` don't get individual doc-level scoring (they're static reference material), they DO benefit from KG tracking at the **collection level**:

**Action KG (`record_action_outcome`):**
```python
# Tracks: "In coding context, score_response on memory_bank worked"
# Key format: "{context_type}|{action_type}|{collection}"
# Example: "coding|score_response|memory_bank" → 85% success rate
```

**Routing KG (`_update_kg_routing`):**
```python
# Tracks: "Queries about 'user preferences' work best on memory_bank"
# Learns concept→collection mappings from outcomes
# Example: "python" concept → books collection (90% success)
```

This means:
- memory_bank facts get surfaced more for queries where they've historically helped
- books get prioritized for technical queries where they've worked
- The learning happens even though individual docs aren't scored

### Data Sharing with Desktop

The KG files are stored in the same data directory as Roampal Desktop:
- `%APPDATA%/Roampal/data/knowledge_graph.json` (Windows prod)
- `%APPDATA%/Roampal_DEV/data/knowledge_graph.json` (Windows dev)

Patterns learned in Desktop are available to roampal-core and vice versa.

---

## CLI Commands

```bash
roampal init           # Configure hooks + MCP + permissions
roampal status         # Check server status
roampal stats          # Show memory statistics
roampal ingest <file>  # Ingest .txt/.md/.pdf into books collection
roampal books          # List all ingested books
roampal remove <title> # Remove a book by title
```

### Book Management

Books are stored in the `books` ChromaDB collection with these operations:

| Operation | What it does |
|-----------|--------------|
| `roampal ingest <file>` | Chunks file, embeds, stores in ChromaDB |
| `roampal books` | Lists all books grouped by title |
| `roampal remove <title>` | Deletes all chunks + cleans Action KG refs |

**Book Removal Flow:**
1. Find all chunks with matching title
2. Delete from ChromaDB `books` collection
3. Clean `context_action_effectiveness` examples referencing those doc_ids
4. Save knowledge_graph.json

**Note:** Removal is by title (not UUID) - simpler for CLI. Duplicate titles get removed together.

**Note:** `roampal start` is no longer required! The MCP server auto-starts the FastAPI hook server
when Claude Code launches. If you need to run the server standalone (e.g., for debugging), you can
still use `roampal start` or `roampal start --dev`.

### Dev Mode

Set `ROAMPAL_DEV=1` environment variable to isolate dev from production:
- **Port:** Production uses 27182, dev uses 27183
- **Data:** Production: `%APPDATA%/Roampal/data`, Dev: `%APPDATA%/Roampal_DEV/data`

**Important:** Both MCP server AND hooks need `ROAMPAL_DEV=1`:
- MCP server gets it from `.mcp.json` env config
- Hooks automatically detect `ROAMPAL_DEV` from environment and use the correct port

You can also set `ROAMPAL_DATA_PATH` environment variable for custom paths.

### What `roampal init` Does

1. Creates `~/.claude/settings.json` with:
   - UserPromptSubmit hook → `python -m roampal.hooks.user_prompt_submit_hook`
   - Stop hook → `python -m roampal.hooks.stop_hook`
   - Auto-allow permissions for all roampal MCP tools

2. Creates `~/.claude/mcp_servers.json` with roampal MCP server

3. Creates data directory at `%APPDATA%/Roampal/data`

---

## Configuration

### Claude Code Settings (auto-generated by `roampal init`)

```json
{
  "hooks": {
    "UserPromptSubmit": ["python", "-m", "roampal.hooks.user_prompt_submit_hook"],
    "Stop": ["python", "-m", "roampal.hooks.stop_hook"]
  },
  "permissions": {
    "allow": [
      "mcp__roampal-core__search_memory",
      "mcp__roampal-core__add_to_memory_bank",
      "mcp__roampal-core__update_memory",
      "mcp__roampal-core__archive_memory",
      "mcp__roampal-core__get_context_insights",
      "mcp__roampal-core__record_response",
      "mcp__roampal-core__score_response"
    ]
  }
}
```

### MCP Server Config (auto-generated)

```json
{
  "roampal-core": {
    "command": "python",
    "args": ["-m", "roampal.mcp.server"],
    "env": {
      "ROAMPAL_DEV": "1"
    }
  }
}
```

---

## Key Design Decisions

### 1. Soft Enforcement via Hook Prompting

**Problem:** LLMs don't reliably call score_response() without prompting.

**Solution:**
- UserPromptSubmit hook injects scoring prompt with previous exchange
- Hook prompt tells LLM to call score_response(outcome) FIRST
- Stop hook logs warning if not called (soft enforcement)
- Prompt injection does 95% of the work - no hard blocking needed
- Separate tools: score_response (scoring) vs record_response (key takeaways)

### 2. Automatic Context vs MCP Tools

**Previous approach:** LLM must remember to call `get_context_insights()` every turn.

**Current approach:** Hooks inject context automatically. MCP tools for deep searches only.

### 3. Score-Based Promotion

**Working → History → Patterns**

Memories earn their way to permanent storage through proven usefulness.

### 4. No Local LLM

Unlike Roampal Desktop, roampal-core uses NO local LLM. The connected external LLM (Claude, GPT) IS the brain. This removes:
- Ollama dependency
- Model download/management
- Local compute requirements

### 5. No Truncation (v0.2.8)

All content is stored and returned in full - no character limits or truncation:
- Full transcripts in session JSONL files
- Full memory content in search results
- Full context injection
- Limit by count (e.g., top 5 facts) not by characters

---

## Dependencies

```toml
[project]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.0.0",
    "mcp>=0.1.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]
```

---

## Future Considerations

1. **Cursor Support**: Hooks work, but Cursor hook format may differ
2. **Dashboard**: Web UI for memory management
3. **Multi-User**: Currently single-user design
4. **Encryption**: Data at rest is not encrypted
5. **KG Service**: ✅ Knowledge Graphs now shared with Roampal Desktop (v0.2.8)

---

## Related Files

- [README.md](README.md) - Quick start guide
- [roampal/cli.py](roampal/cli.py) - CLI implementation
- [roampal/server/main.py](roampal/server/main.py) - FastAPI server
- [roampal/mcp/server.py](roampal/mcp/server.py) - MCP server
- [roampal/hooks/session_manager.py](roampal/hooks/session_manager.py) - Exchange tracking