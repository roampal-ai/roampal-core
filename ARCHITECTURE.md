# Roampal Core - Architecture

## Overview

**One command install. AI coding tools get persistent memory.**

Roampal Core provides hook-based memory injection and outcome learning for AI coding tools. Works with **Claude Code** and **OpenCode**. The LLM behind your coding tool sits in the driver seat — Roampal provides the memory layer, the LLM provides the intelligence.

```bash
pip install roampal
roampal init          # Auto-detects installed tools
roampal init --opencode   # Or configure explicitly
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SHARED FastAPI SERVER                          │
│                    (port 27182, single instance)                  │
│                                                                   │
│  UnifiedMemorySystem → ChromaDB    Hook endpoints                │
│  ScoringService, SearchService     /api/hooks/get-context        │
│  KnowledgeGraphService             /api/hooks/stop               │
│  Cross-encoder (optional)          /api/record-outcome           │
│                                    /api/search, etc.             │
└──────────────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
  MCP (HTTP client)    Hook HTTP calls     Plugin HTTP calls
  Claude Code /        from Claude Code    from OpenCode
  OpenCode MCP         hook subprocesses   TypeScript plugin
```

**Architecture (v0.3.2):** MCP servers are thin HTTP clients — no ChromaDB, PyTorch, or sentence-transformers in the MCP process. All access is serialized through a single shared FastAPI server. The first MCP client to start auto-launches the server; subsequent clients detect it's already running.

`roampal start` is available for standalone use (e.g., OpenCode-only setups where no MCP auto-starts the server).

---

## Hook-Based Outcome Scoring

The key innovation: **hooks prompt the LLM to call score_memories()** via soft enforcement - no blocking, just prompt injection.

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
│      → Hybrid search (vector + BM25) + cross-encoder rerank │
│      → 4 slots: reserved working + reserved history          │
│        + 2 best from all collections (Wilson-ranked)         │
│      → Cache doc_ids for scoring                             │
│    ↓                                                         │
│ 3. LLM SEES (only if scoring required):                      │
│    <roampal-score-required>                                  │
│    Previous: User asked "..." You answered "..."             │
│    Current user message: "..."                               │
│    Call score_memories(outcome="...") FIRST                  │
│    </roampal-score-required>                                 │
│                                                              │
│    ═══ KNOWN CONTEXT ═══                                     │
│    • User preference: Python                                 │
│    ═══ END CONTEXT ═══                                       │
│                                                              │
│    Original user message                                     │
│    ↓                                                         │
│ 4. LLM calls score_memories(outcome) then responds           │
│    ↓                                                         │
│ 5. HOOK: Stop (calls /api/hooks/stop)                       │
│    - Stores exchange with doc_id                             │
│    - Sets assistant_completed=True for next turn             │
│    - SOFT ENFORCE: logs warning if score_memories not called │
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

Call score_memories(outcome="...", related=["doc_ids that were relevant"]) FIRST.
- related is optional: omit to score all, or list only the memories you actually used

Separately, record_response(key_takeaway="...") is OPTIONAL - only for significant learnings.
</roampal-score-required>

═══ KNOWN CONTEXT ═══
• User: backend engineer [id:mb_abc123] (memory_bank)
• Preference: TypeScript, detailed explanations [id:mb_def456] (memory_bank)
• JWT refresh token pattern worked for auth issues [id:patterns_ghi789] (5d, 90% proven, patterns)
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
│   │           ├── search_service.py        # Hybrid search + cross-encoder reranking
│   │           ├── scoring_service.py       # Wilson score calculation
│   │           ├── routing_service.py       # KG-based collection routing
│   │           ├── knowledge_graph_service.py # Dual KG system
│   │           ├── outcome_service.py       # Score updates from outcomes
│   │           ├── promotion_service.py     # Promotion/demotion/cleanup
│   │           ├── memory_bank_service.py   # Permanent user facts
│   │           ├── context_service.py       # Context analysis
│   │           ├── config.py                # MemoryConfig
│   │           ├── memory_types.py          # TypedDicts, enums
│   │           └── content_graph.py         # Content KG entity linking
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI server (port 27182)
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py         # MCP server — thin HTTP client (7 tools)
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── session_manager.py          # Exchange tracking (JSONL)
│   │   ├── user_prompt_submit_hook.py  # Context injection + self-healing
│   │   └── stop_hook.py                # Enforcement + storage + self-healing
│   │
│   └── plugins/
│       └── opencode/
│           └── roampal.ts    # OpenCode TypeScript plugin (hooks + events)
```

---

## Memory Collections

| Collection | Purpose | Scorable | Decay |
|------------|---------|----------|-------|
| `books` | Uploaded reference docs | No | Never |
| `working` | Current session context | Yes | 24h — promotes if useful, deleted otherwise |
| `history` | Past conversations | Yes | 30d + Score-based |
| `patterns` | Proven solutions (promoted from history) | Yes | Demoted if score < 0.4, deleted if < 0.2 |
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
 score     score     score ≥ 0.9
 < 0.2:    ≥ 0.7     AND uses ≥ 3
 deleted   AND       AND success_count ≥ 5
           uses ≥ 2  = PATTERN
           = promote

 working: Age > 24h without promotion = deleted
 history: Age > 30d = deleted (cleanup_old_history)
```

---

## API Endpoints

### Hook Endpoints (FastAPI - Port 27182)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hooks/get-context` | POST | 4-slot context injection: reserved working + reserved history + 2 best matches (Wilson-ranked). Returns `scoring_prompt`, `context_only`, and backward-compatible `formatted_injection`. |
| `/api/hooks/stop` | POST | Stores exchange, logs if score_memories not called |
| `/api/record-outcome` | POST | Records outcome, updates scores |

### Memory API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Hybrid search (vector + BM25 + cross-encoder reranking) across collections. Supports `metadata_filters` and `sort_by`. |
| `/api/memory-bank/add` | POST | Add to memory bank (supports `always_inject` flag) |
| `/api/memory-bank/update` | POST | Update memory bank entry |
| `/api/memory-bank/archive` | POST | Archive memory bank entry |
| `/api/record-response` | POST | Store key takeaway in working memory (MCP tool proxy) |
| `/api/context-insights` | POST | Get user profile + relevant memories (MCP tool proxy) |
| `/api/ingest` | POST | Ingest document into books collection |
| `/api/books` | GET | List all ingested books |
| `/api/remove-book` | POST | Remove a book by title |
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
| `delete_memory` | Delete outdated memories |
| `score_memories` | **SOFT ENFORCED** - Score the previous exchange (worked/failed/partial/unknown) |
| `record_response` | **OPTIONAL** - Store key takeaways when transcript won't capture learning |

### Recommended Workflow

```
1. Hook injects context + previous exchange for scoring
2. score_memories(outcome) → Score the previous exchange
3. Respond to user
4. record_response(key_takeaway) → OPTIONAL, only for significant learnings
```

### score_memories (Soft Enforced)

```json
{
  "outcome": "worked" | "failed" | "partial" | "unknown",
  "memory_scores": {
    "doc_id_1": "worked",
    "doc_id_2": "unknown",
    "doc_id_3": "failed"
  }
}
```

The hook presents the previous exchange, cached memories, and current user message. Based on the user's reaction, score the exchange outcome and each memory individually.

**Exchange Outcome Detection:**
- `worked` = user satisfied, says thanks, moves on to new topic
- `failed` = user corrects you, says "no", "that's wrong"
- `partial` = lukewarm response, "kind of", "I guess"
- `unknown` = no clear signal

**Per-Memory Scoring (`memory_scores` dict):**
- Each cached memory gets its own score: `worked`, `failed`, `partial`, or `unknown`
- `worked` = this memory was helpful for the response
- `failed` = this memory was **misleading** (gave bad advice that led you astray)
- `unknown` = didn't use this memory (neutral, not penalized)
- All cached memories MUST be scored; additional memories from context MAY be scored

Example: 4 memories surfaced, 2 were helpful, 1 was misleading, 1 unused:
```
score_memories(
  outcome="worked",
  memory_scores={
    "mem_abc123": "worked",    // +0.20
    "mem_def456": "worked",    // +0.20
    "mem_ghi789": "failed",    // -0.30
    "mem_jkl012": "unknown"    // 0 (skipped)
  }
)
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

Both hook injection and `search_memory` cache doc_ids by session_id. When `score_memories` is called, it:
1. Scores the most recent unscored exchange
2. Uses `memory_scores` dict to score each cached memory individually
3. Applies per-memory outcomes (+0.20 for worked, -0.30 for failed, 0 for unknown)

**How `memory_scores` works:**

The hook prompt lists all cached memories with their doc_ids. The LLM scores each one:

```python
score_memories(
    outcome="worked",
    memory_scores={
        "doc_id_1": "worked",   # +0.20 — was helpful
        "doc_id_2": "failed",   # -0.30 — was misleading
        "doc_id_3": "unknown"   #  0    — didn't use
    }
)
```

This ensures:
- Hook-injected memories get scored (cached under client's session_id)
- MCP search results get scored (cached under MCP session_id)
- Per-memory granularity: helpful memories are boosted, misleading ones are demoted, unused ones are neutral
- Each MCP client gets a unique `mcp_{uuid}` session_id for cache isolation

---

## Score Updates

When `score_memories(outcome)` is called:

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

### KG Methods (on UnifiedMemorySystem)

These methods live on `UnifiedMemorySystem`, which delegates to `KnowledgeGraphService` internally:

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
# Tracks: "In coding context, score_memories on memory_bank worked"
# Key format: "{context_type}|{action_type}|{collection}"
# Example: "coding|score_memories|memory_bank" → 85% success rate
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
roampal init                # Auto-detect and configure installed tools
roampal init --claude-code  # Configure Claude Code explicitly
roampal init --opencode     # Configure OpenCode explicitly
roampal start               # Start the HTTP server manually
roampal stop                # Stop the HTTP server
roampal status              # Check server status
roampal stats               # Show memory statistics
roampal doctor              # Diagnose installation issues
roampal ingest <file>       # Ingest .txt/.md/.pdf into books collection
roampal books               # List all ingested books
roampal remove <title>      # Remove a book by title
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

**Note:** The first MCP client to start auto-launches the FastAPI server. For standalone use (e.g., OpenCode-only setups or debugging), use `roampal start` or `roampal start --dev`.

### Dev Mode

Set `ROAMPAL_DEV=1` environment variable to isolate dev from production:
- **Port:** Production uses 27182, dev uses 27183
- **Data:** Production: `%APPDATA%/Roampal/data`, Dev: `%APPDATA%/Roampal_DEV/data`

**Important:** Both MCP server AND hooks need `ROAMPAL_DEV=1` configured SEPARATELY:
- MCP server gets it from `~/.claude/mcp.json` env config
- Hooks get it from `~/.claude/settings.json` env config (NOT automatic!)

**CRITICAL:** Hooks run as subprocesses spawned by Claude Code. They do NOT inherit your terminal's env vars. You MUST add `"env": {"ROAMPAL_DEV": "1"}` to settings.json for hooks to connect to DEV server.

Example `~/.claude/settings.json`:
```json
{
  "env": {
    "ROAMPAL_DEV": "1"
  },
  "hooks": { ... }
}
```

Bug (2025-12-26): Missing env section in settings.json caused hooks to silently connect to PROD (27182) while MCP connected to DEV (27183). This went unnoticed for 15 days because MCP still worked.

You can also set `ROAMPAL_DATA_PATH` environment variable for custom paths.

### Dev Mode Implementation (v0.2.0)

**SINGLE SOURCE OF TRUTH:** All code that determines DEV vs PROD mode MUST use the `is_dev_mode()` helper function. Never check `args.dev` directly.

**The Pattern (cli.py):**
```python
def is_dev_mode(args=None) -> bool:
    """
    SINGLE SOURCE OF TRUTH for DEV mode detection.

    Checks (in order):
    1. args.dev flag (if args provided)
    2. ROAMPAL_DEV environment variable

    ALL commands MUST use this. Never check args.dev directly.
    """
    if args is not None and getattr(args, 'dev', False):
        return True
    return os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")


def get_port(args=None) -> int:
    """Get port based on DEV/PROD mode."""
    return DEV_PORT if is_dev_mode(args) else PROD_PORT
```

**Bug History (2025-12-26):** Multiple CLI commands only checked `args.dev`, ignoring the `ROAMPAL_DEV` environment variable. This caused MCP (using env var from `.mcp.json`) to connect to DEV while CLI commands connected to PROD - data ended up in wrong database. Fixed by centralizing all DEV mode checks through `is_dev_mode()`.

**Why This Matters:**
- MCP server gets `ROAMPAL_DEV=1` from `.mcp.json` env config
- Hooks detect `ROAMPAL_DEV` from environment
- CLI commands must also check the env var, not just `--dev` flag
- Single helper prevents this class of bug from recurring

### What `roampal init` Does

Automatically detects and configures installed tools. Use `--claude-code` or `--opencode` to configure explicitly.

**Claude Code** (`~/.claude/` detected):

1. Creates `~/.claude/settings.json` with:
   - UserPromptSubmit hook → `python -m roampal.hooks.user_prompt_submit_hook`
   - Stop hook → `python -m roampal.hooks.stop_hook`
   - Auto-allow permissions for all roampal MCP tools

2. Creates `~/.claude/.mcp.json` with roampal-core MCP server

**OpenCode** (`~/.config/opencode/` detected):
1. Creates `opencode.json` with roampal-core MCP server + `PYTHONPATH` env
2. Installs TypeScript plugin to `~/.config/opencode/plugin/roampal.ts`
3. Plugin handles context injection (via `chat.message` + `system.transform` hooks) and exchange capture (via `event` hook)

**All:**

- Creates data directory at `%APPDATA%/Roampal/data` (Windows) or platform equivalent
- Supports `--dev` flag for isolated development environment

---

## Configuration

### Claude Code Settings (auto-generated by `roampal init`)

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m roampal.hooks.user_prompt_submit_hook"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m roampal.hooks.stop_hook"
          }
        ]
      }
    ]
  },
  "permissions": {
    "allow": [
      "mcp__roampal-core__search_memory",
      "mcp__roampal-core__add_to_memory_bank",
      "mcp__roampal-core__update_memory",
      "mcp__roampal-core__delete_memory",
      "mcp__roampal-core__get_context_insights",
      "mcp__roampal-core__record_response",
      "mcp__roampal-core__score_memories"
    ]
  }
}
```

### MCP Server Config (auto-generated, all clients use the same server.py)

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

### OpenCode Config (`opencode.json`, auto-generated by `roampal init --opencode`)

```json
{
  "mcp": {
    "roampal-core": {
      "command": "python",
      "args": ["-m", "roampal.mcp.server"],
      "env": {
        "PYTHONPATH": "<roampal-core-dir>"
      }
    }
  }
}
```

Plugin installed to `~/.config/opencode/plugins/roampal.ts`. Context injection is handled by the plugin (not hooks):
- `chat.message` → fetches context + caches scoring data
- `experimental.chat.system.transform` → injects memory context into system prompt
- `experimental.chat.messages.transform` → injects scoring prompt via deep-cloned user message (clone avoids mutating UI-visible objects)
- `session.idle` → stores exchange, then runs sidecar scoring ONLY if main LLM didn't call `score_memories` (prevents double-scoring memories)

---

## Key Design Decisions

### 1. Soft Enforcement via Hook Prompting

**Problem:** LLMs don't reliably call score_memories() without prompting.

**Solution:**
- UserPromptSubmit hook injects scoring prompt with previous exchange
- Hook prompt tells LLM to call score_memories(outcome) FIRST
- Stop hook logs warning if not called (soft enforcement)
- Prompt injection does 95% of the work - no hard blocking needed
- Separate tools: score_memories (scoring) vs record_response (key takeaways)

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
    "chromadb>=1.0.0,<2.0.0",
    "sentence-transformers>=2.2.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "mcp>=1.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "rank-bm25>=0.2.0",
    "nltk>=3.8.0",
]
```

---

## Self-Healing

All three entry points (hooks, MCP, plugin) auto-recover if the FastAPI server goes down:

| Entry Point | Recovery Method |
|-------------|----------------|
| Python hooks (`user_prompt_submit_hook`, `stop_hook`) | `_restart_server()` — kills stale port process, spawns fresh server, polls `/api/health` for 15s, retries |
| MCP server (`server.py`) | `_ensure_server_running()` — health check before every tool call, auto-restart with 3s timeout |
| OpenCode plugin (`roampal.ts`) | `restartServer()` — same kill/spawn/poll pattern, `_restartInProgress` guard prevents concurrent restarts |

The `user_prompt_submit_hook` / `chat.message` fires first every turn — if the server is down, it recovers before context injection.

---

## Search Pipeline

The `search()` method delegates to `SearchService` which provides a full retrieval pipeline:

1. **Hybrid search**: Vector similarity + BM25 keyword matching with Reciprocal Rank Fusion
2. **Cross-encoder reranking**: Top-30 candidates re-scored by `ms-marco-MiniLM-L-6-v2` (optional — graceful fallback if unavailable)
3. **Wilson scoring**: Confidence intervals with dynamic weight shifts (NEW → EMERGING → ESTABLISHED → PROVEN)
4. **Collection-specific boosts**: patterns priority, memory_bank 80/20 quality+Wilson blend (after 3 uses), books recency
5. **Entity boost**: Content KG quality-weighted entity matching
6. **KG routing**: Intelligent collection selection when no collections specified

Falls back to inline vector + Wilson scoring if SearchService fails or isn't initialized.

---

## Future Considerations

1. **Cursor Support**: Implemented but blocked by Cursor v2.4.7 bug — context injection doesn't reach the AI. Will go live when Cursor fixes `agent_message`.
2. **OpenCode Support**: ✅ Full support via TypeScript plugin (v0.3.2)
3. **Dashboard**: Web UI for memory management
4. **Multi-User**: Currently single-user design
5. **Encryption**: Data at rest is not encrypted
6. **KG Service**: ✅ Knowledge Graphs shared with Roampal Desktop (v0.2.8)

---

## Related Files

- [README.md](README.md) - Quick start guide
- [roampal/cli.py](roampal/cli.py) - CLI implementation
- [roampal/server/main.py](roampal/server/main.py) - FastAPI server
- [roampal/mcp/server.py](roampal/mcp/server.py) - MCP server
- [roampal/hooks/session_manager.py](roampal/hooks/session_manager.py) - Exchange tracking
- [roampal/plugins/opencode/roampal.ts](roampal/plugins/opencode/roampal.ts) - OpenCode plugin
