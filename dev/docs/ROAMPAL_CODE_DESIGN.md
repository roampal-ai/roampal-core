# Roampal Code - Design Document

**Status:** Draft
**Date:** 2026-02-14
**Author:** Roampal Team

## What Is This?

A standalone CLI coding tool that uses roampal-core as its memory layer. Think Claude Code / OpenCode / Aider, but instead of stuffing the entire conversation history into context, it feeds the LLM:

1. **Last 4 exchanges** (sliding window)
2. **Roampal scored context** (memories that actually matter)
3. **Standard coding tools** (file ops, bash, search)

The thesis: you don't need 200k tokens of conversation history. You need the right 2k tokens of context. Roampal already knows how to pick those.

---

## Why Build This?

### The Problem With Existing Tools

| Tool | Context Strategy | Waste |
|------|-----------------|-------|
| Claude Code | Full conversation history | Burns tokens on stale context |
| OpenCode | Full conversation history | Same |
| Aider | Repo map + full chat | Better, still bloated |

All of them treat every message as equally important. They don't learn. They don't score. They just... keep everything.

### Roampal Code's Approach

- **4-exchange window** = ~2-4k tokens of recent context
- **Roampal injection** = ~500-1500 tokens of scored, relevant memories
- **Total context per turn** = ~5-8k tokens instead of 50-200k
- **Outcome scoring** = context gets better over time

This is Intelligence Over Force applied to coding tools.

---

## Architecture

```
roampal-core/
├── roampal/
│   ├── core modules       <-- Existing (NO changes)
│   │   (memory, scoring, ChromaDB, FastAPI, CLI)
│   │
│   └── code/              <-- New subpackage (all new files)
│       (REPL, LLM client, tools, agents, sidecar)
│       │
│       │ HTTP calls to localhost:27182
│       ▼
│   ┌──────────────┐
│   │ roampal-core │  <-- Running server (existing)
│   │  FastAPI      │
│   └──────────────┘
│       │
│   ┌──────────────┐
│   │ LLM Provider │  <-- Anthropic, OpenAI, Ollama, etc.
│   └──────────────┘
```

### Key Principle: New files only. Zero changes to existing core.

`roampal/code/` lives in the repo but talks to core's running server over HTTP — same as the OpenCode plugin, same as channels. It does NOT import core's internals. The only touch point to existing code is one `roampal code` subcommand added to `cli.py`.

---

## Data Path Convention

Follows the same prod/dev split as roampal-core. Same env var, same logic.

```
# PROD (default)
Windows:  %APPDATA%/Roampal/data          -> port 27182
macOS:    ~/Library/Application Support/Roampal/data
Linux:    ~/.local/share/roampal/data

# DEV (ROAMPAL_DEV=1)
Windows:  %APPDATA%/Roampal_DEV/data      -> port 27183
macOS:    ~/Library/Application Support/Roampal_DEV/data
Linux:    ~/.local/share/roampal_dev/data
```

**`roampal/code/` doesn't manage the data path.** It just connects to the right roampal-core port:
- Default: `http://localhost:27182` (prod)
- With `ROAMPAL_DEV=1`: `http://localhost:27183` (dev)

roampal-core owns the data. `roampal code` is a client. Same memory, same ChromaDB, same scored context — regardless of whether you're using `roampal code`, Claude Code with the MCP plugin, or OpenCode with the TS plugin. All clients share one memory store.

```python
def get_roampal_url():
    """Connect to the right roampal-core instance based on dev/prod mode."""
    if os.environ.get('ROAMPAL_DEV', '').lower() in ('1', 'true', 'yes'):
        return "http://localhost:27183"
    return "http://localhost:27182"
```

---

## Modularity

Everything is swappable. The architecture is a pipeline of interfaces, not a monolith.

```
┌──────────────────────────────────────────────────────────────────┐
│                         roampal-code                              │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ ┌───────┐ │
│  │ Main LLM │  │ Memory   │  │ Tools    │  │Sidecar │ │ Agent │ │
│  │ Provider │  │ Provider │  │ Registry │  │Provider│ │Provider│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ └───┬───┘ │
│       │              │             │             │          │     │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌───▼────┐ ┌───▼───┐ │
│  │Anthropic │  │ Roampal  │  │ Native   │  │Haiku   │ │Sonnet │ │
│  │OpenAI    │  │ HTTP     │  │ + MCP    │  │Ollama  │ │Ollama │ │
│  │Ollama    │  │ Client   │  │ + Custom │  │Groq    │ │Haiku  │ │
│  │Custom    │  │          │  │ + Agent  │  │Custom  │ │Custom │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ └───────┘ │
└──────────────────────────────────────────────────────────────────┘

Three LLM roles, independently configurable:
  Main    = coding brain (quality)
  Agent   = research/exploration (can be cheaper, needs tool use)
  Sidecar = scoring/summarization (cheapest, no tools)
```
```

### What's Swappable

| Layer | Interface | Implementations | Notes |
|-------|-----------|----------------|-------|
| **LLM Provider** | `LLMClient` | Anthropic, OpenAI, Ollama, custom endpoint | Main coding LLM |
| **Sidecar Provider** | `LLMClient` (same interface) | Anthropic Haiku, Ollama small, Groq, custom | Scoring + summarization |
| **Memory Provider** | `MemoryClient` | Roampal HTTP (default) | Could swap for other memory systems |
| **Tools** | `ToolRegistry` | Native tools + MCP tools + custom tools | Pluggable at startup |
| **Permissions** | `PermissionHandler` | Simple y/n, trust levels, auto-approve-all | Controls tool execution |
| **UI** | `OutputRenderer` | Rich terminal, plain text, JSON | How responses display |

### Tool Registry Pattern

Tools are registered, not hardcoded. Adding a tool = drop a file + register it.

```python
class ToolRegistry:
    """Discovers and manages all available tools."""

    def __init__(self):
        self.tools = {}  # name -> {schema, execute_fn, permission_level}

    def register(self, name, schema, execute_fn, permission="auto"):
        """Register a tool. Can be native, MCP, or custom."""
        self.tools[name] = {
            "schema": schema,
            "execute": execute_fn,
            "permission": permission,  # "auto" | "ask" | "deny"
        }

    def register_mcp_tools(self, mcp_loader):
        """Bulk register all tools from connected MCP servers."""
        for tool in mcp_loader.list_tools():
            self.register(
                name=tool["name"],
                schema=tool["input_schema"],
                execute_fn=lambda params, t=tool: mcp_loader.execute_tool(t["name"], params),
                permission="ask",  # MCP tools require approval by default
            )

    def register_custom_tool(self, tool_module):
        """Register a custom tool from a Python module."""
        self.register(
            name=tool_module.NAME,
            schema=tool_module.SCHEMA,
            execute_fn=tool_module.execute,
            permission=tool_module.PERMISSION,
        )

    def get_schemas(self):
        """Get all tool schemas for the LLM API call."""
        return [{"name": n, **t["schema"]} for n, t in self.tools.items()]

    async def execute(self, tool_name, params):
        """Execute a tool by name. Checks permissions first."""
        tool = self.tools[tool_name]
        # Permission check happens here
        return await tool["execute"](params)
```

### Custom Tools

Users can drop custom tools into a directory and they get picked up:

```yaml
# ~/.config/roampal-code/config.yaml
custom_tools_dir: ~/.config/roampal-code/tools/
```

```python
# ~/.config/roampal-code/tools/deploy.py
NAME = "deploy"
PERMISSION = "ask"
SCHEMA = {
    "description": "Deploy the current project to staging",
    "input_schema": {
        "type": "object",
        "properties": {
            "environment": {"type": "string", "enum": ["staging", "production"]}
        },
        "required": ["environment"]
    }
}

async def execute(params):
    env = params["environment"]
    # Custom deployment logic
    result = subprocess.run(["./deploy.sh", env], capture_output=True)
    return {"output": result.stdout.decode()}
```

### Swapping the Memory Provider

Default is Roampal HTTP client. But the interface is clean enough to swap:

```python
class MemoryClient(Protocol):
    """Interface for memory providers. Roampal is default, but could be swapped."""

    async def get_context(self, user_message: str, session_id: str) -> dict:
        """Get context for the current turn (cold start + known context)."""
        ...

    async def record_exchange(self, session_id: str, exchange: dict) -> None:
        """Record an exchange for scoring/summarization."""
        ...

    async def search(self, query: str, limit: int = 5) -> list:
        """Search memory."""
        ...

    async def store(self, content: str, tags: list = None) -> str:
        """Store a permanent fact."""
        ...

    async def update(self, old_content: str, new_content: str) -> None:
        """Update an existing fact."""
        ...

    async def delete(self, content: str) -> None:
        """Delete a fact."""
        ...
```

The Roampal HTTP client implements this. Someone could write a different backend (plain vector DB, file-based, etc.) and plug it in. But Roampal is the whole point, so this is more for clean architecture than practical swapping.

---

## Components

### 1. CLI Interface (`roampal_code/cli.py`)

```
$ roampal-code
> Read the auth module and find the bug in token refresh

[Reading roampal_code/auth.py...]
[Found issue at line 47: token expiry check uses <= instead of <]

The bug is in auth.py:47. The expiry check...

> Fix it

[Editing roampal_code/auth.py:47...]
[Done]

> /commit
```

- Python `prompt_toolkit` for rich terminal input
- Markdown rendering for output (like `rich` library)
- Streaming responses from LLM
- `/slash` commands for common actions

### 2. Conversation Manager (`roampal_code/conversation.py`)

The core differentiator. Instead of keeping everything:

```python
class ConversationManager:
    """Maintains a sliding window of recent exchanges."""

    def __init__(self, window_size=4):
        self.exchanges = deque(maxlen=window_size)
        self.session_id = generate_session_id()

    def add_exchange(self, user_msg, assistant_msg, tool_calls=None):
        self.exchanges.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "tool_calls": tool_calls,  # keep tool context
            "timestamp": now()
        })

    def build_messages(self, roampal_context):
        """Build the message array sent to the LLM."""
        messages = []

        # System prompt with roampal context (no scoring prompt - sidecar handles that)
        system = self._build_system_prompt(roampal_context)

        # Only the last N exchanges - raw, not summarized
        for exchange in self.exchanges:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        return system, messages
```

**What's one "exchange"?**
A complete user turn: the user message, all tool calls/results in the assistant's response loop, and the final assistant text. A single exchange might be: user → assistant (tool_use) → tool_result → assistant (tool_use) → tool_result → assistant (text). The entire tool loop is one exchange.

```python
# One exchange stores the full message sequence
exchange = {
    "user": "Fix the auth bug",
    "messages": [                         # Full message sequence for LLM replay
        {"role": "user", "content": "Fix the auth bug"},
        {"role": "assistant", "content": [tool_use_block, ...]},
        {"role": "user", "content": [tool_result_block, ...]},
        {"role": "assistant", "content": "Fixed. The issue was..."},
    ],
    "summary": "Fixed auth bug",          # For sidecar / session-start context
    "token_count": 3200,                  # Tracked for budget enforcement
    "timestamp": "2026-02-14T10:30:00Z",
}
```

**Per-exchange token limits:**
Individual exchanges can blow up (LLM reads 5 large files, makes 15 tool calls). Without limits, 4 oversized exchanges could hit 100k+ tokens. Safeguard: if an exchange exceeds a configurable max (default ~8k tokens), tool results within it get summarized before storage. The raw exchange is still sent to sidecar for scoring, but the windowed version is compressed.

**What happens to old exchanges?**
They get summarized by the sidecar and stored as memories. So they're not gone — they're compressed and scored. If they were important, they'll surface via Roampal context injection when relevant.

**Why no exchange summaries in the system prompt?**
The main LLM always has 4 raw exchanges in its message window. That IS its recent context. For anything older, it uses `search_memory`. No need to double-inject summaries of things already in the window.

### 3. Roampal Integration (`roampal_code/memory.py`)

Two options for how roampal-code talks to roampal-core:

#### Option A: HTTP Client (Recommended)

Uses the existing FastAPI server, same as the OpenCode plugin. No coupling.

```python
class RoampalClient:
    """HTTP client to roampal-core FastAPI server."""

    def __init__(self, base_url="http://localhost:27182"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def get_context(self, user_message, session_id):
        """Get context injection + scoring prompt."""
        resp = await self.client.post(f"{self.base_url}/api/hooks/get-context", json={
            "user_message": user_message,
            "session_id": session_id,
        })
        return resp.json()
        # Returns: context_only, scoring_prompt, scoring_required, etc.

    async def record_outcome(self, session_id, exchange, memory_scores, summary):
        """Record exchange outcome + per-memory scores."""
        await self.client.post(f"{self.base_url}/api/record-outcome", json={
            "conversation_id": session_id,
            "exchange": exchange,
            "memory_scores": memory_scores,
            "exchange_summary": summary,
        })

    async def search(self, query, collections=None, limit=5):
        """Direct memory search."""
        resp = await self.client.post(f"{self.base_url}/api/search", json={
            "query": query,
            "collections": collections,
            "limit": limit,
        })
        return resp.json()
```

#### Option B: Direct Import

Import `UnifiedMemorySystem` and use it directly. More coupled but no server needed.

```python
from roampal import UnifiedMemorySystem, MemoryConfig

system = UnifiedMemorySystem(data_path=default_path())
await system.initialize()
results = await system.search("auth token pattern", limit=5)
```

**Recommendation:** Option A. It's how the existing ecosystem works, supports multiple clients sharing one memory store, and means roampal-code doesn't need to manage ChromaDB lifecycle.

### 4. Tool System (`roampal_code/tools/`)

Based on research across Claude Code, OpenCode, Codex CLI, Gemini CLI, Amazon Q, Cline, and Aider - every coding CLI converges on the same core tools. Here's what roampal-code needs:

```
roampal_code/tools/
    __init__.py          # Tool registry + schema translation
    file_read.py         # Read files (line numbers, images, PDFs)
    file_write.py        # Create/overwrite files
    file_edit.py         # String replacement edits
    bash.py              # Shell execution with timeout + background support
    glob.py              # File pattern matching (sorted by mod time)
    grep.py              # Content search via regex (ripgrep wrapper)
    list_dir.py          # Directory listing
    agent.py             # Spawn sub-agents (context shields)
    memory.py            # search, add, update, delete memory (NO scoring)
```

Each tool follows a simple pattern:

```python
# Every tool exports these three things
NAME = "read_file"
PERMISSION = "auto"  # auto | ask | deny

SCHEMA = {
    "description": "Read a file's contents with line numbers",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute file path"},
            "offset": {"type": "integer", "description": "Start line (optional)"},
            "limit": {"type": "integer", "description": "Max lines (optional)"},
        },
        "required": ["path"]
    }
}

async def execute(params):
    path = params["path"]
    # ... read file, return content with line numbers
    return {"content": numbered_content}
```

**Tool count for MVP:** 12 built-in tools (+ MCP tools discovered at startup)

| Tool | Permission | Notes |
|------|-----------|-------|
| **File operations** | | |
| `read_file` | auto | Line numbers, image support, PDF support. Must read before edit (enforced). |
| `write_file` | ask | Create/overwrite. Absolute paths only. |
| `edit_file` | ask | Exact string replacement. Fails if match isn't unique (use context). Requires prior read. |
| `list_dir` | auto | Directory contents with optional glob filter. |
| **Search** | | |
| `glob` | auto | Find files by pattern (e.g. `**/*.py`). |
| `grep` | auto | Regex search via ripgrep. Modes: content, files_with_matches, count. |
| **Execution** | | |
| `bash` | ask | Shell commands. Max 600s timeout. Output truncated at 30k chars. Background support. |
| **Agents** | | |
| `agent` | auto | Spawn sub-agent with own context window. For heavy ops: web research, codebase exploration, large file analysis, MCP operations. Agents have web access + read-only tools internally. Returns concise summary. |
| **Roampal Memory** | | |
| `search_memory` | auto | Search past context (when 4-exchange window isn't enough). |
| `add_memory` | ask | Store permanent facts. |
| `update_memory` | ask | Update existing facts (e.g. user corrects a preference). |
| `delete_memory` | ask | Remove outdated/wrong facts. |
| **MCP** | | |
| `[mcp_*]` | ask | Any tools from connected MCP servers. Discovered at startup, prefixed with server name. |

**What's NOT a main LLM tool (and why):**
- ~~`web access`~~ - Agents handle all web operations (search + URL fetching) in their own context. Web content would bloat the 4-exchange window. Main LLM just spawns an agent: "look up X" → gets concise answer back.
- ~~`ask_user`~~ - It's a REPL. The LLM just asks in plain text, user types back. No special tool needed.
- ~~`score_memories`~~ - Sidecar handles all scoring after each exchange.
- ~~`record_response`~~ - Redundant. Sidecar already summarizes and scores every exchange automatically.

**Key design decisions from research:**

1. **read-before-write enforced** - Edit and Write on existing files require a prior Read in the same session. Every tool (Claude Code, OpenCode, Cline) does this. Prevents blind file corruption.

2. **All web operations go through agents** - Main LLM never searches or fetches URLs directly. Agents handle web_search and web_fetch in their own context and return concise results. Prevents context bloat and prompt injection.

3. **bash output truncation** - 30k char limit (Claude Code's number). Full output saved to file if needed. Prevents context blowout from verbose commands.

4. **Full Roampal memory CRUD** - search, add, update, delete. No scoring tools (sidecar handles that). No record_response (sidecar summarizes automatically).

5. **Schema translation** - Tool schemas stored in Anthropic format internally. The `LLMClient` translates to OpenAI format when using OpenAI-compatible providers. One source of truth for tool definitions.

### Tool Result Handling

```python
# Truncation thresholds
MAX_BASH_OUTPUT = 30_000        # chars before truncation
MAX_FILE_LINES = 2_000          # default read limit
MAX_LINE_LENGTH = 2_000         # per-line truncation
MAX_TOOL_RESULT = 50_000        # absolute cap on any tool result

# When output exceeds threshold:
# 1. Save full output to temp file
# 2. Return truncated preview + file path to LLM
# 3. LLM can read the full file if needed
```

### 5. LLM Client (`roampal_code/llm.py`)

Supports multiple providers through a unified interface. Same client class used for both main LLM and sidecar, just different config.

```python
class LLMClient:
    """Multi-provider LLM client. Works with Anthropic, OpenAI-compatible, Ollama."""

    def __init__(self, provider, model, api_key=None, base_url=None):
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            # OpenAI-compatible (covers OpenAI, Groq, Ollama, any custom endpoint)
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(self, system, messages, tools):
        """Stream a response, handling tool calls in a loop."""
        # Provider-specific API call (Anthropic vs OpenAI format)
        # Both support streaming + tool use
        # Tool call loop: execute tools, append results, continue until text response
```

**Two instances at runtime:**
```python
# From config
main_llm = LLMClient(provider="anthropic", model="claude-sonnet-4-5-20250929", ...)
sidecar_llm = LLMClient(provider="ollama", model="llama3.2:3b", base_url="http://localhost:11434/v1")
```

### 6. System Prompt (`roampal_code/prompts.py`)

**Lean by design.** The system prompt is rebuilt every turn. It contains:
1. How to behave + what tools do (static)
2. Cold start profile (from Roampal - user identity, preferences)
3. KNOWN CONTEXT (from Roampal - scored memories relevant to this turn)

All three are assembled into a single system prompt before each LLM call.

```python
def build_system_prompt(cold_start, known_context, mcp_index, cwd,
                        project_instructions=None, git_context=None,
                        session_history=None):
    """Built fresh every turn. Context changes, prompt stays lean."""

    return f"""You are a coding assistant with persistent memory.

## Rules
- Always read a file before editing or overwriting it
- Use the agent tool for: web research, large file analysis, multi-file exploration, MCP operations, anything that would produce large output
- Don't store session-specific facts in memory (the system handles that automatically)
- When the user corrects a stored fact, use update_memory or delete_memory

## Tools
Coding: read_file, write_file, edit_file, bash, glob, grep, list_dir
Memory: search_memory, add_memory, update_memory, delete_memory
Agent: spawn a sub-agent for heavy operations (web, large searches, MCP)
{mcp_index if mcp_index else ""}

## Context
- You see the last 4 exchanges as conversation history
- Relevant memories from past sessions appear below (scored — higher = more reliable)
- Use search_memory if you need more context than what's provided
- Use add_memory for permanent facts worth preserving across sessions

{f"## Project Instructions{chr(10)}{project_instructions}{chr(10)}" if project_instructions else ""}
{f"## Git{chr(10)}{git_context}{chr(10)}" if git_context else ""}
{f"## Recent Session History{chr(10)}{session_history}{chr(10)}" if session_history else ""}
{cold_start}

{known_context}

## Working Directory
{cwd}
"""
```

Still lean. The rules section adds ~4 lines that prevent the most common wasted tool calls (editing without reading) and bad memory habits. Everything else is structural — the template slots for project instructions, git context, and session history only appear when populated.

**Session start context:** On fresh session start (0 exchanges in window), the system prompt includes the last 4 exchange summaries by time from roampal-core. This gives the LLM chronological continuity ("here's what happened last session") without full exchanges. Roampal already stores these summaries (sidecar creates them after each exchange) and has the time-based filtering mechanism. After the first few real exchanges fill the window, these summaries are no longer needed — the raw exchanges take over.

```python
# On session start, fetch recent summaries
recent = await roampal.search(query=None, days_back=1, sort_by="recency", limit=4)
session_history = "\n".join(
    f"  [{r['age']}] {r['summary']}" for r in recent if r.get('summary')
)
# Injected as "## Recent Session History" in system prompt
```

**What's NOT in the system prompt:**
- Scoring instructions (sidecar handles it)
- Exchange summaries mid-session (raw exchanges are in the message window)
- Verbose tool descriptions (tool schemas have their own descriptions)
- Personality directives (Roampal's cold start profile handles user preferences)
- Output formatting rules (cold start profile captures user preferences)
- Safety instructions (permission system handles this — the LLM doesn't need to self-police)
- Code quality rules (go in ROAMPAL.md per-project — not every project wants the same style)

**What IS rebuilt each turn:**
- KNOWN CONTEXT block (Roampal searches relevant memories based on the current user message)
- Cold start profile (stable but included for consistency)
- MCP server index (stable after startup)
- Project instructions from ROAMPAL.md (if present in project root)
- Git context: current branch + dirty/clean status (at session start, refreshed on git ops)
- Session history: last 4 exchange summaries by time (only on fresh session start, before window fills)

---

## Scoring Flow

**Key decision: The main LLM does NOT score. Sidecar handles everything.**

This is simpler than the OpenCode implementation. No `score_memories` tool for the main LLM, no scoring prompt injection, no "did it score or not?" detection. The main LLM just codes. The sidecar does all bookkeeping.

```
1. User sends message
2. roampal-code calls /api/hooks/get-context
   -> Gets: KNOWN CONTEXT (context_only) + cached memory IDs
   -> Ignores: scoring_prompt (not needed - sidecar handles it)
3. Build system prompt: base + cold start profile + KNOWN CONTEXT
4. Send to LLM with last 4 raw exchanges + current user message
5. LLM responds:
   - Uses coding tools to do the work
   - Uses search_memory / add_memory if it needs more context
   - Responds to user
6. Exchange complete. roampal-code triggers sidecar:
   a. Sends previous exchange (user + assistant) to sidecar
   b. Sidecar summarizes exchange (~300 chars)
   c. Sidecar scores exchange outcome (worked/failed/partial/unknown)
   d. Sidecar scores individual cached memories from that turn
   e. Posts results to /api/record-outcome
7. Old exchanges fall off the 4-exchange window
   - But they're preserved as scored summaries in Roampal
   - If they were important, they'll surface via context injection later
```

**Why no LLM scoring?**
- Main LLM never wastes a turn on scoring overhead
- No scoring prompt = cleaner system prompt = better coding performance
- No "did it score?" detection logic (the OpenCode complexity)
- Sidecar is cheap (Haiku/free models) and runs async after each turn
- Sidecar has both the exchange AND the cached memory list, which is enough for attribution

**Why the main LLM doesn't need exchange summaries injected:**
- It always has the last 4 raw exchanges in its message window
- That's the immediate context. For anything older, it has `search_memory`.
- Exchange summaries still get stored by the sidecar - they're just for Roampal's memory system, not for injecting back into the main LLM.

---

### 7. Sidecar Orchestrator (`roampal_code/sidecar.py`)

Runs after every exchange. The main LLM never sees this.

```python
class SidecarOrchestrator:
    """Handles all scoring/summarization after each exchange.
    Auto-approved — runs silently, no user permission prompts."""

    def __init__(self, sidecar_llm, roampal_client, cached_memory_ids):
        self.llm = sidecar_llm                      # Cheap model (Haiku, Ollama, etc.)
        self.roampal = roampal_client
        self.cached_memory_ids = cached_memory_ids   # From get-context

    async def process_exchange(self, user_msg, assistant_msg, cached_memories):
        """Called after each LLM response. Runs async, no user interaction."""

        # 1. Sidecar LLM generates summary + scores
        scoring_prompt = self._build_scoring_prompt(
            user_msg, assistant_msg, cached_memories
        )
        result = await self.llm.chat(
            system="You are a scoring assistant. Evaluate the exchange and score memories.",
            messages=[{"role": "user", "content": scoring_prompt}],
            tools=None,  # Sidecar has no tools
        )
        parsed = self._parse_scoring_response(result)

        # 2. Post results to roampal-core
        await self.roampal.record_outcome(
            exchange={"user": user_msg, "assistant": assistant_msg},
            memory_scores=parsed["memory_scores"],    # {id: worked/failed/partial/unknown}
            summary=parsed["exchange_summary"],        # ~300 char summary
            outcome=parsed["exchange_outcome"],        # worked/failed/partial/unknown
        )

    def _build_scoring_prompt(self, user_msg, assistant_msg, cached_memories):
        """Build the prompt that asks sidecar to score the exchange."""
        memory_block = "\n".join(
            f"  [{m['id']}] {m['content'][:200]}" for m in cached_memories
        )
        return f"""Score this coding exchange. Respond ONLY with JSON, no other text.

User: {user_msg[:500]}
Assistant: {assistant_msg[:500]}

Memories injected into the assistant's context:
{memory_block}

Score each memory:
- worked = this memory was helpful to the response
- partial = somewhat helpful
- unknown = wasn't used
- failed = was MISLEADING (caused errors or wrong info)

Respond with this exact JSON structure:
{{
  "exchange_summary": "<~300 char summary of what happened>",
  "exchange_outcome": "worked|failed|partial|unknown",
  "memory_scores": {{
    "<memory_id>": "worked|failed|partial|unknown"
  }}
}}"""
```

**What the sidecar sees:**
- The user message and assistant response (full exchange)
- The cached memories that were injected into context (content + IDs)
- It determines: was this exchange successful? Which memories helped?

**What the sidecar does NOT need:**
- The tool call details (it's scoring the exchange, not the tool usage)
- The system prompt
- Previous exchanges (it scores one exchange at a time)

**Auto-approved:** Sidecar operations never prompt the user. Scoring and summarization happen silently after every exchange — same pattern as the OpenCode sidecar implementation. The user doesn't know or care that scoring is happening. It just works in the background.

### 8. Agent System (`roampal_code/agent.py`)

Agents are **context shields**. With a 4-exchange window, any heavy operation that dumps thousands of tokens into context is a problem. Agents solve this by running in their own context window and returning concise results.

**Why agents are Phase 1, not Phase 3:**
Without agents, one `web_fetch` or large `grep` result eats your entire window budget. The 4-exchange window only works if individual exchanges stay lean. Agents are what make the small window viable.

```python
class AgentRunner:
    """Spawns sub-agents with their own context. Returns concise results."""

    def __init__(self, llm_client, tool_registry, roampal_client):
        self.llm = llm_client        # Can be main LLM or a separate cheaper model
        self.tools = tool_registry    # Read-only subset by default
        self.roampal = roampal_client

    async def run(self, task_description, tool_subset=None, max_turns=10):
        """Run an agent task. Returns a concise summary string."""

        # Agent gets:
        # - A task description (from main LLM)
        # - Read-only tools by default (read, glob, grep, list_dir, search_memory)
        # - Web access (search + fetch) for internet research
        # - Its own context window (independent from main)
        # - Access to same Roampal memory

        # Agent does NOT get:
        # - The 4-exchange window (no conversation history)
        # - Write tools (unless explicitly granted)
        # - The ability to spawn more agents (no recursion)

        system = f"""You are a research agent. Complete the task and return a concise summary.
Keep your final answer under 500 tokens. Be specific and actionable.

Task: {task_description}"""

        messages = []
        for turn in range(max_turns):
            response = await self.llm.chat(system, messages, tools)
            # If text-only response: agent is done, return the text
            # If tool calls: execute, append results, continue loop

        return final_response  # Concise summary goes back to main LLM
```

**What agents handle (context-heavy operations):**

| Operation | Without Agent | With Agent |
|-----------|-------------|-----------|
| Web research | 5-20k tokens raw search results/HTML | Agent searches/fetches, returns ~500 token answer |
| Codebase exploration | Multiple glob/grep/read cycles in main context | Agent explores, returns findings |
| Large file analysis | 2k lines dumped into context | Agent reads, answers specific question |
| Multi-file search | 10k+ tokens of grep matches | Agent filters, returns relevant hits |
| Documentation lookup | Entire doc page in context | Agent reads docs, returns what's relevant |

**Agent as a tool:**

The main LLM sees agents as just another tool:

```python
AGENT_TOOL_SCHEMA = {
    "name": "agent",
    "description": "Spawn a sub-agent for tasks that would produce large results. "
                   "The agent runs in its own context and returns a concise summary. "
                   "Use for: web fetching, codebase exploration, reading large files, "
                   "multi-file search, documentation lookup, and MCP server operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the agent should do. Be specific."
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: specific tools the agent can use. "
                              "Default: read-only tools (read_file, glob, grep, "
                              "list_dir, search_memory) + web access. "
                              "Use 'mcp:servername' to give agent access to an MCP server."
            }
        },
        "required": ["task"]
    }
}
```

**Agent model configuration:**

```yaml
# ~/.config/roampal-code/config.yaml
agent:
  provider: anthropic
  model: claude-sonnet-4-5-20250929  # Same as main by default
  # Or cheaper:
  # provider: ollama
  # model: qwen2.5-coder:14b
  max_turns: 10                       # Max tool call rounds per agent
  max_concurrent: 3                   # Max parallel agents
```

Three separately configurable LLM roles:
- **Main** - the coding brain (quality matters most)
- **Agent** - research/exploration (can be cheaper, still needs tool use)
- **Sidecar** - scoring/summarization (cheapest, no tools needed)

**Parallel agent execution:**

The main LLM can spawn multiple agents in a single turn:
```
Main LLM response:
  tool_call: agent(task="fetch React useEffect docs, summarize cleanup")
  tool_call: agent(task="search codebase for existing useEffect usage patterns")

Both run concurrently. Both return concise summaries.
Main LLM gets both results, continues with full picture.
```

**Safety rails:**
- Agents cannot spawn agents (no recursion, depth = 1)
- Agents get read-only tools by default (main LLM can grant write access per-task)
- Agent output hard-capped at 2k tokens (the agent's system prompt asks for 500, but truncation enforces 2k as absolute max)
- `max_concurrent` prevents runaway parallel spawning
- Each agent has a `max_turns` limit to prevent infinite tool loops

---

### 9. MCP Tool Import (`roampal_code/mcp_client.py`)

roampal-code connects to external MCP servers at startup. But it does NOT dump all tool schemas into the main LLM's context. That would defeat the lean context strategy.

**The problem with naive MCP loading:**

```
5 MCP servers × 10 tools each = 50 tool schemas = ~10-15k tokens
Our total context budget is ~20k tokens
Tool schemas would be ~50% of all context. Wasteful.
```

**The solution: MCP server index + agent routing.**

The main LLM sees a lightweight index of what's available. When it needs an MCP tool, it spawns an agent with access to that specific server.

```python
class MCPToolLoader:
    """Connects to MCP servers per the MCP protocol spec.
       Builds lightweight index for main LLM.
       Full tool schemas only loaded into agent context on demand."""

    def __init__(self, config):
        self.servers = {}          # name -> MCP client session
        self.capabilities = {}     # name -> server capabilities (from init)
        self.server_index = {}     # name -> short description (for main LLM)
        self.tool_cache = {}       # name -> full tool schemas (for agents)

    async def connect_servers(self, mcp_config):
        """Connect to all configured MCP servers at startup.

        Per MCP spec, each connection follows a 3-step handshake:
        1. Client sends 'initialize' request (with our capabilities + protocol version)
        2. Server responds with its capabilities (tools, resources, etc.)
        3. Client sends 'initialized' notification (no response expected)

        Only AFTER this handshake can we call tools/list or tools/call.
        """
        for name, server_config in mcp_config.items():
            try:
                session = await self._init_server(name, server_config)
                self.servers[name] = session
            except Exception as e:
                # Server failure doesn't block other servers or the CLI
                log.warning(f"MCP server '{name}' failed to connect: {e}")

    async def _init_server(self, name, server_config):
        """Initialize one MCP server with proper handshake."""
        session = await connect_mcp(server_config)  # Spawns subprocess (stdio)

        # Step 1+2: initialize request -> server capabilities response
        init_result = await session.initialize(
            protocol_version="2024-11-05",
            capabilities={"roots": {"listChanged": True}},
            client_info={"name": "roampal-code", "version": "0.1.0"},
        )
        self.capabilities[name] = init_result.get("capabilities", {})

        # Step 3: initialized notification (no id, no response expected)
        await session.send_notification("notifications/initialized")

        # Only discover tools if server declared the 'tools' capability
        if "tools" not in self.capabilities[name]:
            self.server_index[name] = {
                "description": server_config.get("description", ""),
                "tools": [],
                "has_tools": False,
            }
            return session

        # Discover tools with pagination (nextCursor handling)
        all_tools = []
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            all_tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

        self.tool_cache[name] = all_tools

        # Build lightweight index entry
        tool_names = [t["name"] for t in all_tools]
        self.server_index[name] = {
            "description": server_config.get("description", ""),
            "tools": tool_names,
            "has_tools": True,
            "list_changed": self.capabilities[name].get("tools", {}).get("listChanged", False),
        }

        return session

    async def handle_notification(self, server_name, method):
        """Handle server notifications (e.g., tool list changed)."""
        if method == "notifications/tools/list_changed":
            # Re-fetch tool list for this server
            await self._refresh_tools(server_name)

    async def execute_tool(self, server_name, tool_name, params):
        """Execute a tool. Handles the MCP two-tier error model:
        - JSON-RPC errors (protocol level): raised as exceptions
        - Tool errors (isError=true): returned with error flag for LLM context
        """
        session = self.servers[server_name]
        result = await session.call_tool(tool_name, params)

        # Two-tier error model: isError=true is NOT a protocol error
        # It's a successful response where the tool itself failed
        if result.get("isError", False):
            return {
                "error": True,
                "content": result.get("content", [{"type": "text", "text": "Tool error"}]),
            }
        return {"error": False, "content": result.get("content", [])}

    async def shutdown_all(self):
        """Graceful shutdown per MCP spec (stdio transport):
        1. Close server's stdin
        2. Wait for process exit (timeout)
        3. SIGTERM if still running
        4. SIGKILL as last resort
        """
        for name, session in self.servers.items():
            await session.close_stdin()
            if not await session.wait_exit(timeout=5):
                await session.send_signal("SIGTERM")
                if not await session.wait_exit(timeout=5):
                    await session.send_signal("SIGKILL")

    # ... get_index_for_system_prompt() and get_tools_for_agent() unchanged
```

**MCP spec compliance notes:**
- **Tool descriptions are untrusted.** MCP servers can put anything in tool descriptions, including prompt injection. Since we route MCP through agents (not the main LLM), this is partially mitigated — but agent prompts should still not blindly trust descriptions from untrusted servers.
- **`listChanged` notifications** mean the tool list can change at runtime. If a server declared this capability, we listen for `notifications/tools/list_changed` and re-fetch.
- **Content types:** Tool results can include text, images, audio, and resource links. Agents must handle or gracefully degrade for non-text content types.

**How it works at runtime:**

```
System prompt includes (~300 tokens):
  "Available MCP servers (use agent tool to access):
    github: GitHub operations (create_pr, list_issues, get_repo, +8 more)
    postgres: Database queries (query, list_tables, describe_table)
    slack: Messaging (send_message, list_channels, search)"

User: "create a PR for this change"

Main LLM sees "github" in the MCP index, spawns agent:
  agent(task="use github MCP to create a PR titled 'Fix auth bug'
              with the staged changes on branch fix-auth",
        tools=["mcp:github"])

Agent loads full github tool schemas in ITS context (~1-2k tokens)
Agent calls github.create_pr with the right params
Agent returns: "Created PR #47 'Fix auth bug' at github.com/..."

Main LLM gets the ~50 token summary. Context stays lean.
```

**What the main LLM sees:** ~300 token index listing server names + tool names
**What agents see:** Full tool schemas for the specific server they need
**Token savings:** 10-15k tokens of MCP schemas never enter main context

**Config:**
```yaml
# ~/.config/roampal-code/config.yaml
mcp_servers:
  github:
    description: "GitHub operations"     # Shows in main LLM's index
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
  postgres:
    description: "Database queries"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
```

**roampal-core itself is NOT connected as MCP** - it's an HTTP client (simpler, already works).

**Note:** The `mcp` Python package is already a dependency of roampal-core, so the MCP protocol client is available without extra dependencies.

---

## Package Structure

Lives inside roampal-core as a subpackage. Same pattern as channels.

```
roampal-core/
    roampal/
        ...                      # Existing core modules (untouched)
        cli.py                   # Add one subcommand: `roampal code`
        code/                    # NEW — all new files
            __init__.py
            repl.py              # REPL loop + streaming display
            conversation.py      # Sliding window manager
            llm.py               # Multi-provider LLM client
            agent.py             # Sub-agent runner (context shields)
            memory.py            # Roampal HTTP client
            sidecar.py           # Post-exchange scoring orchestrator
            mcp_client.py        # MCP server connections + tool discovery
            prompts.py           # System prompt templates
            permissions.py       # Tool approval system (allow/ask/deny + patterns)
            config.py            # Settings (3 LLM roles, window size, etc.)
            setup_wizard.py      # First-run interactive model picker
            tools/
                __init__.py      # Tool registry + executor + schema translation
                file_read.py     # read_file (line numbers, images, PDFs)
                file_write.py    # write_file (create/overwrite)
                file_edit.py     # edit_file (string replacement, read-before-write enforced)
                bash.py          # bash (timeout, truncation, background support)
                glob.py          # glob (pattern matching, sorted by mod time)
                grep.py          # grep (ripgrep wrapper, 3 output modes)
                list_dir.py      # list_dir (directory listing)
                agent.py         # agent (sub-agent spawner, has web access + read tools internally)
                memory.py        # search, add, update, delete memory (NO scoring)
```

**Dependencies:** anthropic, openai, httpx, prompt-toolkit, rich, and mcp are already in core's `pyproject.toml` (or will be added). No separate package needed.

**Entry point:** `roampal code` subcommand in the existing CLI. Handles auto-starting the core server if not running, first-run wizard if no config exists, then launches the REPL.

---

## Configuration

Config file location follows platform conventions (same approach as roampal-core data paths):

```
Windows:  %APPDATA%/roampal-code/config.yaml
macOS:    ~/Library/Application Support/roampal-code/config.yaml
Linux:    ~/.config/roampal-code/config.yaml
```

```yaml
# config.yaml

# --- Main LLM (does the coding) ---
main:
  provider: anthropic              # anthropic | openai | ollama | custom
  model: claude-sonnet-4-5-20250929
  api_key: ${ANTHROPIC_API_KEY}    # Env var reference
  # For local models:
  # provider: ollama
  # model: qwen2.5-coder:32b
  # base_url: http://localhost:11434/v1

# --- Sidecar (scoring + summarization, runs after each exchange) ---
sidecar:
  provider: anthropic              # anthropic | openai | ollama | custom
  model: claude-haiku-4-5-20251001 # Cheap/fast model for scoring
  api_key: ${ANTHROPIC_API_KEY}
  # Or use a free/local option:
  # provider: ollama
  # model: llama3.2:3b
  # base_url: http://localhost:11434/v1
  #
  # provider: custom
  # model: mixtral-8x7b-32768
  # base_url: https://api.groq.com/openai/v1
  # api_key: ${GROQ_API_KEY}

# --- Agent (research/exploration, spawned by main LLM) ---
agent:
  provider: anthropic              # anthropic | openai | ollama | custom
  model: claude-sonnet-4-5-20250929  # Same as main by default, or cheaper
  api_key: ${ANTHROPIC_API_KEY}
  max_turns: 10                      # Max tool call rounds per agent
  max_concurrent: 3                  # Max parallel agents
  # For cheaper agents:
  # provider: ollama
  # model: qwen2.5-coder:14b
  # base_url: http://localhost:11434/v1

# --- General ---
window_size: 4                        # Exchange history window
roampal_url: http://localhost:27182   # Roampal server

permissions:
  allow:
    - read_file
    - glob
    - grep
    - list_dir
    - search_memory
    - agent
    - "bash(git status *)"
    - "bash(git diff *)"
    - "bash(git log *)"
    - "bash(ls *)"
    - "bash(python -m pytest *)"
  deny:
    - "bash(rm -rf *)"
    - "edit(*.env)"
    - "write(*.env)"
  # Everything else defaults to "ask"

# --- MCP Servers (agents load full schemas on demand, main LLM sees index only) ---
mcp_servers:
  github:
    description: "GitHub operations"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
```

**Key: Main and sidecar are independently configurable.**

Users can mix and match:
- Anthropic API for main + Haiku for sidecar (quality + cheap scoring)
- Ollama local for main + Ollama small model for sidecar (fully local, no API costs)
- OpenAI for main + Groq for sidecar (fast scoring)
- Any combination that makes sense for their setup

---

## Permissions Model

Research across 7 coding CLIs reveals three approaches:

| Approach | Used By | Strength | Weakness |
|----------|---------|----------|----------|
| Pattern rules (allow/ask/deny) | Claude Code, OpenCode | Flexible, user-configurable | Bash allowlisting is fundamentally broken with file editing |
| OS-level sandbox | Codex CLI, Gemini (optional) | Actually secure for bash | Complex, platform-specific |
| Scope-based (explicit file add) | Aider | Simple, hard to break | Limiting for exploratory coding |

### roampal-code's approach: Pattern rules (Phase 1) + OS sandbox (Phase 2)

**Phase 1 - Pattern-based rules:**

Three permission levels: **allow** (auto-execute), **ask** (prompt user), **deny** (block).

```
Rule evaluation order:
1. Check deny rules first (deny always wins)
2. Check allow rules
3. Default: ask
```

**Rule syntax** (follows Claude Code / OpenCode conventions):

```yaml
permissions:
  allow:
    - read_file                    # Tool name = allow all uses
    - "bash(git status *)"         # Bash with command pattern (glob)
    - "edit(src/**)"               # Edit scoped to path pattern
  deny:
    - "bash(rm -rf *)"            # Block dangerous commands
    - "edit(*.env)"                # Block editing secrets
    - "write(/etc/**)"            # Block writing outside project
  # Unmatched = ask
```

**Approval persistence:**
- **"Allow once"** - just this call
- **"Allow for session"** - same tool+pattern approved for rest of session
- **"Always allow"** - persisted to config file (like Claude Code's permanent bash approval)

**Built-in safety defaults** (even with no config):
- Read-only tools: auto-approve (read_file, glob, grep, list_dir, search_memory, agent)
- Write tools: ask (write_file, edit_file, add_memory, update_memory, delete_memory)
- Bash: ask (every command, unless pattern-matched in allow list)
- MCP tools: ask (all, unless specifically allowed)
- Sidecar operations: auto-approve (scoring, summarization — no user prompt, runs silently after each exchange)
- `.env` files: deny edit/write by default
- Paths outside working directory: ask

**read-before-write enforcement:**
```
write_file("foo.py", content)  -> ERROR if foo.py exists and wasn't read this session
edit_file("foo.py", old, new)  -> ERROR if foo.py wasn't read this session
write_file("new.py", content)  -> OK (new file, no prior read needed)
```

### The bash problem (honest assessment)

Every coding CLI faces this: allowlisting bash commands doesn't actually prevent arbitrary execution when the LLM can also edit files.

```
LLM edits package.json -> adds malicious script -> runs "npm run build" (allowlisted)
LLM edits Makefile -> adds malicious target -> runs "make" (allowlisted)
LLM edits .eslintrc.js -> adds exec() -> runs "eslint ." (allowlisted)
```

**Phase 1 reality:** Pattern rules are a speed bump, not a wall. They prevent accidental damage (LLM running `rm -rf` by mistake) but not adversarial behavior. For most users this is fine - the LLM isn't adversarial, it's just occasionally careless.

**Phase 2 answer:** OS-level sandboxing (Seatbelt on macOS, Landlock+seccomp on Linux). Codex CLI proves this works. The process literally cannot write outside the workspace even if it tries. This is the real security layer.

### Configuration scope hierarchy

```
1. CLI flags (highest priority)
   roampal code --deny "bash(git push *)"

2. Project config (.roampal-code/config.yaml in project root)
   Team-level rules committed to repo

3. User config (platform-specific, see Configuration section)
   Personal defaults

4. Built-in defaults (lowest priority)
   The safety defaults listed above
```

Higher-priority rules override lower ones. Deny at any level cannot be overridden by allow at a lower level.

---

## MVP Scope

### Phase 1: Walking (v0.1)
- [ ] CLI REPL with prompt_toolkit
- [ ] Streaming responses (essential UX — users can't stare at blank screen for 10-30s)
- [ ] Multi-provider LLM client (Anthropic, OpenAI-compat, Ollama)
- [ ] Separate main LLM + sidecar + agent model configuration
- [ ] 4-exchange sliding window with per-exchange token limits
- [ ] System prompt rebuilt each turn (static instructions + cold start + KNOWN CONTEXT)
- [ ] Session start context: last 4 exchange summaries by time (chronological continuity)
- [ ] 12 built-in tools: read, write, edit, bash, glob, grep, list_dir, agent, search_memory, add_memory, update_memory, delete_memory
- [ ] Agent system: sub-agents as context shields (own context window, read-only by default)
- [ ] MCP server connections (import external tools, lightweight index in main context)
- [ ] Tool schema translation (Anthropic format <-> OpenAI format)
- [ ] Roampal context injection via HTTP
- [ ] Sidecar scoring + summarization after each exchange (async, auto-approved)
- [ ] Permission system: allow/ask/deny with pattern matching
- [ ] Read-before-write enforcement
- [ ] Tool result truncation (30k bash, 2k lines read, 50k absolute cap)
- [ ] Approval persistence (once / session / always)
- [ ] Project-level instructions file (ROAMPAL.md — read and prepend to system prompt)
- [ ] Basic git context at session start (current branch, dirty/clean status)
- [ ] Auto-start roampal-core if not running (seamless — user just types `roampal code`)
- [ ] First-run wizard: interactive model picker with guidance (runs once on first `roampal code`)

### Phase 2: Running (v0.2)
- [ ] Rich markdown output rendering + tool call visualization
- [ ] `/slash` commands (/commit, /status, /clear, /compact)
- [ ] Project-level config (.roampal-code/config.yaml in project root)
- [ ] OS-level sandboxing (Seatbelt macOS, Landlock Linux)
- [ ] Background bash with output retrieval
- [ ] Context compaction (summarize + trim when approaching limits)

### Phase 3: Flying (v0.3+)
- [ ] Git integration (auto-diff context, commit history awareness)
- [ ] Task/todo system
- [ ] LSP integration (go-to-definition, find-references, hover)

---

## Open Questions

1. **Default models:** What ships as default for main LLM, agent, and sidecar? (Config is user-editable either way, and first-run wizard lets users pick)

### Resolved
- ~~Permission model~~ -> Allow/ask/deny with pattern matching (Phase 1), OS sandbox (Phase 2)
- ~~Tool set~~ -> 12 built-in tools: file ops (4), search (2), bash, agent, Roampal memory CRUD (4) + MCP
- ~~Scoring responsibility~~ -> Sidecar only, auto-approved. No scoring tools for main LLM.
- ~~Context strategy~~ -> System prompt rebuilt each turn (cold start + KNOWN CONTEXT). Last 4 raw exchanges in message window. Session start: last 4 summaries by time.
- ~~Model flexibility~~ -> Multi-provider day 1. Three configurable LLM roles: main, agent, sidecar.
- ~~Agents~~ -> Phase 1. Context shields, not just parallelism. Essential for keeping 4-exchange window lean.
- ~~Web access~~ -> Not a main LLM tool. Agents have web search + fetch internally. Main LLM spawns agent for any web research.
- ~~ask_user~~ -> Not needed. It's a REPL - LLM asks in plain text, user responds naturally.
- ~~record_response~~ -> Redundant. Sidecar already summarizes/scores every exchange automatically.
- ~~MCP tool bloat~~ -> Main LLM sees lightweight index (~300 tokens). Agents load full schemas in their own context on demand. No caps needed.
- ~~Exchange structure~~ -> One exchange = full tool-call loop (user → tool calls → final text). Stored with message sequence + token count. Per-exchange token limit with compression for oversized exchanges.
- ~~Session continuity~~ -> Last 4 exchange summaries by time on session start. Raw exchanges take over once window fills.
- ~~Streaming~~ -> Phase 1. Non-negotiable UX requirement.
- ~~Project instructions~~ -> Phase 1. ROAMPAL.md in project root, read and prepended to system prompt.
- ~~Git context~~ -> Phase 1 (basic: branch + dirty status). Phase 3 (advanced: auto-diff, commit history).
- ~~MCP protocol compliance~~ -> 3-step init handshake, capability gating, pagination, two-tier error model, proper shutdown lifecycle. Tool descriptions treated as untrusted.
- ~~Reliability~~ -> request_id idempotency on mutations, error classification for retries (transient/overload/permanent), crash-early on core unavailable, streaming side-effect policy, failure containment, scoring backpressure.
- ~~Test strategy~~ -> Contract-based testing per component. State coverage > line coverage. 5 categories: unit, integration, state coverage, failure mode, sabotage.
- ~~Development guidelines~~ -> Humble Object pattern, dependency inversion, assertions in production, context managers for cleanup, Find Bugs Once.
- ~~Repo location~~ -> Lives in roampal-core as `roampal/code/` subpackage (same pattern as channels). Not a separate repo. Zero changes to existing core modules — new files only.
- ~~CLI command~~ -> `roampal code` subcommand (one addition to cli.py, like channels).
- ~~Auto-start core~~ -> Phase 1. `roampal code` auto-starts the server if not running.
- ~~First-run experience~~ -> Interactive wizard on first run. User picks their models with guidance.

---

## Competitive Positioning

| Feature | Claude Code | OpenCode | Aider | Roampal Code |
|---------|------------|----------|-------|-------------|
| Memory across sessions | File-based (no scoring) | No | No | Yes (scored, outcome-tracked) |
| Context efficiency | Low (full history) | Low | Medium | High (scored window) |
| Learning from outcomes | No | No | No | Yes (Wilson scoring) |
| Token cost per session | High | High | Medium | Low |
| Open source | No | Yes | Yes | Yes |
| Multi-model | No (Claude only) | Yes | Yes | Yes |
| Plugin ecosystem | MCP | Plugins | No | Roampal + MCP |

The pitch: **The coding assistant that remembers what works.**

---

## Token Economics (Back of Napkin)

### Claude Code (typical session, 20 turns, 5 MCP servers)
- System prompt: ~3k tokens
- Tool schemas (built-in + MCP): ~6k tokens
- Conversation history (growing): avg ~50k tokens
- Tool results: ~20k tokens
- **Total input per turn (avg):** ~79k tokens
- **Session total:** ~1.6M input tokens

### Roampal Code (same 20 turns, same 5 MCP servers)
- System prompt + Roampal context: ~3k tokens
- Built-in tool schemas: ~3k tokens
- MCP server index: ~300 tokens (NOT full schemas - agents load those)
- Last 4 exchanges: ~4k tokens
- Tool results: ~8k tokens (lighter - heavy results go through agents)
- **Total input per turn (avg):** ~18k tokens
- **Session total:** ~360k input tokens

**~77% reduction in input tokens.** Context is higher quality (scored), tool results are leaner (agent-summarized), and MCP schemas never enter main context.

---

## Reliability Patterns

Derived from *Designing Data-Intensive Applications* and the MCP protocol spec. These aren't optional hardening — they're what prevents the happy-path demo from falling apart under real usage.

### Idempotency

Every mutating HTTP call to roampal-core includes a client-generated `request_id` (UUID). If a network blip causes a retry, the server deduplicates.

```python
# Every mutation gets a request_id
await roampal.add_memory(
    content="user prefers pytest over unittest",
    tags=["preference"],
    request_id=str(uuid4()),  # Server deduplicates on this
)
```

Without this, a retry on `add_memory` creates a duplicate fact. A retry on `record_outcome` double-scores memories. The `request_id` pattern is cheap insurance.

### Error Classification

Not all errors deserve retries. Classify before acting:

```python
# Retry strategy
TRANSIENT = {500, 502, 503, 504}  # Server errors — retry with backoff
OVERLOAD  = {429}                  # Rate limited — longer backoff, jitter
PERMANENT = {400, 401, 403, 404}  # Client errors — never retry

async def resilient_call(fn, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await fn(*args)
        except HTTPError as e:
            if e.status in PERMANENT:
                raise  # Don't retry bad requests
            if e.status in OVERLOAD:
                await sleep(2 ** attempt + random())  # Exponential + jitter
            elif e.status in TRANSIENT:
                await sleep(0.5 * 2 ** attempt)
            else:
                raise
    raise RetriesExhausted()
```

### Crash Early on Core Unavailable

If roampal-core is unreachable at startup, fail clearly:

```
$ roampal-code
Error: Roampal server not running at localhost:27182
Run 'roampal start' first, or check ROAMPAL_DEV setting.
```

Don't silently proceed without memory. A dead program does less damage than a corrupted one (Pragmatic Programmer, Tip 32).

### Streaming Side-Effect Safety

When the LLM streams a response with tool calls, those calls execute mid-stream. If the stream drops after a `write_file` but before the final response, the file is written but the exchange is incomplete.

**Policy:** Tool calls during streaming are fire-and-forget from the stream's perspective. If the stream drops:
1. Tool calls that already executed are done (can't un-write a file)
2. The partial exchange is NOT added to the 4-exchange window (incomplete)
3. The sidecar does NOT score the partial exchange
4. The user sees: "Stream interrupted. Previous tool calls may have executed."

### Failure Containment

Each component has its own failure domain:

```
MCP server A crashes    → only server A unavailable, B and C still work
Sidecar scoring fails   → main conversation unaffected, scoring queued for retry
Agent hangs             → agent hits max_turns/timeout, main LLM gets error result
LLM API returns 429     → exponential backoff, user sees "rate limited, retrying..."
roampal-core slow       → adaptive timeout (not hardcoded), degrade gracefully
```

**Key principle:** Non-critical paths (scoring, memory writes) are async and buffered. They never block the critical path (user conversation).

### Scoring Backpressure

If the user sends messages faster than sidecar scoring completes:
- Scoring queue is bounded (max 10 pending)
- Oldest unscored exchanges get scored with `outcome: "unknown"` (no per-memory attribution)
- This is acceptable degradation — better to lose some scoring fidelity than to block the user

---

## Development Guidelines

Principles from *Clean Architecture*, *Pragmatic Programmer*, and *DDIA* that apply to how roampal-code should be built.

### Humble Object Pattern

I/O boundaries (LLM API, HTTP to core, MCP subprocesses) are hard to test. Separate testable logic from untestable I/O:

```python
# BAD: Logic mixed with I/O
class LLMClient:
    async def chat(self, system, messages, tools):
        response = await self.client.messages.create(...)  # I/O
        parsed = self._parse_response(response)            # Logic
        translated = self._translate_tool_calls(parsed)    # Logic
        return translated

# GOOD: Logic in testable functions, I/O in thin wrapper
class MessageFormatter:
    """Pure logic. Fully testable without network."""
    def format_request(self, system, messages, tools, provider): ...
    def parse_response(self, raw_response, provider): ...
    def translate_tool_calls(self, parsed): ...

class LLMClient:
    """Thin I/O wrapper. Hard to unit test, but logic is minimal."""
    async def chat(self, system, messages, tools):
        request = self.formatter.format_request(system, messages, tools, self.provider)
        raw = await self._send(request)  # Only I/O
        return self.formatter.parse_response(raw, self.provider)
```

This means schema translation, message formatting, tool-call parsing, and permission checking are all testable without hitting APIs.

### Dependency Flows Inward

The tool registry shouldn't know about the LLM client. The conversation manager shouldn't know about HTTP. Each module depends on abstractions (Protocol classes), not concrete implementations.

```
Tools → ToolRegistry (abstract) ← ConversationManager → MemoryClient (abstract) ← RoampalHTTPClient
                                                       → LLMClient (abstract) ← AnthropicClient
```

**Test smell:** If a unit test for PermissionHandler requires importing LLMClient, the architecture is too coupled.

### Assertions Stay On

States that "can't happen" get asserted, not silently swallowed:

```python
assert len(self.exchanges) <= self.window_size, \
    f"Exchange window overflow: {len(self.exchanges)} > {self.window_size}"
```

Stripping assertions in production is "crossing a high wire without a net because you once made it across in practice." (Pragmatic Programmer, Tip 33)

### Finish What You Start

Every MCP subprocess spawned gets killed. Every temp file created gets cleaned up. Use context managers:

```python
async with MCPToolLoader(config) as mcp:
    # mcp.connect_servers() on enter
    # mcp.shutdown_all() on exit (even on crash)
    await run_session(mcp)
```

### Find Bugs Once

Every bug found manually becomes a permanent test. No exceptions, no "that'll never happen again." The test suite is a ratchet — it only grows.

---

## Test Strategy

Not just "tests pass" — tests that prove the system works under real conditions.

### Per-Component Contracts

| Component | Contract (what it promises) | Key test cases |
|-----------|---------------------------|----------------|
| **ConversationManager** | Window never exceeds N exchanges. Messages array is valid for LLM API. | 5th exchange pushes out 1st. Empty window builds valid messages. Oversized exchange gets compressed. |
| **LLMClient** | Sends valid requests to any provider. Streams responses. Terminates tool loops. | Anthropic↔OpenAI schema translation is lossless. Tool loop hits max iterations. Timeout → clean error. |
| **SidecarOrchestrator** | Scores every exchange async. Never blocks main. Handles malformed responses. | Sidecar returns prose instead of JSON → graceful fallback. Sidecar down → main unaffected. |
| **AgentRunner** | Isolated context. No recursion. Output capped. Concurrent limit enforced. | Agent can't spawn agent. 3 agents + 1 more → queued or rejected. Agent crash → error result to main. |
| **PermissionHandler** | Deny wins. Patterns match correctly. Read-before-write enforced. | `deny bash(rm*)` + `allow bash(*)` → denied. Edit without read → error. Session approval doesn't leak. |
| **MCPToolLoader** | 3-step handshake. Capability gating. Pagination. Two-tier errors. Clean shutdown. | Server without tools capability → no tools/list call. Paginated tools/list → all pages fetched. isError=true → error result (not exception). Shutdown kills subprocess. |
| **ToolRegistry** | Any registered tool is callable. Schemas translate between formats. | Register native + MCP + custom tool → all callable. Anthropic schema → OpenAI schema roundtrip. |

### Test Categories

1. **Unit tests (fast, isolated):** Each module against its contract. Mock I/O boundaries. These run in <1 second.
2. **Integration tests (slower, real connections):** CLI → roampal-core HTTP. LLM streaming with real API key. MCP server subprocess lifecycle.
3. **State coverage tests:** Empty window, full window, oversized exchange. Zero MCP servers, 10 MCP servers. First session (no memories), 100th session (rich context).
4. **Failure mode tests:** roampal-core unreachable. LLM API 429. MCP server crash mid-call. Sidecar returns garbage. Stream drops mid-tool-call.
5. **Sabotage tests:** Deliberately introduce bugs in production code, verify tests catch them. If a test doesn't catch a deliberate bug, the test is worthless.
