# roampal-core v0.2.2 Release Notes

**Release Date:** TBD
**Type:** Feature Addition
**Status:** READY FOR RELEASE

---

## Overview

v0.2.2 adds Cursor IDE support (MCP works fully, hooks prepared but blocked by Cursor regression bug), always-inject identity context, explicit IDE selection flags, ghost registry for safe book deletion, and a new `roampal doctor` diagnostic command.

> **Note:** Cursor hooks are implemented and ready, but Cursor has a known regression bug (since v2.0.64) that prevents `agent_message` context injection from working. MCP tools work perfectly in Cursor. See "Cursor Hooks Support" section for details and forum links.

---

## Features

### 1. Cursor IDE Support

**Problem:** roampal-core only configured Claude Code. Cursor users had to manually set up MCP.

**Files Changed:**
- `roampal/cli.py` - Added `configure_cursor()` function
- Detection logic in `cmd_init()` now checks for `~/.cursor`

**Implementation:**
```python
def configure_cursor(cursor_dir: Path, is_dev: bool = False):
    """Configure Cursor MCP.

    Args:
        cursor_dir: Path to ~/.cursor directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
    """
    mcp_config_path = cursor_dir / "mcp.json"
    mcp_config = {
        "mcpServers": {
            "roampal-core": {
                "command": sys.executable,  # Fixed: was "python"
                "args": ["-m", "roampal.mcp.server"],
                "env": {"ROAMPAL_DEV": "1"} if is_dev else {}
            }
        }
    }
    # Merge with existing config if present
    # ...
```

**Key Fixes:**
- Use `sys.executable` instead of `"python"` for correct venv resolution
- Server name standardized to `"roampal-core"`
- Added `is_dev` parameter for development environment flag
- Merges with existing Cursor MCP config instead of overwriting

**Status:** ✅ IMPLEMENTED

---

### 2. Always-Inject Identity Context

**Problem:** User identity (name, preferences) only surfaces when semantically relevant to the query. So "fix this bug" doesn't include user context.

**Solution:** Add `always_inject` metadata flag for memory_bank entries that should be included in every context injection.

**Files Changed:**
- `roampal/mcp/server.py` - Added `always_inject` to tool schema (line 437)
- `roampal/backend/modules/memory/memory_bank_service.py` - Added `get_always_inject()` method
- `roampal/backend/modules/memory/unified_memory_system.py` - Stores flag and fetches in context

**Implementation:**

Tool schema now includes:
```python
"always_inject": {
    "type": "boolean",
    "default": false,
    "description": "If true, this memory appears in EVERY context (use for core identity only)"
}
```

`memory_bank_service.py` has new method:
```python
def get_always_inject(self) -> List[Dict[str, Any]]:
    """Get all memories marked with always_inject: true."""
    all_items = self.list_all(include_archived=False)
    return [
        item for item in all_items
        if item.get("metadata", {}).get("always_inject", False)
    ]
```

**Status:** ✅ IMPLEMENTED

---

### 3. Explicit IDE Selection Flags

**Problem:** Users wanted to configure only one IDE without auto-detect behavior.

**Solution:** Add `--claude-code` and `--cursor` flags to `roampal init`.

**Files Changed:**
- `roampal/cli.py` - Added arguments and updated `cmd_init()` logic

**Usage:**
```bash
roampal init                    # Auto-detect both (default)
roampal init --claude-code      # Claude Code only
roampal init --cursor           # Cursor only
roampal init --cursor --dev     # Cursor in dev mode
```

**Implementation:**
```python
init_parser.add_argument("--claude-code", action="store_true",
    help="Configure Claude Code only (skip auto-detect)")
init_parser.add_argument("--cursor", action="store_true",
    help="Configure Cursor only (skip auto-detect)")
```

**Behavior:**
- Explicit flags override auto-detect
- Creates config directories if they don't exist when flag is used
- Preserves original auto-detect when no flags given

**Status:** ✅ IMPLEMENTED

---

### 4. Ghost Registry for Books

**Problem:** `roampal remove <title>` directly deletes ChromaDB vectors, which can corrupt HNSW index.

**Solution:** Non-destructive ghost registry approach:

1. On `roampal remove <title>` → save chunk IDs to `ghost_ids.json`
2. On search → filter out ghost IDs from results
3. On list_books → filter out ghosted titles
4. Ghosts become invisible immediately without touching vector DB

**Files Changed:**
- `roampal/backend/modules/memory/unified_memory_system.py`:
  - Added `ghost_registry_path` and `ghost_ids` set to `__init__`
  - Added `_load_ghost_registry()` method
  - Added `_save_ghost_registry()` method
  - Modified `remove_book()` to add IDs to ghost registry instead of deleting
  - Modified `search()` to filter out ghost IDs
  - Modified `list_books()` to filter out ghosted chunks

**Implementation:**
```python
# __init__
self.ghost_registry_path = self.data_path / "ghost_ids.json"
self.ghost_ids: set = self._load_ghost_registry()

def _load_ghost_registry(self) -> set:
    """Load ghost registry from disk."""
    if self.ghost_registry_path.exists():
        with open(self.ghost_registry_path, 'r') as f:
            data = json.load(f)
            return set(data.get("ghost_ids", []))
    return set()

def _save_ghost_registry(self):
    """Save ghost registry to disk."""
    with open(self.ghost_registry_path, 'w') as f:
        json.dump({"ghost_ids": list(self.ghost_ids)}, f, indent=2)

# In remove_book():
self.ghost_ids.update(doc_ids)
self._save_ghost_registry()

# In search():
if doc_id in self.ghost_ids:
    continue  # Skip ghosted IDs
```

**Benefits:**
- No HNSW index corruption risk
- Instant "deletion" (no need to wait for ChromaDB operations)
- Reversible if needed (just remove IDs from ghost registry)
- Future: `roampal compact` can rebuild collection to reclaim space

**Status:** ✅ IMPLEMENTED

---

### 5. ChromaDB Schema Migration

**Problem:** Users upgrading from ChromaDB 0.4.x/0.5.x to 1.x get schema errors (`no such column: collections.topic`) because ChromaDB 1.x added new internal columns.

**Solution:** Auto-migration on startup that safely adds missing columns without affecting data.

**Files Changed:**
- `roampal/backend/modules/memory/unified_memory_system.py` - Added `_migrate_chromadb_schema()` method
- `pyproject.toml` - Changed chromadb pin from `>=0.4.0` to `>=1.0.0,<2.0.0`

**Implementation:**
```python
def _migrate_chromadb_schema(self):
    """Migrate ChromaDB schema for compatibility across versions."""
    sqlite_path = self.data_path / "chromadb" / "chroma.sqlite3"
    if not sqlite_path.exists():
        return  # No existing DB to migrate

    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    # Check for missing columns
    cursor.execute("PRAGMA table_info(collections)")
    existing_columns = {col[1] for col in cursor.fetchall()}

    if 'topic' not in existing_columns:
        cursor.execute("ALTER TABLE collections ADD COLUMN topic TEXT")
        logger.info("ChromaDB migration: Added topic column")

    conn.commit()
    conn.close()
```

**Key Points:**
- Migration runs automatically at start of `initialize()`
- Safe: ALTER TABLE ADD COLUMN doesn't affect existing data
- Idempotent: Checks if column exists before adding
- Non-blocking: Logs warning and continues if migration fails
- Version pin ensures ChromaDB 2.x won't break unexpectedly

**Status:** ✅ IMPLEMENTED

---

### 6. Cursor Hooks Support

**Problem:** Cursor was only getting MCP configuration, not hooks. Claude Code gets both MCP + hooks (UserPromptSubmit for context injection, Stop for scoring enforcement). Cursor users missed the automatic context injection and scoring prompts.

**Discovery:** Cursor 1.7+ added hooks support (October 2025) with similar lifecycle events:
- `beforeSubmitPrompt` → equivalent to Claude Code's `UserPromptSubmit`
- `stop` → equivalent to Claude Code's `Stop`

**Solution:** Update `configure_cursor()` to also create `~/.cursor/hooks.json` alongside `mcp.json`.

---

### ⚠️ KNOWN LIMITATION: Cursor Hooks Context Injection Broken

**As of December 2025, Cursor has a regression bug that prevents hook-based context injection from working.**

The `agent_message` field in `beforeSubmitPrompt` hook output is not being injected into the AI's context. This is a Cursor bug, not a roampal issue.

**Bug Details:**
- Regression introduced in Cursor v2.0.64 (November 2025)
- Still broken in v2.0.77, v2.1.6, v2.1.42, v2.1.46
- Our hooks execute correctly (confirmed via server logs), but Cursor ignores the `agent_message` output

**Forum threads tracking this issue:**
- [Initial regression v2.0.64](https://forum.cursor.com/t/regression-hook-response-fields-usermessage-agentmessage-ignored-in-v2-0-64/141516)
- [Still broken v2.0.77](https://forum.cursor.com/t/regression-hook-response-fields-user-message-agent-message-still-ignored-in-windows-v2-0-77/142589)
- [Still broken v2.1.6](https://forum.cursor.com/t/hooks-still-not-working-properly-in-2-1-6/143417)
- [Completely broken v2.1.46](https://forum.cursor.com/t/hooks-are-not-working-anymore/145016)

**What Works:**
- ✅ **MCP tools work perfectly** - `get_context_insights()`, `search_memory()`, etc. all function correctly
- ✅ Hooks are configured correctly and execute
- ❌ Automatic context injection via hooks does not work

**Workaround for Cursor Users:**
Until Cursor fixes this regression, Cursor users need to manually call `get_context_insights()` at the start of conversations to get memory context. The hooks infrastructure is in place and will work automatically once Cursor fixes their bug.

---

**Files Changed:**

1. **`roampal/cli.py`** - `configure_cursor()` function
   - Add hooks.json creation with Cursor's format
   - Update success message to show hooks configured

2. **`roampal/hooks/user_prompt_submit_hook.py`** - Input field handling
   - Cursor sends `conversation_id`, Claude Code sends `session_id`
   - Need to check both fields for compatibility

3. **`roampal/hooks/stop_hook.py`** - Same field handling
   - Check for both `conversation_id` and `session_id`

4. **`roampal/cli.py`** - `cmd_doctor()` function
   - Add check for Cursor hooks.json

**Implementation:**

`~/.cursor/hooks.json` format (Cursor 1.7+):
```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [
      { "command": "C:\\path\\to\\python.exe -m roampal.hooks.user_prompt_submit_hook" }
    ],
    "stop": [
      { "command": "C:\\path\\to\\python.exe -m roampal.hooks.stop_hook" }
    ]
  }
}
```

Hook input compatibility (both hooks):
```python
# Current (Claude Code only):
conversation_id = input_data.get("session_id", "default")

# New (Claude Code + Cursor):
conversation_id = input_data.get("conversation_id") or input_data.get("session_id", "default")
```

Cursor hook input includes:
- `conversation_id` (string)
- `generation_id` (string)
- `model` (string)
- `hook_event_name` (string)
- `workspace_roots` (array of paths)
- For `beforeSubmitPrompt`: prompt content in context

**Testing Checklist:**
- [x] `roampal init --cursor` creates both mcp.json AND hooks.json
- [x] hooks.json has correct format (version 1, beforeSubmitPrompt, stop)
- [x] Hooks use full Python executable path (sys.executable)
- [x] user_prompt_submit_hook handles Cursor's `conversation_id` field
- [x] stop_hook handles Cursor's `conversation_id` field
- [x] `roampal doctor` checks for Cursor hooks.json
- [ ] ~~Cursor sessions get context injection on prompt submit~~ **BLOCKED BY CURSOR BUG**
- [ ] ~~Cursor sessions get scoring prompts when needed~~ **BLOCKED BY CURSOR BUG**

**Status:** ✅ IMPLEMENTED (hooks configured, awaiting Cursor fix for context injection)

---

### 7. Doctor Command

**Problem:** When MCP tools fail silently, users have no way to diagnose why. Today's `false`/`False` bug would have been impossible to catch without reading logs.

**Solution:** Add `roampal doctor` command that validates the entire installation.

**Files Changed:**
- `roampal/cli.py` - Added `cmd_doctor()` function and argument parser

**Usage:**
```bash
roampal doctor          # Check production mode
roampal doctor --dev    # Check dev mode
```

**What it checks:**
1. Python version (3.10+)
2. Roampal version
3. Claude Code config files (settings.json, .mcp.json)
4. Cursor config files (mcp.json)
5. Data directory existence
6. MCP server module loads (catches syntax errors)
7. Memory system initializes
8. All dependencies installed

**Sample output:**
```
Roampal Doctor - Diagnostics

Python Environment:
  [OK] Python 3.10.0
  [OK] Executable: /usr/bin/python3

Configuration Files:
  [OK] ~/.claude exists
  [OK] settings.json has hooks configured
  [OK] .mcp.json has roampal-core server

MCP Server:
  [OK] MCP server module loads
  [OK] Server module valid

Memory System:
  [OK] Memory system initializes
  [OK] Collections: books, working, history, patterns, memory_bank

==================================================
All checks passed! (19/19)
```

**Status:** ✅ IMPLEMENTED

---

## Files Summary

| File | Change | Status |
|------|--------|--------|
| `roampal/cli.py` | Cursor support, IDE flags, doctor command, Cursor hooks.json | ✅ Done |
| `roampal/__init__.py` | Version bump to 0.2.2 | ✅ Done |
| `pyproject.toml` | Version bump to 0.2.2, chromadb pin to 1.x | ✅ Done |
| `ARCHITECTURE.md` | Cursor documentation | ✅ Done |
| `roampal/mcp/server.py` | always_inject in tool schema | ✅ Done |
| `roampal/backend/modules/memory/memory_bank_service.py` | get_always_inject() method | ✅ Done |
| `roampal/backend/modules/memory/unified_memory_system.py` | always_inject + ghost registry + schema migration | ✅ Done |
| `roampal/hooks/user_prompt_submit_hook.py` | Handle Cursor's conversation_id field | ✅ Done |
| `roampal/hooks/stop_hook.py` | Handle Cursor's conversation_id field | ✅ Done |

---

---

## Bug Fixes

### 8. ChromaDB Ghost Entry Error Handling

**Problem:** When ChromaDB has ghost entries (IDs in the HNSW index but no corresponding document data), calls to `get_fragment()` and `list_all_ids()` would throw "Error executing plan: Internal error: Error finding id" exceptions, causing hook endpoint HTTP 500 errors.

**Root Cause:** ChromaDB's HNSW index can retain IDs for documents that were deleted or never fully written. When these ghost IDs are queried, ChromaDB throws an exception instead of returning empty results.

**Solution:** Added try/except wrappers around ChromaDB calls in `chromadb_adapter.py`.

**Files Changed:**
- `roampal/backend/modules/memory/chromadb_adapter.py` - Added error handling to `get_fragment()` and `list_all_ids()`

**Impact:** Prevents hook endpoint failures and allows graceful degradation when ghost entries exist. Users with corrupted ChromaDB state will see warnings in logs but the system continues to function.

**Status:** ✅ FIXED


## Testing Checklist

- [x] `roampal init` detects Cursor and creates `~/.cursor/mcp.json`
- [x] `roampal init --claude-code` configures only Claude Code
- [x] `roampal init --cursor` configures only Cursor
- [x] `roampal init --cursor --dev` uses dev mode for Cursor
- [x] `add_to_memory_bank(content, always_inject=True)` stores flag
- [x] `get_context_insights()` includes always_inject memories (line 715-720)
- [x] Identity memories appear even for unrelated queries (via always_inject)
- [x] `roampal remove <title>` adds IDs to ghost_ids.json (line 1041-1044)
- [x] Search results exclude ghosted book chunks (line 448-451)
- [x] `roampal list` excludes ghosted books (line 1120-1123)
- [x] `roampal doctor` runs and passes all checks
- [x] `roampal doctor --dev` checks dev mode configuration
- [x] **Cursor Hooks**: `roampal init --cursor` creates hooks.json
- [x] **Cursor Hooks**: hooks.json has version 1 format with beforeSubmitPrompt + stop
- [x] **Cursor Hooks**: user_prompt_submit_hook handles conversation_id (Cursor) and session_id (Claude Code)
- [x] **Cursor Hooks**: stop_hook handles conversation_id (Cursor) and session_id (Claude Code)
- [x] **Cursor Hooks**: `roampal doctor` checks for Cursor hooks.json

---

## Local Testing

```bash
# Editable install
pip install -e C:\roampal-core

# Test init in dev mode (won't touch prod config)
roampal init --dev

# Or build wheel and install
cd C:\roampal-core
pip wheel . -w dist/
pip install dist/roampal-0.2.2-py3-none-any.whl
```

---

## Migration Notes

No breaking changes. Existing data is automatically migrated.

### ChromaDB Schema Migration
Users upgrading from roampal-core < 0.2.2 with data created on ChromaDB 0.4.x/0.5.x will have their schema automatically migrated on first startup. The migration:
- Adds missing `topic` column to ChromaDB's internal `collections` table
- Preserves all existing data
- Is idempotent (safe to run multiple times)

### memory_bank always_inject
Existing memory_bank entries will have `always_inject: False` by default.

Users can manually add identity info with the flag:
```
add_to_memory_bank(content="User's name is [NAME]", tags=["identity"], always_inject=true)
```
