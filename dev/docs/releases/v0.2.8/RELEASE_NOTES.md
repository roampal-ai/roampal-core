# roampal-core v0.2.8 Release Notes

**Release Date:** 2026-01-12
**Type:** Bug Fix

---

## Summary

Three bug fixes + one breaking change:
1. `related=[]` parameter not being honored in MCP fallback path
2. **Wilson score capped at 10 uses** - outcome_history truncation broke long-term learning
3. **FastAPI orphan process** - hook server stayed alive after MCP restart
4. **Per-Memory Scoring (BREAKING)** - replaces `related` with `memory_scores` dict for granular feedback

---

## What Changed

### Wilson Score 10-Use Cap Bug (outcome_service.py, scoring_service.py)

**Issue:** Wilson score calculation was broken for memories with >10 uses because:
1. `outcome_history` was capped at 10 entries (line 149)
2. `uses` counter incremented indefinitely
3. Wilson calculated `successes / uses` where successes came from capped history

**Example of the bug:**
- Memory with 100 uses (90 worked)
- `outcome_history` only had last 10 entries (9 worked)
- Wilson calculated: 9/100 = **9%** instead of **90%**
- Highly-used memories got progressively WORSE scores

**Fix:** Added `success_count` field to track cumulative successes:
```python
# outcome_service.py - Now tracks success_count
if outcome == "worked":
    success_delta = 1.0
    uses += 1
elif outcome == "partial":
    success_delta = 0.5
    uses += 1
elif outcome == "failed":
    success_delta = 0.0
    uses += 1  # Also fixed: failed now increments uses

success_count += success_delta  # Cumulative, never capped
```

**Bonus fix:** Failed outcomes now increment `uses` (was silently skipped before).

**Files Changed:**
| File | Change |
|------|--------|
| `outcome_service.py` | Add `success_count` field, return `success_delta` from `_calculate_score_update()` |
| `scoring_service.py` | `calculate_learned_score()` uses `success_count` when available |
| `test_outcome_service.py` | Update tests for 4-value return, failed incrementing uses |
| `test_scoring_service.py` | Add `success_count` tests |

**Backward Compatibility:** Falls back to parsing `outcome_history` if `success_count` not present.

---

### MCP Fallback `related=[]` Bug (server.py, main.py)

**Issue:** When `related=[]` (empty array meaning "score nothing"), the condition failed and fell through to cache, scoring all memories instead of none.

**Root Cause:**
```python
# Line 782 server.py, Line 882 main.py (buggy)
if related is not None and len(related) > 0:
    doc_ids = related

# The `len(related) > 0` check causes empty array to fail
# Falls through to elif branch which uses cache
```

**When it triggers:**
- User calls `score_response(outcome="worked", related=[])`
- Intent: "this exchange worked, but don't score any memories"
- Actual: all cached memories get scored

**Fix:**
```python
# Both files - remove len() check
if related is not None:
    doc_ids = related  # Empty array = score none, as intended
```

**Files Changed:**
| File | Line | Change |
|------|------|--------|
| `roampal/mcp/server.py` | 782 | Remove `and len(related) > 0` |
| `roampal/server/main.py` | 882 | Remove `and len(request.related) > 0` |

**Priority:** Medium - affects selective scoring accuracy

---

### FastAPI Orphan Process Bug (mcp/server.py, server/main.py)

**Issue:** When Claude Code restarts (killing MCP), the FastAPI hook server survived as an orphan process with stale state. Users had to manually kill Python processes.

**Root Cause:**
```python
# Line 252 mcp/server.py (buggy)
creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# CREATE_NO_WINDOW fully detaches the subprocess from parent
# When MCP dies, FastAPI keeps running indefinitely
```

**Symptoms:**
- Hook calls stale FastAPI with old memory state
- Port 27182 already in use on restart
- "Zombie" Python processes accumulating

**Fix:** Three-layer defense (bulletproof):

**Layer 1: MCP-side cleanup (mcp/server.py)**
```python
# Windows: STARTUPINFO hides window but stays in process tree
if sys.platform == "win32":
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    kwargs["startupinfo"] = startupinfo

# Plus atexit handler for graceful shutdown
atexit.register(_cleanup_fastapi)
```

**Layer 2: FastAPI-side parent monitor (server/main.py)**
```python
# No external dependencies - uses ctypes (Windows) / os.kill (Unix)
def _is_parent_alive(parent_pid: int) -> bool:
    if sys.platform == "win32":
        # Windows: ctypes + kernel32 OpenProcess/GetExitCodeProcess
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, parent_pid)
        # ... check exit code
    else:
        # Unix: signal 0 checks if process exists
        os.kill(parent_pid, 0)

async def _monitor_parent_process():
    parent_pid = os.getppid()
    while True:
        await asyncio.sleep(2)  # Check every 2 seconds
        if not _is_parent_alive(parent_pid):
            os._exit(0)  # Immediate exit when parent dies
```

**Layer 3: Graceful cleanup on normal shutdown**
- atexit handler terminates child on graceful exit
- asyncio task cleanup in lifespan context

**Files Changed:**
| File | Change |
|------|--------|
| `roampal/mcp/server.py` | Replace `CREATE_NO_WINDOW` with `STARTUPINFO`, add `_cleanup_fastapi()` + atexit |
| `roampal/server/main.py` | Add `_is_parent_alive()`, `_monitor_parent_process()`, wire into lifespan |

**Platform Behavior:**
| Platform | Before v0.2.8 | After v0.2.8 |
|----------|---------------|--------------|
| Windows | FastAPI orphaned on SIGKILL | FastAPI dies within 2s |
| Linux/macOS | FastAPI orphaned on SIGKILL | FastAPI dies within 2s |

**Why bulletproof:**
- atexit handles graceful shutdown
- Parent monitor catches SIGKILL/crash (atexit skipped)
- No external dependencies (stdlib only: ctypes, os.kill)
- 2-second polling = fast detection, low overhead

**Priority:** High - caused user-facing issues requiring manual process kills

---

### Per-Memory Scoring (BREAKING CHANGE)

**Issue:** Claude kept defaulting to `related=[]`, breaking the learning loop. The old design had too much ambiguity - Claude had to decide which memories to include, and often chose none.

**Solution:** Force explicit scoring of each cached memory individually.

---

#### Simple Explanation

**Before (v0.2.7):**
```
Hook: "Score the exchange"
Claude: score_response(outcome="worked", related=[])  # Lazy default, scores nothing
```

**After (v0.2.8):**
```
Hook: "Score each of these memories:"
- memory_bank_abc: ___
- history_xyz: ___
- patterns_123: ___

Claude MUST fill in each one:
score_response(
    outcome="worked",
    memory_scores={
        "memory_bank_abc": "worked",   # This one helped
        "history_xyz": "partial",       # Somewhat useful
        "patterns_123": "unknown"       # Didn't use it
    }
)
```

**Key changes:**
1. **No more lazy `related=[]`** - must explicitly score each cached memory
2. **Granular feedback** - different memories can get different outcomes
3. **Freedom to add more** - can score any memory in context, not just cached ones

---

#### Files Changed

**1. MCP Tool Schema (mcp/server.py, server/main.py)**

Add `memory_scores` parameter:
```python
types.Tool(
    name="score_response",
    inputSchema={
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["worked", "failed", "partial", "unknown"],
                "description": "Exchange outcome"
            },
            "memory_scores": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": ["worked", "failed", "partial", "unknown"]
                },
                "description": "REQUIRED: Score for each cached memory (doc_id -> outcome). Can include additional memories from context."
            }
        },
        "required": ["outcome", "memory_scores"]
    }
)
```

**2. Hook Prompt (hooks/session_manager.py)**

Change `build_scoring_prompt()` to list cached memories requiring explicit scoring:
```python
# Old
scoring_instruction = """Call score_response(outcome="...", related=["doc_ids"]) FIRST."""

# New
cached_memory_list = "\n".join([f"- {mem['id']}: ___" for mem in surfaced_memories])
scoring_instruction = f"""Score each cached memory individually:
{cached_memory_list}

Call score_response(
    outcome="...",
    memory_scores={{
        "doc_id": "worked|failed|partial|unknown",
        ...
    }}
)

You MUST score every cached memory above.
You MAY add scores for other memories in your context."""
```

**3. Backend Processing (mcp/server.py, server/main.py)**

Process individual scores instead of applying same outcome to all:
```python
# Old
for doc_id in doc_ids:
    await memory_system.record_outcome(doc_id, outcome)

# New
memory_scores = arguments.get("memory_scores", {})
for doc_id, mem_outcome in memory_scores.items():
    await memory_system.record_outcome(doc_id, mem_outcome)
```

**4. Tool Description Update**

Remove `related` parameter, update description to explain `memory_scores`.

---

#### Migration

- `related` parameter deprecated (ignored if provided)
- `memory_scores` is now required
- Old clients calling without `memory_scores` will get validation error

**Priority:** High - fixes fundamental learning loop issue

---

## Testing

```bash
# Test related=[] fix
1. Start fresh session
2. Send message (memories get cached)
3. Call score_response(outcome="worked", related=[])
4. Verify "0 memories updated" instead of "N memories updated"

# Test lifecycle fix
1. Note FastAPI process ID from MCP startup log: "Started FastAPI hook server on port 27182 (PROD mode, pid=XXXXX)"
2. Close Claude Code (or restart MCP)
3. Check if FastAPI process died: `tasklist | findstr XXXXX` (Windows) or `ps aux | grep XXXXX` (Unix)
4. Should return nothing (process terminated)
```

---

## Version Bump

- `pyproject.toml`: 0.2.7 → 0.2.8
- `roampal/__init__.py`: 0.2.7 → 0.2.8
