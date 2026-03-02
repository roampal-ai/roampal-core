# v0.3.8 Release Notes

**Status:** In progress
**Platforms:** All (Claude Code, OpenCode, CLI)
**Theme:** Docker support + memory_bank scoring transparency + version consistency + thread-safety fix

---

## Overview

v0.3.8 adds Docker support (required for awesome-mcp-servers listing via Glama), fixes a metadata display gap where memory_bank facts hid their Wilson scores, and fixes the stale FastAPI version string. Additionally, v0.3.7 introduced 100% Wilson scoring for proven facts — this release surfaces that data so users can actually see the scores driving retrieval.

---

## 1. Docker Support

### Dockerfile + .dockerignore for Glama verification

Required by awesome-mcp-servers PR #1915. Glama's automated verification builds the Docker image, boots the server, and confirms the health endpoint responds before approving directory listings.

**Dockerfile:**
- Base: `python:3.12-slim` with `build-essential` for native deps
- Installs from PyPI (`pip install roampal`) — always gets latest version, no per-release updates needed
- Uses `ROAMPAL_DATA_PATH=/data` env var for persistent storage volume
- Health check on `/api/health` (returns 503 when embedding service unhealthy)
- `start-period=120s` — embedding model (~90MB) downloads on first health check
- Binds to `0.0.0.0:27182` via uvicorn `--factory` flag

**.dockerignore:** Excludes `__pycache__`, `.git`, `data/`, `dev/`, `venv/`, `.env`, etc.

**Usage:**
```bash
docker build -t roampal-core .
docker run -p 27182:27182 -v roampal-data:/data roampal-core
```

**Files:**
| File | Changes |
|------|---------|
| `Dockerfile` | NEW |
| `.dockerignore` | NEW |

---

## 2. Memory Bank Scoring Transparency

### memory_bank search results now show Wilson/uses/outcomes

v0.3.7 moved memory_bank to 100% Wilson scoring for proven facts (3+ uses) — but the search result formatter never displayed that data. The MCP `search_memory` tool showed `imp:` and `conf:` for memory_bank, while history/patterns showed `w:`, uses, and outcome history. This meant the Wilson scores driving retrieval ranking were invisible to users and the LLM.

In older versions, memory_bank weighting was too aggressive — the combination of `importance * confidence` boosting with early Wilson score accumulation caused memory_bank facts to dominate retrieval slots over more contextually relevant memories from history and patterns. v0.3.7 addressed this by switching to pure Wilson ranking for proven facts. This release completes that work by making the scores visible so users can audit retrieval behavior.

**Before:**
```
[memory_bank] (29d, imp:0.9, conf:1.0) User preference: always use...
```

**After (1+ uses):**
```
[memory_bank] (29d, imp:0.9, conf:1.0, w:0.72, 8 uses, [YYY]) User preference: always use...
```

**Implementation:**
- After existing `imp:/conf:` display, check `uses >= 1`
- Compute Wilson via `wilson_score_lower(success_count, uses)`, append `w:`, uses count, and outcome history
- Fresh memory_bank facts (0 uses) show only `imp:/conf:` — clean display until scoring data exists

**File:** `roampal/mcp/server.py` (search result formatter, `search_memory` tool handler)

---

## 3. FastAPI Version String Fix

### Hardcoded version → `__version__` import

`create_app()` in `server/main.py` had `version="0.3.7"` hardcoded. This was stale since v0.3.7.1 and would drift on every release.

**Fix:** Replace hardcoded string with `__version__` import so the version stays in sync with `__init__.py`.

**File:** `roampal/server/main.py`

---

## 4. ContentGraph Thread-Safety Fix

### `to_dict()` crash on concurrent graph mutation

`ContentGraph.save_to_file()` runs via `asyncio.to_thread()` in the debounced KG save path. While serializing, it iterates `self.entities.items()` directly — if another coroutine adds/removes entities concurrently, Python raises `RuntimeError: dictionary changed size during iteration`.

**Trigger:** Any concurrent memory operation (store, score, ingest) that modifies the content graph while a debounced save is serializing it to disk.

**Fix:** Snapshot all three dicts (`entities`, `relationships`, `metadata`) at the top of `to_dict()` before iterating. `dict()` / `list()` creates a shallow copy immune to concurrent mutation.

**File:** `roampal/backend/modules/memory/content_graph.py`

---

## Version Bumps Required

| File | Field | From | To |
|------|-------|------|----|
| `pyproject.toml` | `version` | `0.3.7.1` | `0.3.8` |
| `roampal/__init__.py` | `__version__` | `0.3.7.1` | `0.3.8` |
| `roampal/server/main.py` | `FastAPI(version=)` | `"0.3.7"` (hardcoded) | `__version__` (import) |

---

## Files Modified

| File | Changes |
|------|---------|
| `Dockerfile` | NEW — Docker support for Glama listing |
| `.dockerignore` | NEW — excludes dev artifacts from image |
| `roampal/mcp/server.py` | memory_bank Wilson/uses/outcomes display in search results |
| `roampal/backend/modules/memory/content_graph.py` | Thread-safe `to_dict()` — snapshot dicts before iteration |
| `roampal/server/main.py` | Version from `__version__` import (was hardcoded) |
| `pyproject.toml` | Version bump to 0.3.8 |
| `roampal/__init__.py` | Version bump to 0.3.8 |

---

## Verification

- [ ] `docker build -t roampal-core .` succeeds
- [ ] `docker run -p 27182:27182 roampal-core` → `/api/health` returns 200
- [ ] `search_memory` for memory_bank facts with 1+ uses shows `w:` and uses count
- [ ] `search_memory` for fresh memory_bank facts (0 uses) shows only `imp:/conf:`
- [ ] FastAPI `/docs` shows correct version string
- [ ] ContentGraph saves without `RuntimeError` under concurrent memory operations
- [ ] All existing tests pass
