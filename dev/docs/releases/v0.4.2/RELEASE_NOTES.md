# v0.4.2 Release Notes

**Platforms:** Claude Code, OpenCode, Core Backend
**Theme:** Hook reliability, embedding performance, OpenCode plugin correctness

---

## Overview

Fixes intermittent "UserPromptSubmit hook error" caused by redundant embedding calls, adds ONNX Runtime backend for faster/lighter inference, corrects version check when roampal-cli is co-installed, fixes two OpenCode plugin hook registration issues deferred from v0.4.1, and adds session.idle debounce to protect the scoring loop from subagent noise.

---

## Changes

### Bug Fixes

#### 1. Fix intermittent "UserPromptSubmit hook error" (embedding cache)
**Files:** `embedding_service.py`, `hooks/user_prompt_submit_hook.py`
**Priority:** High

**Bug:** `get_context_for_injection()` calls `search()` 3 times sequentially (reserved working slot, reserved history slot, best-match slots), each generating an embedding for the **same query text**. Each `embed_text()` call runs sentence-transformers `encode()` on CPU via `asyncio.to_thread`. Three sequential encodes take ~12s on tested hardware, which intermittently exceeds the hook's 5-second `urlopen` timeout. When the timeout is hit, the hook exits 1 with no stdout, so Claude Code shows "UserPromptSubmit hook error" and the user's message gets **zero memory context** that turn.

**Root cause introduced in:** v0.3.2 (reserved history slot added a 3rd `search()` call; v0.3.1 had 2 calls). No embedding cache has ever existed, so the same query was always re-encoded on every `search()` call.

**Fix:**
1. Added `_embed_cache` (dict, max 32 entries) to `EmbeddingService.embed_text()`. Same text returns cached embedding instantly — the 3 search calls now only encode once (~4s total instead of ~12s).
2. Bumped hook `urlopen` timeout from 5s to 10s as safety margin.

**Affects:** All clients (Claude Code, OpenCode, any hook/MCP caller). The server-side bottleneck was client-agnostic.

#### 2. Fix version check when roampal-cli is co-installed
**Files:** `hooks/user_prompt_submit_hook.py`, `mcp/server.py`, `server/main.py`
**Priority:** Low

**Bug:** `from roampal import __version__` can return the wrong version when roampal-cli is also installed, because both packages share the `roampal` namespace. CLI's `__init__.py` hardcodes `0.1.0`, shadowing core's actual version. This causes a false "update available" notification on every message.

**Fix:** Use `importlib.metadata.version("roampal")` which reads pip package metadata directly — always correct regardless of namespace conflicts. Applied to all three version check paths:
- `hooks/user_prompt_submit_hook.py` — hook injection (fixed in v0.4.1.2)
- `mcp/server.py` — MCP server `_check_for_updates()` — two call sites (fixed in v0.4.2)
- `server/main.py` — top-level import and `_get_installed_version()` rewrite with file fallback (fixed in v0.4.2)

---

### Performance

#### 3. ONNX Runtime embedding backend
**Files:** `embedding_service.py`, `pyproject.toml`
**Priority:** Medium

**Change:** Model loading now tries ONNX backend first, falls back to PyTorch automatically. Users with `pip install roampal[onnx]` get faster inference and lower RAM. Users without ONNX Runtime installed see no change — PyTorch fallback is transparent.

**Implementation:**
- Bumped `sentence-transformers>=3.2.0` in `pyproject.toml` (ONNX backend support)
- Added `onnx = ["onnxruntime>=1.14.0", "optimum>=1.14.0"]` optional dependency group
- `EmbeddingService.model` property: checks for `onnxruntime`, tries `backend="onnx"`, catches any exception, falls back to PyTorch
- Logs which backend loaded (`_backend` attribute: `"onnx"` or `"pytorch"`)

**Research findings (v0.4.1 investigation):**
- Pre-built ONNX files exist on HuggingFace for `paraphrase-multilingual-mpnet-base-v2` (no auto-export needed)
- Unquantized ONNX produces ~4e-6 per-dimension difference from PyTorch — negligible for cosine similarity. Safe against existing ChromaDB collections.
- int8 quantized variants produce meaningfully different vectors — NOT used

---

### OpenCode Plugin

#### 4. Fix `experimental.session.compacting` hook registration
**File:** `roampal.ts`
**Priority:** Medium

**Bug:** Handler was inside the `event:` switch as a `case`, but the SDK defines `experimental.session.compacting` as a **top-level hook** with `(input, output)` signature. The `Event` union type does NOT include this event. It may have worked by accident or not at all.

**Fix:** Removed `case "experimental.session.compacting"` from event switch. Added top-level hook registration with proper `(input: { sessionID }, output: { context, prompt })` signature. Logic unchanged — fetches 4 recent exchanges from working collection and pushes into `output.context[]`.

#### 5. Add `tool.execute.after` for score detection
**File:** `roampal.ts`
**Priority:** Medium

**Change:** Added `tool.execute.after` hook that detects when `score_memories` or `score_response` MCP tools complete. When detected, clears `pendingScoringData` for the session so the sidecar doesn't double-score on `session.idle`.

**Replaces:** Previously relied on HTTP round-trip (`GET /api/hooks/check-scored`) or had no detection at all (removed in v0.3.7). This is cleaner — in-plugin, no network call, immediate.

**Requires:** OpenCode v0.15.14+ for MCP tool calls to trigger this hook.

#### 6. Debounce `session.idle` to protect scoring loop from subagent noise
**File:** `roampal.ts`
**Priority:** Medium
**Source:** OpenCode issue #13334 (open, no upstream fix)

**Bug:** OpenCode fires `session.idle` when subagents finish, not just the main agent. Without debounce, a premature idle stores an incomplete assistant response, clears plugin state, and the real complete response is lost. This corrupts the scoring loop — on the next turn, the sidecar is fed the incomplete response as "what the LLM said" and scores it accordingly.

**Fix:** 1.5s debounce on `session.idle`. Timer is cancelled whenever new `message.part.updated` events arrive (meaning the LLM is still streaming). Only when no parts arrive for 1.5s does the idle handler run. Timer also cleaned up on `session.deleted`.

**Why sidecar scoring itself is safe:** `pendingScoringData` contains the **previous** exchange (set during `chat.message`), so even a premature idle would score the correct already-complete exchange. The debounce specifically protects exchange storage consistency.

---

## Dropped

### `tool.definition` hook for hiding scoring tool
**Originally item 5 in v0.4.1**

**Investigated and dropped:** `tool.definition` hook does NOT exist in the OpenCode plugin API. The `tool` key only adds new tools — cannot hide or modify existing MCP tool definitions. **Solved differently in v0.4.1:** MCP server now checks `ROAMPAL_PLATFORM` + `ROAMPAL_SIDECAR_DISABLED` and omits `score_memories` from `list_tools()` when sidecar is active. No plugin hook needed.

---

## Carried Forward

### Longer Term
- **Lighter multilingual embedding model** — `multilingual-e5-small` (118MB vs 420MB) would save ~300MB RAM but requires full re-embed migration
- **Custom fine-tuned embeddings** — outcome data provides natural training pairs (v1.0 scope)
- **Content graph relationship traversal in search** — entity relationships stored but not yet used
- **Ghost entry self-healing at adapter level** — deferred from v0.4.0

---

## Files Changed

| File | Changes |
|------|---------|
| `embedding_service.py` | ONNX backend with PyTorch fallback; embedding cache (32 entries); `_backend` attribute |
| `pyproject.toml` | Bump sentence-transformers>=3.2.0; add `[onnx]` optional dep |
| `roampal.ts` | Compacting hook to top-level; tool.execute.after; session.idle 1.5s debounce |
| `hooks/user_prompt_submit_hook.py` | importlib.metadata version check; timeout 5s→10s (both primary and retry) |
| `mcp/server.py` | importlib.metadata version check (2 call sites) |
| `server/main.py` | importlib.metadata top-level import + `_get_installed_version()` rewrite with file fallback |
