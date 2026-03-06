# v0.4.1.1 Hotfix Release Notes

**Platforms:** Claude Code, OpenCode (via Core Backend)
**Theme:** Remove silent `claude -p` subprocess spawning

---

## Overview

Community feedback identified that the sidecar fallback chain could silently spawn full `claude -p` CLI processes when all other backends (Custom, Haiku API, Zen, Ollama/LM Studio) were unavailable. Each subprocess loaded the entire Claude Code runtime (~30-60s, high memory), creating unexpected cross-product behavior between OpenCode and Claude Code.

---

## Fix

### Remove `claude -p` subprocess fallback from sidecar
**File:** `sidecar_service.py`

**Problem:** Backend #5 in `_call_llm()` ran `subprocess.run(["claude", "-p", ...])` as a last resort for summarization and scoring. If a user had Claude Code installed but no API key, no Zen access, and no local model, every sidecar call would spawn a heavy CLI process. In OpenCode, the idle-time auto-summarization (`/api/memory/auto-summarize-one`) could trigger this repeatedly.

**Fix:** Removed `_call_claude()` function entirely. Removed `subprocess`, `shutil`, `sys` imports (no longer needed). Updated `get_backend_info()` to return `"none available"` instead of `"claude -p (slow)"`. If all backends fail, `_call_llm()` now returns `None` and logs an error — no silent process spawning.

**Fallback chain (updated):**
1. `ROAMPAL_SIDECAR_URL` configured → custom endpoint (user's choice)
2. `ANTHROPIC_API_KEY` → direct Haiku API (~3s, ~$0.001)
3. Zen free models (~5s, $0)
4. Ollama / LM Studio local (~5-14s, $0)
5. ~~`claude -p` CLI~~ → **removed** — fails gracefully

---

## Files Modified

| File | Changes |
|------|---------|
| `sidecar_service.py` | Removed `_call_claude()` (~60 lines), removed from `_call_llm()` chain, removed from `get_backend_info()`, removed unused imports (`subprocess`, `shutil`, `sys`) |

---

## Verification

**484 tests passing. 0 failures.**

- [x] `_call_claude` function removed
- [x] `_call_llm` fallback chain ends at Ollama/LM Studio
- [x] `get_backend_info()` no longer references `claude -p`
- [x] No remaining `subprocess` usage in `sidecar_service.py`
