# v0.3.9 Release Notes

**Status:** Ready
**Platforms:** OpenCode (sidecar + fallback scoring paths), All (storage safety)
**Theme:** Scoring accuracy fix + memory storage safety cap

---

## Overview

v0.3.9 fixes redundant memory truncation in both OpenCode scoring paths and adds a 2000-character safety cap at memory storage endpoints. The sidecar scorer truncated memory content to 200 characters and the fallback scoring prompt truncated to 120 characters, discarding content the scoring LLM needs to evaluate memory usefulness. Present since v0.3.5.

---

## 1. Remove Memory Truncation in Sidecar Scoring Prompt

### `scoreExchangeViaLLM()` truncated memory content to 200 chars

The sidecar scoring prompt in `roampal.ts` used `m.content.slice(0, 200)` when building the memory section. This is the primary scoring path — it runs on every exchange for every OpenCode user with a working sidecar.

**Impact:** Every sidecar-scored memory since v0.3.5 was evaluated with content beyond 200 characters removed. Memories where key details appeared later in the text were most likely to be mis-scored, introducing drift in Wilson score rankings over time.

**Fix:** Remove `.slice(0, 200)`. Pass full memory content to the scoring prompt.

**File:** `roampal/plugins/opencode/roampal.ts`

---

## 2. Remove Memory Truncation in Fallback Scoring Prompt

### `build_scoring_prompt_simple()` truncated memory content to 120 chars

When the sidecar scorer is unavailable, the plugin falls back to injecting a scoring prompt into the main LLM's system prompt. `build_scoring_prompt_simple()` in `session_manager.py` truncated each memory to 120 characters before inclusion.

**Impact:** OpenCode sessions where the sidecar was unavailable produced lower-quality scores due to the LLM seeing incomplete memory content.

**Fix:** Remove the truncation. Pass full memory content through to the scoring prompt.

**File:** `roampal/hooks/session_manager.py`

---

## 3. Memory Storage Safety Cap (2000 chars)

### Enforce maximum memory size at storage endpoints

Memory content had no size limit at the storage layer. While tool descriptions recommend concise entries, there was no enforcement — arbitrarily large content could be stored and subsequently injected into scoring prompts and context windows.

**Fix:** Add a 2000-character cap at all user-facing storage endpoints. Content exceeding the limit is truncated with a warning logged. This cap is intentionally generous — normal memories are a few hundred characters; anything over 2000 is either misuse or content that belongs in the books collection.

**Endpoints:**
- `/api/memory-bank/add` — caps `content`
- `/api/memory-bank/update` — caps `new_content`
- `/api/record-response` — caps `key_takeaway`

Exchange storage and book chunk ingestion are system-controlled and unaffected.

**File:** `roampal/server/main.py`

---

## Version Bumps Required

| File | Field | From | To |
|------|-------|------|----|
| `pyproject.toml` | `version` | `0.3.8` | `0.3.9` |
| `roampal/__init__.py` | `__version__` | `0.3.8` | `0.3.9` |

---

## Files Modified

| File | Changes |
|------|---------|
| `roampal/plugins/opencode/roampal.ts` | Remove `.slice(0, 200)` in sidecar scoring prompt |
| `roampal/hooks/session_manager.py` | Remove 120-char truncation in `build_scoring_prompt_simple()` |
| `roampal/server/main.py` | 2000-char safety cap on memory storage endpoints |
| `pyproject.toml` | Version bump to 0.3.9 |
| `roampal/__init__.py` | Version bump to 0.3.9 |

---

## Verification

- [x] Sidecar scoring prompt passes full memory content (no `.slice()`)
- [x] `build_scoring_prompt_simple()` passes full memory content (no truncation)
- [x] Memory storage endpoints enforce 2000-char cap with warning log
- [x] Exchange storage and book ingestion unaffected by cap
- [x] All existing tests pass (527 passed)
