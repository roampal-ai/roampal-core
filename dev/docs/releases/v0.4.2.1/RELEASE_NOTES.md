# v0.4.2.1 Hotfix Release Notes

**Platforms:** All (CLI / Init)
**Theme:** Safe config updates — `roampal init` no longer wipes user configs

---

## Overview

`roampal init` was overwriting user configurations on every run. The hooks dict in Claude Code's `settings.json` was replaced entirely (deleting non-roampal hooks), and config updates for all tools required `--force` even for routine upgrades. This hotfix makes init merge-safe: it only touches roampal-owned entries and always applies updates without requiring `--force`.

---

## Critical

### 1. Claude Code hooks replaced instead of merged
**File:** `cli.py` (`configure_claude_code`)

**Problem:** `settings["hooks"] = { roampal hooks }` replaced the entire hooks dict on every `roampal init`, even without `--force`. Any non-roampal hooks (linters, formatters, custom scripts) were silently deleted.

**Fix:** Load existing hooks first, then merge only roampal entries (UserPromptSubmit, Stop, SessionStart) using `dict.update()`. Non-roampal hooks are preserved.

---

## High

### 2. MCP config updates blocked without --force
**Files:** `cli.py` (`configure_claude_code`, `configure_cursor`, `configure_opencode`)

**Problem:** When the roampal-core MCP config changed between versions (e.g., env vars, args), `roampal init` printed a warning and skipped the update unless `--force` was passed. Users running `pip install --upgrade roampal && roampal init` would silently keep stale MCP configs.

**Fix:** Always update the roampal-core MCP entry when it differs from the expected config. The entry is roampal-owned — no user data is at risk.

### 3. OpenCode plugin not updated without --force
**File:** `cli.py` (`configure_opencode`)

**Problem:** When the bundled `roampal.ts` plugin source differed from the deployed copy, init skipped the update and printed "Use --force to overwrite." Users upgrading roampal wouldn't get plugin fixes (like the debounce fix in v0.4.2) without knowing to pass `--force`.

**Fix:** Always update the plugin when source content differs. The plugin is roampal-owned with no user customizations to preserve.

### 4. Cursor hooks replaced instead of merged
**File:** `cli.py` (`configure_cursor`)

**Problem:** Same pattern as Claude Code — when Cursor hooks differed, the update was blocked without `--force`, and with `--force` it only updated roampal entries (already correct). But the gate was unnecessary for roampal-owned hooks.

**Fix:** Always update roampal's `beforeSubmitPrompt` and `stop` hooks when they differ. Non-roampal hooks are preserved.

---

## Low

### 5. Update message recommends --force
**File:** `cli.py` (update check)

**Problem:** The update prompt printed `pip install --upgrade roampal && roampal init --force`, encouraging users to force-overwrite configs on every update.

**Fix:** Changed to `pip install --upgrade roampal && roampal init`.

---

## What --force still does

The `--force` flag is not removed. It retains two edge-case uses:
- Re-prompts sidecar model selection (skipped if already configured)
- Resets the email signup marker (so the optional signup prompt reappears)

---

## Files Modified

| File | Changes |
|------|---------|
| `cli.py` | Hooks merge instead of replace; removed `--force` gates on MCP, plugin, and hooks updates; updated upgrade message |
| `pyproject.toml` | Version bump to 0.4.2.1 |
| `__init__.py` | Version bump to 0.4.2.1 |

---

## Verification

**526 tests passing. 0 failures.**

- [x] Claude Code hooks merge preserves non-roampal entries
- [x] MCP config updates applied without --force on all 3 tools
- [x] OpenCode plugin updated when source differs
- [x] Cursor hooks updated without --force, non-roampal hooks preserved
- [x] Update message no longer recommends --force
- [x] --force still works for sidecar re-prompt and email reset
