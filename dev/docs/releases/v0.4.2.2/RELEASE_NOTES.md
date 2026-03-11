# v0.4.2.2 Hotfix Release Notes

**Platforms:** OpenCode
**Theme:** Fix sidecar config wipe during `roampal init`

---

## Overview

`roampal init` was replacing the entire OpenCode MCP environment dict on every run, deleting user-configured sidecar scoring variables (`ROAMPAL_SIDECAR_URL`, `ROAMPAL_SIDECAR_MODEL`, `ROAMPAL_SIDECAR_KEY`). This was a pre-existing bug triggered by `--force` in v0.4.2, but v0.4.2.1 removed the `--force` gate on MCP updates, exposing the wipe to all users running `roampal init`.

---

## Critical

### 1. OpenCode MCP environment dict replaced instead of merged
**File:** `cli.py` (`configure_opencode`)

**Problem:** `expected_env` was built with only roampal's base variables (`PYTHONPATH`, `ROAMPAL_PLATFORM`). When the MCP config was written, the entire `environment` dict was replaced with `expected_env`, deleting any sidecar variables the user had configured via `roampal init` (sidecar model selection).

**Impact:** Users who ran `roampal init` after configuring a sidecar scorer lost their sidecar config silently. Scoring would fall back to the default (Zen community models or no scoring).

**Fix:** Merge environment dicts — existing vars are preserved, roampal's base vars are updated on top. `{**existing_env, **expected_env}` ensures sidecar vars survive while base config stays current.

### 2. Idempotency check compared full environment including sidecar vars
**File:** `cli.py` (`configure_opencode`)

**Problem:** The "already configured" check compared `existing_env == expected_env`. Since `existing_env` includes sidecar variables and `expected_env` does not, this comparison never matched. The MCP config was rewritten on every `roampal init`, even when nothing changed.

**Fix:** Compare only roampal's base keys (`PYTHONPATH`, `ROAMPAL_PLATFORM`, `ROAMPAL_DEV`) when checking idempotency. Sidecar variables are ignored in the comparison.

---

## Scope

Claude Code and Cursor MCP configs are **not affected** — their `env` dict contains only `ROAMPAL_DEV` (or empty), with no sidecar variables stored there. The sidecar config is OpenCode-specific, stored in `opencode.json` under `mcp.roampal-core.environment`.

---

## Files Modified

| File | Changes |
|------|---------|
| `cli.py` | Environment dict merged instead of replaced; idempotency check compares base keys only |
| `pyproject.toml` | Version bump to 0.4.2.2 |
| `__init__.py` | Version bump to 0.4.2.2 |

---

## Verification

**526 tests passing. 0 failures.**

- [x] Existing sidecar vars preserved after `roampal init`
- [x] Base vars (PYTHONPATH, ROAMPAL_PLATFORM) updated correctly
- [x] Idempotency check passes when only sidecar vars differ
- [x] Fresh install (no existing config) works correctly
- [x] Claude Code and Cursor configs unaffected
