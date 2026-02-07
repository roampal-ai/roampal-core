# v0.3.3.1 Release Notes

**Date:** 2026-02-06
**Platforms:** Claude Code, OpenCode, Cursor

## Summary

Fixes tool detection on fresh installs where the coding tool is installed but never launched (no config directory yet). Previously `roampal init` would report "No AI coding tools detected" and skip email collection.

## Changes

### 1. Robust Tool Detection for Fresh Installs

**Problem:** `roampal init` auto-detection only checked for config directories (`~/.claude`, `~/.config/opencode`, `~/.cursor`). On a fresh machine where the tool was installed but never launched, no config dir exists — so `roampal init` reported "No AI coding tools detected" even though the tools were installed. This also meant `collect_email()` was never reached (early return), so the email prompt was skipped too.

**Fix:** Auto-detection now checks three sources per tool:

| Tool | Config Dir | PATH Binary | Platform-Specific |
|------|-----------|-------------|-------------------|
| Claude Code | `~/.claude` | `claude` | — |
| Cursor | `~/.cursor` | `cursor` | Windows: `%LOCALAPPDATA%\Programs\cursor`, macOS: `/Applications/Cursor.app` |
| OpenCode | `~/.config/opencode` | `opencode` | Windows: `%LOCALAPPDATA%\opencode`, macOS: `/Applications/OpenCode.app` |

Any single match triggers detection and configuration.

### 2. Safe Directory Creation

**Problem:** If a tool was detected via PATH (binary found) but its config directory didn't exist yet, `configure_claude_code()` and `configure_cursor()` would crash with `FileNotFoundError` when trying to write config files.

**Fix:** Both functions now create the config directory (`~/.claude`, `~/.cursor`) if it doesn't exist before writing. OpenCode's `configure_opencode()` already did this.

### 3. `--force` Resets Email Marker

**Problem:** If the first `roampal init` wrote a "skipped" email marker (e.g., non-interactive terminal or user pressing Enter), subsequent runs on the same version would skip the email prompt with no way to re-trigger it.

**Fix:** `roampal init --force` now deletes the email marker file before calling `collect_email()`.

## Files Changed

| File | Change |
|------|--------|
| `roampal/__init__.py` | Version bump 0.3.3 → 0.3.3.1 |
| `pyproject.toml` | Version bump 0.3.3 → 0.3.3.1 |
| `roampal/cli.py` | Multi-source tool detection, mkdir safety, force email reset |
| `roampal/backend/modules/memory/tests/unit/test_cli.py` | Updated test mocks for new detection logic |
