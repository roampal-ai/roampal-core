# v0.3.3 Release Notes

**Date:** 2026-02-06
**Platforms:** Claude Code, OpenCode

## Summary

Patch release fixing a critical OpenCode packaging bug, email collection logic, and adding persistent memory awareness to context injection.

## Changes

### 1. Smart Email Collection Marker

**Problem:** The email marker file only stored the version string (e.g., `0.3.2`). On every version update, `roampal init` would re-prompt for email — even if the user already provided it on a previous version.

**Fix:** Marker now stores `version:status` (e.g., `0.3.2:provided` or `0.3.2:skipped`).

**New behavior:**

| Scenario | Action |
|----------|--------|
| First install (no marker) | Ask for email |
| Re-run init on same version | Skip (already asked) |
| Update + previously **provided** email | Skip (already have it) |
| Update + previously **skipped** | Ask again (one more chance) |
| Legacy marker (just version, no status) | Treat as skipped |

**Files changed:**
- `roampal/cli.py` — `collect_email()` and `_write_email_marker()` updated

### 2. Persistent Memory Awareness in Context Injection

**Problem:** The KNOWN CONTEXT block was injected without telling the AI it has persistent memory. When users asked "do you remember me?" or referenced past conversations, the AI had no instruction to use the injected context — it might deny having memory even though the memories were right there in context.

**Fix:** Added a preamble before the KNOWN CONTEXT block:

```
You have persistent memory about this user via Roampal. The context below was retrieved from past conversations. If the user references past interactions or asks if you remember them, use this context — you DO remember.

═══ KNOWN CONTEXT ═══
...
```

This applies to both Claude Code (via hooks stdout) and OpenCode (via system prompt injection).

**Files changed:**
- `roampal/backend/modules/memory/unified_memory_system.py` — `format_context_injection()` updated

### 3. OpenCode Plugin Packaging Fix (Critical)

**Problem:** The OpenCode TypeScript plugin (`roampal.ts`) was NOT included in PyPI packages. The `roampal/plugins/` directories had no `__init__.py` files and `pyproject.toml` had no `package_data` config for `.ts` files. When users ran `pip install roampal`, the plugin source file was missing — `configure_opencode()` would print "Plugin source not found" and OpenCode users got MCP tools only, with no context injection, exchange capture, or scoring.

**Verified:** Downloaded `roampal-0.3.2-py3-none-any.whl` from PyPI — zero `.ts` files in the package.

**Fix:**
- Added `roampal/plugins/__init__.py` and `roampal/plugins/opencode/__init__.py` so setuptools discovers the directories as packages
- Added `[tool.setuptools.package-data]` section to `pyproject.toml` to include `*.ts` files

**Verified:** v0.3.3 wheel build includes `roampal/plugins/opencode/roampal.ts`.

**Files changed:**
- `roampal/plugins/__init__.py` — new (empty, for package discovery)
- `roampal/plugins/opencode/__init__.py` — new (empty, for package discovery)
- `pyproject.toml` — added `[tool.setuptools.package-data]` section

### Backwards Compatibility

Old markers (plain version string like `0.3.2`) are handled gracefully — parsed as `version:skipped`, so upgrading users who skipped email will be asked once on v0.3.3. Users who already provided email via the old format will also be asked once (since old format can't distinguish), but this is a one-time occurrence.

## Files Changed

| File | Change |
|------|--------|
| `roampal/__init__.py` | Version bump 0.3.2 → 0.3.3 |
| `pyproject.toml` | Version bump 0.3.2 → 0.3.3 + package-data for .ts files |
| `roampal/plugins/__init__.py` | New — package discovery for plugins dir |
| `roampal/plugins/opencode/__init__.py` | New — package discovery for opencode plugin dir |
| `roampal/cli.py` | Smart email marker with `version:status` format |
| `roampal/backend/modules/memory/unified_memory_system.py` | Persistent memory awareness preamble in context injection |
