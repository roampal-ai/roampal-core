# roampal-core v0.1.10 Release Notes

**Release Date:** December 2025

## Summary

Added update notifications to MCP hook so Claude Code users see when updates are available.

## Changes

### 1. Update notifications in MCP hook

Previously, update notifications only appeared when running CLI commands like `roampal stats`. Claude Code users (who use MCP tools, not CLI) never saw update notifications.

**Fix:** Added `check_for_updates_cached()` to `user_prompt_submit_hook.py`. Now when a newer version is available, Claude Code users see:

```
<roampal-update-available>Roampal update: 0.1.9 -> 0.1.10. Run: pip install --upgrade roampal</roampal-update-available>
```

The check is cached per session to avoid hitting PyPI on every message.

## Files Changed

```
roampal/hooks/user_prompt_submit_hook.py
  - Lines 40-74: Added check_for_updates_cached() function
  - Lines 130-133: Added update notification output

roampal/__init__.py
  - Version bump: 0.1.9 -> 0.1.10

pyproject.toml
  - Version bump: 0.1.9 -> 0.1.10
```

## Upgrade Notes

No breaking changes. After upgrading, restart Claude Code. Update notifications will now appear in your Claude Code sessions when newer versions are available.

## Previous Version

- v0.1.9: ChromaDB collection name fix, improved search result formatting