# v0.3.4.1 Hotfix

**Date:** 2026-02-08
**Platforms:** OpenCode (Linux)
**Triggered by:** GitHub Issue #1 follow-up comment by @whccx

## Problem

After the v0.3.4 deep clone fix, the reporter (Linux) still saw text leaking into the OpenCode input box:

```
[roampal] Stored exchange for session ses_3bff97d64ffeHFSH7LsvlB...
```

This is a separate issue from the original `messages.transform` mutation bug. The plugin had 9 `console.log()` calls for status messages (plugin loaded, session created, exchange stored, etc.). On Linux, OpenCode routes plugin stdout into the TUI — so `console.log` output appears in the input box.

On Windows, plugin stdout is swallowed (invisible), which is why this was never caught during development.

## Root Cause

Platform-specific behavior in OpenCode's TUI:
- **Windows:** Plugin subprocess stdout goes to NUL. `console.log` is invisible.
- **Linux:** Plugin subprocess stdout feeds into the TUI. `console.log` renders in the input box.

The plugin already had a `debugLog()` function that writes to a file (`roampal_plugin_debug.log`) instead of stdout. The 9 `console.log` calls were leftover status messages that predated the `debugLog` function.

## Fix

Replace all `console.log()` calls with `debugLog()`:

| Line | Before | After |
|------|--------|-------|
| 160 | `console.log(\`[roampal] Killed stale server process ${pid}\`)` | `debugLog(...)` |
| 170 | `console.log(\`[roampal] Killed stale server process ${pid}\`)` | `debugLog(...)` |
| 191 | `console.log(\`[roampal] Starting fresh server on port ${port}\`)` | `debugLog(...)` |
| 202 | `console.log("[roampal] Server restarted successfully")` | `debugLog(...)` |
| 558 | `console.log(\`[roampal] Plugin loaded...\`)` | `debugLog(...)` |
| 803 | `console.log(\`[roampal] Session created: ${sid}\`)` | `debugLog(...)` |
| 830 | `console.log(\`[roampal] Main LLM called score_response...\`)` | `debugLog(...)` |
| 869 | `console.log(\`[roampal] Stored exchange for session ${sid}\`)` | `debugLog(...)` |
| 934 | `console.log(\`[roampal] Session deleted: ${sid}\`)` | `debugLog(...)` |

Also updated comment on line 36:
```
// Before: "Debug logging to file (console.log is invisible from plugins)"
// After:  "Debug logging to file (console.log leaks into OpenCode UI on some platforms)"
```

## Files Changed

| File | Change |
|------|--------|
| `roampal/plugins/opencode/roampal.ts` | 9x `console.log()` → `debugLog()`, updated comment |

## Why We Missed It

MEMORY.md had: "console.log from plugins does NOT appear in OpenCode log files" — true on Windows, false on Linux. All development and testing was done on Windows. The reporter (@whccx) is the first Linux user to file feedback.

## Testing

Cannot reproduce on Windows (stdout is swallowed). Fix is mechanical — `debugLog` writes to file, never touches stdout. Verified zero `console.log` calls remain in plugin source (only the comment reference).

## Lesson Learned

Never use `console.log()` in OpenCode plugins. Always use file-based logging (`debugLog`). Plugin stdout behavior is platform-dependent.
