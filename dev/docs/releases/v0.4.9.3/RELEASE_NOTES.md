# Roampal Core v0.4.9.3 — CLI Sidecar Config Hotfix

**Release Date:** 2026-04-15  
**Theme:** Fix CLI sidecar commands to load config from opencode.json

---

## What Changes for Users

1. **`roampal sidecar test` now works** — Previously failed with "All backends failed" even when sidecar was configured
2. **CLI loads config from opencode.json** — Sidecar commands now use the same configuration as the MCP server
3. **Consistent behavior** — `roampal sidecar setup` → `roampal sidecar test` now works as expected

---

## The Bug

**Before v0.4.9.3:**
- User runs `roampal sidecar setup` → config written to `opencode.json`
- User runs `roampal sidecar test` → **FAILS** with "All backends failed"
- CLI wasn't loading config from `opencode.json` before sidecar commands

**After v0.4.9.3:**
- `roampal sidecar test` loads config from `opencode.json` → **WORKS**
- CLI and MCP server use same configuration → **CONSISTENT**

---

## Changes

### 1. CLI Sidecar Command Fix
**File:** `roampal/cli.py:3054-3070`

Added config loading to `cmd_sidecar()`:
```python
def cmd_sidecar(args):
    """Configure sidecar scoring model."""
    subcommand = args.sidecar_command or "setup"
    
    # v0.4.9.3: Load sidecar config from opencode.json before any sidecar command
    # This ensures CLI commands use the same config as the MCP server
    if subcommand in ["test", "status"]:
        _check_sidecar_configured()
    
    # ... rest of function
```

### 2. What `_check_sidecar_configured()` Does:
- Reads `ROAMPAL_SIDECAR_URL`, `ROAMPAL_SIDECAR_MODEL`, `ROAMPAL_SIDECAR_KEY` from `opencode.json`
- Injects them into `os.environ`
- Updates `sidecar_service.CUSTOM_URL`, `CUSTOM_MODEL`, `CUSTOM_KEY` module variables
- Ensures sidecar service uses user-configured endpoint

---

## Verification

**✅ Fixed Behavior:**
```bash
# Setup sidecar
roampal sidecar setup  # Configures gpt-oss:20b via Ollama

# Test sidecar - NOW WORKS!
roampal sidecar test
# Output: "Sidecar is working correctly."

# Check status - shows correct config
roampal sidecar status
# Output: "Scoring model: gpt-oss:20b"
```

**✅ Direct Test Works:**
```python
# Before: Would fail
from roampal.sidecar_service import test_sidecar_scoring
result = test_sidecar_scoring()  # ✅ Now works

# After: Loads config and succeeds
```

---

## Root Cause Analysis

**Problem:** CLI commands and MCP server had different config loading behavior:
- **MCP server:** Loads config from `opencode.json` on startup
- **CLI commands:** Weren't loading config before sidecar operations
- **Result:** `roampal sidecar test` failed even when sidecar was configured

**Solution:** Unified config loading path for CLI sidecar commands.

---

## Impact

**Affected Users:**
- Anyone using `roampal sidecar setup` + `roampal sidecar test`
- Users with custom sidecar configurations (OpenAI API, Groq, etc.)
- Development/testing workflows

**Not Affected:**
- MCP server operation (already worked)
- Manual env var configuration (`ROAMPAL_SIDECAR_URL=... roampal ...`)

---

## Migration

**No migration needed** — fix is backward compatible. Existing `opencode.json` configs now work with CLI commands.

---

## Reference

- **Bug report:** CLI sidecar test fails after setup
- **Fix:** Add config loading to `cmd_sidecar()` in CLI
- **Files changed:** `roampal/cli.py` (1 file)
- **Test coverage:** Manual verification of `roampal sidecar test` success