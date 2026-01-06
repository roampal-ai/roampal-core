# roampal-core v0.2.5 Release Notes

**Release Date:** 2026-01-06
**Type:** Critical Bug Fix + Init Robustness

---

## Summary

Fixes critical bug where `roampal init` wrote MCP configuration to an invalid location that Claude Code never reads.

**The Problem:** CLI users running `roampal init` had their MCP config written to `~/.claude/.mcp.json`, which is NOT a valid Claude Code config location. Claude Code only reads from `~/.claude.json` or project-level `.mcp.json`.

**The Fix:** `roampal init` now writes to `~/.claude.json` (user scope) and includes migration logic to clean up the old broken config.

---

## Impact

**Who was affected:**
- Users who installed roampal via CLI (`pip install roampal`) and ran `roampal init`
- MCP server would not load in Claude Code sessions

**Who was NOT affected:**
- Users who manually configured their `.mcp.json` in project roots
- Users guided by Claude to fix their config manually

---

## What Changed

### 1. Correct MCP Config Location (cli.py)

`configure_claude_code()` now writes to `~/.claude.json` with root-level `mcpServers`.

**Before (broken):**
```python
# Wrote to ~/.claude/.mcp.json - INVALID LOCATION
mcp_config_path = claude_dir / ".mcp.json"
```

**After (fixed):**
```python
# Writes to ~/.claude.json - VALID user scope location
claude_json_path = Path.home() / ".claude.json"
claude_json["mcpServers"]["roampal-core"] = roampal_server_config
```

### 2. Migration for Existing Users

Automatically cleans up the old broken config at `~/.claude/.mcp.json`:

```python
# Migration: Clean up old broken config
old_mcp_config_path = claude_dir / ".mcp.json"
if old_mcp_config_path.exists():
    # Remove roampal-core from old location
    # Delete file if empty
```

### 3. Updated Doctor Check

`roampal doctor` now checks the correct location:

```python
# Before: checked ~/.claude/.mcp.json (wrong)
# After: checks ~/.claude.json mcpServers (correct)
```

### 4. Idempotent Init (Robustness)

`roampal init` is now idempotent and safe to run multiple times:

```python
# Check if already correctly configured
existing = claude_json.get("mcpServers", {}).get("roampal-core", {})
if existing.get("args") == ["-m", "roampal.mcp.server"]:
    print("✓ roampal-core already configured correctly")
    return  # No changes needed

# Validate python can import roampal before writing
result = subprocess.run([sys.executable, "-c", "import roampal"], ...)
if result.returncode != 0:
    print("Error: roampal not found - check your venv")
    return
```

**CLI Flags:**
- `roampal init` - Safe: skip if already correct, warn if different
- `roampal init --force` - Overwrite existing config
- `roampal status` - Show current MCP config state without modifying

### 5. Status Command

New `roampal status` command shows MCP configuration state:

```bash
$ roampal status
roampal-core MCP Configuration:
  Location: ~/.claude.json (user scope)
  Command:  python -m roampal.mcp.server
  Status:   ✓ Configured correctly
```

---

## Valid Claude Code MCP Config Locations

| Location | Scope | Description |
|----------|-------|-------------|
| `~/.claude.json` (mcpServers) | User | Global, all projects |
| `~/.claude.json` (projects.{path}.mcpServers) | Local | Per-project override |
| `.mcp.json` (project root) | Project | Shared with team |
| `~/.claude/.mcp.json` | **INVALID** | Never read by Claude Code |

---

## Files Changed

| File | Change |
|------|--------|
| `roampal/cli.py` | Fixed MCP config location, migration, idempotent init, --force flag, status command |
| `pyproject.toml` | Version 0.2.4 → 0.2.5 |
| `roampal/__init__.py` | Version bump |

---

## Testing

Manual CLI testing performed on Windows. Cross-platform paths use `Path.home()` and `sys.platform` checks for Mac/Linux compatibility.

| Scenario | Command | Result |
|----------|---------|--------|
| Config matches | `roampal init --dev` | `[OK] roampal-core already configured correctly` |
| Config differs (env mismatch) | `roampal init` (prod mode when dev exists) | Warning + `Use --force to overwrite` |
| Force overwrite | `roampal init --force` | `Updated MCP server in: ... (forced)` |
| Status command | `roampal status` | Shows MCP config location, command, env, status |
| Doctor command | `roampal doctor` | `All checks passed! (21/21)` |

---

## Upgrade Instructions

```bash
pip install --upgrade roampal
roampal init
```

The init command will:
1. Write correct config to `~/.claude.json`
2. Clean up old broken config at `~/.claude/.mcp.json`
3. Verify with `roampal doctor`

### Manual Fix (if needed)

If `roampal init` doesn't work, add this to `~/.claude.json`:

```json
{
  "mcpServers": {
    "roampal-core": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "roampal.mcp.server"]
    }
  }
}
```

---

## Verification

After upgrading, run:

```bash
roampal doctor
```

Should show:
```
[OK] Claude Code MCP: roampal-core configured in ~/.claude.json
```

