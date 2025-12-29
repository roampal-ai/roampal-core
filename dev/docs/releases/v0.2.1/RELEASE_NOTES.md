# roampal-core v0.2.1 Release Notes

**Release Date:** December 28, 2025
**Type:** Hotfix

---

## Summary

Emergency fix for MCP tool loading failure affecting all fresh installs of v0.2.0.

---

## Bug Fixed

### MCP "Method not found" Error (P0)

**Problem:** Fresh installs of v0.2.0 failed to load MCP tools. Claude Code's MCP client calls `prompts/list` and `resources/list` during initialization - when these returned errors, tool loading failed silently.

**Symptoms:**
```
prompts/list  → {"error":{"code":-32601,"message":"Method not found"}}
resources/list → {"error":{"code":-32601,"message":"Method not found"}}
```

Users who ran `pip install roampal && roampal init` got a broken setup with no roampal tools available.

**Root Cause:** The MCP server only registered `list_tools()` and `call_tool()` handlers. The MCP protocol also expects `list_prompts()` and `list_resources()` handlers, even if they return empty lists.

**Fix:** Added missing handlers in `roampal/mcp/server.py`:

```python
@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List available prompts (none currently)."""
    return []

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List available resources (none currently)."""
    return []
```

---

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Version bump 0.2.0 → 0.2.1 |
| `roampal/__init__.py` | Version bump 0.2.0 → 0.2.1 |
| `roampal/mcp/server.py` | Added `list_prompts()` and `list_resources()` handlers |

---

## Upgrade Instructions

```bash
pip install --upgrade roampal
```

No configuration changes required.

---

## Lesson Learned

Always test fresh installs, not just upgrades. The v0.2.0 release was tested by upgrading from v0.1.x, which masked this bug because Claude Code cached the MCP connection. Fresh installs hit the full initialization path.
