# Roampal Core v0.5.5.1 — 2026-04-28

**Release Date:** 2026-04-28
**Type:** Hotfix
**Platforms:** OpenCode (Windows Desktop app)
**Triggered by:** GitHub Issues #10, #11

## Summary

Two bugs surfaced in v0.5.5 by user marcusyoung (Issues #10 and #11):

- **#10 Critical** — `ROAMPAL_PROFILE` is ignored by OpenCode Desktop. Switching projects in the Desktop UI doesn't apply the project-specific profile. CLI is unaffected.
- **#11 Medium** — `pip install --upgrade roampal && roampal init --force` doesn't replace `~\.config\opencode\plugins\roampal.ts` on Windows even when the source is newer.

Both fixes are plugin/CLI only — no server-side changes.

## Issue #10 — Profiles with OpenCode Desktop App (Critical)

### Symptom
`ROAMPAL_PROFILE` works correctly with the OpenCode CLI but not the Desktop app. Switching projects in the Desktop UI doesn't apply the project-specific profile — the plugin keeps sending the stale profile header from the first project opened in the workspace.

### Root Cause
The `ROAMPAL_PROFILE` constant is resolved **once at module load time** in the OpenCode plugin, then cached forever:

**`roampal/plugins/opencode/roampal.ts:94`**
```typescript
const ROAMPAL_PROFILE = resolveRoampalProfile()  // ← called ONCE at startup
debugLog(`[v0.5.4] Resolved profile: ${ROAMPAL_PROFILE || "(default — no header)"}`)

function roampalHeaders(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" }
  if (ROAMPAL_PROFILE) {  // ← uses stale cached value
    h["X-Roampal-Profile"] = ROAMPAL_PROFILE
  }
  return h
}
```

When OpenCode Desktop loads a workspace with multiple projects (e.g. `research`, `lcca-gui`, `lcca-data`):
1. Desktop spawns the plugin as a **singleton** — one process for the entire workspace
2. `resolveRoampalProfile()` runs once and caches the profile from whichever project was "active" at startup
3. When the user switches to a different project in the UI, Desktop spawns a new MCP subprocess with the correct `ROAMPAL_PROFILE` env var — **but the plugin process never reloads**
4. All plugin-initiated HTTP calls (scoring, context injection, search) continue sending the original stale profile header

The CLI works because each `opencode` invocation spawns a fresh plugin process, so `resolveRoampalProfile()` runs fresh every time.

Note: `resolveRoampalProfile()` itself has correct priority logic (env var → project config → global config). The function is fine — the problem is **when** it's called.

### Why the naive fix doesn't work
Moving `resolveRoampalProfile()` into `roampalHeaders()` (so it's re-resolved on every request) does not solve the problem for Desktop. The plugin process has its own `cwd` and its own `process.env` — neither changes when Desktop switches projects. `process.env.ROAMPAL_PROFILE` stays whatever Desktop set at workspace launch, and `process.cwd()` stays the initial project directory.

The real fix requires the plugin to either:
- **Option A:** Detect project switches via Desktop events and re-resolve
- **Option B:** Accept the profile as a parameter on every hook handler (Desktop passes the active project to each hook invocation)
- **Option C:** Route profile resolution through the MCP subprocess (which DOES get the correct env), rather than through the plugin's own singleton env

### Files affected
| File | Role |
|------|------|
| `roampal/plugins/opencode/roampal.ts:94` | `ROAMPAL_PROFILE` cached at module load |
| `roampal/plugins/opencode/roampal.ts:97-103` | `roampalHeaders()` uses cached value |
| `roampal/mcp/server.py:161-163` | `_mcp_profile_name` cached at module load (benign — MCP subprocess restarts per-project) |
| `roampal/server/main.py:650-662` | `_resolve_profile_name()` resolves per-request (correct) |

### Impact
All Desktop users with per-project profiles are affected. The plugin always operates on the default profile (or whichever profile was active when the workspace opened), even after switching to a different project. Scoring results, context injections, and search results all hit the wrong profile.

---

## Issue #11 — Upgrade Process Doesn't Replace OpenCode Plugin on Windows (Medium)

### Symptom
Running `pip install --upgrade roampal && roampal init --force` on Windows 11 does not replace `~\\.config\\opencode\\plugins\\roampal.ts` even when the new version has a different plugin file. The user must manually download the plugin from the repository.

### Root Cause
Two defects in the `configure_opencode` function at `roampal/cli.py:1014-1045`:

**Defect A: `--force` flag has zero effect on plugin installation**

The `force` parameter is received at line 898 but only used for MCP config scope detection at line 952:
```python
if force or not user_config_file.exists():
    config_path = user_config_file
```

The plugin installation block (lines 1017-1044) never references the `force` parameter. The overwrite decision relies entirely on content comparison:
```python
if existing_content == source_content:
    plugin_needs_write = False  # "already installed" — even with --force
```

**Defect B: `shutil.copy` has no error handling**

```python
if plugin_needs_write:
    if plugin_source.exists():
        shutil.copy(plugin_source, plugin_file)  # ← unguarded — crashes on lock/permission error
        print(f"  {GREEN}Installed plugin: {plugin_file}{RESET}")
    else:
        print(f"  {RED}Plugin source not found: {plugin_source}{RESET}")
```

If OpenCode Desktop is running and holds a file lock on the plugin, `shutil.copy` throws an unhandled exception (typically `PermissionError` or `OSError`). On Windows, the Desktop app's Node.js/Electron process may keep the plugin file open, preventing writes.

Additionally, the error message "Plugin source not found" is misleading — it only checks if the source exists in the installed package. If the package was partially installed or installed in an unusual layout (e.g. zip-included, editable install, pipx), the source path resolves differently.

### What the user expected
`roampal init --force` should forcefully replace the plugin file regardless of content comparison. The flag name implies "do it anyway" — but the code silently ignores it for plugins.

### Files affected
| File | Line | Issue |
|------|------|-------|
| `roampal/cli.py` | 1020-1034 | Content comparison ignores `force` flag |
| `roampal/cli.py` | 1041 | `shutil.copy` lacks error handling |
| `roampal/cli.py` | 898 | `force` param accepted but unused for plugin path |

---

## Fixes

### Fix #10 — Full fix: Profile resolution via session lookup
In `roampal/plugins/opencode/roampal.ts`:

1. **Remove cached constant.** `resolveRoampalProfile()` moved from module scope into the new `_cachedProfile` lookup that `roampalHeaders()` reads — profile re-resolved per exchange instead of once at plugin load.

2. **Per-message session lookup (the actual fix).** The plugin stores the OpenCode `client` reference at module level. `refreshProfile(sessionID)` calls `client.session.get({path: {id: sessionID}})`, which returns the session's `directory` field. **Sessions are bound to the project they were created in**, so even though the Desktop plugin is a singleton across project switches, each `chat.message` arrives with a sessionID whose session reflects the user's actually-active project.

   The OpenCode SDK uses URL templating (`/session/{id}`), so the sessionID must be passed under `path`, not at the top level. Other parameter shapes return a `must start with "ses"` validation error because the literal `{id}` placeholder leaks through.

3. **Refresh on every exchange.** `refreshProfile()` is called:
   - At plugin load (no sessionID, falls back to `project.current()` then `process.cwd()`)
   - At the start of every `chat.message` hook with the sessionID, before any API calls

4. **Fallback chain with timeouts.** Both `session.get()` and `project.current()` are wrapped in `Promise.race` with a 2s timeout — mirrors the existing `client.app.agents()` pattern at plugin load to prevent hangs from non-resolving API methods. `project.current()` is kept as a CLI-friendly fallback (it works correctly outside Desktop's singleton workspace model). If both fail, falls back to `process.cwd()`.

5. **`worktree=/` guard.** Desktop's `project.current()` was observed returning `worktree=/` (filesystem root) in some startup states — these responses are now rejected so we don't try to read `/opencode.json`.

This ensures that when a user switches projects in the Desktop UI and sends a message, the plugin reads the active session's `directory`, loads that project's `opencode.json`, and applies the correct profile header.

### Fix #10 — Approaches that did NOT work (recorded for posterity)

- **Naive re-resolve in `roampalHeaders()`.** Doesn't help — `process.env.ROAMPAL_PROFILE` and `process.cwd()` are frozen at plugin spawn and don't change when Desktop switches projects.
- **`client.project.current()`.** Returns the workspace's primary project, not the currently active one. Verified on Desktop 1.14.29: switching between two projects yielded the same project ID across all `chat.message` events.
- **`client.project.list()` / `client.session.current()`.** Don't exist on Desktop 1.14.29 (probes returned no data).

### Fix #11 — Force flag + error handling in plugin installation
In `roampal/cli.py` `configure_opencode()`:

1. **Line 1020:** Bind `force` to `plugin_needs_write` — when `force=True`, always overwrite regardless of content match:
   ```python
   plugin_needs_write = True
   if not force and plugin_file.exists():
       # content comparison...
   ```

2. **Line 1041:** Wrap `shutil.copy` in try/except with a clear error message:
   ```python
   try:
       shutil.copy(plugin_source, plugin_file)
       print(f"  {GREEN}Installed plugin: {plugin_file}{RESET}")
   except (OSError, PermissionError) as e:
       print(f"  {RED}Failed to install plugin: {e}{RESET}")
       print(f"  {YELLOW}Close OpenCode Desktop and try again, or copy manually:
       print(f"  {YELLOW}  cp {plugin_source} {plugin_file}{RESET}")
   ```

---

## Fix validation for #10

End-to-end verified on Desktop 1.14.29 (2026-04-28):

**Setup:** Two project directories with distinct profiles in their `opencode.json`:
- `C:\test-roampal-a` → `mcp.roampal-core.environment.ROAMPAL_PROFILE = "test-a"`
- `C:\test-roampal-b` → `mcp.roampal-core.environment.ROAMPAL_PROFILE = "test-b"`

**Test:** Both projects added to Desktop workspace. Sent a message from project A, switched to project B in the UI, sent another message.

**Plugin debug log:**
```
[v0.5.5.1] refreshProfile: worktree=C:\test-roampal-b → profile=test-b
[v0.5.5.1] refreshProfile: worktree=C:\test-roampal-a → profile=test-a
```

The session's `directory` field correctly tracked the active project across UI switches, and the resolved profile flipped accordingly. The corresponding `X-Roampal-Profile` header is now sent on all subsequent FastAPI calls (scoring, context injection, search) for that exchange.

**Server side (`server/main.py:650–662`)** was already correct in v0.5.4 — it reads `X-Roampal-Profile` per-request and lazily creates `UnifiedMemorySystem` per profile via `get_memory_for_request()`. No server-side changes were needed; the fix is plugin-only.
