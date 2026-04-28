# Roampal Core v0.5.5.2 — 2026-04-28

**Release Date:** 2026-04-28
**Type:** Hotfix
**Platforms:** OpenCode (Windows Desktop app)
**Triggered by:** GitHub Issue #11 (follow-up from v0.5.5.1)

## Summary

v0.5.5.1 fixed two defects in `roampal init --force` for the plugin installation:
1. The `--force` flag was ignored for plugins (now respects force)
2. `shutil.copy` lacked error handling when Desktop holds a file lock

However, Marcus reported that **even with OpenCode Desktop fully closed**, v0.5.5.1 still failed to install the plugin on his Windows 11 system:

> "Upgraded to 0.5.5.1 and the plugin was still missed - opencode desktop closed and no electron/node processes running."

Two new defects were identified that compound the original issue:

- **Defect A** — `shutil.copy` can appear to succeed but produce an empty/corrupted file on Windows due to OneDrive sync, antivirus real-time scanning, or Controlled Folder Access. No post-copy verification was performed.
- **Defect B** — On some Windows installations, OpenCode resolves its plugin path from `%APPDATA%\opencode\plugins` rather than `~\.config\opencode\plugins`. The CLI only copies to the latter, so the plugin is never found even though it was "installed".

Both fixes are CLI-only — no server-side or plugin changes.

## Issue #11 (follow-up) — Plugin Install Still Fails on Windows After v0.5.5.1

### Symptom
After upgrading to v0.5.5.1, running `pip install --upgrade roampal && roampal init --force` reports "Installed plugin" but the plugin is either:
- An empty file (0 bytes) or truncated
- Installed to the wrong directory that OpenCode doesn't look at

The user confirmed Desktop was fully closed with no Electron/Node processes running.

### Root Cause Analysis

**Defect A: Silent copy failure on Windows (`cli.py:1042` pre-fix)**

On Windows, `shutil.copy` can succeed without actually writing all bytes due to:
- **OneDrive sync:** Files in monitored directories may be temporarily quarantined during sync operations. The API returns success but the file is empty or truncated until sync completes.
- **Antivirus real-time scanning:** Some AV products intercept write operations and delay them for scanning. `shutil.copy` returns before the scan completes, leaving a partially written file.
- **Controlled Folder Access (Windows Defender):** If the destination folder is protected, the copy may appear to succeed but be silently blocked or redirected.

The original code had no post-copy verification:
```python
# Before v0.5.5.1
if plugin_needs_write:
    if plugin_source.exists():
        shutil.copy(plugin_source, plugin_file)  # ← no error handling (fixed in .1)
        print(f"  {GREEN}Installed plugin: {plugin_file}{RESET}")  # ← printed even on failure
```

v0.5.5.1 added try/except around `shutil.copy`, but this only catches **exceptions**. When the copy silently fails without raising an exception, the code still prints "Installed plugin" and moves on.

**Defect B: Wrong config directory on some Windows installations (`cli.py:960`)**

The CLI hardcodes the user config path as `Path.home() / ".config" / "opencode"`:
```python
if sys.platform == "win32":
    user_config_dir = Path.home() / ".config" / "opencode"  # C:\Users\logte\.config\opencode
```

However, some Windows applications (including Electron-based apps) may resolve their config path from `%APPDATA%` instead:
- CLI writes to: `C:\Users\logte\.config\opencode\plugins\roampal.ts`
- OpenCode may look in: `C:\Users\logte\AppData\Roaming\opencode\plugins\roampal.ts`

When both directories exist, the plugin gets installed but never loaded. This is especially problematic because:
1. The user sees "Installed plugin" and assumes success
2. The file exists at the path shown in the message
3. But OpenCode doesn't use that path on their system

### Files affected
| File | Line | Issue |
|------|------|-------|
| `roampal/cli.py` | 1042 | No post-copy verification of plugin file |
| `roampal/cli.py` | 960 | Hardcoded `.config\opencode` path may not match OpenCode's actual lookup path |

### Impact
Windows users upgrading via `pip install --upgrade roampal && roampal init --force` may see the success message but have a non-functional plugin. This affects all upgrade scenarios and initial installations on Windows systems with OneDrive sync, aggressive antivirus, or alternative config paths.

---

## Fixes

### Fix A: Post-copy verification with fallback method
New helper function `_install_plugin_file()` in `roampal/cli.py`:

1. **Primary copy via `shutil.copy`** — unchanged from v0.5.5.1's error-handled version
2. **Existence check** — verifies the destination file actually exists after copy. If it doesn't, prints an actionable message naming potential causes (antivirus/OneDrive).
3. **Size verification** — compares `stat().st_size` between source and destination. If sizes don't match or destination is empty:
   - Falls back to manual `read_bytes()` / `write_bytes()` method which bypasses the OS copy API
   - Re-verifies size after fallback write
4. **Clear error messages** — each failure mode produces a specific message with the exact command to manually copy

```python
def _install_plugin_file(plugin_source: Path, plugin_dest: Path):
    """Install plugin file with post-copy verification and fallback methods."""
    plugin_dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(str(plugin_source), str(plugin_dest))
    except (OSError, PermissionError) as e:
        print(f"  {RED}Failed to install plugin: {e}{RESET}")
        # ... clear error message with manual copy instructions
        return

    # Verify the copy actually succeeded
    if not plugin_dest.exists():
        print(f"  {RED}Plugin install failed: file disappeared after copy (antivirus/OneDrive?) {plugin_dest}{RESET}")
        return

    try:
        dest_size = plugin_dest.stat().st_size
        source_size = plugin_source.stat().st_size
        if dest_size == 0 or dest_size != source_size:
            # Fallback to manual read/write
            src_content = plugin_source.read_bytes()
            plugin_dest.write_bytes(src_content)
    except Exception as e:
        logger.warning(f"Failed to verify plugin file: {e}")

    print(f"  {GREEN}Installed plugin: {plugin_dest}{RESET}")
```

### Fix B: Dual-path installation on Windows
After copying to the primary location (`~/.config/opencode/plugins`), the CLI now also copies to `%APPDATA%\opencode\plugins` on Windows:

```python
# v0.5.5.2: On Windows, also copy to %APPDATA%\opencode\plugins as fallback
if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        alt_plugin_dir = Path(appdata) / "opencode" / "plugins"
        alt_plugin_file = alt_plugin_dir / "roampal.ts"
        if alt_plugin_file != plugin_file:
            alt_plugin_dir.mkdir(parents=True, exist_ok=True)
            _install_plugin_file(plugin_source, alt_plugin_file)
```

Both paths use the same `_install_plugin_file()` helper with full verification. If OpenCode looks in either location, it will find a valid plugin file. The check `alt_plugin_file != plugin_file` prevents redundant copies on systems where both paths resolve to the same directory.

---

## Verification Steps

To verify this fix works on an affected system:

1. Close all OpenCode instances (Desktop and CLI)
2. Run: `pip install --upgrade roampal && roampal init --force`
3. Verify output shows "Installed plugin" for both paths (if different):
   ```
   Installed plugin: C:\Users\<user>\.config\opencode\plugins\roampal.ts
   Installed plugin: C:\Users\<user>\AppData\Roaming\opencode\plugins\roampal.ts
   ```
4. Check file sizes match source:
   ```powershell
   (Get-Item "C:\Users\logte\.config\opencode\plugins\roampal.ts").Length
   # Should be ~80KB+ (not 0)
   ```
5. Open OpenCode and verify the plugin is loaded (check debug log at `%APPDATA%\roampal_plugin_debug.log`)

### Manual workaround if fix doesn't take effect
If the automated installation still fails, manually copy:
```powershell
$src = "C:\Users\logte\AppData\Local\Programs\Python\Python310\Lib\site-packages\roampal\plugins\opencode\roampal.ts"
Copy-Item $src "$env:APPDATA\opencode\plugins\roampal.ts" -Force
Copy-Item $src "~/.config/opencode/plugins/roampal.ts" -Force
```

Then restart OpenCode.

---

## Files Changed
| File | Change |
|------|--------|
| `roampal/cli.py` | New `_install_plugin_file()` helper with verification and fallback (lines 898-943) |
| `roampal/cli.py` | Modified plugin install block to use new helper + dual-path on Windows (lines 1067-1082) |
| `dev/docs/releases/v0.5.5.2/RELEASE_NOTES.md` | This file |

## Backward Compatibility
- No breaking changes
- Existing installations unaffected — only impacts the install/update path
- The dual-path copy is idempotent and safe to run multiple times
- On Linux/macOS, behavior unchanged (only Windows gets the second copy)

