# v0.3.7.1 Release Notes

**Status:** Hotfix implemented. Server restart verified — no more false update spam.
**Platforms:** All (Claude Code, OpenCode, Cursor)
**Theme:** Fix stale version detection in long-running server

---

## Bug: False "update available" spam after upgrading

### Symptom

After upgrading to 0.3.7 via `pip install --upgrade roampal`, every new conversation injects:

```
<roampal-update>
A newer version of roampal is available: 0.3.7 (user has 0.3.6.1).
</roampal-update>
```

The user IS on 0.3.7. The spam persists even after `roampal stop && roampal start`.

### Root cause

Two bugs in `_check_for_updates()` in `roampal/server/main.py`:

1. **Stale import cache:** `from roampal import __version__` in a long-running server returns the value loaded at process startup. If the user upgrades without restarting the server (or the self-healing hook restarts it from a stale parent process), the imported `__version__` stays at the old value forever. Python's module cache (`sys.modules`) never re-reads the source file.

2. **Infinite cache TTL:** `_update_check_cache` was set once and never expired. Even if the version resolved correctly on a subsequent restart, a stale cache from a previous check persisted for the entire server lifetime. There was no mechanism to re-check after a pip upgrade.

### Why restarting didn't always fix it

The self-healing hook (`user_prompt_submit_hook.py`) auto-restarts the server when it's down. It uses `sys.executable` from the hook process, which may inherit module cache state from the parent shell. In practice, the server sometimes restarted with the correct version and sometimes didn't — depending on when during the boot sequence `from roampal import __version__` was first evaluated relative to pip's file writes.

### Fix

Two changes to `roampal/server/main.py`:

**1. `_get_installed_version()` (NEW):** Reads `__version__` directly from `roampal/__init__.py` on disk, bypassing Python's `sys.modules` cache. Parses the `__version__ = "X.Y.Z"` line from the source file. Falls back to the cached import only if file read fails.

```python
def _get_installed_version() -> str:
    try:
        init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
        with open(init_path, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    from roampal import __version__
    return __version__
```

**2. TTL on `_update_check_cache`:** Cache expires after `_CACHE_TTL_SECONDS` (30 minutes, same TTL used for search/injection caches). After expiry, the next cold start re-reads the version from disk and re-checks PyPI. This means a pip upgrade is picked up within 30 minutes without any server restart.

```python
_update_check_cache: Optional[tuple] = None
_update_check_time: float = 0.0

def _check_for_updates() -> tuple:
    global _update_check_cache, _update_check_time
    if _update_check_cache is not None and (time.time() - _update_check_time) < _CACHE_TTL_SECONDS:
        return _update_check_cache
    current = _get_installed_version()
    # ... PyPI check using `current` instead of cached import ...
```

**Added:** `import time` (was not previously imported in main.py).

---

## Files Modified

| File | Changes |
|------|---------|
| `roampal/server/main.py` | Added `import time`, added `_update_check_time` global, new `_get_installed_version()`, rewrote `_check_for_updates()` with TTL and disk-based version read |

---

## Verification

- Killed server with stale 0.3.6.1 cache, restarted with new code
- `GET /api/hooks/get-context` returns no `<roampal-update>` tag
- `_get_installed_version()` resolves to `C:\roampal-core\roampal\__init__.py` and parses `0.3.7` correctly
- TTL reuses existing `_CACHE_TTL_SECONDS = 1800` (30 minutes)
