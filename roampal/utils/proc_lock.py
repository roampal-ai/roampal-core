"""
Cross-process single-flight helper (Round 2 Item 7 / Task 23, amended by
v0.6.0 review fix 6).

A restart of the shared server must happen at most once no matter how
many restarter processes race (Claude Code hooks, the MCP server, the
OpenCode plugin). A plain O_EXCL lock file with mtime-based staleness
does the job without OS-specific advisory locks (hooks and the plugin
run in environments where fcntl/msvcrt juggling is unnecessary risk).

Contract:
- acquire(port) -> str path on success, None when someone else holds it
  (waits up to `wait` seconds for FIRST-COME holders, never spins into
  a second restart).
- release(port) removes the file (best effort, OWNERSHIP-CHECKED: only
  when the file still holds our token — another restarter that broke our
  lock and re-acquired must keep its own).
- A crashed restarter (file left behind) is recovered after `ttl` by
  mtime: the next acquirer breaks the stale lock and proceeds. fix 6:
  the TTL must exceed the restarter's WORST-CASE hold (port cleanup up
  to ~10s + 1s pause + up-to-15s health wait ≈ 26s), or a waiter steals
  a LIVE lock mid-restart and a second server gets spawned.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

# fix 6: was 15.0 — shorter than the restarter's worst-case hold (~26s:
# 5s netstat + 5s taskkill + 1s pause + 15s health-wait), which let a
# waiter break a LIVE lock and spawn a second server. 45s bounds crashed-
# holder recovery while staying > every legitimate hold.
LOCK_STALE_TTL_SECONDS = 45.0
_LOCK_WAIT_SECONDS = 10.0

# port -> uuid written by THIS process's live acquire(). release() only
# unlinks when the file still carries OUR uuid (fix 6: ownership check —
# the pid+uuid in the file were previously written but never compared).
_HELD_TOKENS: Dict[int, str] = {}


def _config_dir() -> Path:
    """Same location the registry/active-profile file live in."""
    from roampal.profile_manager import _config_dir as _pm_config_dir

    return _pm_config_dir()


def _lock_path(port: int) -> Path:
    """One lock file per server port (dev 27183 vs prod 27182)."""
    return _config_dir() / f"server_restart_{int(port)}.lock"


def acquire(port: int, *, ttl: float = LOCK_STALE_TTL_SECONDS, wait: float = _LOCK_WAIT_SECONDS):
    """Wait (up to `wait` seconds) to become the single restarter.

    Returns the lock file path (truthy) on success; None when the wait
    expired — the caller should treat the server as 'restarting in
    progress' and wait for its health, never spawn a second one.
    """
    path = _lock_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait
    while True:
        token = uuid.uuid4().hex
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {token}".encode("utf-8"))
            os.close(fd)
            _HELD_TOKENS[int(port)] = token
            return path
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > ttl:
                    # Crashed holder — break the stale lock and retry.
                    # fix 6: TTL now exceeds any live restarter's hold, so
                    # a break here really means the holder is gone.
                    path.unlink()
                    continue
            except OSError:
                pass
        if time.time() >= deadline:
            return None
        time.sleep(0.05)


def release(port: int) -> None:
    """Ownership-checked unlock: remove the lock file only when it still
    carries THIS process's token. If another restarter broke our lock and
    re-acquired, ITS lock is left untouched (the previous unconditional
    unlink could delete a successor's live lock)."""
    port = int(port)
    token = _HELD_TOKENS.pop(port, None)
    path = _lock_path(port)
    if token is None:
        try:
            path.unlink()
        except OSError:
            pass
        return
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content.split(" ", 1)[-1] == token:
            path.unlink()
    except OSError:
        pass  # already gone — nothing to do
