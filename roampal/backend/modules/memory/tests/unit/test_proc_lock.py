"""
proc_lock single-flight tests (Round 2 Item 7 / Task 23).

The cross-process lock that makes CC hooks, the MCP server and the
OpenCode plugin produce EXACTLY ONE visible restart:
- two concurrent acquirers -> exactly one holds the lock
- second acquirer waits -> gives up after the wait expires
- crashed holder (stale lock past TTL) -> recovered by the next acquirer
"""

import os
import sys
import threading
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import pytest

import roampal.profile_manager as pm
from roampal.utils import proc_lock


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Point the lock dir (profile-manager config dir) at a temp tree."""
    monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
    return tmp_path / "config"


def _acquire(port=29191, **kwargs):
    return proc_lock.acquire(port, **kwargs)


class TestSingleFlight:
    def test_two_threads_exactly_one_acquirer(self, isolated_config):
        """Two concurrent restarters -> exactly one holds the lock."""
        barrier = threading.Barrier(2)
        results = {}

        def contender(name):
            barrier.wait()
            results[name] = proc_lock.acquire(29191, wait=0.5)

        threads = [threading.Thread(target=contender, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results.values() if r is not None]
        assert len(winners) == 1, (
            f"single-flight violated: {len(winners)} acquirers hold the lock: {results}"
        )

    def test_releaselets_second_in(self, isolated_config):
        first = _acquire()
        assert first is not None
        try:
            # Release -> the waiting contender can now acquire.
            proc_lock.release(29191)
            second = _acquire(wait=1.0)
            assert second is not None
        finally:
            proc_lock.release(29191)

    def test_stale_lock_recovered_after_ttl(self, isolated_config):
        # Simulate a crashed holder: lock file exists with a far-past mtime.
        lock = isolated_config / "server_restart_29192.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("999 dead-pid", encoding="utf-8")
        stale = time.time() - 60
        os.utime(lock, (stale, stale))

        # Default TTL is 15s -> the stale lock must be broken, not waited on.
        got = _acquire(29192, ttl=proc_lock.LOCK_STALE_TTL_SECONDS, wait=2.0)
        assert got is not None, "stale lock was never recovered"
        proc_lock.release(29192)

    def test_fresh_lock_held_until_ttl(self, isolated_config):
        first = _acquire(29193)
        assert first is not None
        try:
            # A fresh lock from a LIVE holder must NOT be broken by a skipper.
            got = proc_lock.acquire(29193, ttl=3600.0, wait=0.3)
            assert got is None, "second acquirer broke a live holder's lock"
        finally:
            proc_lock.release(29193)

    def test_per_port_namespacing(self, isolated_config):
        a = _acquire(27182)  # prod
        b = _acquire(27183)  # dev
        assert a is not None and b is not None, "ports share a lock namespace"
        proc_lock.release(27182)
        proc_lock.release(27183)


class TestOwnershipCheckedRelease:
    """v0.6.0 review fix 6: release() must not delete a successor's lock,
    and the TTL must exceed the restarter's worst-case hold."""

    def test_release_only_removes_our_own_lock(self, isolated_config):
        """If our lock was broken and re-acquired by someone else, our
        release must leave THEIR lock intact (the old unconditional
        unlink deleted the successor's live lock)."""
        first = _acquire(29191)
        assert first is not None
        # Simulate the steal: another process broke our lock and acquired.
        lock_path = isolated_config / "server_restart_29191.lock"
        lock_path.unlink()
        lock_path.write_text("4242 foreign-token", encoding="utf-8")

        proc_lock.release(29191)  # our token no longer matches
        assert lock_path.exists(), "release deleted a foreign holder's lock"

        # Our own lock IS removed by our release.
        second = proc_lock.acquire(29192, wait=0.2)
        assert second is not None
        proc_lock.release(29192)
        assert not (isolated_config / "server_restart_29192.lock").exists()

    def test_release_without_acquire_still_unlinks(self, isolated_config):
        """Legacy best-effort path: release with no held token removes
        whatever is there (old callers / best-effort semantics)."""
        lock = isolated_config / "server_restart_29193.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("1 2", encoding="utf-8")
        proc_lock.release(29193)
        assert not lock.exists()

    def test_ttl_exceeds_worst_case_hold(self):
        """fix 6: the default TTL must exceed the restarter's worst-case
        hold (~26s = 5s netstat + 5s taskkill + 1s pause + 15s health wait),
        or a waiter can steal a LIVE lock and spawn a second server."""
        assert proc_lock.LOCK_STALE_TTL_SECONDS >= 30.0

    def test_live_holder_not_stolen_at_full_ttl(self, isolated_config):
        """A waiter using the DEFAULT TTL must never break a lock whose
        mtime is younger than the TTL (the live long-restart scenario)."""
        first = _acquire(29194)
        assert first is not None
        try:
            # Age the lock to just under the default TTL.
            lock = isolated_config / "server_restart_29194.lock"
            near_stale = time.time() - (proc_lock.LOCK_STALE_TTL_SECONDS - 2)
            os.utime(lock, (near_stale, near_stale))
            got = proc_lock.acquire(29194, ttl=proc_lock.LOCK_STALE_TTL_SECONDS, wait=0.3)
            assert got is None, "waiter broke a live holder's lock below TTL"
        finally:
            proc_lock.release(29194)
