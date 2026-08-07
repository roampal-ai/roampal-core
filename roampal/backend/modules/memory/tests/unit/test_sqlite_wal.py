"""Tests for _configure_sqlite_wal in chromadb_adapter.py.

v0.5.8 Item 1: SQLite WAL journaling + FULL durability on ChromaDB catalog.
These tests use only stdlib sqlite3 — no ChromaDB import, so they stay fast.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _create_minimal_sqlite(db_path: Path) -> None:
    """Create a minimal chroma.sqlite3 file with one table so the helper has something to configure."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS dummy (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


@pytest.mark.wal
class TestConfigureSqliteWal:
    """v0.5.8 Item 1: unit tests for _configure_sqlite_wal."""

    def test_wal_mode_applied_to_existing_db(self, tmp_path):
        """WAL journal mode is set on an existing SQLite file."""
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        _create_minimal_sqlite(db_file)

        result = _configure_sqlite_wal(str(tmp_path))

        # Reconnect and verify journal mode is WAL
        conn2 = sqlite3.connect(str(db_file))
        mode = conn2.execute("PRAGMA journal_mode").fetchone()[0]
        conn2.close()

        assert mode == "wal"
        assert result is None  # helper returns nothing

    def test_wal_mode_persists_across_connections(self, tmp_path):
        """WAL setting persists in DB header — second connection sees it without re-applying."""
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        _create_minimal_sqlite(db_file)
        _configure_sqlite_wal(str(tmp_path))

        # Open a brand-new connection without re-running the helper
        conn2 = sqlite3.connect(str(db_file))
        mode = conn2.execute("PRAGMA journal_mode").fetchone()[0]
        conn2.close()

        assert mode == "wal"  # header persistence — release-notes acceptance criterion

    def test_synchronous_full(self, tmp_path):
        """The WAL configuration connection requests synchronous=FULL (value 2)."""
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        _create_minimal_sqlite(db_file)
        _configure_sqlite_wal(str(tmp_path))

        # synchronous is connection-local, so verify the helper requests FULL
        # on its own configuration connection rather than claiming persistence.
        configured_conn = MagicMock()
        with patch(
            "roampal.backend.modules.memory.chromadb_adapter.sqlite3.connect",
            return_value=configured_conn,
        ):
            _configure_sqlite_wal(str(tmp_path))

        configured_conn.execute.assert_any_call("PRAGMA synchronous=FULL")

    def test_missing_db_file_silent_noop(self, tmp_path):
        """When chroma.sqlite3 doesn't exist yet, helper returns silently without creating it."""
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        result = _configure_sqlite_wal(str(tmp_path))

        assert result is None
        assert not (tmp_path / "chroma.sqlite3").exists()  # no file created

    def test_sqlite_failure_logs_warning_not_raise(self, tmp_path):
        """If sqlite3.connect raises, helper logs WARNING and does NOT raise — startup must never block."""
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        _create_minimal_sqlite(db_file)

        with patch(
            "roampal.backend.modules.memory.chromadb_adapter.sqlite3.connect",
            side_effect=Exception("simulated locked DB"),
        ):
            # Must not raise — startup must proceed even if WAL config fails
            result = _configure_sqlite_wal(str(tmp_path))

        assert result is None  # returns silently on failure too

    def test_uncommitted_write_does_not_corrupt(self, tmp_path):
        """WAL mode: uncommitted writes are rolled back cleanly; committed data survives.

        Honest approximation of a hard kill — true SIGKILL mid-write can't be done portably in pytest.
        We simulate by starting an uncommitted transaction and closing without commit (which rolls back).
        The key assertion is that the DB header remains intact and integrity_check passes.
        """
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        _create_minimal_sqlite(db_file)
        _configure_sqlite_wal(str(tmp_path))

        # Batch 1: commit a row
        conn_a = sqlite3.connect(str(db_file))
        conn_a.execute("INSERT INTO dummy (id) VALUES (42)")
        conn_a.commit()

        # Batch 2: start uncommitted insert, then close without committing (simulates hard kill)
        conn_b = sqlite3.connect(str(db_file))
        conn_b.execute("INSERT INTO dummy (id) VALUES (99)")
        # No commit — just close. In WAL mode this rolls back the uncommitted write.
        conn_b.close()

        # Reopen and verify: batch 1 present, batch 2 absent, DB healthy
        conn_c = sqlite3.connect(str(db_file))
        count = conn_c.execute("SELECT COUNT(*) FROM dummy WHERE id=42").fetchone()[0]
        missing = conn_c.execute("SELECT COUNT(*) FROM dummy WHERE id=99").fetchone()[0]
        integrity = conn_c.execute("PRAGMA integrity_check").fetchone()[0]
        conn_c.close()

        assert count == 1, "Committed data must survive"
        assert missing == 0, "Uncommitted write must be absent"
        assert integrity == "ok", "DB header and catalog must remain intact after uncommitted close"

    def test_hard_kill_during_write_preserves_catalog(self, tmp_path):
        """A separate process killed during a write must not corrupt committed metadata.

        Closing a connection normally is not a crash test: SQLite gets a chance to
        roll the transaction back. This test uses a real child process and kills it
        while a large transaction is being written, which exercises SQLite's crash
        recovery path instead.
        """
        from roampal.backend.modules.memory.chromadb_adapter import (
            _configure_sqlite_wal,
        )

        db_file = tmp_path / "chroma.sqlite3"
        marker_file = tmp_path / "writer-active"
        _create_minimal_sqlite(db_file)

        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE catalog (id INTEGER PRIMARY KEY, document TEXT)")
        conn.execute("INSERT INTO catalog (id, document) VALUES (1, 'committed metadata')")
        conn.commit()
        conn.close()
        _configure_sqlite_wal(str(tmp_path))

        child_code = r'''
import os
import sqlite3
import sys
import time

db_file, marker_file = sys.argv[1:]
conn = sqlite3.connect(db_file)
conn.execute("BEGIN IMMEDIATE")
for row_id in range(2, 10000):
    conn.execute(
        "INSERT INTO catalog (id, document) VALUES (?, ?)",
        (row_id, "in-flight catalog metadata " + ("x" * 4096)),
    )
    if row_id == 2:
        open(marker_file, "w").close()
    if row_id % 100 == 0:
        time.sleep(0.001)
time.sleep(30)  # Keep the uncommitted transaction open until the parent kills us.
'''
        child = subprocess.Popen([sys.executable, "-c", child_code, str(db_file), str(marker_file)])
        try:
            deadline = time.monotonic() + 10
            while not marker_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker_file.exists(), "writer did not reach the active transaction"
            child.kill()  # SIGKILL on POSIX; TerminateProcess on Windows
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

        recovered = sqlite3.connect(str(db_file))
        try:
            assert recovered.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert recovered.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert recovered.execute(
                "SELECT document FROM catalog WHERE id = 1"
            ).fetchone()[0] == "committed metadata"
            assert recovered.execute(
                "SELECT COUNT(*) FROM catalog WHERE id >= 2"
            ).fetchone()[0] == 0
        finally:
            recovered.close()
