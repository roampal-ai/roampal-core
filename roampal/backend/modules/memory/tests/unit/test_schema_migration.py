"""
Unit Tests for ChromaDB Schema Migration - v0.2.2

Tests the schema migration that adds missing 'topic' columns
when users upgrade from ChromaDB 0.4.x/0.5.x to 1.x.

These tests ensure:
1. Migration adds missing columns without data loss
2. Migration is idempotent (safe to run multiple times)
3. Migration handles edge cases gracefully
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
import sqlite3

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


class TestSchemaMigration:
    """Test ChromaDB schema migration - v0.2.2 safety for upgrades."""

    @pytest.fixture
    def ums_with_old_schema(self, tmp_path):
        """Create UMS with simulated old ChromaDB schema (missing topic column)."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        chromadb_path = data_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create old schema WITHOUT topic column
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        # Minimal old schema - collections table without 'topic'
        cursor.execute("""
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dimension INTEGER
            )
        """)
        cursor.execute("INSERT INTO collections VALUES ('test-id', 'test-collection', 384)")

        # Minimal old schema - segments table without 'topic'
        cursor.execute("""
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT
            )
        """)
        cursor.execute("INSERT INTO segments VALUES ('seg-1', 'vector', 'collection')")

        conn.commit()
        conn.close()

        return UnifiedMemorySystem(data_path=str(data_path))

    def test_migration_adds_missing_topic_to_collections(self, ums_with_old_schema, tmp_path):
        """Migration should add topic column to collections table."""
        # Run migration
        ums_with_old_schema._migrate_chromadb_schema()

        # Verify column was added
        sqlite_path = tmp_path / "data" / "chromadb" / "chroma.sqlite3"
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(collections)")
        columns = {col[1] for col in cursor.fetchall()}
        conn.close()

        assert "topic" in columns, "Migration should add topic column to collections"

    def test_migration_adds_missing_topic_to_segments(self, ums_with_old_schema, tmp_path):
        """Migration should add topic column to segments table."""
        # Run migration
        ums_with_old_schema._migrate_chromadb_schema()

        # Verify column was added
        sqlite_path = tmp_path / "data" / "chromadb" / "chroma.sqlite3"
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(segments)")
        columns = {col[1] for col in cursor.fetchall()}
        conn.close()

        assert "topic" in columns, "Migration should add topic column to segments"

    def test_migration_preserves_existing_data(self, ums_with_old_schema, tmp_path):
        """Migration should not corrupt existing data."""
        # Run migration
        ums_with_old_schema._migrate_chromadb_schema()

        # Verify data is intact
        sqlite_path = tmp_path / "data" / "chromadb" / "chroma.sqlite3"
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        cursor.execute("SELECT id, name FROM collections")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "test-id"
        assert rows[0][1] == "test-collection"

    def test_migration_idempotent(self, ums_with_old_schema, tmp_path):
        """Running migration twice should not error."""
        # Run migration twice
        ums_with_old_schema._migrate_chromadb_schema()
        ums_with_old_schema._migrate_chromadb_schema()  # Should not throw

        # Still works
        sqlite_path = tmp_path / "data" / "chromadb" / "chroma.sqlite3"
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM collections")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_migration_skips_if_no_db(self, tmp_path):
        """Migration should skip gracefully if no ChromaDB exists."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        # No chromadb folder created

        ums = UnifiedMemorySystem(data_path=str(data_path))
        # Should not throw
        ums._migrate_chromadb_schema()

    def test_migration_with_new_schema_is_noop(self, tmp_path):
        """Migration on DB with topic column already should be a no-op."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        chromadb_path = data_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create new schema WITH topic column already
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dimension INTEGER,
                topic TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT,
                topic TEXT
            )
        """)
        conn.commit()
        conn.close()

        ums = UnifiedMemorySystem(data_path=str(data_path))
        # Should not throw or duplicate column
        ums._migrate_chromadb_schema()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
