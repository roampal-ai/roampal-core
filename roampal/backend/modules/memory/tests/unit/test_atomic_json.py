"""Tests for roampal.utils.atomic_json.write_json_atomic().

v0.5.3 Section 10: Crash-safe atomic JSON writes for profiles.json
and session state files.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestWriteJsonAtomic:
    """Section 10: write_json_atomic() correctness tests."""

    def test_creates_file_on_fresh_path(self, tmp_path):
        """Path doesn't exist → after call, file exists and round-trips through json.load."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "new_file.json"
        assert not dest.exists()

        write_json_atomic(dest, {"key": "value", "nested": {"a": 1}})

        assert dest.exists()
        loaded = json.loads(dest.read_text())
        assert loaded == {"key": "value", "nested": {"a": 1}}

    def test_replaces_existing_file(self, tmp_path):
        """Path exists with old content → after call, file contains new content."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "existing.json"
        dest.write_text(json.dumps({"old": "data"}))

        write_json_atomic(dest, {"new": "content"})

        loaded = json.loads(dest.read_text())
        assert loaded == {"new": "content"}
        assert "old" not in loaded

    def test_leaves_no_tmp_files_on_success(self, tmp_path):
        """After a successful call, no *.tmp sibling remains in the parent dir."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "no_tmp.json"
        write_json_atomic(dest, {"data": 123})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Found leftover tmp files: {tmp_files}"

    def test_preserves_original_on_exception(self, tmp_path):
        """Monkeypatch os.replace to raise OSError → original file byte-for-byte unchanged;
        no *.tmp file left behind (unlink path in the except runs)."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "preserve.json"
        original_content = json.dumps({"original": "data"})
        dest.write_text(original_content)

        with patch("roampal.utils.atomic_json.os.replace") as mock_replace:
            mock_replace.side_effect = OSError("simulated disk error")

            with pytest.raises(OSError, match="simulated disk error"):
                write_json_atomic(dest, {"should_not_write": True})

        # Original file must be unchanged
        assert dest.read_text() == original_content

        # No tmp file should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Found leftover tmp files: {tmp_files}"

    def test_creates_parent_dirs(self, tmp_path):
        """Path's parent doesn't exist → after call, parents created and file written."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "nonexistent" / "subdir" / "deep.json"
        assert not dest.parent.exists()

        write_json_atomic(dest, {"deep": True})

        assert dest.exists()
        loaded = json.loads(dest.read_text())
        assert loaded == {"deep": True}

    def test_indent_none_produces_compact(self, tmp_path):
        """Pass indent=None; output has no newlines or extra whitespace."""
        from roampal.utils.atomic_json import write_json_atomic

        dest = tmp_path / "compact.json"
        write_json_atomic(dest, {"a": 1, "b": [1, 2, 3]}, indent=None)

        content = dest.read_text()
        assert "\n" not in content.strip()
        assert " " not in content.replace('{"a": 1,"b": [1, 2, 3]}', "")
        # Should be compact: {"a":1,"b":[1,2,3]} or similar without newlines
        loaded = json.loads(content)
        assert loaded == {"a": 1, "b": [1, 2, 3]}
