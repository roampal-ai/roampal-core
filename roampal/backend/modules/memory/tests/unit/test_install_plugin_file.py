"""Tests for _install_plugin_file() (cli.py).

v0.5.6 Item 1: Regression tests for all failure modes in the plugin installer.
Covers shutil.copy failures, missing destination, size mismatches, fallback paths.
Pure-Python tests — no Windows-only code needed; runs cleanly on Linux/macOS CI.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
import shutil


class TestInstallPluginFile:
    """Regression tests for _install_plugin_file()."""

    @pytest.fixture
    def source_file(self, tmp_path):
        src = tmp_path / "source.ts"
        src.write_text("console.log('hello');\n")
        return src

    def test_happy_path_shutil_copy_succeeds(self, source_file, tmp_path, capsys):
        """Primary shutil.copy works, dest matches source size — assert success."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"
        _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert dest.exists()
        assert dest.stat().st_size == source_file.stat().st_size
        assert "Installed plugin:" in captured.out

    def test_copy_raises_permission_error(self, source_file, tmp_path, capsys):
        """Monkeypatch shutil.copy to raise — assert error message printed."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"

        original_shutil_copy = shutil.copy

        def mock_copy(src, dst):
            raise PermissionError(13, "Access denied")

        with patch("shutil.copy", side_effect=mock_copy):
            _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert not dest.exists()
        assert "Failed to install plugin:" in captured.out
        assert "Possible causes:" in captured.out
        assert "OpenCode Desktop is running" in captured.out
        assert "Controlled Folder Access" in captured.out

    def test_copy_succeeds_but_destination_missing(self, source_file, tmp_path, capsys):
        """Monkeypatch shutil.copy to be a no-op — assert file disappeared message."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"

        def mock_copy_noop(src, dst):
            pass  # Pretend copy succeeded but didn't write anything

        with patch("shutil.copy", side_effect=mock_copy_noop):
            _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert not dest.exists()
        assert "file disappeared after copy" in captured.out
        assert "Possible causes:" in captured.out
        assert "Antivirus quarantined" in captured.out

    def test_copy_succeeds_but_writes_zero_bytes(self, source_file, tmp_path, capsys):
        """Monkeypatch shutil.copy to write empty file — assert fallback corrects it."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"

        def mock_copy_empty(src, dst):
            Path(dst).write_bytes(b"")  # Simulates partial/truncated write

        with patch("shutil.copy", side_effect=mock_copy_empty):
            _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert dest.exists()
        expected_size = source_file.stat().st_size
        assert dest.stat().st_size == expected_size
        assert "Installed plugin:" in captured.out

    def test_copy_succeeds_but_writes_wrong_size(self, source_file, tmp_path, capsys):
        """Monkeypatch to write half the source — assert fallback corrects it."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"
        src_content = source_file.read_bytes()
        half_size = len(src_content) // 2

        def mock_copy_half(src, dst):
            Path(dst).write_bytes(src_content[:half_size])  # Half the content

        with patch("shutil.copy", side_effect=mock_copy_half):
            _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert dest.exists()
        assert dest.stat().st_size == len(src_content)
        assert "Installed plugin:" in captured.out

    def test_both_copy_and_fallback_produce_wrong_size(self, source_file, tmp_path, capsys):
        """Monkeypatch both copy and write_bytes — assert size mismatch error."""
        from roampal.cli import _install_plugin_file

        dest = tmp_path / "plugins" / "roampal.ts"
        src_content = source_file.read_bytes()
        half_size = len(src_content) // 2

        def mock_copy_half(src, dst):
            Path(dst).write_bytes(src_content[:half_size])

        original_write_bytes = Path.write_bytes

        def mock_write_bytes_wrong(self_path, data):
            # Simulates write_bytes also failing — writes half the requested data
            with open(str(self_path), "wb") as f:
                f.write(data[:len(data) // 2])

        with patch("shutil.copy", side_effect=mock_copy_half):
            with patch.object(Path, "write_bytes", new=mock_write_bytes_wrong):
                _install_plugin_file(source_file, dest)

        captured = capsys.readouterr()
        assert "file size mismatch after fallback copy" in captured.out


class TestVerifyPluginInstallTargets:
    """v0.5.6: Cross-target verification catches the silent stale-destination case
    that v0.5.5.2's dual-path install introduced (the actual symptom from issue #11).
    """

    @pytest.fixture
    def source_file(self, tmp_path):
        src = tmp_path / "source.ts"
        src.write_text("// v0.5.6 plugin\nconsole.log('hi');\n")
        return src

    def test_all_targets_fresh_no_warning(self, source_file, tmp_path, capsys):
        """Both targets match source — no warning should print."""
        from roampal.cli import _verify_plugin_install_targets

        a = tmp_path / "a.ts"
        b = tmp_path / "b.ts"
        a.write_bytes(source_file.read_bytes())
        b.write_bytes(source_file.read_bytes())

        _verify_plugin_install_targets(source_file, [a, b])
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
        assert "STALE" not in captured.out

    def test_one_stale_one_fresh_warns_with_repair(self, source_file, tmp_path, capsys):
        """Marcus's case: one target has stale content, the other is fresh.
        Warning lists the stale path and a copy command using the fresh one as donor."""
        from roampal.cli import _verify_plugin_install_targets

        fresh = tmp_path / "fresh.ts"
        stale = tmp_path / "stale.ts"
        fresh.write_bytes(source_file.read_bytes())
        stale.write_bytes(b"// old plugin code\n")

        _verify_plugin_install_targets(source_file, [fresh, stale])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "1 of 2 location(s)" in captured.out
        assert str(stale) in captured.out
        assert str(fresh) in captured.out
        # Repair command should suggest copying the fresh donor over the stale path.
        assert f"cp {fresh} {stale}" in captured.out
        assert "restart OpenCode Desktop" in captured.out

    def test_all_stale_falls_back_to_source_donor(self, source_file, tmp_path, capsys):
        """Edge case: every target diverged. Repair command points at source instead."""
        from roampal.cli import _verify_plugin_install_targets

        a = tmp_path / "a.ts"
        b = tmp_path / "b.ts"
        a.write_bytes(b"// stale a\n")
        b.write_bytes(b"// stale b\n")

        _verify_plugin_install_targets(source_file, [a, b])
        captured = capsys.readouterr()
        assert "2 of 2 location(s)" in captured.out
        assert f"cp {source_file} {a}" in captured.out
        assert f"cp {source_file} {b}" in captured.out

    def test_unreadable_target_treated_as_stale(self, source_file, tmp_path, capsys):
        """A target that can't be read (vanished, permission denied) is stale."""
        from roampal.cli import _verify_plugin_install_targets

        good = tmp_path / "good.ts"
        good.write_bytes(source_file.read_bytes())
        missing = tmp_path / "never_written.ts"  # does not exist on disk

        _verify_plugin_install_targets(source_file, [good, missing])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert str(missing) in captured.out

    def test_unreadable_source_skips_verification(self, tmp_path, capsys):
        """If the source itself can't be read, verification skips quietly — caller already failed earlier."""
        from roampal.cli import _verify_plugin_install_targets

        absent_source = tmp_path / "no_such_source.ts"
        target = tmp_path / "a.ts"
        target.write_bytes(b"anything")

        _verify_plugin_install_targets(absent_source, [target])
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


class TestOpenCodePluginHardlink:
    """v0.5.6 Fix E: On Windows, AppData plugin path is a hardlink to the
    canonical .config copy — same inode, divergence becomes structurally
    impossible. Falls back to a real copy if os.link is unsupported.

    Tests run on any OS by mocking sys.platform and Path.home(); os.link
    works on Linux/macOS too, so the happy path is exercisable in CI.
    """

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        """Redirect Path.home() and APPDATA to tmp dirs; force win32 branch."""
        userprofile = tmp_path / "user"
        appdata = userprofile / "AppData" / "Roaming"
        userprofile.mkdir(parents=True)
        appdata.mkdir(parents=True)
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("pathlib.Path.home", lambda: userprofile)
        monkeypatch.setenv("APPDATA", str(appdata))
        return userprofile, appdata

    def test_hardlink_branch_produces_same_inode(self, fake_home):
        """Both install paths share an inode after configure_opencode(force=True)."""
        from roampal.cli import configure_opencode

        userprofile, appdata = fake_home
        configure_opencode(is_dev=False, force=True)

        canonical = userprofile / ".config" / "opencode" / "plugins" / "roampal.ts"
        alt = appdata / "opencode" / "plugins" / "roampal.ts"

        assert canonical.exists(), f"canonical not written: {canonical}"
        assert alt.exists(), f"alt not written: {alt}"
        # Same inode = same physical file = no divergence possible.
        assert canonical.stat().st_ino == alt.stat().st_ino
        # nlink=2 confirms exactly two names refer to the inode.
        assert canonical.stat().st_nlink == 2

    def test_hardlink_reflects_canonical_changes_at_alt(self, fake_home):
        """Writing to canonical is visible at alt without a second copy step."""
        from roampal.cli import configure_opencode

        userprofile, appdata = fake_home
        configure_opencode(is_dev=False, force=True)

        canonical = userprofile / ".config" / "opencode" / "plugins" / "roampal.ts"
        alt = appdata / "opencode" / "plugins" / "roampal.ts"
        marker = b"// hardlink test marker\n"
        original = canonical.read_bytes()
        try:
            canonical.write_bytes(original + marker)
            assert marker in alt.read_bytes()
        finally:
            canonical.write_bytes(original)

    def test_falls_back_to_copy_when_os_link_raises(self, fake_home):
        """Cross-volume / OneDrive / AV: os.link raises → copy fallback runs.

        Both paths still exist with matching content; inodes differ (not the same
        physical file) — that's the explicit fallback contract."""
        from roampal.cli import configure_opencode

        userprofile, appdata = fake_home

        def boom(_src, _dst):
            raise OSError("simulated cross-volume link failure")

        with patch("os.link", side_effect=boom):
            configure_opencode(is_dev=False, force=True)

        canonical = userprofile / ".config" / "opencode" / "plugins" / "roampal.ts"
        alt = appdata / "opencode" / "plugins" / "roampal.ts"
        assert canonical.exists() and alt.exists()
        assert canonical.read_bytes() == alt.read_bytes()
        # Distinct inodes — they're real copies, not links.
        assert canonical.stat().st_ino != alt.stat().st_ino

    def test_replaces_pre_existing_alt_file(self, fake_home):
        """If alt path already has stale content from a prior install, the
        hardlink replaces it cleanly (alt.unlink() before os.link())."""
        from roampal.cli import configure_opencode

        userprofile, appdata = fake_home
        alt = appdata / "opencode" / "plugins" / "roampal.ts"
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text("// pre-existing stale plugin\n")
        stale_inode = alt.stat().st_ino

        configure_opencode(is_dev=False, force=True)

        canonical = userprofile / ".config" / "opencode" / "plugins" / "roampal.ts"
        # Stale file is gone — replaced by the hardlink.
        assert alt.stat().st_ino == canonical.stat().st_ino
        assert alt.stat().st_ino != stale_inode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
