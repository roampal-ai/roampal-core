"""Unit tests for roampal.profile_manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roampal.profile_manager import (
    DEFAULT_PROFILE,
    InvalidProfileNameError,
    Profile,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    ProfileRegistry,
    active_profile_name,
    profile_slug,
    resolve_data_path,
    system_default_data_path,
)


# --- profile_slug ----------------------------------------------------------


class TestProfileSlug:
    def test_basic(self):
        assert profile_slug("Work") == "work"

    def test_spaces_to_underscore(self):
        assert profile_slug("Work Project") == "work_project"

    def test_special_chars_to_underscore(self):
        assert profile_slug("my-project/2024!") == "my_project_2024"

    def test_collapses_consecutive_underscores(self):
        assert profile_slug("a   b") == "a_b"
        assert profile_slug("a---b") == "a_b"

    def test_strips_leading_trailing(self):
        assert profile_slug("__work__") == "work"
        assert profile_slug("---work---") == "work"

    def test_max_30_chars(self):
        name = "a" * 50
        assert len(profile_slug(name)) == 30

    def test_unicode_non_ascii_becomes_underscores(self):
        # Non-ASCII letters get stripped to underscores, then collapsed.
        # "café" -> "caf_" (é is non-ascii) -> "caf"
        assert profile_slug("café") == "caf"

    def test_empty_string_raises(self):
        with pytest.raises(InvalidProfileNameError):
            profile_slug("")

    def test_none_raises(self):
        with pytest.raises(InvalidProfileNameError):
            profile_slug(None)  # type: ignore

    def test_all_special_chars_raises(self):
        with pytest.raises(InvalidProfileNameError):
            profile_slug("---")

    def test_non_string_raises(self):
        with pytest.raises(InvalidProfileNameError):
            profile_slug(123)  # type: ignore


# --- ProfileRegistry -------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    """Fresh registry at a temp path per test."""
    return ProfileRegistry(registry_path=tmp_path / "profiles.json")


class TestRegistryLoad:
    def test_missing_file_starts_empty(self, registry):
        assert registry.list() == []

    def test_corrupt_file_starts_empty(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg_path.write_text("not valid json{{{")
        reg = ProfileRegistry(registry_path=reg_path)
        assert reg.list() == []

    def test_non_object_json_starts_empty(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg_path.write_text('["not", "an", "object"]')
        reg = ProfileRegistry(registry_path=reg_path)
        assert reg.list() == []

    def test_loads_existing_entries(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg_path.write_text(
            json.dumps({"work": "/custom/path", "home": None})
        )
        reg = ProfileRegistry(registry_path=reg_path)
        profiles = reg.list()
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"work", "home"}

    def test_skips_invalid_names(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg_path.write_text(json.dumps({"---": "/x", "valid": None}))
        reg = ProfileRegistry(registry_path=reg_path)
        names = {p.name for p in reg.list()}
        assert names == {"valid"}  # "---" slugifies to empty, skipped


class TestRegistryCreate:
    def test_create_new_profile(self, registry):
        p = registry.create("work")
        assert p.name == "work"
        assert p.slug == "work"
        assert p.path is None

    def test_create_persists(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg1 = ProfileRegistry(registry_path=reg_path)
        reg1.create("work")
        reg2 = ProfileRegistry(registry_path=reg_path)
        assert reg2.exists("work")

    def test_create_duplicate_raises(self, registry):
        registry.create("work")
        with pytest.raises(ProfileAlreadyExistsError):
            registry.create("work")

    def test_cannot_create_default(self, registry):
        with pytest.raises(ProfileAlreadyExistsError):
            registry.create(DEFAULT_PROFILE)

    def test_invalid_name_raises(self, registry):
        with pytest.raises(InvalidProfileNameError):
            registry.create("---")


class TestRegistryRegister:
    def test_register_with_path(self, registry):
        p = registry.register("existing", "/some/custom/path")
        assert p.path == "/some/custom/path"

    def test_register_persists(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg1 = ProfileRegistry(registry_path=reg_path)
        reg1.register("existing", "/some/path")
        reg2 = ProfileRegistry(registry_path=reg_path)
        got = reg2.get("existing")
        assert got is not None
        assert got.path == "/some/path"

    def test_register_empty_path_raises(self, registry):
        with pytest.raises(Exception):  # ProfileError
            registry.register("bad", "")

    def test_register_whitespace_path_raises(self, registry):
        with pytest.raises(Exception):
            registry.register("bad", "   ")

    def test_register_duplicate_raises(self, registry):
        registry.create("work")
        with pytest.raises(ProfileAlreadyExistsError):
            registry.register("work", "/some/path")


class TestRegistryDelete:
    def test_delete_removes_from_registry(self, registry):
        registry.create("work")
        registry.delete("work")
        assert not registry.exists("work")

    def test_delete_unknown_raises(self, registry):
        with pytest.raises(ProfileNotFoundError):
            registry.delete("ghost")

    def test_delete_persists(self, tmp_path):
        reg_path = tmp_path / "profiles.json"
        reg1 = ProfileRegistry(registry_path=reg_path)
        reg1.create("work")
        reg1.delete("work")
        reg2 = ProfileRegistry(registry_path=reg_path)
        assert not reg2.exists("work")

    def test_delete_with_destroy_data(self, tmp_path):
        reg = ProfileRegistry(registry_path=tmp_path / "profiles.json")
        data_dir = tmp_path / "data_to_destroy"
        data_dir.mkdir()
        (data_dir / "marker.txt").write_text("x")
        reg.register("trash", str(data_dir))
        reg.delete("trash", destroy_data=True)
        assert not data_dir.exists()

    def test_delete_without_destroy_keeps_data(self, tmp_path):
        reg = ProfileRegistry(registry_path=tmp_path / "profiles.json")
        data_dir = tmp_path / "keep_me"
        data_dir.mkdir()
        (data_dir / "marker.txt").write_text("x")
        reg.register("work", str(data_dir))
        reg.delete("work")  # destroy_data defaults False
        assert data_dir.exists()
        assert (data_dir / "marker.txt").exists()


class TestRegistryResolve:
    def test_default_resolves_to_system_default(self, registry):
        path = registry.resolve(DEFAULT_PROFILE)
        assert path == str(system_default_data_path())

    def test_none_resolves_to_system_default(self, registry):
        path = registry.resolve(None)
        assert path == str(system_default_data_path())

    def test_registered_custom_path(self, registry):
        registry.register("work", "/custom/dir")
        assert registry.resolve("work") == "/custom/dir"

    def test_registered_null_path_auto_locates(self, registry):
        registry.create("home")
        resolved = registry.resolve("home")
        expected = str(system_default_data_path() / "home")
        assert resolved == expected

    def test_unknown_name_raises(self, registry):
        with pytest.raises(ProfileNotFoundError):
            registry.resolve("does_not_exist")

    def test_slug_used_for_auto_location(self, registry):
        # "My Project!" slug = "my_project"
        registry.create("My Project!")
        resolved = registry.resolve("My Project!")
        assert resolved.endswith("my_project") or resolved.endswith(
            "my_project/"
        ) or resolved.endswith("my_project\\")


# --- Module-level helpers --------------------------------------------------


class TestResolveDataPath:
    def test_env_override_wins(self, monkeypatch, registry):
        monkeypatch.setenv("ROAMPAL_DATA_PATH", "/env/override")
        registry.create("work")
        # Even with a registered profile, env override takes precedence.
        assert resolve_data_path("work", registry=registry) == "/env/override"

    def test_no_env_uses_registry(self, monkeypatch, registry):
        monkeypatch.delenv("ROAMPAL_DATA_PATH", raising=False)
        registry.register("work", "/my/dir")
        assert resolve_data_path("work", registry=registry) == "/my/dir"

    def test_default_profile_resolves_to_system_default(self, monkeypatch, registry):
        monkeypatch.delenv("ROAMPAL_DATA_PATH", raising=False)
        assert resolve_data_path(DEFAULT_PROFILE, registry=registry) == str(
            system_default_data_path()
        )


class TestActiveProfileName:
    def test_defaults_to_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setenv("APPDATA", str(tmp_path))  # isolate config dir
        assert active_profile_name() == DEFAULT_PROFILE

    def test_env_var_respected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROAMPAL_PROFILE", "work")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert active_profile_name() == "work"

    def test_empty_env_var_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROAMPAL_PROFILE", "   ")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert active_profile_name() == DEFAULT_PROFILE


class TestActiveProfileFile:
    """Cover the persisted active-profile marker + precedence."""

    def _setup_isolated_config(self, monkeypatch, tmp_path):
        """Redirect the config dir to tmp_path across platforms."""
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # macOS path is derived from Path.home(); patch HOME for good measure.
        monkeypatch.setenv("HOME", str(tmp_path))

    def test_file_read_missing_returns_none(self, monkeypatch, tmp_path):
        from roampal.profile_manager import read_active_profile_file
        self._setup_isolated_config(monkeypatch, tmp_path)
        assert read_active_profile_file() is None

    def test_file_write_then_read(self, monkeypatch, tmp_path):
        from roampal.profile_manager import (
            read_active_profile_file,
            write_active_profile_file,
        )
        self._setup_isolated_config(monkeypatch, tmp_path)
        write_active_profile_file("work")
        assert read_active_profile_file() == "work"

    def test_file_clear(self, monkeypatch, tmp_path):
        from roampal.profile_manager import (
            clear_active_profile_file,
            read_active_profile_file,
            write_active_profile_file,
        )
        self._setup_isolated_config(monkeypatch, tmp_path)
        write_active_profile_file("work")
        removed = clear_active_profile_file()
        assert removed is True
        assert read_active_profile_file() is None

    def test_clear_when_no_file_returns_false(self, monkeypatch, tmp_path):
        from roampal.profile_manager import clear_active_profile_file
        self._setup_isolated_config(monkeypatch, tmp_path)
        assert clear_active_profile_file() is False

    def test_active_name_precedence_env_wins(self, monkeypatch, tmp_path):
        from roampal.profile_manager import write_active_profile_file
        self._setup_isolated_config(monkeypatch, tmp_path)
        write_active_profile_file("from_file")
        monkeypatch.setenv("ROAMPAL_PROFILE", "from_env")
        assert active_profile_name() == "from_env"

    def test_active_name_file_beats_default(self, monkeypatch, tmp_path):
        from roampal.profile_manager import write_active_profile_file
        self._setup_isolated_config(monkeypatch, tmp_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        write_active_profile_file("from_file")
        assert active_profile_name() == "from_file"

    def test_active_name_source_env(self, monkeypatch, tmp_path):
        from roampal.profile_manager import active_profile_source
        self._setup_isolated_config(monkeypatch, tmp_path)
        monkeypatch.setenv("ROAMPAL_PROFILE", "x")
        assert active_profile_source() == "env"

    def test_active_name_source_file(self, monkeypatch, tmp_path):
        from roampal.profile_manager import (
            active_profile_source,
            write_active_profile_file,
        )
        self._setup_isolated_config(monkeypatch, tmp_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        write_active_profile_file("x")
        assert active_profile_source() == "file"

    def test_active_name_source_default(self, monkeypatch, tmp_path):
        from roampal.profile_manager import active_profile_source
        self._setup_isolated_config(monkeypatch, tmp_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        assert active_profile_source() == "default"
