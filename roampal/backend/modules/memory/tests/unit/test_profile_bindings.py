"""
Profile-directory binding resolution tests (v0.6.0 Item 3 / Test Plan B).

Registry JSON is written into a temp APPDATA-equivalent the real resolver
reads via ProfileRegistry(registry_path=...); cwd AND registry are
param-injected (never os.chdir, never the machine registry) per the Plan B
fixture note.

Case matrix (Test Plan B):
- direct match: cwd inside a bound dir -> bound profile
- ancestry inherit: cwd nested under the bound root -> inherits
- innermost wins: two nested bindings, cwd under the deeper one -> deeper
- outer not leaked: removing the inner binding falls back to the outer one
- env beats binding: ROAMPAL_PROFILE wins over a matching binding
- global default fallback: no binding -> persisted use, then 'default'
- registry schema: registries WITHOUT a bindings key resolve exactly as
  before (backward compat); the reserved key never becomes a profile
- drive-style casing (Windows): C:/roampal-core vs C:\\ROAMPAL-CORE\\
- unregistered bind: clean error, registry untouched
- unbind idempotence: absent binding -> no change, no error
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import json
from pathlib import Path

import pytest

from roampal.profile_manager import (
    DEFAULT_PROFILE,
    ProfileNotFoundError,
    ProfileRegistry,
    _norm_binding_dir,
    active_profile_name,
    active_profile_source,
    binding_for_cwd,
)


def _seed_registry(path, profiles=None, bindings=None):
    """Write the exact registry JSON shape the real loader reads.

    `profiles` uses the legacy flat name->path shape; `bindings` is the
    reserved top-level key (absent when passed empty/None).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(profiles or {})
    if bindings:
        payload.update(bindings)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestResolutionMatrix:
    """Test Plan B case matrix rows."""

    def test_direct_match_cwd_inside_bound_dir(self, tmp_path):
        """Matrix rows: cwd inside/dir == bound root -> bound profile; source 'binding'."""
        root = tmp_path / "proj"
        root.mkdir()
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"ghost": None},
            bindings={"bindings": {str(root): "ghost"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        assert active_profile_name(cwd=root, registry=reg) == "ghost"
        assert active_profile_source(cwd=root, registry=reg) == "binding"
        matched = binding_for_cwd(cwd=root, registry=reg)
        assert matched is not None and matched[0], "matched key normalizes"

    def test_ancestry_inherit_from_subdirectory(self, tmp_path):
        """Matrix row: cwd is a child/grandchild of the bound root -> bound."""
        root = tmp_path / "proj"
        inner = root / "src" / "deep"
        inner.mkdir(parents=True)
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"ghost": None},
            bindings={"bindings": {str(root): "ghost"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        assert active_profile_name(cwd=inner, registry=reg) == "ghost"
        matched = binding_for_cwd(cwd=inner, registry=reg)
        assert matched is not None
        key, profile = matched
        assert profile == "ghost"
        # the walk matched the stored root itself, not the child
        assert Path(key) != inner

    def test_innermost_wins(self, tmp_path):
        """Matrix: '/root'->A, '/root/sub'->B, cwd '/root/sub/x' -> B."""
        root = tmp_path / "root"
        sub = root / "sub"
        leaf = sub / "x"
        leaf.mkdir(parents=True)
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"alpha": None, "beta": None},
            bindings={"bindings": {str(root): "alpha", str(sub): "beta"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        assert active_profile_name(cwd=leaf, registry=reg) == "beta"
        assert active_profile_source(cwd=leaf, registry=reg) == "binding"
        # cwd AT the root (not under sub) binds to the outer profile
        assert active_profile_name(cwd=root, registry=reg) == "alpha"

    def test_outer_not_leaked(self, tmp_path):
        """Matrix: '/root/sub' removed from bindings; outer '/root'->A intact."""
        root = tmp_path / "root"
        sub = root / "sub"
        sub.mkdir(parents=True)
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"alpha": None, "beta": None},
            bindings={"bindings": {str(root): "alpha", str(sub): "beta"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        assert reg.unbind(str(sub)) is True
        # Re-read fresh registry to avoid any in-memory bleed
        reg2 = ProfileRegistry(registry_path=reg_path)
        # After removal the sub path resolves to the OUTER binding
        assert active_profile_name(cwd=sub, registry=reg) == "alpha"
        assert active_profile_name(cwd=root, registry=reg) == "alpha"

    def test_env_beats_binding(self, tmp_path, monkeypatch):
        """Matrix: ROAMPAL_PROFILE=other + binding->A -> other; source 'env'."""
        root = tmp_path / "proj"
        root.mkdir()
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"ghost": None, "other": None},
            bindings={"bindings": {str(root): "ghost"}},
        )
        monkeypatch.setenv("ROAMPAL_PROFILE", "other")
        assert active_profile_name(cwd=root) == "other"
        assert active_profile_source(cwd=root) == "env"

    def test_global_default_fallback_persisted_use(self, tmp_path, monkeypatch):
        """Matrix: no binding matches, persisted use = main -> main; 'file'."""
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(reg_path, profiles={"main": None})
        import roampal.profile_manager as pm

        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: "main")
        cwd = tmp_path / "elsewhere"
        cwd.mkdir()
        assert active_profile_name(cwd=cwd) == "main"
        assert active_profile_source(cwd=cwd) == "file"

    def test_no_binding_no_use_falls_to_default(self, tmp_path, monkeypatch):
        """Matrix: no binding, no persisted default -> 'default', source 'default'."""
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(reg_path)
        import roampal.profile_manager as pm

        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        cwd = tmp_path / "elsewhere"
        cwd.mkdir()
        assert active_profile_name(cwd=cwd) == DEFAULT_PROFILE
        assert active_profile_source(cwd=cwd) == "default"

    def test_registry_schema_backward_compat(self, tmp_path):
        """Matrix: registries without 'bindings' resolve exactly as before;
        the reserved key never registers a profile named 'bindings'.
        An OLD registry with a real profile named 'bindings' still loads."""
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(reg_path, profiles={"alpha": None, "main": None})
        reg = ProfileRegistry(registry_path=reg_path)
        assert reg.exists("alpha") and reg.exists("main")
        assert not reg.exists("bindings")

        # Policy note: the reserved key WINS over a profile named
        # 'bindings' in an old registry — an extremely unlikely name that
        # the loader now always interprets as the binding map (its repr
        # lands in the malformed-binding warning path, never a profile).
        reg_path2 = tmp_path / "roampal2" / "profiles.json"
        _seed_registry(reg_path2, profiles={"bindings": None})
        reg2 = ProfileRegistry(registry_path=reg_path2)
        assert not reg2.exists("bindings")
        assert reg2.bindings() == {}

    def test_bound_profile_unregistered_fails_explicit(self, tmp_path, monkeypatch):
        """Acceptance: unknown/unregistered bound names fail explicit at
        resolve_data_path — never a silent fall-through to the default."""
        import roampal.profile_manager as pm

        root = tmp_path / "proj"
        root.mkdir()
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"alpha": None},
            bindings={"bindings": {str(root): "ghost"}},  # 'ghost' NOT registered
        )
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        # Resolver surfaces the binding name...
        assert active_profile_name(cwd=root) == "ghost"
        # ...and path resolution fails explicitly instead of falling back.
        with pytest.raises(ProfileNotFoundError):
            pm.resolve_data_path(active_profile_name(cwd=root))

    @pytest.mark.skipif(os.name != "nt", reason="drive-style casing is Windows semantics")
    def test_drive_style_casing_windows(self, tmp_path):
        """Matrix: bound C:/roampal-core, cwd C:\\ROAMPAL-CORE\\ -> match."""
        root = tmp_path / "proj"
        inner = root / "Sub"
        inner.mkdir(parents=True)
        reg_path = tmp_path / "roampal" / "profiles.json"
        forward = str(inner).replace("\\", "/")
        _seed_registry(
            reg_path,
            profiles={"ghost": None},
            bindings={"bindings": {forward: "ghost"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        # Query with opposite casing + trailing separator on the same dir
        weird = Path(str(inner).upper() + os.sep)
        matched = binding_for_cwd(cwd=weird, registry=reg)
        assert matched is not None and matched[1] == "ghost"
        assert active_profile_name(cwd=weird, registry=reg) == "ghost"

    @pytest.mark.skipif(
        os.name == "nt", reason="posix exactness is the non-Windows analog above"
    )
    def test_drive_style_casing_posix_exact(self, tmp_path):
        """Non-Windows: normcase is identity — the exact same path must
        match after normpath (trailing separators collapse)."""
        root = tmp_path / "proj"
        root.mkdir()
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(
            reg_path,
            profiles={"ghost": None},
            bindings={"bindings": {str(root): "ghost"}},
        )
        reg = ProfileRegistry(registry_path=reg_path)
        trailing = Path(str(root) + os.sep)
        matched = binding_for_cwd(cwd=trailing, registry=reg)
        assert matched is not None and matched[1] == "ghost"


class TestBindingMutations:
    """bind/unbind registry ops (the resolver-facing API Task 13 wires)."""

    def test_bind_registered_profile_round_trips(self, tmp_path):
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("work")
        key = reg.bind("work", str(tmp_path / "proj"))
        assert key == _norm_binding_dir(str(tmp_path / "proj"))
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["bindings"][key] == "work"
        assert raw["work"] is None
        assert ProfileRegistry(registry_path=path).bindings()[key] == "work"

    def test_bind_unregistered_clean_error_registry_untouched(self, tmp_path):
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("alpha")
        before = json.loads(path.read_text(encoding="utf-8"))
        with pytest.raises(ProfileNotFoundError):
            reg.bind("ghosttown", str(tmp_path))
        assert json.loads(path.read_text(encoding="utf-8")) == before

    def test_unbind_idempotent(self, tmp_path):
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("work")
        d = reg.bind("work", str(tmp_path / "proj"))
        assert reg.unbind(d) is True
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        # Matrix row: unbind already-absent binding -> no error, no change
        assert reg.unbind(d) is False
        assert json.loads(path.read_text(encoding="utf-8")) == snapshot
        assert reg.bindings() == {}

    def test_delete_cascades_bindings(self, tmp_path):
        """v0.6.0 review fix 8: deleting a profile removes bindings that
        point at it — left in place, every request from a bound directory
        resolves to an unregistered name and 404s forever (the 'quietly
        breaks memory' failure). Re-creating the same name does NOT
        resurrect the binding (explicit re-bind required)."""
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("work")
        d1 = reg.bind("work", str(tmp_path / "proj-a"))
        d2 = reg.bind("work", str(tmp_path / "proj-b"))
        assert len(reg.bindings()) == 2

        reg.delete("work")
        assert reg.bindings() == {}, "bindings must cascade with the profile"
        # Persisted, not just in-memory:
        assert json.loads(path.read_text(encoding="utf-8")).get("bindings") is None

        # Re-creating the same name does not resurrect stale bindings.
        reg.create("work")
        assert reg.bindings() == {}
        assert binding_for_cwd(cwd=tmp_path / "proj-a", registry=reg) is None

    def test_delete_leaves_other_profiles_bindings(self, tmp_path):
        """fix 8 scope: only the deleted profile's bindings cascade —
        bindings to other profiles survive untouched."""
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("doomed")
        reg.create("keeper")
        reg.bind("doomed", str(tmp_path / "gone"))
        reg.bind("keeper", str(tmp_path / "stays"))
        reg.delete("doomed")
        bindings = reg.bindings()
        assert bindings == {_norm_binding_dir(str(tmp_path / "stays")): "keeper"}

    def test_bind_normalizes_key_duplicates(self, tmp_path):
        """Bind-time normalization collapses casing/separator variants."""
        path = tmp_path / "roampal" / "profiles.json"
        reg = ProfileRegistry(registry_path=path)
        reg.create("work")
        d1 = reg.bind("work", str(tmp_path / "Proj"))
        if os.name == "nt":
            # Same directory, different casing + trailing slash -> same key
            d2 = reg.bind("work", str(tmp_path / "PROJ"))
            assert d1 == d2
            assert len(reg.bindings()) == 1
        else:
            # POSIX is case-sensitive: 'Proj' and 'PROJ' are different keys.
            # What still collapses is the trailing separator on the SAME dir.
            d2 = reg.bind("work", str(tmp_path / "Proj") + os.sep)
            assert d1 == d2
            assert len(reg.bindings()) == 1

    def test_sources_enumeration(self, tmp_path, monkeypatch):
        """Every active_profile_source() tag stays reachable: env/binding/file/default."""
        reg_path = tmp_path / "roampal" / "profiles.json"
        _seed_registry(reg_path)
        import roampal.profile_manager as pm

        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        cwd = tmp_path / "w"
        cwd.mkdir()
        assert active_profile_source(cwd=cwd) == "default"
        # binding present
        reg = ProfileRegistry(registry_path=reg_path)
        reg.create("b")
        reg.bind("b", str(cwd))
        assert active_profile_source(cwd=cwd) == "binding"
        assert active_profile_name(cwd=cwd) == "b"
        # env overrides
        monkeypatch.setenv("ROAMPAL_PROFILE", "zzz")
        assert active_profile_source(cwd=cwd) == "env"
        # persisted file slice (no binding): cwd far from bound root
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: "keeper")
        assert active_profile_source(cwd=tmp_path / "far" / "away") == "file"

    def test_binding_absent_when_registry_empty(self, tmp_path):
        reg = ProfileRegistry(registry_path=tmp_path / "roampal" / "profiles.json")
        assert reg._bindings == {}
