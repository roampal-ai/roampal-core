"""
Profile Manager for Roampal Core.

Profiles are named isolated memory stores. A profile has:
  - A name (user-facing)
  - A slug (filesystem-safe derived from the name)
  - An optional custom path (for registering pre-existing directories)

The registry lives at <config_dir>/profiles.json and maps name -> optional path.
Resolution rules:
  1. If name is "default" (or None/empty) and not registered, resolve to the
     system default data path (same as pre-v0.5.2 behaviour).
  2. If name is registered with an explicit path, use that path (lets existing
     users register directories they already maintain via ROAMPAL_DATA_PATH).
  3. If name is registered with no path, auto-locate at <default_base>/<slug>/.
  4. If name is unknown, raise ProfileNotFoundError.

Environment variables continue to be honored for backward compatibility:
  - ROAMPAL_DATA_PATH (absolute path override, highest precedence)
  - ROAMPAL_DEV (dev-mode toggle: Roampal_DEV/data vs Roampal/data)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_PROFILE = "default"
_REGISTRY_FILENAME = "profiles.json"
_ACTIVE_FILENAME = "active_profile"
_MAX_SLUG_LEN = 30


# --- Errors ----------------------------------------------------------------


class ProfileError(Exception):
    """Base class for profile errors."""


class ProfileNotFoundError(ProfileError):
    """Requested profile is not registered."""


class ProfileAlreadyExistsError(ProfileError):
    """Attempted to create/register a profile name that is already taken."""


class InvalidProfileNameError(ProfileError):
    """Profile name cannot be slugified to a valid identifier."""


# --- Data types ------------------------------------------------------------


@dataclass
class Profile:
    """A registered memory profile."""

    name: str
    slug: str
    path: Optional[str] = None  # None -> auto-located under default base
    resolved_path: Optional[str] = None  # computed on resolve()

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"name": self.name, "slug": self.slug, "path": self.path}


# --- Slug / path helpers ---------------------------------------------------


def profile_slug(name: str) -> str:
    """Slugify a profile name.

    Lowercase, replace non-alphanum with `_`, collapse consecutive `_`, strip
    leading/trailing `_`, cap at 30 chars.

    Raises InvalidProfileNameError if the result is empty.
    """
    if not name or not isinstance(name, str):
        raise InvalidProfileNameError("Profile name must be a non-empty string")
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")[:_MAX_SLUG_LEN]
    if not slug:
        raise InvalidProfileNameError(
            f"Profile name {name!r} slugifies to an empty string"
        )
    return slug


def _system_default_base() -> Path:
    """Return the base directory for Roampal data (minus the per-profile subdir)."""
    dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
    app_folder = "Roampal_DEV" if dev_mode else "Roampal"

    if os.name == "nt":  # Windows
        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        return Path(appdata) / app_folder / "data"
    if sys.platform == "darwin":  # macOS
        return (
            Path.home() / "Library" / "Application Support" / app_folder / "data"
        )
    # Linux
    xdg_data = os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
    )
    folder = app_folder.lower()  # Linux convention: lowercase app dirs
    return Path(xdg_data) / folder / "data"


def system_default_data_path() -> Path:
    """The data_path the 'default' profile resolves to when not explicitly registered."""
    return _system_default_base()


def neutral_spawn_dir() -> Path:
    """Task 36: a neutral working directory for subprocesses that must never
    resolve ``import roampal`` through the caller's cwd.

    For ``python -m roampal...``, sys.path[0] is the cwd — a project containing
    a local ``roampal/`` directory shadows the installed package and feeds the
    shared server the WRONG code. The data base directory is guaranteed off of
    any Python package tree and is created if missing (the server expects it).
    Combined with ``spawn_isolation_flags()`` on the command line, neither cwd
    nor PYTHONPATH can influence module resolution."""
    base = _system_default_base()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        user_home = Path.home()
        try:
            user_home.mkdir(exist_ok=True)
        except OSError:  # pragma: no cover - home missing is pathological
            user_home = Path(tempfile.gettempdir())
        base = user_home
    return base


def spawn_isolation_flags() -> List[str]:
    """v0.6.0 review fix 1: interpreter flags every roampal spawn/written
    command uses instead of ``-I``.

    ``-I`` implies ``-s`` (skip USER site-packages), which silently breaks
    Microsoft Store Python (always installs to user site), ``pip install
    --user``, and system Pythons whose site-packages is not writable — the
    spawned server could not ``import roampal`` at all.

    Replacement contract:
    - ``-E``: ignore PYTHONPATH/PYTHONHOME env shadowing. Supported by every
      Python this package installs on (3.10+), so it is safe to write from
      the TS plugin, which cannot version-check the interpreter.
    - ``-P`` (3.11+ only): keep the cwd off sys.path for ``-m`` launches in
      PROJECT directories (init-written hook/MCP commands). NOT used from
      the plugin: a 3.10 interpreter hard-fails on the unknown flag.
    Runtime spawn sites pin a neutral cwd (neutral_spawn_dir), so ``-P``
    there is belt-and-suspenders against a polluted data dir.
    """
    flags = ["-E"]
    if sys.version_info >= (3, 11):
        flags.append("-P")
    return flags


# ============================================================================
# Server-launch pin persistence (v0.6.0 review fix 5)
# ============================================================================

def _server_pin_path(port: int) -> Path:
    """Per-port pin file: the profile a server on this port was launched with
    (`roampal start --profile X`). Spawn sites read it so a respawn of the
    pinned server re-passes --profile instead of silently dropping the pin."""
    return _config_dir() / f"server_pin_{int(port)}.txt"


def read_server_pin(port: int) -> Optional[str]:
    """The pinned profile for this port, or None. Tolerant: a stale,
    empty, or unreadable pin file counts as unpinned. Line 1 of the file
    is the profile (line 2, if present, is the launching pid — visibility
    data for `profile show`)."""
    try:
        lines = _server_pin_path(port).read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line:
                return line
        return None
    except (OSError, ValueError):
        return None


def write_server_pin(port: int, profile_name: str) -> None:
    """Persist the launch pin. Called by `roampal start --profile X`.

    Line 1: profile name (what read_server_pin returns). Line 2: the
    launching process's pid, recorded for visibility — a pin whose owner
    died without a graceful shutdown lingers (nothing deletes it), so
    `profile show` surfaces it with its origin."""
    path = _server_pin_path(port)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{profile_name}\n{os.getpid()}\n", encoding="utf-8")
    except OSError:
        pass  # best effort — an unpinned fallback is never fatal


def clear_server_pin(port: int) -> None:
    """Remove the pin (bare `roampal start`, `roampal stop`, explicit
    shutdown of the pinned foreground server)."""
    try:
        _server_pin_path(port).unlink()
    except OSError:
        pass


def _config_dir() -> Path:
    """Location of the profiles.json registry. Always a stable per-user path."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        return Path(appdata) / "Roampal"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Roampal"
    xdg_config = os.environ.get(
        "XDG_CONFIG_HOME", str(Path.home() / ".config")
    )
    return Path(xdg_config) / "roampal"


def _registry_path() -> Path:
    return _config_dir() / _REGISTRY_FILENAME


def _active_profile_file() -> Path:
    return _config_dir() / _ACTIVE_FILENAME


def read_active_profile_file() -> Optional[str]:
    """Read the persisted active-profile name, if set.

    Returns None if the file doesn't exist or is empty/unreadable.
    """
    path = _active_profile_file()
    if not path.exists():
        return None
    try:
        name = path.read_text(encoding="utf-8").strip()
        return name or None
    except OSError:
        return None


def write_active_profile_file(name: str) -> None:
    """Persist the active profile name to disk."""
    path = _active_profile_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")


def clear_active_profile_file() -> bool:
    """Remove the persisted active-profile marker.

    Returns True if a file was removed, False if none existed.
    """
    path = _active_profile_file()
    if path.exists():
        path.unlink()
        return True
    return False


# --- Registry --------------------------------------------------------------


class ProfileRegistry:
    """Load/save/query the profiles.json registry."""

    # v0.6.0 Item 3: reserved top-level key holding the directory→profile
    # binding map (design "TBD during implementation" — resolved here).
    # The map is kept OUT of the profile namespace: the loader pulls it out
    # before iterating profile entries, so it never registers as a profile
    # named 'bindings'. Registries without the key behave exactly as before.
    BINDINGS_KEY = "bindings"

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or _registry_path()
        self._profiles: Dict[str, Profile] = {}
        self._bindings: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._profiles = {}
            self._bindings = {}
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to read profile registry at %s: %s. Starting empty.",
                self._path,
                e,
            )
            self._profiles = {}
            self._bindings = {}
            return

        if not isinstance(raw, dict):
            logger.warning(
                "Profile registry at %s is not a JSON object. Ignoring.", self._path
            )
            self._profiles = {}
            self._bindings = {}
            return

        # Bindings (reserved key; absent in pre-Item 3 registries).
        raw_bindings = raw.get(self.BINDINGS_KEY, {})
        # F6: legacy registries could hold a REAL profile named 'bindings'
        # (create accepted the name before the key was reserved) — such an
        # entry silently vanished on reload because the same top-level key
        # is now the binding map. Profile-entry shape here is either a
        # plain path string (register) or null (create) — NOT dir→name
        # pairs whose values are short non-path slugs. Warn loudly when
        # the shape says "legacy profile", not bindings.
        if raw_bindings is None or (
            isinstance(raw_bindings, dict)
            and any(v is None or (isinstance(v, str) and os.path.isabs(v)) for v in raw_bindings.values())
        ):
            logger.warning(
                "Registry at %s carries a legacy 'bindings' entry that looks "
                "like a PROFILE named 'bindings'. That name is now reserved "
                "(it is the registry's binding-map key) and the entry was "
                "dropped. Remove it from %s manually if you did not intend it.",
                self._path,
                self._path,
            )
        bindings: Dict[str, str] = {}
        if isinstance(raw_bindings, dict):
            for d, pname in raw_bindings.items():
                if not isinstance(d, str) or not isinstance(pname, str):
                    logger.warning(
                        "Skipping malformed binding entry %r -> %r", d, pname
                    )
                    continue
                try:
                    bindings[_norm_binding_dir(d)] = pname
                except (OSError, ValueError) as e:
                    logger.warning("Skipping binding %r: %s", d, e)
        else:
            logger.warning("Registry %r key is not an object. Ignoring.", self.BINDINGS_KEY)

        profiles: Dict[str, Profile] = {}
        for name, path in raw.items():
            if name == self.BINDINGS_KEY:
                continue
            try:
                slug = profile_slug(name)
            except InvalidProfileNameError:
                logger.warning("Skipping invalid profile name in registry: %r", name)
                continue
            if path is not None and not isinstance(path, str):
                logger.warning(
                    "Profile %r has non-string path %r. Treating as unset.",
                    name,
                    path,
                )
                path = None
            profiles[name] = Profile(name=name, slug=slug, path=path)
        self._profiles = profiles
        self._bindings = bindings

    def _save(self) -> None:
        from roampal.utils.atomic_json import write_json_atomic

        payload = {name: profile.path for name, profile in self._profiles.items()}
        if self._bindings:
            payload[self.BINDINGS_KEY] = self._bindings
        write_json_atomic(self._path, payload)

    # --- Queries ---

    def list(self) -> List[Profile]:
        """All registered profiles, sorted by name."""
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def get(self, name: str) -> Optional[Profile]:
        return self._profiles.get(name)

    def exists(self, name: str) -> bool:
        return name in self._profiles

    # --- Mutations ---

    def create(self, name: str) -> Profile:
        """Register a new profile with auto-located path (path=None)."""
        if name == DEFAULT_PROFILE:
            raise ProfileAlreadyExistsError(
                f"Profile name {DEFAULT_PROFILE!r} is reserved"
            )
        if name == self.BINDINGS_KEY:
            # F6: `profile create bindings` was accepted, but the reserved
            # top-level key swallowed it on reload — the profile silently
            # vanished. Reject it and preserve the registry untouched.
            raise ProfileAlreadyExistsError(
                f"Profile name {self.BINDINGS_KEY!r} is reserved "
                f"(it is the registry's binding map key, not a profile)"
            )
        if name in self._profiles:
            raise ProfileAlreadyExistsError(f"Profile {name!r} already exists")
        slug = profile_slug(name)
        profile = Profile(name=name, slug=slug, path=None)
        self._profiles[name] = profile
        self._save()
        return profile

    def register(self, name: str, path: str) -> Profile:
        """Register an existing directory as a named profile."""
        if name == DEFAULT_PROFILE:
            raise ProfileAlreadyExistsError(
                f"Profile name {DEFAULT_PROFILE!r} is reserved"
            )
        if name == self.BINDINGS_KEY:
            raise ProfileAlreadyExistsError(
                f"Profile name {self.BINDINGS_KEY!r} is reserved "
                f"(it is the registry's binding map key, not a profile)"
            )
        if name in self._profiles:
            raise ProfileAlreadyExistsError(f"Profile {name!r} already exists")
        if not isinstance(path, str) or not path.strip():
            raise ProfileError(f"Path must be a non-empty string, got {path!r}")
        slug = profile_slug(name)
        profile = Profile(name=name, slug=slug, path=path)
        self._profiles[name] = profile
        self._save()
        return profile

    def delete(self, name: str, *, destroy_data: bool = False) -> Optional[str]:
        """Remove a profile from the registry.

        If destroy_data is True, recursively delete the resolved data directory.
        Returns the resolved path that was (or would have been) destroyed, or
        None if the profile did not exist.

        v0.6.0 review fix 8: bindings pointing at the deleted profile are
        CASCADED (removed). Left in place, every request from a bound
        directory resolves to an unregistered name and the server answers
        404 forever — the "quietly breaks memory" failure. Callers should
        tell the user which directories were unbound; they now fall back
        to use -> default.
        """
        profile = self._profiles.pop(name, None)
        if profile is None:
            raise ProfileNotFoundError(f"Profile {name!r} is not registered")
        resolved = self.resolve(profile=profile)
        unbound = [d for d, pname in self._bindings.items() if pname == name]
        for d in unbound:
            del self._bindings[d]
        self._save()
        if destroy_data and resolved and Path(resolved).exists():
            shutil.rmtree(resolved)
        return resolved

    # --- Bindings (v0.6.0 Item 3 / Plan B) ---

    def bindings(self) -> Dict[str, str]:
        """Copy of the directory→profile binding map (normalized keys)."""
        return dict(self._bindings)

    def bind(self, name: str, directory: str) -> str:
        """Bind `directory` (or its ancestors) to profile `name`.

        The profile must already be registered — unregistered names raise
        ProfileNotFoundError and the registry is left untouched. F6 fix:
        DEFAULT_PROFILE ("default") is the one exception — it is always
        resolvable, and binding a directory to literal default is the way
        to pin a project back to default while a persisted `use` (or a
        parent-directory binding) points elsewhere.

        The binding map is directory-keyed so the ancestry walk's
        innermost-wins ordering falls out naturally.

        Returns the normalized key stored.
        """
        if name != DEFAULT_PROFILE and not self.exists(name):
            raise ProfileNotFoundError(f"Profile {name!r} is not registered")
        if not isinstance(directory, str) or not directory.strip():
            raise ProfileError(f"Directory must be a non-empty string, got {directory!r}")
        key = _norm_binding_dir(directory)
        self._bindings[key] = name
        self._save()
        return key

    def unbind(self, directory: str) -> bool:
        """Remove a binding. Idempotent: False (no change) when absent."""
        if not isinstance(directory, str) or not directory.strip():
            raise ProfileError(f"Directory must be a non-empty string, got {directory!r}")
        key = _norm_binding_dir(directory)
        if key not in self._bindings:
            return False
        del self._bindings[key]
        self._save()
        return True

    # --- Resolution ---

    def resolve(
        self, name: Optional[str] = None, *, profile: Optional[Profile] = None
    ) -> str:
        """Return the absolute data path for a given profile name.

        If an explicit Profile is passed, use it directly (avoids a second lookup).
        Raises ProfileNotFoundError for unknown names.
        """
        if profile is None:
            if not name or name == DEFAULT_PROFILE:
                return str(system_default_data_path())
            profile = self._profiles.get(name)
            if profile is None:
                raise ProfileNotFoundError(f"Profile {name!r} is not registered")
        if profile.path:
            return profile.path
        # Auto-located under default base
        return str(system_default_data_path() / profile.slug)


# --- Module-level convenience ----------------------------------------------


def resolve_data_path(
    profile_name: Optional[str] = None, *, registry: Optional[ProfileRegistry] = None
) -> str:
    """Resolve a profile name to an absolute data path.

    Honors env vars for backward compatibility (ROAMPAL_DATA_PATH takes precedence).
    """
    env_override = os.environ.get("ROAMPAL_DATA_PATH")
    if env_override:
        return env_override

    reg = registry or ProfileRegistry()
    return reg.resolve(profile_name)


def _norm_binding_dir(path: str) -> str:
    """Normalize a binding directory for storage and comparison.

    Windows: `os.path.normcase` lowercases + backslash-ifies, so
    C:/roampal-core and C:\\ROAMPAL-CORE\\ are the same key (Test Plan B
    drive-style casing). POSIX: normcase is identity — paths stay
    case-sensitive, matching filesystem semantics.
    """
    resolved = str(Path(path).resolve())
    return os.path.normcase(os.path.normpath(resolved))


def _quiet_bindings_map(registry_path: Optional[Path] = None) -> Dict[str, str]:
    """Read ONLY the bindings map, without the profile loader's warnings.

    The binding step runs on every active_profile_name()/source() call,
    including for legacy malformed registries (golden outputs capture the
    loader warnings — see the profile_show goldens). A second, louder
    load would double those warnings and break byte-goldens, so this
    reader stays silent and extracts just the reserved key.
    """
    path = registry_path or _registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    raw_bindings = raw.get(ProfileRegistry.BINDINGS_KEY, {})
    if not isinstance(raw_bindings, dict):
        return {}
    out: Dict[str, str] = {}
    for d, pname in raw_bindings.items():
        if not isinstance(d, str) or not isinstance(pname, str):
            logger.debug("Skipping malformed binding entry %r -> %r", d, pname)
            continue
        try:
            out[_norm_binding_dir(d)] = pname
        except (OSError, ValueError) as e:
            logger.debug("Skipping binding %r: %s", d, e)
    return out


def binding_for_cwd(
    cwd: Optional[Path] = None,
    *,
    registry: Optional[ProfileRegistry] = None,
    registry_path: Optional[Path] = None,
) -> Optional[tuple]:
    """Resolve the cwd-ancestry binding step (Plan B, resolution level 3).

    Walks UP from `cwd` (or Path.cwd()); the first ancestor (innermost
    wins naturally, since the walk starts at cwd) whose normalized path
    IS a stored binding key wins.

    Returns (bound_dir, profile_name) or None. Injected `registry_path`
    / `cwd` keep tests free of chdir/global state (Plan B fixture note);
    injected `registry` reuses an already-loaded instance's map.
    """
    if registry is not None:
        bindings = registry._bindings
    else:
        bindings = _quiet_bindings_map(registry_path)
    if not bindings:
        return None
    checked = cwd if cwd is not None else Path.cwd()
    current = Path(os.path.normcase(os.path.normpath(str(checked.resolve()))))
    while True:
        key = str(current)
        if key in bindings:
            return key, bindings[key]
        if current.parent == current:
            return None
        current = current.parent


def active_profile_name(
    cwd: Optional[Path] = None, *, registry: Optional[ProfileRegistry] = None
) -> str:
    """Return the profile name that should be used when none is specified.

    Precedence (highest wins):
      1. ROAMPAL_PROFILE env var (per-shell / per-project-config)
      2. cwd-ancestry binding (v0.6.0 Item 3) — inserted between env and
         the persisted default per the Plan B precedence table
      3. Persisted active_profile file (user-global default)
      4. DEFAULT_PROFILE

    The per-command --profile flag is handled by callers before this is checked.
    A bound profile that is not registered still wins at this layer —
    resolution failures surface explicitly (ProfileNotFoundError from
    resolve_data_path) instead of silently falling through. Injected
    `cwd`/`registry` keep tests free of chdir/global state (Plan B).
    """
    env = os.environ.get("ROAMPAL_PROFILE", "").strip()
    if env:
        return env
    try:
        bound = binding_for_cwd(cwd=cwd, registry=registry)
        if bound is not None:
            return bound[1]
    except (OSError, ValueError) as e:
        logger.debug("Binding resolution skipped: %s", e)
    persisted = read_active_profile_file()
    if persisted:
        return persisted
    return DEFAULT_PROFILE


def active_profile_source(
    cwd: Optional[Path] = None, *, registry: Optional[ProfileRegistry] = None
) -> str:
    """Return a human-readable tag of where the active profile came from.

    One of: 'env', 'binding', 'file', 'default'.
    """
    if os.environ.get("ROAMPAL_PROFILE", "").strip():
        return "env"
    try:
        if binding_for_cwd(cwd=cwd, registry=registry) is not None:
            return "binding"
    except (OSError, ValueError):
        pass
    if read_active_profile_file():
        return "file"
    return "default"


def persisted_profile_fallback() -> str:
    """Round 2 Item 6 / Task 17: the SERVER-side headerless resolution.

    The shared server never guesses its request profile from its own cwd
    or a spawner's env (audit finding F1): a request with no identity
    header resolves the persisted `profile use` name, falling back to
    DEFAULT_PROFILE.

    Deliberately binding-free and env-free — active_profile_name()'s env
    and cwd-ancestry steps answer CLIENT-side questions (whose shell /
    which project). The server's own cwd depends on which client spawned
    (or last respawned) it, so neither of those steps may run there.
    Directory-bound requests arrive as explicit headers (Task 19).
    """
    return read_active_profile_file() or DEFAULT_PROFILE


def profile_header_value(cwd: Optional[Path] = None) -> str:
    """THE one helper every client seam resolves the profile header through.

    v0.6.0 Task 14 wiring: stdio MCP (`_get_mcp_profile_name`), the
    OpenCode plugin (worktree/header resolution, comment-tied), and the
    Claude Code / Cursor hook CLI paths (roampal context) all read
    the active-profile name through active_profile_name()'s full
    precedence walk — env var still beats cwd binding.

    Round 2 Item 6 / Task 18: clients ALWAYS name their profile — the
    helper returns "default" (a full header value) instead of None, so
    every seam sends X-Roampal-Profile with an explicit default and the
    shared server never resolves a profile for the client from its own
    cwd/env (F1)."""
    return active_profile_name(cwd)
