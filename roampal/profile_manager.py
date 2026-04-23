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

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or _registry_path()
        self._profiles: Dict[str, Profile] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._profiles = {}
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
            return

        if not isinstance(raw, dict):
            logger.warning(
                "Profile registry at %s is not a JSON object. Ignoring.", self._path
            )
            self._profiles = {}
            return

        profiles: Dict[str, Profile] = {}
        for name, path in raw.items():
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

    def _save(self) -> None:
        from roampal.utils.atomic_json import write_json_atomic

        payload = {name: profile.path for name, profile in self._profiles.items()}
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
        """
        profile = self._profiles.pop(name, None)
        if profile is None:
            raise ProfileNotFoundError(f"Profile {name!r} is not registered")
        resolved = self.resolve(profile=profile)
        self._save()
        if destroy_data and resolved and Path(resolved).exists():
            shutil.rmtree(resolved)
        return resolved

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


def active_profile_name() -> str:
    """Return the profile name that should be used when none is specified.

    Precedence (highest wins):
      1. ROAMPAL_PROFILE env var (per-shell / per-project-config)
      2. Persisted active_profile file (user-global default)
      3. DEFAULT_PROFILE

    The per-command --profile flag is handled by callers before this is checked.
    """
    env = os.environ.get("ROAMPAL_PROFILE", "").strip()
    if env:
        return env
    persisted = read_active_profile_file()
    if persisted:
        return persisted
    return DEFAULT_PROFILE


def active_profile_source() -> str:
    """Return a human-readable tag of where the active profile came from.

    One of: 'env', 'file', 'default'.
    """
    if os.environ.get("ROAMPAL_PROFILE", "").strip():
        return "env"
    if read_active_profile_file():
        return "file"
    return "default"
