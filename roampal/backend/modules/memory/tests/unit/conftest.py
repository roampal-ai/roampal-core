"""
Pytest configuration for unit tests.

Path setup is managed by root conftest.py - no duplicate needed here.

Task 26 / Item 8: unit tests must never read (or chance upon) real user
data. The autouse fixture below points HOME / USERPROFILE / APPDATA /
LOCALAPPDATA and every XDG_* var at pytest-managed temp dirs for the
duration of each test. Tests that need specific values still setenv AFTER
this fixture runs (later monkeypatch.setenv wins); tests that patch
Path.home directly are unaffected.
"""

import os

import pytest

# Original (real) user-env values, captured at conftest import time —
# BEFORE the autouse fixture mutates anything — so tests whose contract
# legitimately needs the real environment can opt back in per-test.
_REAL_HF_HUB_CACHE = os.path.expanduser(
    os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
)

# Original (real) user-env values, captured at conftest import time —
# BEFORE the autouse fixture mutates anything — so tests whose contract
# legitimately needs the real environment can opt back in per-test.
_ORIGINAL_USER_ENV = {
    key: value
    for key, value in os.environ.items()
    if key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
    or key.startswith("XDG_")
    or key.startswith("HF_")
    or key.startswith("HUGGINGFACE")
}

# Boolean flags are not paths: the tmp rewrite below would turn e.g.
# HF_HUB_OFFLINE=1 into HF_HUB_OFFLINE=<tmp>/xdg/hf-hub-offline — a
# garbage but still truthy value, which silently made every forked test
# child run in HF offline mode against an empty cache (first CI run,
# 2026-09-26). Flags pass through untouched; only PATH-valued cache vars
# get the rewrite.
_HF_FLAG_KEYS = {
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_DATASETS_OFFLINE",
}


@pytest.fixture
def unsandboxed_home(monkeypatch):
    """OPT-IN override for the Task 26 sandbox.

    Request this fixture ONLY when the contract under test is about the
    real user environment itself (e.g. the golden HF-cache skip-check).
    Everything else must run inside the autouse sandbox."""
    for key, value in _ORIGINAL_USER_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _sandbox_user_env(monkeypatch, tmp_path):
    home = tmp_path / "userhome"
    home.mkdir(parents=True, exist_ok=True)
    appdata = tmp_path / "appdata"
    appdata.mkdir(parents=True, exist_ok=True)

    # Fixed-name vars: point at temp when set (keep absence where the
    # platform convention omits them, e.g. HOME on Windows Python).
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        if key in os.environ:
            monkeypatch.setenv(key, str(appdata if key.endswith(("APPDATA", "LOCALAPPDATA")) else home))
    # XDG_* and the HF/HUGGINGFACE cache family: replace any real value
    # with a tmp equivalent (HF_ vars callers may preset on CI runners —
    # without this a real model cache could be WRITTEN to or read from).
    # Absent vars stay absent; roampal's XDG/HF fallbacks resolve through
    # the USERPROFILE/HOME we just sandboxed via their rewritten dirs.
    for key in list(os.environ):
        if key.startswith("XDG_") or key.startswith("HF_") or key.startswith("HUGGINGFACE"):
            if key in _HF_FLAG_KEYS:
                continue
            monkeypatch.setenv(key, str(tmp_path / "xdg" / key.lower().replace("_", "-")))
    # Task 28 CI opt-in: when the workflow sets ROAMPAL_TEST_PRIME_HF_CACHE=1
    # (and the primed cache exists), forked test children resolve the primed
    # model cache instead of this empty sandbox one. Without it, every
    # forked child misses the cache; under huggingface_hub 2.x even the
    # offline miss constructs the httpx client, which on macOS aborts the
    # child (urllib's getproxies_macosx_sysconf is not fork-safe). Read-only
    # in practice: HF_HUB_OFFLINE passes through the flag list above, so no
    # child can write to the real cache.
    if os.environ.get("ROAMPAL_TEST_PRIME_HF_CACHE") == "1" and os.path.isdir(_REAL_HF_HUB_CACHE):
        monkeypatch.setenv("HF_HUB_CACHE", _REAL_HF_HUB_CACHE)
        monkeypatch.setenv("HF_HOME", os.path.dirname(_REAL_HF_HUB_CACHE))
