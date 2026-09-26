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
_ORIGINAL_USER_ENV = {
    key: value
    for key, value in os.environ.items()
    if key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
    or key.startswith("XDG_")
    or key.startswith("HF_")
    or key.startswith("HUGGINGFACE")
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
            monkeypatch.setenv(key, str(tmp_path / "xdg" / key.lower().replace("_", "-")))
