"""Task 26 / Item 8 red-check probe: unit tests must run with HOME,
USERPROFILE, APPDATA and all XDG_* env vars pointed INSIDE the pytest tmp
tree. Written first WITHOUT the conftest fixture so its failure documents
the real-data exposure; the autouse conftest fixture turns it green."""

import os
import pathlib


def test_unit_tests_never_see_the_real_home(tmp_path):
    # A var is sandboxed when it is either absent (all lookups fall through
    # to the pytest tmp supplies) or points inside the pytest tmp tree.
    def sandboxed(var):
        v = os.environ.get(var)
        return v is None or str(tmp_path) in v

    assert sandboxed("HOME"), f"HOME leaks real data: {os.environ.get('HOME')}"
    assert sandboxed("USERPROFILE"), "USERPROFILE not sandboxed"
    assert sandboxed("APPDATA"), "APPDATA not sandboxed"
    unsandboxed_xdg = [
        k for k, w in os.environ.items()
        if k.startswith("XDG_") and str(tmp_path) not in w
    ]
    assert not unsandboxed_xdg, f"unsandboxed XDG_ vars remain: {unsandboxed_xdg}"
