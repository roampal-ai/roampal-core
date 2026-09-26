"""v0.6.0 smoke-test fix: every CLI request that touches profile data names
its profile (X-Roampal-Profile).

Found by the Task 34 live smoke (2026-09-26): `roampal stats` run inside a
folder bound to `smoke060` reported `data\\main` — the shared server resolves
a headerless request to persisted `use` -> default (Task 17), and only
`context` and `score` sent the header. stats / books / ingest / remove /
summarize / retag read or wrote the wrong profile in a bound folder, and
`ingest`'s offline fallback ignored the profile entirely.
"""

import argparse
import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import roampal.profile_manager as pm

CLI_DIR = Path(__file__).resolve().parents[5] / "cli"

# Calls that legitimately carry no profile: liveness probes (/api/health is
# profile-independent) and the init signup webhook (not the Roampal server).
_ALLOWED_HEADERLESS = {
    ("server.py", "cmd_status"),  # /api/health probe
    ("update_check.py", "collect_email"),  # external webhook
}


def _headerless_calls():
    missing = []
    for path in sorted(CLI_DIR.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "httpx"
                    and node.func.attr in ("get", "post", "put", "delete")
                ):
                    continue
                if any(k.arg == "headers" for k in node.keywords):
                    continue
                segment = ast.get_source_segment(src, node) or ""
                if "/api/health" in segment:
                    continue
                if (path.name, fn.name) in _ALLOWED_HEADERLESS:
                    continue
                missing.append(f"{path.relative_to(CLI_DIR)}:{node.lineno} ({fn.name})")
    return missing


def test_every_cli_server_call_names_its_profile():
    missing = _headerless_calls()
    # Dedupe: nested functions are walked from each enclosing def.
    assert not sorted(set(missing)), (
        "CLI server calls without X-Roampal-Profile (use profile_headers()): "
        f"{sorted(set(missing))}"
    )


def test_guard_is_not_vacuous():
    """The scan must actually see the CLI's httpx calls."""
    count = 0
    for path in CLI_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count += sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "httpx"
        )
    assert count >= 10


@pytest.fixture
def bound_project(tmp_path, monkeypatch):
    """A registered profile `smoke` bound to a project folder, cwd inside it."""
    reg_path = tmp_path / "cfg" / "roampal" / "profiles.json"
    reg_path.parent.mkdir(parents=True)
    data = tmp_path / "data" / "smoke"
    data.mkdir(parents=True)
    reg_path.write_text(json.dumps({"smoke": str(data)}), encoding="utf-8")
    monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    pm.ProfileRegistry().bind("smoke", str(project))
    monkeypatch.chdir(project / "sub")
    monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
    return project


def _ok(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_stats_in_bound_folder_sends_bound_profile(bound_project):
    from roampal.cli.server import cmd_stats

    with patch("httpx.get", return_value=_ok({"data_path": "x", "collections": {}})) as g, \
            patch("roampal.cli.server.print_update_notice"):
        cmd_stats(argparse.Namespace(host=None, port=None, dev=False, json_output=True))
    assert g.call_args.kwargs["headers"] == {"X-Roampal-Profile": "smoke"}


def test_books_and_remove_send_bound_profile(bound_project):
    from roampal.cli.memory_cmds import cmd_books, cmd_remove

    with patch("httpx.get", return_value=_ok({"books": []})) as g:
        cmd_books(argparse.Namespace(dev=False))
    assert g.call_args.kwargs["headers"] == {"X-Roampal-Profile": "smoke"}

    with patch("httpx.post", return_value=_ok({"removed": 1})) as p:
        cmd_remove(argparse.Namespace(title="t", dev=False, port=None))
    assert p.call_args.kwargs["headers"] == {"X-Roampal-Profile": "smoke"}


def test_ingest_server_path_and_offline_fallback_use_bound_profile(bound_project):
    import httpx

    from roampal.cli.memory_cmds import cmd_ingest

    doc = bound_project / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")
    args = argparse.Namespace(
        file=str(doc), title=None, chunk_size=1000, chunk_overlap=200, dev=False
    )

    with patch("httpx.post", return_value=_ok({"chunks": 1})) as p:
        cmd_ingest(args)
    assert p.call_args.kwargs["headers"] == {"X-Roampal-Profile": "smoke"}

    # Server down -> direct storage must open the bound profile, not the
    # system default data dir.
    captured = {}

    class FakeUMS:
        def __init__(self, data_path=None, profile_name=None, **_):
            captured["data_path"] = data_path
            captured["profile_name"] = profile_name

        async def initialize(self):
            pass

        async def store_book(self, **_):
            return ["id1"]

    with patch("httpx.post", side_effect=httpx.ConnectError("down")), \
            patch("roampal.backend.modules.memory.UnifiedMemorySystem", FakeUMS):
        cmd_ingest(args)
    assert captured == {"data_path": None, "profile_name": "smoke"}
