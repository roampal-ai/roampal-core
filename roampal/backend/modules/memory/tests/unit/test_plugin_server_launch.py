"""Task 41 — the OpenCode plugin restarts the server with Roampal's own Python.

`restartServer` used to guess the interpreter from PATH (`where pythonw` →
`pythonw`, else a VBS running bare `python`; `python3` on Unix). The first
Python on PATH is often a different install (dev box 2026-09-25: the repo
.venv first → a plugin restart would start a 0.5.9 server under 0.6.0
clients) and has no Roampal at all for pipx/venv installs.

Now `roampal init`'s recorded MCP command (`mcp["roampal-core"].command` =
[<python>, <flags...>, "-m", "roampal.mcp.server"]) drives the respawn: same
interpreter, same flags (module swapped to roampal.server.main), PATH only as
a debugLogged fallback. The decision lives in one pure function,
`resolveServerLaunch(mcpCommand, platform, fileExists, port)` — sliced out of
the plugin source and executed under Node (skip when node < 22.13), same
pattern as test_sidecar_privacy.py.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PLUGIN_PATH = Path(__file__).resolve().parents[5] / "plugins" / "opencode" / "roampal.ts"


@pytest.fixture
def plugin_source():
    return PLUGIN_PATH.read_text(encoding="utf-8")


# ============================================================================
# Structural checks
# ============================================================================


class TestResolverStructure:
    def test_restart_server_calls_the_resolver(self, plugin_source):
        body = plugin_source.split("async function restartServer(", 1)[1]
        assert "resolveServerLaunch(" in body
        assert "_loadMcpCommand()" in body

    def test_no_bare_path_python_spawns(self, plugin_source):
        """Every spawn goes through the resolver's exe. The only PATH-python
        references left (`where pythonw`, the VBS `python` cmdline) sit inside
        the fallback branch."""
        assert 'spawn("python3"' not in plugin_source
        assert 'spawn("pythonw"' not in plugin_source
        fallback_body = plugin_source.split("if (launch.usedFallback) {", 1)[1]
        assert "where pythonw" in fallback_body

    def test_config_source_is_the_recorded_mcp_command(self, plugin_source):
        assert 'config?.mcp?.["roampal-core"]?.command' in plugin_source
        assert "opencode.json" in plugin_source

    def test_fallback_is_debug_logged(self, plugin_source):
        body = plugin_source.split("async function restartServer(", 1)[1]
        assert "FALLBACK to PATH" in body


# ============================================================================
# Resolver behavior: run the REAL decision code under Node
# ============================================================================

_NODE = shutil.which("node")

_RUNNER = r"""
import { stripTypeScriptTypes } from "node:module"
import { readFileSync, writeFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
const [pluginPath, casePath, outPath] = process.argv.slice(2)
const src = readFileSync(pluginPath, "utf8")
const start = src.indexOf("function resolveServerLaunch(")
const endMarker = "async function restartServer("
const end = src.indexOf(endMarker)
if (start < 0 || end < 0) { console.error("markers not found"); process.exit(3) }
const slice = src.slice(start, end)
const cases = JSON.parse(readFileSync(casePath, "utf8"))
const body = `${slice}
const cases = ${JSON.stringify(cases)}
const results = []
for (const c of cases) {
  const exists = new Set(c.exists)
  const r = resolveServerLaunch(c.command, c.platform, (p) => exists.has(p), c.port)
  results.push(r)
}
console.log(JSON.stringify(results))
`
writeFileSync(outPath, stripTypeScriptTypes(body))
await import(pathToFileURL(outPath).href)
"""


def _node_supports_strip():
    if not _NODE:
        return False
    probe = subprocess.run(
        [_NODE, "--input-type=module", "-e",
         "import { stripTypeScriptTypes } from 'node:module'; if (!stripTypeScriptTypes) process.exit(1)"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0


def _resolve(tmp_path, cases):
    runner = tmp_path / "runner.mjs"
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps(cases), encoding="utf-8")
    runner.write_text(_RUNNER, encoding="utf-8")
    proc = subprocess.run(
        [_NODE, "--no-warnings", str(runner), str(PLUGIN_PATH), str(case_file),
         str(tmp_path / "resolved.mjs")],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not _node_supports_strip(), reason="needs node >= 22.13 (stripTypeScriptTypes)")
class TestResolveServerLaunch:
    def test_windows_configured_python_with_sibling_pythonw(self, tmp_path):
        """Case 1: configured python.exe + sibling pythonw.exe → spawn that
        pythonw directly, flags carried from the config."""
        py = r"C:\Users\x\AppData\Local\Programs\Python\Python310\python.exe"
        (result,) = _resolve(tmp_path, [{
            "command": [py, "-E", "-P", "-m", "roampal.mcp.server"],
            "platform": "win32",
            "exists": [py, r"C:\Users\x\AppData\Local\Programs\Python\Python310\pythonw.exe"],
            "port": 27182,
        }])
        assert result["usedFallback"] is False
        assert result["useVbs"] is False
        assert result["exe"] == py.replace("python.exe", "pythonw.exe")
        assert result["args"] == ["-E", "-P", "-m", "roampal.server.main", "--port", "27182"]

    def test_windows_no_sibling_pythonw_uses_vbs(self, tmp_path):
        """Case 2: no sibling pythonw.exe → the configured python.exe through
        the hidden VBS launcher."""
        py = r"C:\Tools\roampal venv\python.exe"
        (result,) = _resolve(tmp_path, [{
            "command": [py, "-E", "-m", "roampal.mcp.server"],
            "platform": "win32",
            "exists": [py],
            "port": 27183,
        }])
        assert result["usedFallback"] is False
        assert result["useVbs"] is True
        assert result["exe"] == py
        assert result["args"] == ["-E", "-m", "roampal.server.main", "--port", "27183"]

    def test_unix_configured_interpreter_spawned_directly(self, tmp_path):
        """Case 3: pipx/venv absolute path on Linux/macOS → exactly that exe,
        no VBS, no console-window gymnastics."""
        py = "/Users/x/.local/pipx/venvs/roampal/bin/python"
        (result,) = _resolve(tmp_path, [{
            "command": [py, "-E", "-P", "-m", "roampal.mcp.server"],
            "platform": "linux",
            "exists": [py],
            "port": 27182,
        }])
        assert result["usedFallback"] is False
        assert result["useVbs"] is False
        assert result["exe"] == py

    def test_flags_carried_and_default_flag(self, tmp_path):
        """Case 4: config flags are carried verbatim; a config without flags
        (pre-0.6.0 shape) gets -E."""
        py = "/usr/bin/python3"
        (with_flags,) = _resolve(tmp_path, [{
            "command": [py, "-E", "-P", "-m", "roampal.mcp.server"],
            "platform": "linux", "exists": [py], "port": 27182,
        }])
        (no_flags,) = _resolve(tmp_path, [{
            "command": [py, "-m", "roampal.mcp.server"],
            "platform": "linux", "exists": [py], "port": 27182,
        }])
        assert with_flags["args"][:2] == ["-E", "-P"]
        assert no_flags["args"][0] == "-E"

    def test_missing_config_or_interpreter_falls_back_to_path(self, tmp_path):
        """Case 5: no config / no command / interpreter file missing →
        usedFallback with today's PATH exe and -E flags."""
        (no_config,) = _resolve(tmp_path, [{
            "command": None, "platform": "win32", "exists": [], "port": 27182,
        }])
        (no_file,) = _resolve(tmp_path, [{
            "command": [r"C:\gone\python.exe", "-E", "-m", "roampal.mcp.server"],
            "platform": "win32", "exists": [], "port": 27182,
        }])
        (unix_missing,) = _resolve(tmp_path, [{
            "command": ["/gone/python"], "platform": "linux", "exists": [], "port": 27182,
        }])
        assert no_config == {
            "exe": "pythonw", "args": ["-E", "-m", "roampal.server.main", "--port", "27182"],
            "usedFallback": True, "useVbs": False,
        }
        assert no_file["usedFallback"] is True and no_file["exe"] == "pythonw"
        assert unix_missing["usedFallback"] is True and unix_missing["exe"] == "python3"

    def test_non_string_or_empty_command_falls_back(self, tmp_path):
        (garbage,) = _resolve(tmp_path, [{
            "command": [42], "platform": "win32", "exists": [], "port": 27182,
        }])
        (empty,) = _resolve(tmp_path, [{
            "command": [], "platform": "win32", "exists": [], "port": 27182,
        }])
        assert garbage["usedFallback"] is True
        assert empty["usedFallback"] is True


# ============================================================================
# The whole plugin still parses (type-strip + node --check), with a broken
# control file proving the check is not vacuous.
# ============================================================================


def _node_check(tmp_path, source_text) -> subprocess.CompletedProcess:
    """Type-strip then `node --check` the result. Returns the FIRST failing
    step (strip throws on syntax errors) so a broken source cannot pass."""
    stripped = tmp_path / "plugin_stripped.mjs"
    runner = tmp_path / "strip.mjs"
    src_file = tmp_path / "src.ts"
    src_file.write_text(source_text, encoding="utf-8")
    runner.write_text(
        'import { stripTypeScriptTypes } from "node:module"\n'
        'import { readFileSync, writeFileSync } from "node:fs"\n'
        'const [srcPath, outPath] = process.argv.slice(2)\n'
        'writeFileSync(outPath, stripTypeScriptTypes(readFileSync(srcPath, "utf8")))\n',
        encoding="utf-8",
    )
    stripped_src = subprocess.run(
        [_NODE, "--no-warnings", str(runner), str(src_file), str(stripped)],
        capture_output=True, text=True, timeout=60,
    )
    if stripped_src.returncode != 0:
        return stripped_src
    # --check parses the file as a module (.mjs) without executing it —
    # process.env / fetch references are fine, imports are never resolved.
    return subprocess.run(
        [_NODE, "--check", str(stripped)], capture_output=True, text=True, timeout=60,
    )


@pytest.mark.skipif(not _node_supports_strip(), reason="needs node >= 22.13 (stripTypeScriptTypes)")
class TestWholePluginParses:
    def test_full_plugin_type_strips_and_parses(self, tmp_path, plugin_source):
        proc = _node_check(tmp_path, plugin_source)
        assert proc.returncode == 0, proc.stderr

    def test_broken_control_file_fails_the_parse_check(self, tmp_path, plugin_source):
        broken = plugin_source.replace(
            "function resolveServerLaunch(",
            "function resolveServerLaunch(((",  # deliberate syntax error
            1,
        )
        proc = _node_check(tmp_path, broken)
        assert proc.returncode != 0
