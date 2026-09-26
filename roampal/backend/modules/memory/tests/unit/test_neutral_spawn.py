"""Task 36 / Item 7 / v0.6.0 review fix 1: shared-server spawn sites must
never let the caller's cwd (or PYTHONPATH) influence `import roampal`.

Red-check mechanism: with bare `python -m roampal...`, cwd is sys.path[0],
so a project containing a local `roampal/` package shadows the installed
one. Every site that spawns the server (MCP health/launcher, both CC hooks,
the OpenCode plugin) and every hook/MCP command `roampal init` writes must
therefore run with `-E` (PYTHONPATH ignored) and a neutral cwd, plus `-P`
(3.11+) wherever the command runs in a PROJECT directory.

v0.6.0 review fix 1: `-I` was replaced by `-E [-P]` — `-I` implies `-s`,
which hides USER site-packages and silently broke Microsoft Store Python,
`pip install --user`, and non-writable system site-packages installs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SHADOW_ROAMPAL_INIT = "SENTINEL_SHOULD_HAVE_BEEN_SHADOWED"


class TestIsolatedFlagBlocksCwdShadowing(unittest.TestCase):
    """The interpreter-level contract the spawn sites rely on."""

    @classmethod
    def setUpClass(cls):
        base = Path(tempfile.mkdtemp(prefix="roampal_task36_shadow_"))
        (base / "roampal").mkdir()
        # Shadow package that prints a marker; the real roampal prints its file.
        (base / "roampal" / "__init__.py").write_text(
            "print('SHADOWED'); raise SystemExit(1)\n", encoding="utf-8"
        )
        cls.shadow_dir = base
        cls.addClassCleanup(shutil.rmtree, base, ignore_errors=True)

    def _import_target(self, extra_flags):
        proc = subprocess.run(
            [sys.executable, *extra_flags, "-c", "import roampal"],
            capture_output=True, text=True, cwd=str(self.shadow_dir), timeout=30,
        )
        return proc

    def test_bare_invocation_cwd_shadows_installed_package(self):
        # Control: the ORIGINAL behavior is dangerous — proves the test
        # setup actually exercises shadowing.
        proc = self._import_target([])
        self.assertIn("SHADOWED", proc.stdout, "setup broken: shadow not picked up")

    def test_E_alone_still_shadows_why_P_is_needed(self):
        # -E ignores PYTHONPATH but NOT the cwd — documents why init-written
        # commands (which run in project dirs) need -P on 3.11+.
        proc = self._import_target(["-E"])
        self.assertIn("SHADOWED", proc.stdout)

    def test_spawn_flags_match_interpreter_capabilities(self):
        # -E [-P] = the _spawn_isolation_flags() contract: user/site-packages
        # stay visible (no -s). On 3.11+ the cwd also cannot shadow (-P);
        # on 3.10 -P does not exist, so a project-local roampal/ shadows a
        # -c/-m launch from its own dir — the documented v0.5.9-equivalent
        # residual (runtime spawns are unaffected: neutral cwd).
        from roampal.profile_manager import spawn_isolation_flags

        proc = self._import_target(spawn_isolation_flags())
        if sys.version_info >= (3, 11):
            self.assertNotIn("SHADOWED", proc.stdout)
            self.assertNotIn("SHADOWED", proc.stderr)
        else:
            self.assertIn("SHADOWED", proc.stdout)


class TestSpawnSitesAreNeutral(unittest.TestCase):
    """Every shared-server spawn site runs
    `python [-E -P] -m roampal.server.main` with cwd set to the neutral
    data dir."""

    def _assert_neutral_spawn(self, popen_kwargs, cmd, module_name):
        from roampal.profile_manager import spawn_isolation_flags

        expected_prefix = [sys.executable, *spawn_isolation_flags()]
        self.assertEqual(
            cmd[: len(expected_prefix)], expected_prefix,
            f"{module_name}: python must run with spawn_isolation_flags() "
            f"({expected_prefix}); got {cmd}",
        )
        self.assertIn("-m", cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "roampal.server.main")
        cwd = popen_kwargs.get("cwd")
        self.assertIsNotNone(cwd, f"{module_name}: spawn must pin a neutral cwd")
        self.assertFalse(
            Path(cwd).name == "roampal" or (Path(cwd) / "roampal").is_dir(),
            f"{module_name}: cwd {cwd} is package-shadowing-adjacent",
        )

    def test_mcp_server_spawn_is_neutral(self):
        import roampal.mcp.server as mcp

        captured = {}

        def fake_popen(cmd, *a, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return mock.Mock(pid=1)

        with mock.patch.object(subprocess, "Popen", fake_popen), \
             mock.patch.object(mcp.subprocess, "Popen", fake_popen), \
             mock.patch.object(mcp, "_is_port_in_use", return_value=False), \
             mock.patch.dict(os.environ, {"ROAMPAL_INSPECT_ONLY": ""}):
            # reset any "already started" latch from a previous import —
            # saved/restored so the module global is left as it was (Task 22
            # hygiene: tests must not leave process-wide state mutated)
            old_latch, old_proc = mcp._fastapi_started, mcp._fastapi_process
            mcp._fastapi_started = False
            try:
                mcp._start_fastapi_server()
            finally:
                mcp._fastapi_started, mcp._fastapi_process = old_latch, old_proc

        self._assert_neutral_spawn(captured["kw"], captured["cmd"], "mcp/server.py")

    def _spawn_via_hook(self, hook_module):
        hook = __import__(hook_module, fromlist=["_spawn_fresh_server"])
        captured = {}

        def fake_popen(cmd, *a, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return mock.Mock(pid=1)

        netstat = mock.Mock(returncode=0)
        netstat.stdout = f"  TCP    127.0.0.1:27182   0.0.0.0:0   LISTENING   4242"
        with mock.patch.object(hook.subprocess, "run", return_value=netstat), \
             mock.patch.object(hook.subprocess, "Popen", fake_popen):
            hook._spawn_fresh_server("http://127.0.0.1:27182", 27182)

        self._assert_neutral_spawn(captured["kw"], captured["cmd"], hook_module)
        return captured

    def test_stop_hook_spawn_is_neutral(self):
        self._spawn_via_hook("roampal.hooks.stop_hook")

    def test_user_prompt_submit_hook_spawn_is_neutral(self):
        self._spawn_via_hook("roampal.hooks.user_prompt_submit_hook")


class TestSpawnRepassesLaunchPin(unittest.TestCase):
    """Review fix 5: a respawn of a pinned server re-passes --profile from
    the per-port pin file, preserving its routing identity."""

    def test_mcp_spawn_repasses_pin(self):
        import roampal.mcp.server as mcp
        import roampal.profile_manager as pm

        port = mcp._get_port()
        pm.write_server_pin(port, "work")
        self.addCleanup(pm.clear_server_pin, port)

        captured = {}

        def fake_popen(cmd, *a, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return mock.Mock(pid=1)

        with mock.patch.object(subprocess, "Popen", fake_popen), \
             mock.patch.object(mcp.subprocess, "Popen", fake_popen), \
             mock.patch.object(mcp, "_is_port_in_use", return_value=False), \
             mock.patch.dict(os.environ, {"ROAMPAL_INSPECT_ONLY": ""}):
            old_latch, old_proc = mcp._fastapi_started, mcp._fastapi_process
            mcp._fastapi_started = False
            try:
                mcp._start_fastapi_server()
            finally:
                mcp._fastapi_started, mcp._fastapi_process = old_latch, old_proc

        cmd = captured["cmd"]
        self.assertIn("--profile", cmd)
        self.assertEqual(cmd[cmd.index("--profile") + 1], "work")

    def test_hook_spawn_repasses_pin(self):
        from roampal.hooks import user_prompt_submit_hook as ups
        import roampal.profile_manager as pm

        pm.write_server_pin(27182, "work")
        self.addCleanup(pm.clear_server_pin, 27182)

        captured = {}

        def fake_popen(cmd, *a, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return mock.Mock(pid=1)

        netstat = mock.Mock(returncode=0)
        netstat.stdout = f"  TCP    127.0.0.1:27182   0.0.0.0:0   LISTENING   4242"
        with mock.patch.object(ups.subprocess, "run", return_value=netstat), \
             mock.patch.object(ups.subprocess, "Popen", fake_popen):
            ups._spawn_fresh_server("http://127.0.0.1:27182", 27182)

        cmd = captured["cmd"]
        self.assertIn("--profile", cmd)
        self.assertEqual(cmd[cmd.index("--profile") + 1], "work")

    def test_unpinned_spawn_has_no_profile_flag(self):
        from roampal.hooks import user_prompt_submit_hook as ups
        import roampal.profile_manager as pm

        pm.clear_server_pin(27182)

        captured = {}

        def fake_popen(cmd, *a, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return mock.Mock(pid=1)

        netstat = mock.Mock(returncode=0)
        netstat.stdout = f"  TCP    127.0.0.1:27182   0.0.0.0:0   LISTENING   4242"
        with mock.patch.object(ups.subprocess, "run", return_value=netstat), \
             mock.patch.object(ups.subprocess, "Popen", fake_popen):
            ups._spawn_fresh_server("http://127.0.0.1:27182", 27182)

        self.assertNotIn("--profile", captured["cmd"])


class TestInitWritesIsolatedCommands(unittest.TestCase):
    """`roampal init` writes MCP/hook commands that don't put cwd on sys.path."""

    def test_hook_command_builder_includes_isolation_flags(self):
        from roampal.cli.setup import _build_hook_command
        from roampal.profile_manager import spawn_isolation_flags

        flags = " ".join(spawn_isolation_flags())
        for is_dev in (False, True):
            cmd = _build_hook_command("stop_hook", is_dev)
            self.assertIn(f" {flags} -m roampal.hooks.stop_hook", cmd, repr(cmd))

    def test_expected_mcp_args_lists_use_isolation_flags(self):
        # Every written `args` for the roampal MCP server routes through
        # _mcp_args() (claude root, claude project, cursor) plus the opencode
        # list form — no hardcoded args lists may bypass the helper.
        src = Path("roampal/cli/setup.py").read_text(encoding="utf-8")
        self.assertEqual(
            src.count('"args": ["-m", "roampal.mcp.server"]'), 0,
            "at least one MCP config write bypasses _mcp_args()",
        )
        self.assertEqual(
            src.count('"args": ["-I", "-m", "roampal.mcp.server"]'), 0,
            "a MCP config write still uses the -I shape (breaks user-site installs)",
        )
        self.assertGreaterEqual(src.count('"args": _mcp_args()'), 3)
        self.assertIn('"command": [sys.executable, *_mcp_args()]', src)
        # setup.py's helper must delegate to profile_manager's contract.
        self.assertIn(
            "from roampal.profile_manager import spawn_isolation_flags", src
        )
        # The contract itself: -E, never -I.
        pm = Path("roampal/profile_manager.py").read_text(encoding="utf-8")
        self.assertIn('flags = ["-E"]', pm)
        self.assertNotIn('flags = ["-I"]', pm)

    def test_plugin_spawns_use_E(self):
        ts = Path("roampal/plugins/opencode/roampal.ts").read_text(encoding="utf-8")
        # Task 41: flags come from the config (`roampal init` writes -E [-P]
        # for 3.11+); the plugin cannot version-check an interpreter itself.
        # The hard-coded -E only survives as the no-config FALLBACK shape.
        self.assertIn('args: ["-E", ...SERVER_MODULE_ARGS]', ts)
        self.assertNotIn('"-I"', ts)
        self.assertIn("cwd: dataBase", ts)  # pythonw / VBS-wscript / python3 sites
        # wscript fallback also gets the neutral cwd (Shell.Run inherits it)
        self.assertIn('spawn("wscript", [vbsPath], { detached: true, stdio: "ignore", env: childEnv, cwd: dataBase }', ts)


if __name__ == "__main__":
    unittest.main()
