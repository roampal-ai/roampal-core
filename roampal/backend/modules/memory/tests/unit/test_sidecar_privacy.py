"""Sidecar privacy + non-interactive setup (v0.6.0 Tasks 38 and 39).

Task 38 — the OpenCode plugin sent exchange text to Zen (opencode.ai) whenever
no sidecar was configured, from v0.3.7 on. v0.5.3 decided "no explicit choice
= scoring off" but only fixed the server side; the plugin never read the
recorded opt-in. These tests pin the plugin to that rule: Zen only after
ROAMPAL_SIDECAR_PRIORITY=zen, nothing at all when nothing was chosen.

Task 39 — `roampal sidecar setup` could only be driven by typing into a menu,
so an LLM doing an install could not configure scoring. The non-interactive
flags (--list/--json, --model, --url/--key-env, --go, --zen, --auto) are
tested through cmd_sidecar against a temp opencode.json.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PLUGIN_PATH = Path(__file__).resolve().parents[5] / "plugins" / "opencode" / "roampal.ts"


@pytest.fixture
def plugin_source():
    return PLUGIN_PATH.read_text(encoding="utf-8")


# ============================================================================
# Task 38 — plugin structure
# ============================================================================

class TestPluginZenOnlyAfterOptIn:
    def test_zen_targets_gated_on_explicit_opt_in(self, plugin_source):
        assert "if (ZEN_OPTED_IN) {" in plugin_source
        # The v0.3.7 default (Zen whenever no custom URL) must be gone.
        assert "if (!CUSTOM_SIDECAR_URL) {" not in plugin_source
        assert "Default (no setup): Zen" not in plugin_source

    def test_opt_in_read_from_recorded_priority(self, plugin_source):
        assert "ROAMPAL_SIDECAR_PRIORITY" in plugin_source
        assert '.includes("zen")' in plugin_source

    def test_scorer_entry_short_circuits_when_off(self, plugin_source):
        body = plugin_source.split("async function scoreExchangeViaLLM(", 1)[1]
        head = body.split("if (scoringQueueRunning)", 1)[0]
        assert "if (SIDECAR_OFF) return true" in head

    def test_queues_and_drainer_gated_on_off(self, plugin_source):
        assert "if (SIDECAR_OFF || pendingScoringQueue.size === 0) return" in plugin_source
        assert "if (SIDECAR_OFF || pendingSummaryQueue.size === 0) return" in plugin_source
        assert "if (!SIDECAR_OFF && backgroundDrainTimer === null)" in plugin_source

    def test_model_is_told_scoring_is_off(self, plugin_source):
        assert "scoring: off — no scoring model chosen" in plugin_source
        assert "scoring: zen (free, best-effort) | For reliable scoring" not in plugin_source


# ============================================================================
# Task 38 — plugin behavior: run the REAL config-decision code under Node
# ============================================================================

_NODE = shutil.which("node")

_RUNNER = r"""
import { stripTypeScriptTypes } from "node:module"
import { readFileSync, writeFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { join } from "node:path"
const [pluginPath, outPath] = process.argv.slice(2)
const src = readFileSync(pluginPath, "utf8")
const start = src.indexOf("function _loadSidecarConfig(")
const endMarker = "const SIDECAR_OFF ="
const end = src.indexOf("\n", src.indexOf(endMarker))
if (start < 0 || end < 0) { console.error("markers not found"); process.exit(3) }
const slice = src.slice(start, end)
const body = `import { readFileSync } from "node:fs"
import { join } from "node:path"
const debugLog = () => {}
${slice}
console.log(JSON.stringify({ off: SIDECAR_OFF, zen: ZEN_OPTED_IN, custom: CUSTOM_SIDECAR_CONFIGURED }))
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


def _decide(tmp_path, env_block):
    """Run the plugin's sidecar decision against an opencode.json whose
    roampal-core environment is `env_block` (None = no config file)."""
    home = tmp_path / "home"
    cfg_dir = home / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    if env_block is not None:
        (cfg_dir / "opencode.json").write_text(
            json.dumps({"mcp": {"roampal-core": {"environment": env_block}}}),
            encoding="utf-8",
        )
    runner = tmp_path / "runner.mjs"
    runner.write_text(_RUNNER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROAMPAL_SIDECAR")}
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    env.pop("XDG_CONFIG_HOME", None)
    proc = subprocess.run(
        [_NODE, "--no-warnings", str(runner), str(PLUGIN_PATH), str(tmp_path / "decision.mjs")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not _node_supports_strip(), reason="needs node >= 22.13 (stripTypeScriptTypes)")
class TestPluginDecisionBehavior:
    def test_no_config_file_means_off(self, tmp_path):
        assert _decide(tmp_path, None) == {"off": True, "zen": False, "custom": False}

    def test_nothing_chosen_means_off_not_zen(self, tmp_path):
        """The v0.3.7-v0.6.0 leak: a fresh or 'Skip' config sent data to Zen."""
        d = _decide(tmp_path, {"ROAMPAL_PLATFORM": "opencode"})
        assert d == {"off": True, "zen": False, "custom": False}

    def test_explicit_zen_opt_in(self, tmp_path):
        d = _decide(tmp_path, {"ROAMPAL_SIDECAR_PRIORITY": "zen"})
        assert d == {"off": False, "zen": True, "custom": False}

    def test_custom_model(self, tmp_path):
        d = _decide(tmp_path, {
            "ROAMPAL_SIDECAR_URL": "http://localhost:11434/v1",
            "ROAMPAL_SIDECAR_MODEL": "qwen3:1.7b",
        })
        assert d == {"off": False, "zen": False, "custom": True}

    def test_custom_wins_over_zen_opt_in(self, tmp_path):
        d = _decide(tmp_path, {
            "ROAMPAL_SIDECAR_URL": "http://localhost:11434/v1",
            "ROAMPAL_SIDECAR_MODEL": "qwen3:1.7b",
            "ROAMPAL_SIDECAR_PRIORITY": "zen",
        })
        assert d == {"off": False, "zen": False, "custom": True}

    def test_disabled_flag_turns_everything_off(self, tmp_path):
        d = _decide(tmp_path, {
            "ROAMPAL_SIDECAR_PRIORITY": "zen",
            "ROAMPAL_SIDECAR_DISABLED": "true",
        })
        assert d["off"] is True

    def test_priority_without_zen_is_not_an_opt_in(self, tmp_path):
        d = _decide(tmp_path, {"ROAMPAL_SIDECAR_PRIORITY": "ollama,lmstudio"})
        assert d == {"off": True, "zen": False, "custom": False}


# ============================================================================
# Task 39 — non-interactive `roampal sidecar setup`
# ============================================================================

OLLAMA = [{"name": "qwen3:1.7b", "size_gb": 1.2, "source": "ollama"},
          {"name": "gemma4:31b", "size_gb": 18.0, "source": "ollama"}]
LOCAL = [{"name": "qwen3.6-35b-a3b", "port": 1234, "server_label": "LM Studio", "source": "local"}]


@pytest.fixture
def oc_config(tmp_path, monkeypatch):
    """A user-global opencode.json with roampal-core configured."""
    home = tmp_path / "home"
    cfg = home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({
        "mcp": {"roampal-core": {"type": "local", "command": ["python"],
                                 "environment": {"ROAMPAL_PLATFORM": "opencode"}}},
        "provider": {"groq": {"name": "Groq", "options": {
            "baseURL": "https://api.groq.com/openai/v1", "apiKey": "gsk_secret"},
            "models": {"llama-3.1-8b-instant": {}}}},
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return cfg


def _env(cfg):
    return json.loads(cfg.read_text(encoding="utf-8"))["mcp"]["roampal-core"]["environment"]


def _run(argv_extra, *, go=None, capsys=None):
    import argparse
    from roampal.cli import cmd_sidecar

    ns = argparse.Namespace(
        sidecar_command="setup", scope="user", list=False, json=False, model=None,
        url=None, key_env=None, go=None, zen=False, auto=False,
    )
    for k, v in argv_extra.items():
        setattr(ns, k, v)
    with patch("roampal.cli._detect_ollama_models", return_value=OLLAMA), \
         patch("roampal.cli._detect_local_servers", return_value=LOCAL), \
         patch("roampal.cli._detect_opencode_go", return_value=go), \
         patch("roampal.cli._check_sidecar_configured", return_value=True), \
         patch("builtins.input", side_effect=AssertionError("must not prompt")):
        return cmd_sidecar(ns)


class TestNonInteractiveSetup:
    def test_list_json_is_machine_readable_and_secret_free(self, oc_config, capsys):
        rc = _run({"list": True, "json": True})
        out = capsys.readouterr().out
        data = json.loads(out)  # stdout is ONLY the JSON document
        assert rc == 0
        kinds = [o["kind"] for o in data["options"]]
        assert kinds[:3] == ["local", "local", "local"]
        assert {"api", "custom", "zen", "off"} <= set(kinds)
        assert data["recommended_local"] == "qwen3:1.7b"
        zen = next(o for o in data["options"] if o["kind"] == "zen")
        assert zen["data_leaves_machine"] is True and zen["command"] == "roampal sidecar setup --zen"
        local = next(o for o in data["options"] if o["kind"] == "local")
        assert local["data_leaves_machine"] is False
        assert "gsk_secret" not in out  # API keys are never printed
        assert _env(oc_config) == {"ROAMPAL_PLATFORM": "opencode"}  # --list writes nothing

    def test_model_picks_detected_local_model(self, oc_config):
        assert _run({"model": "qwen3.6-35b-a3b"}) == 0
        env = _env(oc_config)
        assert env["ROAMPAL_SIDECAR_URL"] == "http://localhost:1234/v1"
        assert env["ROAMPAL_SIDECAR_MODEL"] == "qwen3.6-35b-a3b"
        assert "ROAMPAL_SIDECAR_PRIORITY" not in env

    def test_unknown_model_fails_and_writes_nothing(self, oc_config, capsys):
        assert _run({"model": "does-not-exist"}) == 1
        assert "--list" in capsys.readouterr().out
        assert _env(oc_config) == {"ROAMPAL_PLATFORM": "opencode"}

    def test_auto_picks_smallest_local_never_cloud(self, oc_config):
        assert _run({"auto": True}) == 0
        env = _env(oc_config)
        assert env["ROAMPAL_SIDECAR_MODEL"] == "qwen3:1.7b"
        assert env["ROAMPAL_SIDECAR_URL"] == "http://localhost:11434/v1"

    def test_auto_without_local_models_fails(self, oc_config):
        import argparse
        from roampal.cli import cmd_sidecar

        ns = argparse.Namespace(sidecar_command="setup", scope="user", list=False, json=False,
                                model=None, url=None, key_env=None, go=None, zen=False, auto=True)
        with patch("roampal.cli._detect_ollama_models", return_value=[]), \
             patch("roampal.cli._detect_local_servers", return_value=[]), \
             patch("roampal.cli._check_sidecar_configured", return_value=True):
            assert cmd_sidecar(ns) == 1
        assert _env(oc_config) == {"ROAMPAL_PLATFORM": "opencode"}

    def test_zen_is_an_explicit_recorded_opt_in(self, oc_config):
        assert _run({"zen": True}) == 0
        env = _env(oc_config)
        assert env["ROAMPAL_SIDECAR_PRIORITY"] == "zen"
        assert "ROAMPAL_SIDECAR_URL" not in env

    def test_custom_endpoint_reads_key_from_env_var(self, oc_config, monkeypatch):
        monkeypatch.setenv("MY_SIDECAR_KEY", "sk-test-123")
        rc = _run({"url": "https://api.example.com/v1", "model": "small-1", "key_env": "MY_SIDECAR_KEY"})
        assert rc == 0
        env = _env(oc_config)
        assert env["ROAMPAL_SIDECAR_URL"] == "https://api.example.com/v1"
        assert env["ROAMPAL_SIDECAR_MODEL"] == "small-1"
        assert env["ROAMPAL_SIDECAR_KEY"] == "sk-test-123"

    def test_custom_endpoint_with_empty_key_env_fails(self, oc_config, monkeypatch):
        monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
        rc = _run({"url": "https://api.example.com/v1", "model": "m", "key_env": "MISSING_KEY_VAR"})
        assert rc == 1
        assert "ROAMPAL_SIDECAR_URL" not in _env(oc_config)

    def test_go_model(self, oc_config):
        go = {"url": "https://opencode.ai/zen/go/v1", "key": "go-key", "models": ["glm-5.1"]}
        assert _run({"go": "glm-5.1"}, go=go) == 0
        env = _env(oc_config)
        assert env["ROAMPAL_SIDECAR_MODEL"] == "glm-5.1"
        assert env["ROAMPAL_SIDECAR_URL"] == "https://opencode.ai/zen/go/v1"

    def test_two_choices_at_once_is_rejected(self, oc_config):
        assert _run({"zen": True, "auto": True}) == 1
        assert _env(oc_config) == {"ROAMPAL_PLATFORM": "opencode"}

    def test_choosing_a_model_clears_an_old_zen_opt_in(self, oc_config):
        assert _run({"zen": True}) == 0
        assert _run({"model": "qwen3:1.7b"}) == 0
        assert "ROAMPAL_SIDECAR_PRIORITY" not in _env(oc_config)
