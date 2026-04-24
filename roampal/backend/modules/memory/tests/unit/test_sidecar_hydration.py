"""
Unit tests for v0.5.4 _hydrate_sidecar_from_opencode_config.

The FastAPI server is launched independently of the OpenCode plugin
process and does not inherit ROAMPAL_SIDECAR_URL/MODEL/KEY env vars set
in opencode.json's mcp.roampal-core.environment block. v0.5.3 §11
claimed server-side noun_tags extraction would fall back to the sidecar
when the plugin doesn't send them — but extract_tags() reads
sidecar_service.CUSTOM_URL which was empty because nothing populated it.

These tests exercise the new server-startup hydration that reads
opencode.json and populates os.environ + sidecar_service module globals
before any request handler runs.
"""

import sys
import os
import json
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
    ),
)


@pytest.fixture(autouse=True)
def _isolate_sidecar_env(monkeypatch):
    """Each test starts with sidecar env vars unset and module globals empty."""
    monkeypatch.delenv("ROAMPAL_SIDECAR_URL", raising=False)
    monkeypatch.delenv("ROAMPAL_SIDECAR_MODEL", raising=False)
    monkeypatch.delenv("ROAMPAL_SIDECAR_KEY", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    import roampal.sidecar_service as svc
    monkeypatch.setattr(svc, "CUSTOM_URL", "")
    monkeypatch.setattr(svc, "CUSTOM_MODEL", "")
    monkeypatch.setattr(svc, "CUSTOM_KEY", "")
    yield


def _write_opencode_json(home: Path, env: dict):
    """Place an opencode.json at the location _hydrate looks for."""
    if sys.platform == "win32":
        config_dir = home / ".config" / "opencode"
    else:
        config_dir = home / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "opencode.json"
    path.write_text(json.dumps({
        "mcp": {
            "roampal-core": {
                "environment": env,
            }
        }
    }))
    return path


def test_hydrate_populates_module_globals_from_opencode_json(tmp_path, monkeypatch):
    """The happy path: sidecar config in opencode.json → module globals set."""
    _write_opencode_json(tmp_path, {
        "ROAMPAL_SIDECAR_URL": "http://localhost:1234/v1",
        "ROAMPAL_SIDECAR_MODEL": "qwen-test",
        "ROAMPAL_SIDECAR_KEY": "sk-test-key",
    })

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == "http://localhost:1234/v1"
    assert svc.CUSTOM_MODEL == "qwen-test"
    assert svc.CUSTOM_KEY == "sk-test-key"
    assert os.environ["ROAMPAL_SIDECAR_URL"] == "http://localhost:1234/v1"


def test_hydrate_no_op_when_env_vars_already_set(tmp_path, monkeypatch):
    """If env vars are pre-set, hydration must not overwrite them."""
    monkeypatch.setenv("ROAMPAL_SIDECAR_URL", "http://preset:9999/v1")
    monkeypatch.setenv("ROAMPAL_SIDECAR_MODEL", "preset-model")

    _write_opencode_json(tmp_path, {
        "ROAMPAL_SIDECAR_URL": "http://from-config:1234/v1",
        "ROAMPAL_SIDECAR_MODEL": "config-model",
    })

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()

    # Pre-set env wins, opencode.json is ignored.
    assert os.environ["ROAMPAL_SIDECAR_URL"] == "http://preset:9999/v1"
    assert os.environ["ROAMPAL_SIDECAR_MODEL"] == "preset-model"


def test_hydrate_no_op_when_opencode_json_missing(tmp_path):
    """Missing opencode.json → silent no-op, no env mutation."""
    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()  # must not raise

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == ""
    assert svc.CUSTOM_MODEL == ""
    assert "ROAMPAL_SIDECAR_URL" not in os.environ


def test_hydrate_no_op_when_opencode_json_invalid(tmp_path):
    """Malformed opencode.json → silent no-op, no env mutation."""
    if sys.platform == "win32":
        config_dir = tmp_path / ".config" / "opencode"
    else:
        config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "opencode.json").write_text("{ not valid json }}}")

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()  # must not raise

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == ""
    assert svc.CUSTOM_MODEL == ""


def test_hydrate_no_op_when_sidecar_block_missing(tmp_path):
    """Valid opencode.json but no roampal-core mcp block → no-op."""
    if sys.platform == "win32":
        config_dir = tmp_path / ".config" / "opencode"
    else:
        config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "opencode.json").write_text(json.dumps({"mcp": {}}))

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == ""


def test_hydrate_skips_when_url_present_but_model_missing(tmp_path):
    """Both URL and MODEL must be present — partial config is treated as no-config."""
    _write_opencode_json(tmp_path, {
        "ROAMPAL_SIDECAR_URL": "http://localhost:1234/v1",
        # ROAMPAL_SIDECAR_MODEL missing
    })

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == ""
    assert svc.CUSTOM_MODEL == ""


def test_hydrate_skips_key_when_not_provided(tmp_path):
    """ROAMPAL_SIDECAR_KEY is optional — local LM Studio etc. doesn't need one."""
    _write_opencode_json(tmp_path, {
        "ROAMPAL_SIDECAR_URL": "http://localhost:1234/v1",
        "ROAMPAL_SIDECAR_MODEL": "qwen-test",
        # No ROAMPAL_SIDECAR_KEY
    })

    with patch("roampal.server.main.Path.home", return_value=tmp_path):
        from roampal.server.main import _hydrate_sidecar_from_opencode_config
        _hydrate_sidecar_from_opencode_config()

    import roampal.sidecar_service as svc
    assert svc.CUSTOM_URL == "http://localhost:1234/v1"
    assert svc.CUSTOM_MODEL == "qwen-test"
    assert svc.CUSTOM_KEY == ""
    assert "ROAMPAL_SIDECAR_KEY" not in os.environ
