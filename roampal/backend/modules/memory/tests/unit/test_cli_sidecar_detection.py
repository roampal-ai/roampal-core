"""Local-model detection for the sidecar picker (v0.6.0 refactor regression).

The CLI split moved the detectors into roampal/cli/sidecar/ without the
monolith's module-level `import urllib.request`. Each detector wraps its
probe in `except Exception: return []`, so the NameError was swallowed and
`roampal init` / `roampal sidecar setup` silently offered no Ollama or
LM Studio models even with both running. These tests drive the detectors
through a faked urlopen and fail if detection comes back empty.
"""

import io
import json
import urllib.request
from pathlib import Path

import pytest


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(routes):
    def fake(req, timeout=None, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                return _Resp(json.dumps(body).encode("utf-8"))
        raise OSError(f"connection refused: {url}")

    return fake


def test_detect_ollama_models_finds_running_ollama(monkeypatch):
    from roampal.cli import sidecar

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _fake_urlopen({
            "http://localhost:11434/api/tags": {
                "models": [
                    {"name": "qwen3:8b", "size": 5 * 1024**3, "details": {"family": "qwen3"}},
                    {"name": "nomic-embed-text", "size": 1, "details": {"family": "nomic-bert"}},
                ]
            }
        }),
    )
    models = sidecar._detect_ollama_models()
    assert [m["name"] for m in models] == ["qwen3:8b"]


def test_detect_local_servers_finds_lm_studio(monkeypatch):
    from roampal.cli import sidecar

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _fake_urlopen({
            "http://localhost:1234/v1/models": {"data": [{"id": "qwen3.6-35b-a3b"}]},
        }),
    )
    servers = sidecar._detect_local_servers()
    assert any(s["name"] == "qwen3.6-35b-a3b" and s["port"] == 1234 for s in servers), servers


def test_cli_package_has_no_undefined_names():
    """Guard for the whole refactor-bug class: a name the old monolith got
    from a module-level import, used in a group module that never imports
    it. Runtime tests miss these whenever the use sits behind a broad
    `except Exception`."""
    pyflakes_api = pytest.importorskip("pyflakes.api")
    from pyflakes import reporter as pyflakes_reporter

    cli_dir = Path(__file__).resolve().parents[5] / "cli"
    assert cli_dir.is_dir(), cli_dir

    out, err = io.StringIO(), io.StringIO()
    pyflakes_api.checkRecursive([str(cli_dir)], pyflakes_reporter.Reporter(out, err))
    undefined = [line for line in out.getvalue().splitlines() if "undefined name" in line]
    assert not undefined, "\n".join(undefined)
