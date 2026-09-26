"""Task 44: `roampal init` keeps an already-configured scoring model.

Found in the Task 34 live check (2026-09-26): `init --force --opencode` — the
documented upgrade path for OpenCode users — dropped straight into the full
model menu with no hint that a model (qwen3.6-35b-a3b via LM Studio) was
already configured, and its "Skip" option claimed scoring was now disabled
while writing nothing.
"""

import json
from io import StringIO
from unittest.mock import patch

import pytest

import roampal.cli.sidecar as sc


def _config(tmp_path, env):
    p = tmp_path / "opencode.json"
    p.write_text(json.dumps({"mcp": {"roampal-core": {"type": "local", "environment": env}}}),
                 encoding="utf-8")
    return p


LMSTUDIO = {"ROAMPAL_SIDECAR_URL": "http://localhost:1234/v1",
            "ROAMPAL_SIDECAR_MODEL": "qwen3.6-35b-a3b"}


class TestDescribe:
    def test_custom_model(self):
        cfg = {"mcp": {"roampal-core": {"environment": LMSTUDIO}}}
        assert sc._describe_configured_sidecar(cfg) == "qwen3.6-35b-a3b @ localhost:1234"

    def test_zen_opt_in(self):
        cfg = {"mcp": {"roampal-core": {"environment": {"ROAMPAL_SIDECAR_PRIORITY": "zen"}}}}
        assert "Zen" in sc._describe_configured_sidecar(cfg)

    @pytest.mark.parametrize("env", [{}, {"ROAMPAL_SIDECAR_MODEL": "x"},
                                     {"ROAMPAL_SIDECAR_PRIORITY": "ollama,lmstudio"}])
    def test_nothing_configured(self, env):
        assert sc._describe_configured_sidecar({"mcp": {"roampal-core": {"environment": env}}}) is None

    def test_scope_helper_tolerates_missing_or_bad_config(self, tmp_path):
        bad = tmp_path / "opencode.json"
        bad.write_text("{not json", encoding="utf-8")
        with patch.object(sc, "_get_opencode_config_path", return_value=bad):
            assert sc.configured_sidecar_for_scope("user") is None
        with patch.object(sc, "_get_opencode_config_path", return_value=tmp_path / "nope.json"):
            assert sc.configured_sidecar_for_scope("user") is None


def _onboard(config_path, answer):
    out = StringIO()
    with patch.object(sc, "_get_opencode_config_path", return_value=config_path), \
            patch.object(sc, "_sidecar_model_picker") as picker, \
            patch("builtins.input", return_value=answer) as ask, \
            patch("sys.stdout", out):
        sc._prompt_smart_onboarding(scope="user")
    return picker, ask, out.getvalue()


class TestInitOnboarding:
    @pytest.mark.parametrize("answer", ["", "y", "Y", "yes"])
    def test_configured_model_is_kept_without_the_menu(self, tmp_path, answer):
        path = _config(tmp_path, LMSTUDIO)
        before = path.read_text(encoding="utf-8")
        picker, ask, out = _onboard(path, answer)
        picker.assert_not_called()
        assert "qwen3.6-35b-a3b @ localhost:1234 is already configured" in out
        assert "Kept." in out
        assert path.read_text(encoding="utf-8") == before

    def test_answering_no_opens_the_menu(self, tmp_path):
        picker, _, _ = _onboard(_config(tmp_path, LMSTUDIO), "n")
        picker.assert_called_once()

    def test_unconfigured_goes_straight_to_the_menu(self, tmp_path):
        picker, ask, out = _onboard(_config(tmp_path, {}), "")
        picker.assert_called_once()
        ask.assert_not_called()  # no keep prompt
        assert "already configured" not in out


class TestSkipMessage:
    def test_skip_with_a_model_says_it_stays(self, capsys):
        sc._print_skip_message("qwen3.6-35b-a3b @ localhost:1234")
        out = capsys.readouterr().out
        assert "stays" in out and "qwen3.6-35b-a3b" in out
        assert "disabled" not in out

    def test_skip_without_a_model_says_scoring_is_off(self, capsys):
        sc._print_skip_message(None)
        assert "disabled" in capsys.readouterr().out
