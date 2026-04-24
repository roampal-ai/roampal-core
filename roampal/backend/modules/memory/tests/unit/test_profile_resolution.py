"""
Tests for _resolve_profile_name (v0.5.4 per-request profile resolver).

Pins the contract that handlers depend on: a request's profile is resolved by
priority X-Roampal-Profile header > active_profile_name() (which itself reads
ROAMPAL_PROFILE env > persisted file > DEFAULT_PROFILE).

The receiving end of the cross-client header propagation chain — every client
(MCP server, OpenCode plugin, Python hooks) sends the header; this function
decides which profile bucket the request lands in.
"""

from unittest.mock import MagicMock, patch
import pytest

from roampal.server.main import _resolve_profile_name


def _request_with_header(value):
    """Build a minimal Request mock with the given X-Roampal-Profile header value."""
    req = MagicMock()
    req.headers = {"X-Roampal-Profile": value} if value is not None else {}
    # MagicMock dict-style get for headers
    req.headers = MagicMock()
    if value is not None:
        req.headers.get = MagicMock(return_value=value)
    else:
        req.headers.get = MagicMock(return_value=None)
    return req


class TestProfileResolution:
    def test_header_returns_header_value(self):
        """Any non-empty X-Roampal-Profile header wins over env/file."""
        req = _request_with_header("ghost")
        assert _resolve_profile_name(req) == "ghost"

    def test_header_research(self):
        """Different header value routes to different profile."""
        req = _request_with_header("research")
        assert _resolve_profile_name(req) == "research"

    def test_no_header_falls_back_to_active_profile_name(self):
        """When header missing, active_profile_name() decides."""
        req = _request_with_header(None)
        with patch("roampal.profile_manager.active_profile_name", return_value="qr"):
            assert _resolve_profile_name(req) == "qr"

    def test_no_header_default_returns_string_default(self):
        """When active_profile_name() returns DEFAULT_PROFILE, normalize to 'default' string."""
        req = _request_with_header(None)
        with patch("roampal.profile_manager.active_profile_name", return_value="default"):
            assert _resolve_profile_name(req) == "default"

    def test_header_empty_string_falls_through(self):
        """Empty header should fall through to active_profile_name (truthy check)."""
        req = _request_with_header("")
        with patch("roampal.profile_manager.active_profile_name", return_value="research"):
            assert _resolve_profile_name(req) == "research"

    def test_header_takes_precedence_over_env(self):
        """Header beats env even when both are set."""
        req = _request_with_header("ghost")
        with patch("roampal.profile_manager.active_profile_name", return_value="research"):
            assert _resolve_profile_name(req) == "ghost"
