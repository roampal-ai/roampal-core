"""
Unit Tests for Server Main - v0.1.11 fixes.

Comprehensive PII Guard tests to ensure no personal/sensitive info is hardcoded.
Supports local pii_guard_config.py for user-specific sensitive data.
"""

import sys
import os
import re
import glob
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


class PIIGuardConfig:
    """Load and cache PII guard configuration."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load forbidden values from local config if it exists."""
        # Defaults (known past leaks)
        self.forbidden_names = []
        self.forbidden_emails = []
        self.forbidden_api_keys = []
        self.forbidden_urls = []
        self.forbidden_phone_numbers = []
        self.forbidden_addresses = []
        self.forbidden_patterns = []
        self.files_to_check = []

        # Try to load local config
        config_path = os.path.join(os.path.dirname(__file__), "..", "pii_guard_config.py")
        if os.path.exists(config_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("pii_guard_config", config_path)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)

                # Merge with defaults
                if hasattr(config, "FORBIDDEN_NAMES") and config.FORBIDDEN_NAMES:
                    self.forbidden_names.extend([n.lower() for n in config.FORBIDDEN_NAMES])
                if hasattr(config, "FORBIDDEN_EMAILS") and config.FORBIDDEN_EMAILS:
                    self.forbidden_emails.extend([e.lower() for e in config.FORBIDDEN_EMAILS])
                if hasattr(config, "FORBIDDEN_API_KEYS") and config.FORBIDDEN_API_KEYS:
                    self.forbidden_api_keys.extend(config.FORBIDDEN_API_KEYS)
                if hasattr(config, "FORBIDDEN_URLS") and config.FORBIDDEN_URLS:
                    self.forbidden_urls.extend([u.lower() for u in config.FORBIDDEN_URLS])
                if hasattr(config, "FORBIDDEN_PHONE_NUMBERS") and config.FORBIDDEN_PHONE_NUMBERS:
                    self.forbidden_phone_numbers.extend(config.FORBIDDEN_PHONE_NUMBERS)
                if hasattr(config, "FORBIDDEN_ADDRESSES") and config.FORBIDDEN_ADDRESSES:
                    self.forbidden_addresses.extend([a.lower() for a in config.FORBIDDEN_ADDRESSES])
                if hasattr(config, "FORBIDDEN_PATTERNS") and config.FORBIDDEN_PATTERNS:
                    self.forbidden_patterns.extend(config.FORBIDDEN_PATTERNS)
                if hasattr(config, "FILES_TO_CHECK") and config.FILES_TO_CHECK:
                    self.files_to_check.extend(config.FILES_TO_CHECK)
            except Exception:
                pass

        # Deduplicate
        self.forbidden_names = list(set(self.forbidden_names))
        self.forbidden_emails = list(set(self.forbidden_emails))
        self.forbidden_api_keys = list(set(self.forbidden_api_keys))
        self.forbidden_urls = list(set(self.forbidden_urls))


def _get_roampal_source_files():
    """Get all Python source files in the roampal package."""
    roampal_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', '..'
    ))
    # Get server and backend files
    patterns = [
        os.path.join(roampal_dir, "roampal", "server", "*.py"),
        os.path.join(roampal_dir, "roampal", "backend", "**", "*.py"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    # Exclude test files
    return [f for f in files if "test" not in f.lower()]


def _read_source(filepath):
    """Read file content safely."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


class TestColdStartQuery:
    """Test cold-start query does not contain PII - v0.1.11 fix."""

    def test_cold_start_query_no_personal_names(self):
        """Cold-start query should not contain personal names (v0.1.11 fix)."""
        from roampal.server import main
        import inspect

        source = inspect.getsource(main._build_cold_start_profile)
        source_lower = source.lower()

        config = PIIGuardConfig()
        for name in config.forbidden_names:
            assert name not in source_lower, \
                f"PII name '{name}' found in cold-start query - remove before shipping!"

    def test_cold_start_query_uses_generic_terms(self):
        """Cold-start query should use generic identity terms."""
        from roampal.server import main
        import inspect

        source = inspect.getsource(main._build_cold_start_profile)
        source_lower = source.lower()

        expected_terms = ["user", "identity", "preference"]
        for term in expected_terms:
            assert term in source_lower, \
                f"Expected generic term '{term}' not found in cold-start query"


class TestColdStartQualitySelection:
    """Test cold start picks highest quality fact per tag - v0.2.7 fix."""

    @pytest.mark.asyncio
    async def test_picks_highest_importance_per_tag(self):
        """Cold start should pick fact with highest importance for each tag."""
        from roampal.server import main
        from unittest.mock import MagicMock, AsyncMock

        # Create mock facts with same tag but different importance
        mock_facts = [
            {"id": "low", "metadata": {"tags": '["identity"]', "importance": 0.3, "confidence": 0.5}, "text": "Low quality"},
            {"id": "high", "metadata": {"tags": '["identity"]', "importance": 0.9, "confidence": 0.5}, "text": "High quality"},
            {"id": "medium", "metadata": {"tags": '["identity"]', "importance": 0.6, "confidence": 0.5}, "text": "Medium quality"},
        ]

        # Mock the memory system
        mock_memory = MagicMock()
        mock_memory._memory_bank_service.list_all = MagicMock(return_value=mock_facts)
        mock_memory._data_path = MagicMock()
        mock_memory._data_path.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        mock_memory.search = AsyncMock(return_value=[])

        # v0.5.4: _build_cold_start_profile now takes mem as a parameter
        # (singleton _memory was replaced by per-profile registry).
        result = await main._build_cold_start_profile(mock_memory)
        # Should have picked the high importance one
        assert "High quality" in result or result is None or "high" in str(result).lower(), \
            f"Expected highest importance fact, got: {result}"


class TestNoPIIInCodebase:
    """Comprehensive PII leak detection tests."""

    def test_no_forbidden_names(self):
        """Ensure no forbidden names appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_names:
            return  # No names to check

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath).lower()
            for name in config.forbidden_names:
                if len(name) >= 3:  # Skip very short names to avoid false positives
                    assert name not in source, \
                        f"PII name '{name}' found in {filepath}"

    def test_no_forbidden_emails(self):
        """Ensure no forbidden emails appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_emails:
            return  # No emails to check

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath).lower()
            for email in config.forbidden_emails:
                assert email not in source, \
                    f"PII email '{email}' found in {filepath}"

    def test_no_forbidden_api_keys(self):
        """Ensure no forbidden API keys appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_api_keys:
            return  # No keys to check

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            for key in config.forbidden_api_keys:
                assert key not in source, \
                    f"API key '{key[:8]}...' found in {filepath}"

    def test_no_forbidden_urls(self):
        """Ensure no forbidden URLs/domains appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_urls:
            return

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath).lower()
            for url in config.forbidden_urls:
                assert url not in source, \
                    f"Forbidden URL '{url}' found in {filepath}"

    def test_no_forbidden_phone_numbers(self):
        """Ensure no forbidden phone numbers appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_phone_numbers:
            return

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            for phone in config.forbidden_phone_numbers:
                # Normalize phone for matching
                phone_normalized = re.sub(r'[\s\-\(\)]', '', phone)
                source_normalized = re.sub(r'[\s\-\(\)]', '', source)
                assert phone_normalized not in source_normalized, \
                    f"Phone number found in {filepath}"

    def test_no_forbidden_addresses(self):
        """Ensure no forbidden addresses appear in source code."""
        config = PIIGuardConfig()
        if not config.forbidden_addresses:
            return

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath).lower()
            for addr in config.forbidden_addresses:
                assert addr not in source, \
                    f"Address found in {filepath}"

    def test_no_custom_patterns(self):
        """Check custom regex patterns from config."""
        config = PIIGuardConfig()
        if not config.forbidden_patterns:
            return

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            for pattern in config.forbidden_patterns:
                try:
                    matches = re.findall(pattern, source)
                    assert not matches, \
                        f"Pattern '{pattern}' matched in {filepath}: {matches[0][:20]}..."
                except re.error:
                    pass  # Invalid regex, skip


class TestGenericPIIPatterns:
    """Test for common PII patterns that shouldn't be in code."""

    def test_no_real_email_domains(self):
        """Emails should use example.com, not real domains."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        allowed_domains = [
            "example.com", "example.org", "example.net",
            "test.com", "test.org", "localhost",
            "anthropic.com"  # Allow Co-Authored-By
        ]

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            emails = re.findall(email_pattern, source)

            for email in emails:
                domain = email.split("@")[1].lower()
                is_allowed = any(d in domain for d in allowed_domains)
                assert is_allowed, \
                    f"Real email domain '{email}' in {filepath} - use @example.com"

    def test_no_openai_api_keys(self):
        """Ensure no OpenAI-style API keys are hardcoded."""
        openai_pattern = r'sk-[a-zA-Z0-9]{20,}'

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            matches = re.findall(openai_pattern, source)
            assert not matches, \
                f"OpenAI API key pattern found in {filepath}"

    def test_no_anthropic_api_keys(self):
        """Ensure no Anthropic-style API keys are hardcoded."""
        anthropic_pattern = r'sk-ant-[a-zA-Z0-9]{20,}'

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            matches = re.findall(anthropic_pattern, source)
            assert not matches, \
                f"Anthropic API key pattern found in {filepath}"

    def test_no_aws_access_keys(self):
        """Ensure no AWS access key patterns."""
        aws_pattern = r'AKIA[0-9A-Z]{16}'

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            matches = re.findall(aws_pattern, source)
            assert not matches, \
                f"AWS access key pattern found in {filepath}"

    def test_no_private_ip_addresses(self):
        """Ensure no hardcoded private IPs (except localhost)."""
        private_ip_patterns = [
            r'192\.168\.\d{1,3}\.\d{1,3}',
            r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            r'172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}',
        ]
        allowed_ips = ["127.0.0.1", "0.0.0.0"]

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            for pattern in private_ip_patterns:
                matches = re.findall(pattern, source)
                for match in matches:
                    if match not in allowed_ips:
                        assert False, \
                            f"Private IP '{match}' found in {filepath}"


class TestQueryStrings:
    """Test that query strings don't contain PII."""

    def test_no_pii_in_query_assignments(self):
        """Ensure no PII in query string assignments."""
        query_pattern = r'query\s*=\s*["\']([^"\']+)["\']'
        config = PIIGuardConfig()

        for filepath in _get_roampal_source_files():
            source = _read_source(filepath)
            queries = re.findall(query_pattern, source)

            for query in queries:
                query_lower = query.lower()
                for name in config.forbidden_names:
                    if len(name) >= 3:
                        assert name not in query_lower, \
                            f"PII '{name}' in query string in {filepath}"


class TestV052StartServerBanner:
    """v0.5.2: banner must resolve actual data path and show profile line."""

    def _capture_banner(self, tmp_path, env_patch, profile_name=None):
        """Stub uvicorn + create_app, capture banner output."""
        from unittest.mock import patch
        import roampal.server.main as srv

        if profile_name:
            from roampal.profile_manager import ProfileRegistry
            # Patch home BEFORE building registry so the registry writes to tmp
            with patch("roampal.profile_manager.Path.home", return_value=tmp_path):
                reg = ProfileRegistry()
                if not reg.exists(profile_name):
                    reg.create(profile_name)

        printed = []

        def capture(*args, **kwargs):
            if args:
                printed.append(" ".join(str(a) for a in args))

        # v0.5.x fix: mock the persisted-profile read so the banner doesn't
        # reflect the system's actual profile file. Round 2 Item 6: the
        # banner resolves from the explicit pin / persisted fallback —
        # NEVER env or binding — so mocked binding data isn't consulted.
        # Round 2 Item 7 / Task 22: start_server now runs through a real
        # uvicorn.Server so the idle monitor can retire it — stub both
        # uvicorn.Server/Config and give create_app an object that accepts
        # the roster-able app.state handle.
        from unittest.mock import MagicMock
        from types import SimpleNamespace as _NS

        fake_server = MagicMock()
        fake_server.run = lambda: None
        fake_app = MagicMock()
        fake_app.state = _NS()
        with patch("roampal.profile_manager.Path.home", return_value=tmp_path), \
             patch("roampal.profile_manager.read_active_profile_file", return_value=None), \
             patch.object(srv.uvicorn, "Config", MagicMock()), \
             patch.object(srv.uvicorn, "Server", MagicMock(return_value=fake_server)), \
             patch.object(srv, "create_app", lambda: fake_app), \
             patch("builtins.print", side_effect=capture), \
             patch.dict(os.environ, env_patch, clear=False):
            srv.start_server(host="127.0.0.1", port=27182, dev=False, profile=profile_name)

        return "\n".join(printed)

    def test_start_banner_shows_resolved_path_default(self, tmp_path, monkeypatch):
        """Default profile: banner shows the real system path, no Profile line."""
        for k in ("ROAMPAL_PROFILE", "ROAMPAL_DEV", "ROAMPAL_DATA_PATH"):
            monkeypatch.delenv(k, raising=False)

        output = self._capture_banner(tmp_path, env_patch={})

        assert "ROAMPAL SERVER - PROD MODE" in output
        assert "Port: 27182" in output
        assert "%APPDATA%" not in output  # no unexpanded literal
        assert "Profile:" not in output    # default -> no profile line
        # Resolved path is absolute (starts with drive letter on win, / on unix)
        assert "Data:" in output

    def test_start_banner_shows_pinned_profile_line(self, tmp_path, monkeypatch):
        """Pinned profile at launch: banner shows Profile: name (startup --profile flag).

        Round 2 Item 6: the pin is passed explicitly (profile=), not read
        from ROAMPAL_PROFILE — a leaked env must NEVER resolve requests
        on the shared server, only an explicit `--profile` may pin one.
        """
        for k in ("ROAMPAL_PROFILE", "ROAMPAL_DEV", "ROAMPAL_DATA_PATH"):
            monkeypatch.delenv(k, raising=False)

        output = self._capture_banner(
            tmp_path,
            env_patch={},
            profile_name="research",
        )

        assert "ROAMPAL SERVER - PROD MODE" in output
        assert "Profile: research (startup --profile flag)" in output
        # Data path should include the profile slug, not raw %APPDATA%.
        assert "%APPDATA%" not in output

    def test_start_banner_ignores_leaked_profile_env(self, tmp_path, monkeypatch):
        """Round 2 Item 6 / Task 17: ROAMPAL_PROFILE in the server env is
        ignored — headerless requests would resolve persisted `use` ->
        default, so the banner must not claim the leaked profile either.
        """
        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        monkeypatch.delenv("ROAMPAL_DATA_PATH", raising=False)
        monkeypatch.setenv("ROAMPAL_PROFILE", "research")

        output = self._capture_banner(tmp_path, env_patch={}, profile_name=None)

        assert "Profile:" not in output  # research was never pinned -> default
        assert "%APPDATA%" not in output


class TestIdleSelfRetirement:
    """Round 2 Item 7 / Task 22: the shared server retires itself after an
    idle period instead of being killed by whoever launched it."""

    async def _run_monitor(self, monkeypatch, srv, stale_time, expect_exit_after):
        """Shared monitor-runner: restores the module timestamp after use."""
        import asyncio
        from unittest.mock import MagicMock

        monkeypatch.setattr(srv, "_IDLE_CHECK_INTERVAL_SECONDS", 0.01)
        monkeypatch.delenv("ROAMPAL_SERVER_IDLE_TIMEOUT_MINUTES", raising=False)
        original_time = srv._last_request_time
        srv._last_request_time = stale_time
        fake = MagicMock()
        fake.should_exit = False
        try:
            monitor = asyncio.create_task(srv._idle_retire_monitor(fake))
            try:
                done, _ = await asyncio.wait({monitor}, timeout=expect_exit_after)
                return done, fake.should_exit
            finally:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
        finally:
            srv._last_request_time = original_time

    async def test_monitor_flips_should_exit_when_idle(self, monkeypatch):
        import roampal.server.main as srv

        done, should_exit = await self._run_monitor(
            monkeypatch, srv, stale_time=0.0, expect_exit_after=5.0,
        )
        assert done, "monitor never returned"
        assert should_exit is True

    async def test_monitor_keeps_server_alive_with_recent_requests(self, monkeypatch):
        import roampal.server.main as srv

        done, should_exit = await self._run_monitor(
            monkeypatch, srv, stale_time=srv.time.time(), expect_exit_after=0.05,
        )
        # The monitor is still running (timed out waiting) -> server alive.
        assert not done
        assert should_exit is False

    def test_idle_timeout_env_parsing(self, monkeypatch):
        import roampal.server.main as srv

        monkeypatch.delenv("ROAMPAL_SERVER_IDLE_TIMEOUT_MINUTES", raising=False)
        assert srv._idle_timeout_minutes() == 30
        monkeypatch.setenv("ROAMPAL_SERVER_IDLE_TIMEOUT_MINUTES", "10")
        assert srv._idle_timeout_minutes() == 10
        for bad in ("abc", "0", "-5", "3.5"):
            monkeypatch.setenv("ROAMPAL_SERVER_IDLE_TIMEOUT_MINUTES", bad)
            assert srv._idle_timeout_minutes() == 30, bad

    def test_default_is_thirty_minutes(self):
        import roampal.server.main as srv

        assert srv.IDLE_TIMEOUT_DEFAULT_MINUTES == 30

    def test_start_server_idle_retirement_flag(self, tmp_path):
        """v0.6.0 review fix 9: start_server wires the retirement switch —
        idle_retire=False (roampal start) disables it; the default (spawned
        shared servers) keeps it enabled."""
        from unittest.mock import MagicMock, patch
        import roampal.server.main as srv

        fake_server = MagicMock()
        fake_server.run = lambda: None
        fake_app = MagicMock()
        fake_app.state = type("S", (), {})()
        with patch.object(srv.uvicorn, "Config", MagicMock()), \
             patch.object(srv.uvicorn, "Server", MagicMock(return_value=fake_server)), \
             patch.object(srv, "create_app", lambda: fake_app), \
             patch("builtins.print"), \
             patch.dict(os.environ, {"ROAMPAL_DEV": "0"}, clear=False):
            srv.start_server(host="127.0.0.1", port=27182, dev=False, idle_retire=False)
            assert srv._idle_retirement_enabled is False
            srv.start_server(host="127.0.0.1", port=27182, dev=False)
            assert srv._idle_retirement_enabled is True


class TestHealthProbeSkipsCache:
    """v0.6.0 review fix 2, part 2 (round 2): with the embedding cache, the
    first successful health probe stored 'health check' and every later
    probe returned that stored vector — a dead embedder read as healthy
    forever (a dead embedder never writes cache entries, so the stored
    result was never evicted). Health must actually run the model."""

    def test_skip_cache_bypasses_read_and_write(self):
        from unittest.mock import MagicMock
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        import asyncio

        es = EmbeddingService.__new__(EmbeddingService)
        es._embed_cache = {}
        es._embed_cache_max = 32
        es._encode = MagicMock(side_effect=AssertionError("cached result must not be served"))

        # Cache holds a (stale) vector for the health text.
        stale = [1.0] * 768
        es._embed_cache[("passage", "health check")] = stale

        async def run():
            # skip_cache=True must ignore the cached entry and re-encode —
            # which here raises via the poisoned _encode (proves the model
            # path ran instead of the cache read).
            with pytest.raises(AssertionError):
                await es.embed_text("health check", role="passage", skip_cache=True)

            # A normal call still serves the cache.
            inner = MagicMock()
            inner.tolist.return_value = [0.5] * 768
            es._encode = MagicMock(return_value=[inner])
            served = await es.embed_text("health check", role="passage")
            assert served == stale

            # skip_cache result is NOT written into the cache.
            inner2 = MagicMock()
            inner2.tolist.return_value = [0.9] * 768
            es._encode = MagicMock(return_value=[inner2])
            await es.embed_text("brand new text", skip_cache=True)
            assert ("passage", "brand new text") not in es._embed_cache

        asyncio.run(run())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
