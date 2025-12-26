"""
Unit Tests for Server Main - v0.1.11 fixes.

Comprehensive PII Guard tests to ensure no personal/sensitive info is hardcoded.
Supports local pii_guard_config.py for user-specific sensitive data.
"""

import sys
import os
import re
import glob
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
        self.forbidden_names = ["logan", "logte"]
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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
