"""
PII Guard Configuration Template.

Copy this file to pii_guard_config.py (which is gitignored) and add your
personal information to check against. This prevents accidental leaks
of ANY personal/sensitive info in the codebase.

The test_server_main.py tests will read from pii_guard_config.py if it exists
and check that none of these values appear in the codebase.
"""

# ============================================================================
# PERSONAL IDENTIFIERS
# ============================================================================
# Add your personal names, usernames, handles, etc.
FORBIDDEN_NAMES = [
    # "your_first_name",
    # "your_last_name",
    # "your_username",
    # "your_nickname",
    # "your_github_handle",
    # "your_twitter_handle",
]

# ============================================================================
# EMAIL ADDRESSES
# ============================================================================
# Add your personal email addresses
FORBIDDEN_EMAILS = [
    # "your.email@gmail.com",
    # "work.email@company.com",
]

# ============================================================================
# API KEYS & SECRETS
# ============================================================================
# Add any API keys or secrets that should NEVER appear in code
# Even partial keys are useful to detect
FORBIDDEN_API_KEYS = [
    # "sk-1234567890abcdef",  # OpenAI key prefix
    # "your_anthropic_key_prefix",
    # "your_aws_access_key",
]

# ============================================================================
# URLS & DOMAINS
# ============================================================================
# Personal domains, internal URLs, etc.
FORBIDDEN_URLS = [
    # "mycompany.internal",
    # "192.168.1.100",  # Internal IPs
]

# ============================================================================
# PHONE NUMBERS
# ============================================================================
FORBIDDEN_PHONE_NUMBERS = [
    # "555-123-4567",
    # "+1234567890",
]

# ============================================================================
# PHYSICAL ADDRESSES
# ============================================================================
FORBIDDEN_ADDRESSES = [
    # "123 Main St",
    # "Apartment 4B",
]

# ============================================================================
# CUSTOM PATTERNS (regex)
# ============================================================================
# Add regex patterns for custom sensitive data
FORBIDDEN_PATTERNS = [
    # r"my-company-\d+",  # Company-specific patterns
]

# ============================================================================
# FILES TO CHECK (optional)
# ============================================================================
# If empty, checks default critical files
# Add specific files to always scan
FILES_TO_CHECK = [
    # "path/to/sensitive/file.py",
]

# ============================================================================
# EXAMPLE CONFIG
# ============================================================================
# Example of what a filled config might look like:
#
# FORBIDDEN_NAMES = ["john", "johndoe", "jdoe123"]
# FORBIDDEN_EMAILS = ["john.doe@gmail.com"]
# FORBIDDEN_API_KEYS = ["sk-abc123"]
# FORBIDDEN_URLS = ["internal.mycompany.com"]
# FORBIDDEN_PHONE_NUMBERS = ["555-867-5309"]
# FORBIDDEN_ADDRESSES = ["742 Evergreen Terrace"]
