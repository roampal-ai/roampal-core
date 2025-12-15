"""
Pytest configuration for memory module tests.

Sets up import paths for roampal-core.
"""

import sys
import os

# Add roampal-core root to path
CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, CORE_ROOT)
