"""
Pytest configuration for Roampal Core benchmark tests.

These benchmarks test the memory system's:
- Learning effectiveness (outcome scoring)
- Adversarial resistance (poisoning, confusion)
- Performance under stress (high volume, concurrent access)
- Quality metrics (MRR, nDCG, precision)
"""
import pytest
import sys
import os

# Add roampal-core root to path for imports
core_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if core_root not in sys.path:
    sys.path.insert(0, core_root)


@pytest.fixture(scope="session")
def benchmark_data_dir():
    """Return path to benchmark test data."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope="session")
def ab_test_data_dir():
    """Return path to A/B test data."""
    return os.path.join(os.path.dirname(__file__), 'ab_test_data')