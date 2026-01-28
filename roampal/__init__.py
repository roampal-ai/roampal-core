"""
Roampal Core - Persistent Memory for AI Coding Tools

One command install. AI coding tools get persistent memory.

Usage:
    pip install roampal
    roampal init
    roampal start

How it works:
    1. Roampal hooks intercept messages BEFORE the AI sees them
    2. Relevant memories are automatically injected
    3. The AI learns what works and what doesn't
    4. You see your original message; the AI sees your message + context
"""

__version__ = "0.3.1"

from roampal.backend.modules.memory import (
    UnifiedMemorySystem,
    MemoryConfig,
)

__all__ = [
    "UnifiedMemorySystem",
    "MemoryConfig",
    "__version__",
]
