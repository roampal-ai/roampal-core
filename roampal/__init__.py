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

__version__ = "0.4.9"


# Lazy imports: chromadb/onnxruntime are heavy and crash in minimal
# environments (e.g., Glama Docker inspection).  The MCP server only needs
# stdlib + mcp, so we defer the backend imports until someone actually asks
# for UnifiedMemorySystem or MemoryConfig.
def __getattr__(name):
    if name == "UnifiedMemorySystem":
        from roampal.backend.modules.memory import UnifiedMemorySystem

        return UnifiedMemorySystem
    if name == "MemoryConfig":
        from roampal.backend.modules.memory import MemoryConfig

        return MemoryConfig
    raise AttributeError(f"module 'roampal' has no attribute {name}")


__all__ = [
    "UnifiedMemorySystem",
    "MemoryConfig",
    "__version__",
]
