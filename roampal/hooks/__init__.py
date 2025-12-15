"""
Roampal Hooks Module

Manages session tracking and hook scripts for enforced outcome scoring.

Hook Scripts:
- user_prompt_submit_hook.py: Injects scoring prompt + memories BEFORE LLM
- stop_hook.py: Stores exchange, enforces record_response AFTER LLM

Usage:
  roampal init  # Configures hooks and permissions automatically
"""

from .session_manager import SessionManager

__all__ = ["SessionManager"]
