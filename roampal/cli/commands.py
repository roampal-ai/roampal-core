"""Dispatch table for the v0.6.0 CLI refactor (Task 3).

Single source of truth mapping a parsed subcommand to its handler.
Replaces the monolith's hand-written if/elif chain in
`_monolith_impl.main()`; the structural invariant suite
(`test_cli_dispatch_table.py`) asserts parser-set == dispatch-set, which is
what would have caught the v0.5.9 registry-truncation bug.

Table shape: `{command: (handler, uses_exit_code)}` —
- handler: the cmd_* function; `None` for "help" (prints parser help)
- uses_exit_code: True for handlers returning an int exit code; False for
  handlers whose return is ignored (treated as success).

Handlers are resolved LAZILY through the roampal.cli package proxy at
dispatch time (not at import) — handlers relocated to group modules
(Tasks 5-10, complete) resolve the same way.
"""

# "command" -> ("cmd_<name>" in _monolith_impl, uses_exit_code)
_TABLE_SPEC = {
    "init": ("cmd_init", True),
    "start": ("cmd_start", False),
    "stop": ("cmd_stop", True),
    "status": ("cmd_status", True),
    "stats": ("cmd_stats", True),
    "ingest": ("cmd_ingest", True),
    "remove": ("cmd_remove", True),
    "books": ("cmd_books", False),
    "summarize": ("cmd_summarize", False),
    "context": ("cmd_context", False),
    "retag": ("cmd_retag", False),
    "sidecar": ("cmd_sidecar", True),  # v0.6.0 Task 39: setup exits 1 on errors
    "doctor": ("cmd_doctor", True),
    "profile": ("cmd_profile", True),
    "reembed": ("cmd_reembed", True),
    # "help" prints parser help — handler is None (see _dispatch_table)
}


def _dispatch_table():
    """Resolve the dispatch table at call time (handlers not importable at
    module import because the monolith module is large).

    Handlers resolve through the roampal.cli package proxy (getattr on the
    cached impl module first, then any moved group module), so a command
    keeps dispatching whether its cmd_* still lives in _monolith_impl or
    has been relocated to its group module.
    """
    import roampal.cli as pkg

    table = {"help": (None, False)}
    for command, (attr, uses_exit_code) in _TABLE_SPEC.items():
        table[command] = (getattr(pkg, attr), uses_exit_code)
    return table


def dispatch_table():
    """Public accessor; returns a new resolution each call (cheap)."""
    return _dispatch_table()
