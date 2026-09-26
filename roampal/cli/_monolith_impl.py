"""
Roampal CLI - One command install for AI coding tools

Usage:
    pip install roampal
    roampal init          # Configure Claude Code / OpenCode
    roampal start         # Start the memory server
    roampal stop          # Stop the memory server
    roampal status        # Check server status
    roampal doctor        # Diagnose installation issues
"""

import argparse
import sys

# v0.6.0 Task 10: the impl is now the thin entry point - argparse wiring in
# main() plus dispatch. All cmd_* groups and shared helpers live in
# roampal.cli.* group modules / _common.
from roampal.cli._common import set_no_input as _set_no_input_flag
from roampal.cli.commands import dispatch_table


def main():
    """Main CLI entry point."""
    from roampal import __version__

    parser = argparse.ArgumentParser(
        prog="roampal",
        description="roampal - Persistent memory for AI coding assistants",
        allow_abbrev=False,
        epilog="""commands:
  Setup:
    init              Initialize for Claude Code / OpenCode
    doctor            Diagnose installation and configuration

  Server:
    start             Start the memory server
    stop              Stop the memory server
    status            Check server status (--json for scripting)
    stats             Show memory statistics (--json for scripting)

  Memory:
    ingest <file>     Add documents to books collection
    books             List all ingested books
    remove <title>    Remove a book by title
    summarize         Summarize long memories (retroactive cleanup)

  Scoring (OpenCode):
    sidecar status    Check scoring model configuration
    sidecar setup     Configure scoring model
    sidecar test      Test scoring model response format
    sidecar disable   Remove scoring model configuration

  Advanced:
    context           Output recent exchange context

examples:
  roampal init --claude-code    Set up for Claude Code
  roampal init --no-input       Non-interactive setup (CI/scripts)
  roampal status --json         Machine-readable status
  roampal stats --json          Machine-readable statistics""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"roampal {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize Roampal for Claude Code / OpenCode"
    )
    init_parser.add_argument(
        "--dev",
        action="store_true",
        help="Initialize for DEV mode (separate data directory)",
    )
    init_parser.add_argument(
        "--claude-code",
        action="store_true",
        help="Configure Claude Code only (skip auto-detect)",
    )
    init_parser.add_argument(
        "--cursor", action="store_true", help="Configure Cursor only (skip auto-detect)"
    )
    init_parser.add_argument(
        "--opencode",
        action="store_true",
        help="Configure OpenCode only (skip auto-detect)",
    )
    init_parser.add_argument(
        "--force", "-f", action="store_true", help="Force overwrite existing config"
    )
    init_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Non-interactive mode (skip all prompts, use defaults)",
    )
    init_parser.add_argument(
        "--scope",
        choices=["user", "project", "both"],
        default=None,
        help="Config scope: user (global), project (local), or both. Default: auto-detect.",
    )

    # start command
    start_parser = subparsers.add_parser("start", help="Start the memory server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    start_parser.add_argument("--port", type=int, default=27182, help="Server port")
    start_parser.add_argument(
        "--dev", action="store_true", help="Dev mode - use separate data directory"
    )
    start_parser.add_argument(
        "--profile",
        default=None,
        help="Named memory profile (v0.5.1). See 'roampal profile --help'.",
    )

    # stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the memory server")
    stop_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 27182 prod, 27183 dev)",
    )
    stop_parser.add_argument("--dev", action="store_true", help="Stop dev server")

    # status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    status_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 27182 prod, 27183 dev)",
    )
    status_parser.add_argument(
        "--dev", action="store_true", help="Check dev server status"
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON (for scripting)",
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    stats_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 27182 prod, 27183 dev)",
    )
    stats_parser.add_argument(
        "--dev", action="store_true", help="Show dev server stats"
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON (for scripting)",
    )

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a document into the books collection"
    )
    ingest_parser.add_argument("file", help="File to ingest (.txt, .md, .pdf)")
    ingest_parser.add_argument("--title", help="Document title (defaults to filename)")
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Characters per chunk (default: 1000)",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)",
    )
    ingest_parser.add_argument(
        "--dev", action="store_true", help="Dev mode - use separate data directory"
    )

    # remove command
    remove_parser = subparsers.add_parser(
        "remove", help="Remove a book from the books collection"
    )
    remove_parser.add_argument("title", help="Title of the book to remove")
    remove_parser.add_argument("--dev", action="store_true", help="Use dev server")

    # books command
    books_parser = subparsers.add_parser("books", help="List all books in memory")

    # summarize command (v0.3.6)
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize existing long memories"
    )
    summarize_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )
    summarize_parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Summarize memories over this length (default: 400)",
    )
    summarize_parser.add_argument(
        "--collection",
        choices=["working", "history"],
        help="Target specific collection",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max memories to summarize (for batching)",
    )
    summarize_parser.add_argument("--dev", action="store_true", help="Use dev server")
    summarize_parser.add_argument("--port", type=int, default=None, help="Server port")

    # context command (v0.3.6)
    context_parser = subparsers.add_parser(
        "context", help="Output recent exchange context"
    )
    context_parser.add_argument(
        "--recent-exchanges",
        action="store_true",
        help="Output last 4 exchange summaries",
    )
    context_parser.add_argument("--dev", action="store_true", help="Use dev server")
    context_parser.add_argument("--port", type=int, default=None, help="Server port")

    # sidecar command (v0.3.7)
    sidecar_parser = subparsers.add_parser(
        "sidecar", help="Configure sidecar scoring model"
    )
    sidecar_sub = sidecar_parser.add_subparsers(dest="sidecar_command")
    sidecar_setup = sidecar_sub.add_parser(
        "setup", help="Auto-detect and configure a scorer"
    )
    # v0.6.0 Task 39: non-interactive choices (LLM-driven installs, scripts).
    # Each records the user's choice; --list shows every choice first.
    sidecar_setup.add_argument(
        "--list", action="store_true",
        help="List every scoring choice and the command that selects it (no changes)",
    )
    sidecar_setup.add_argument(
        "--json", action="store_true", help="With --list: machine-readable output"
    )
    sidecar_setup.add_argument(
        "--model", default=None,
        help="Use this detected model (from --list), or the model name with --url",
    )
    sidecar_setup.add_argument(
        "--url", default=None, help="Custom OpenAI-compatible endpoint (needs --model)"
    )
    sidecar_setup.add_argument(
        "--key-env", dest="key_env", default=None,
        help="With --url: read the API key from this environment variable",
    )
    sidecar_setup.add_argument(
        "--go", default=None, metavar="MODEL", help="Use this OpenCode Go model"
    )
    sidecar_setup.add_argument(
        "--zen", action="store_true",
        help="Opt in to free Zen cloud models (exchange text is sent to opencode.ai)",
    )
    sidecar_setup.add_argument(
        "--auto", action="store_true",
        help="Use the recommended detected LOCAL model (never a cloud or paid one)",
    )
    sidecar_setup.add_argument(
        "--scope",
        choices=["user", "project", "both"],
        default=None,
        help="Config scope: user (global), project (local), or both. Default: auto-detect.",
    )
    sidecar_sub.add_parser("test", help="Test sidecar with sample exchange")
    # status and disable get --scope for scope-aware config reading
    sidecar_status_cmd = sidecar_sub.add_parser(
        "status", help="Show current sidecar configuration"
    )
    sidecar_status_cmd.add_argument(
        "--scope",
        choices=["user", "project", "both"],
        default=None,
        help="Config scope: user (global), project (local), or both. Default: auto-detect.",
    )
    sidecar_disable = sidecar_sub.add_parser("disable", help="Remove sidecar fallback")
    sidecar_disable.add_argument(
        "--scope",
        choices=["user", "project", "both"],
        default=None,
        help="Config scope to clear: user (global), project (local), or both. Default: auto-detect.",
    )

    # init command — add --scope flag
    retag_parser = subparsers.add_parser(
        "retag", help="Re-extract tags on memories using sidecar LLM"
    )
    retag_parser.add_argument(
        "--collection",
        choices=["working", "history", "patterns", "memory_bank", "all"],
        default="all",
        help="Which collection to retag (default: all)",
    )
    retag_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of memories to retag (for testing)",
    )
    retag_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying memories",
    )
    retag_parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use (default: uses configured sidecar)",
    )
    retag_parser.add_argument("--dev", action="store_true", help="Use dev server")
    retag_parser.add_argument("--port", type=int, default=None, help="Server port")

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor", help="Diagnose installation and configuration"
    )
    doctor_parser.add_argument(
        "--dev", action="store_true", help="Check dev mode configuration"
    )
    doctor_parser.add_argument(
        "--profile", default=None, help="Named memory profile (v0.5.1)"
    )

    # profile command (v0.5.1) — manage named memory profiles
    profile_parser = subparsers.add_parser(
        "profile", help="Manage named memory profiles"
    )
    profile_sub = profile_parser.add_subparsers(dest="profile_command")
    profile_sub.add_parser("list", help="List registered profiles")
    profile_sub.add_parser("show", help="Show the active profile and its path")
    p_use = profile_sub.add_parser(
        "use", help="Persistently set the active profile (global default)"
    )
    p_use.add_argument("name", help="Profile name (use 'default' to clear)")
    profile_sub.add_parser(
        "unuse", help="Clear the persisted active profile (revert to default)"
    )
    p_switch = profile_sub.add_parser(
        "switch",
        help="Same as `use` (the running server keeps running; sessions switch on their next exchange)",
    )
    p_switch.add_argument("name", help="Profile name (use 'default' to revert)")
    p_create = profile_sub.add_parser("create", help="Create a new profile")
    p_create.add_argument("name", help="Profile name (e.g. 'work', 'project-a')")
    p_register = profile_sub.add_parser(
        "register", help="Register an existing directory as a named profile"
    )
    p_register.add_argument("name", help="Profile name")
    p_register.add_argument(
        "--path", required=True, help="Existing data directory to register"
    )
    p_delete = profile_sub.add_parser(
        "delete", help="Remove a profile from the registry"
    )
    p_delete.add_argument("name", help="Profile name to delete")
    p_delete.add_argument(
        "--destroy-data",
        action="store_true",
        help="Also delete the profile's data directory (irreversible)",
    )
    # v0.6.0 Task 13: directory bindings (Item 3)
    p_bind = profile_sub.add_parser(
        "bind", help="Bind a directory (cwd by default) to a profile"
    )
    p_bind.add_argument("name", help="Profile name")
    p_bind.add_argument(
        "--path", default=None, help="Directory to bind (default: cwd)"
    )
    p_unbind = profile_sub.add_parser(
        "unbind", help="Remove a directory binding"
    )
    p_unbind.add_argument(
        "name", nargs="?", default=None, help="Verify the binding points to this profile"
    )
    p_unbind.add_argument(
        "--path", default=None, help="Directory to unbind (default: cwd)"
    )

    # help command (alias for --help)
    reembed_parser = subparsers.add_parser(
        "reembed", help="Re-embed stored vectors after an embedder model change"
    )
    reembed_parser.add_argument("--profile", help="Target profile (default: active)")
    reembed_parser.add_argument("--collection", help="Only re-embed this collection")
    reembed_parser.add_argument("--force", action="store_true",
                                help="Re-embed even if metadata says up to date")
    reembed_parser.add_argument("--dry-run", action="store_true",
                                help="Report what would change without writing")

    subparsers.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    # Set global non-interactive flag from --no-input
    # (v0.6.0 Task 3: via roampal.cli._common accessors)
    if getattr(args, "no_input", False):
        _set_no_input_flag(True)

    # Dispatch commands — v0.6.0 Task 3: the if/elif chain is replaced by the
    # dispatch table in roampal/cli/commands.py. Semantics identical:
    #   - functions return exit codes (None → success → 0)
    #   - "help" prints parser help rather than calling a cmd_ function
    #   - unknown command → parser help (argparse makes this unreachable in
    #     practice, but the behavior is preserved verbatim)
    exit_code = 0
    func, uses_exit_code = dispatch_table().get(args.command, (None, False))
    if func is None:
        parser.print_help()
    elif uses_exit_code:
        exit_code = func(args) or 0
    else:
        func(args)
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
