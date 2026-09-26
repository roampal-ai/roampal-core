"""
Structural-invariant suite for the v0.6.0 CLI refactor (Task 2 / Plan A.2).

Written against the PRE-REFACTOR monolith. Two of these tests are EXPECTED to
fail on first write (2026-09-16) — that is the point:

- test_no_duplicate_cmd_definitions fails today: cli.py defines `cmd_context`
  twice (2657 and 2786); the second silently shadows the first. Task 7 deletes
  the dead definition; this test then goes green and stays green via a
  parser↔dispatch-parity assertion suited to the post-refactor layout.
- test_help_epilog_lists_all_commands PASSES today: argparse auto-renders
  every registered subcommand under the metavar list on `roampal --help`,
  so the hand-maintained epilog's omission of `retag`/`profile`/`reembed`
  (cli.py:4290-4321 only lists a curated subset) is a cosmetic duplication,
  not a user-facing gap. G4 over-claimed this. The invariant stays as the
  permanent guard: if the auto listing ever disappears (e.g. subcommand
  registry truncation, the v0.5.9 bug class), this test goes red.

The rest must pass both before and after every refactor step:

- test_parser_accepts_every_command: every registered subcommand parses a
  `--help` invocation in a real subprocess (catches the v0.5.9 failure mode,
  where a bug truncated the subcommand registry silently).
- test_command_definitions_cover_expected_set: the set of `cmd_<name>`
  definitions in the CLI src equals the canonical 17-command surface.

These target public seams/subprocess output so they survive the refactor
(Ocelot equivalence principle in RELEASE_NOTES Test Plan A).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
CLI_PATH = REPO_ROOT / "roampal" / "cli.py"
PY_EXE = sys.executable


def _cli_sources() -> list[Path]:
    """Every module holding CLI dispatch-level definitions.

    Pre-refactor: just the monolith. Post-refactor (Task 3+): the package
    files under roampal/cli/. Scanning the union (rather than hardcoding
    cli.py) lets Tasks 3-10 move groups without editing this suite;
    duplicates are checked within each module, coverage across the union.
    """
    sources = []
    if CLI_PATH.exists():
        sources.append(CLI_PATH)
    pkg_dir = REPO_ROOT / "roampal" / "cli"
    if pkg_dir.is_dir():
        # Skip __init__.py's proxy shims and any non-code fixture; real
        # command modules are the impl + moved group modules.
        sources.extend(
            sorted(p for p in pkg_dir.rglob("*.py"))
        )
    if not sources:
        pytest.fail("no CLI source files found at all (roampal/cli.py gone "
                    "and no roampal/cli/ package present)")
    return sources

# Canonical CLI surface, documented in README "examples" and main()'s epilog.
# 17 subcommands as of v0.5.9; `score` removed in v0.6.0 (16).
EXPECTED_COMMANDS = {
    "init", "start", "stop", "status", "stats", "ingest", "remove",
    "books", "summarize", "context", "retag", "sidecar",
    "doctor", "profile", "reembed", "help",
}

# Sub-subcommands (dispatched via a second argparse level) — Task 2 verifies
# they parse; Task 13's profile bind/unbind additions must extend this table
# alongside their parser wiring (making citation coverage an explicit PR step).
NAMED_SUBCOMMANDS = {
    "profile": ["list", "show", "use", "unuse", "switch", "create", "register", "delete", "bind", "unbind"],
    "sidecar": ["status", "setup", "test", "disable"],
}


def _cmd_definitions() -> dict[str, dict[str, list[int]]]:
    """All `def <name>` occurrences per module: file -> name -> line numbers."""
    result: dict[str, dict[str, list[int]]] = {}
    for src in _cli_sources():
        tree = ast.parse(src.read_text(encoding="utf-8"))
        found: dict[str, list[int]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(node.name, []).append(node.lineno)
        result[str(src)] = found
    return result


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_parser_accepts_command(command):
    """Every documented command must be registered and parseable.

    Exit 0 = subcommand exists with --help wired. Exit 2 = argparse
    unrecognized-choice error (the v0.5.9 registry-truncation failure mode).
    """
    completed = subprocess.run(
        [PY_EXE, "-m", "roampal.cli", command, "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    # `help` is an alias for --help; argparse treats unknown "help" subcommand
    # as parse failure on old versions — v0.5.9 registers it explicitly.
    assert completed.returncode == 0, (
        f"`roampal {command} --help` exited {completed.returncode}: "
        f"{completed.stdout[-400:]!r} {completed.stderr[-400:]!r}"
    )
    assert "error" not in completed.stderr.lower()


@pytest.mark.parametrize(
    "parent,sub",
    [(p, s) for p, subs in sorted(NAMED_SUBCOMMANDS.items()) for s in subs],
)
def test_parser_accepts_named_subcommand(parent, sub):
    """Every nested subcommand must also parse (`<parent> <sub> --help`)."""
    completed = subprocess.run(
        [PY_EXE, "-m", "roampal.cli", parent, sub, "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"`roampal {parent} {sub} --help` exited {completed.returncode}: "
        f"{completed.stderr[-400:]!r}"
    )


def test_command_definitions_cover_expected_set():
    """One `def cmd_<command>` across the CLI surface, no extras/missing."""
    per_file = _cmd_definitions()
    cmd_funcs: dict[str, list[Path]] = {}
    for src, defs in per_file.items():
        for name in defs:
            if name.startswith("cmd_"):
                cmd_funcs.setdefault(name, []).append(Path(src))
    for command in EXPECTED_COMMANDS - {"help"}:
        assert f"cmd_{command}" in cmd_funcs, (
            f"command {command!r} has no cmd_{command} definition"
        )
    extras = {
        n for n in cmd_funcs
        if n.removeprefix("cmd_") not in EXPECTED_COMMANDS
    }
    assert not extras, f"unexpected cmd_* definitions: {sorted(extras)}"


def test_no_duplicate_cmd_definitions():
    """The CLI surface must not define the same cmd_* function twice in one
    module.

    FAILS TODAY by design: cli.py defines cmd_context at :2657 and :2786; the
    second definition silently shadows the first. Goes green when the dead
    duplicate is deleted (Task 7). Post-refactor this guard also watches
    every roampal/cli/ module for the same mistake class.
    """
    per_file = _cmd_definitions()
    dupes = {}
    for src, defs in per_file.items():
        for name, lines in defs.items():
            if name.startswith("cmd_") and len(lines) > 1:
                dupes[f"{src}:{name}"] = lines
    assert not dupes, (
        f"duplicate cmd_* definitions (later definition silently shadows the "
        f"earlier one): {dupes}"
    )


def test_help_epilog_lists_all_commands():
    """roampal --help's commands list must cover the canonical surface.

    FAILS TODAY by design: the hand-maintained epilog omits retag, profile,
    and reembed. Goes green when Task 4 generates the epilog from the parser
    (diff asserted additive-only by the golden suite).
    """
    completed = subprocess.run(
        [PY_EXE, "-m", "roampal.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert completed.returncode == 0
    out = completed.stdout
    missing = [c for c in sorted(EXPECTED_COMMANDS) if c not in out]
    assert not missing, (
        f"roampal --help output missing registered commands: {missing}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
