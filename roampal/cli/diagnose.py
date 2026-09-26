"""Diagnose group (v0.6.0 Task 10): doctor + reembed.

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 10
split. Shared helpers (is_dev_mode, colors, logger) live in _common.py;
all model/backend imports stay function-local as in the monolith.
"""

import json
import os
import sys
from pathlib import Path

from roampal.cli._common import (
    BLUE,
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    logger,
    is_dev_mode,
)


def cmd_doctor(args):
    """Diagnose Roampal installation and configuration."""
    import asyncio

    print(f"{BOLD}Roampal Doctor - Diagnostics{RESET}\n")

    is_dev = is_dev_mode(args)
    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    print(f"Mode: {mode_str}\n")

    # v0.5.2: honor --profile flag and ROAMPAL_PROFILE env. Propagate --dev via
    # env so profile path resolution targets the Roampal_DEV base.
    if is_dev:
        os.environ["ROAMPAL_DEV"] = "1"
    profile_flag = getattr(args, "profile", None)
    if profile_flag:
        os.environ["ROAMPAL_PROFILE"] = profile_flag
    from roampal.profile_manager import (
        DEFAULT_PROFILE,
        ProfileNotFoundError,
        active_profile_name,
        active_profile_source,
        resolve_data_path,
    )
    profile_name = active_profile_name()
    profile_source = active_profile_source()

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def check_pass(msg):
        nonlocal checks_passed
        checks_passed += 1
        print(f"  {GREEN}[OK]{RESET} {msg}")

    def check_fail(msg):
        nonlocal checks_failed
        checks_failed += 1
        print(f"  {RED}[FAIL]{RESET} {msg}")

    def check_warn(msg):
        nonlocal checks_warned
        checks_warned += 1
        print(f"  {YELLOW}[WARN]{RESET} {msg}")

    # 1. Check Python version
    print(f"{BOLD}Python Environment:{RESET}")
    py_version = sys.version_info
    if py_version >= (3, 10):
        check_pass(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor} (need 3.10+)")

    check_pass(f"Executable: {sys.executable}")

    # 2. Check roampal version
    print(f"\n{BOLD}Roampal Version:{RESET}")
    try:
        from roampal import __version__

        check_pass(f"roampal v{__version__}")
    except ImportError as e:
        check_fail(f"Cannot import roampal: {e}")

    # 3. Check config files
    print(f"\n{BOLD}Configuration Files:{RESET}")
    home = Path.home()

    # Claude Code configs
    claude_dir = home / ".claude"
    if claude_dir.exists():
        check_pass(f"~/.claude exists")

        settings_path = claude_dir / "settings.json"
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                if "hooks" in settings:
                    check_pass("settings.json has hooks configured")
                else:
                    check_warn("settings.json missing hooks (run 'roampal init')")
            except json.JSONDecodeError as e:
                check_fail(f"settings.json invalid JSON: {e}")
        else:
            check_warn("settings.json not found (run 'roampal init')")

        # Check ~/.claude.json for user-scope MCP (v0.2.5+)
        claude_json_path = home / ".claude.json"
        if claude_json_path.exists():
            try:
                claude_json = json.loads(claude_json_path.read_text())
                if (
                    "mcpServers" in claude_json
                    and "roampal-core" in claude_json["mcpServers"]
                ):
                    check_pass("~/.claude.json has roampal-core server (user scope)")
                else:
                    check_warn(
                        "~/.claude.json missing roampal-core (run 'roampal init')"
                    )
            except json.JSONDecodeError as e:
                check_fail(f"~/.claude.json invalid JSON: {e}")
        else:
            check_warn("~/.claude.json not found (run 'roampal init')")

        # Check for old broken config location
        old_mcp_path = claude_dir / ".mcp.json"
        if old_mcp_path.exists():
            try:
                old_mcp = json.loads(old_mcp_path.read_text())
                if "mcpServers" in old_mcp and "roampal-core" in old_mcp["mcpServers"]:
                    check_warn(
                        "Old config at ~/.claude/.mcp.json (run 'roampal init' to migrate)"
                    )
            except Exception as e:
                logger.debug(f"Could not parse old .mcp.json: {e}")
    else:
        check_warn("~/.claude not found (Claude Code not installed?)")

    # Cursor configs
    cursor_dir = home / ".cursor"
    if cursor_dir.exists():
        check_pass("~/.cursor exists")

        mcp_path = cursor_dir / "mcp.json"
        if mcp_path.exists():
            try:
                mcp = json.loads(mcp_path.read_text())
                if "mcpServers" in mcp and "roampal-core" in mcp["mcpServers"]:
                    check_pass("mcp.json has roampal-core server")
                else:
                    check_warn(
                        "mcp.json missing roampal-core (run 'roampal init --cursor')"
                    )
            except json.JSONDecodeError as e:
                check_fail(f"mcp.json invalid JSON: {e}")
        else:
            check_warn("mcp.json not found (run 'roampal init --cursor')")

        # Cursor hooks (1.7+)
        hooks_path = cursor_dir / "hooks.json"
        if hooks_path.exists():
            try:
                hooks = json.loads(hooks_path.read_text())
                if "hooks" in hooks and "beforeSubmitPrompt" in hooks["hooks"]:
                    check_pass("hooks.json has beforeSubmitPrompt configured")
                else:
                    check_warn(
                        "hooks.json missing beforeSubmitPrompt (run 'roampal init --cursor')"
                    )
                if "hooks" in hooks and "stop" in hooks["hooks"]:
                    check_pass("hooks.json has stop hook configured")
                else:
                    check_warn(
                        "hooks.json missing stop hook (run 'roampal init --cursor')"
                    )
            except json.JSONDecodeError as e:
                check_fail(f"hooks.json invalid JSON: {e}")
        else:
            check_warn(
                "hooks.json not found (run 'roampal init --cursor' for Cursor 1.7+)"
            )

    # 4. Check data directory
    # v0.5.2: resolve via profile_manager so --profile / ROAMPAL_PROFILE /
    # active_profile file all influence the reported path. Falls back to the
    # system default when profile is 'default'.
    print(f"\n{BOLD}Data Directory:{RESET}")
    try:
        data_dir = Path(resolve_data_path(profile_name))
        if profile_name != DEFAULT_PROFILE:
            check_pass(f"Profile: {profile_name} (source: {profile_source})")
    except ProfileNotFoundError:
        check_fail(
            f"Profile {profile_name!r} is not registered (source: {profile_source})"
        )
        data_dir = None

    if data_dir is not None:
        if data_dir.exists():
            check_pass(f"Data directory exists: {data_dir}")
            chromadb_path = data_dir / "chromadb"
            if chromadb_path.exists():
                check_pass("ChromaDB directory exists")
            else:
                check_warn("ChromaDB not initialized yet (first use will create it)")
        else:
            check_warn(f"Data directory not created: {data_dir}")

    # 5. Check MCP server can start and list tools
    print(f"\n{BOLD}MCP Server:{RESET}")
    try:
        # Import the server module - this validates the code compiles
        import roampal.mcp.server as mcp_module

        check_pass("MCP server module loads")

        # Check that tools are defined (this catches syntax errors like false/False)
        # The actual list_tools is a decorated async function, so we check the TOOLS dict
        if hasattr(mcp_module, "TOOLS"):
            tool_count = len(mcp_module.TOOLS)
            check_pass(f"Tools defined: {tool_count}")
        else:
            # Try to find tools another way - check if the server starts
            check_pass("Server module valid (tools loaded at runtime)")

    except SyntaxError as e:
        check_fail(f"MCP server syntax error: {e}")
    except NameError as e:
        check_fail(f"MCP server name error: {e}")
    except Exception as e:
        check_fail(f"MCP server import failed: {e}")

    # 6. Check memory system initialization
    # v0.5.2: pass the same profile-resolved data path + profile name used by
    # the Data Directory check so doctor exercises the user's configured store.
    print(f"\n{BOLD}Memory System:{RESET}")
    if data_dir is None:
        check_warn("Skipping memory system init (unresolved data directory)")
    else:
        try:

            async def test_memory():
                from roampal.backend.modules.memory import UnifiedMemorySystem

                data_path = str(data_dir)
                profile_for_mem = (
                    profile_name if profile_name != DEFAULT_PROFILE else None
                )
                mem = UnifiedMemorySystem(
                    data_path=data_path, profile_name=profile_for_mem
                )
                await mem.initialize()
                return mem

            mem = asyncio.run(test_memory())
            check_pass("Memory system initializes")

            # Check collections
            if hasattr(mem, "collections") and mem.collections:
                collection_names = list(mem.collections.keys())
                check_pass(f"Collections: {', '.join(collection_names)}")
            else:
                check_warn("No collections found")

        except Exception as e:
            check_fail(f"Memory system failed: {e}")

    # 7. Check dependencies
    print(f"\n{BOLD}Dependencies:{RESET}")
    deps = [
        ("chromadb", "chromadb"),
        ("onnxruntime", "onnxruntime"),
        ("tokenizers", "tokenizers"),
        ("mcp", "mcp"),
        ("httpx", "httpx"),
        ("fastapi", "fastapi"),
    ]

    for import_name, display_name in deps:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "?")
            check_pass(f"{display_name} v{version}")
        except ImportError:
            check_fail(f"{display_name} not installed")

    # v0.4.5: Hint about legacy torch/sentence-transformers no longer needed
    try:
        import torch as _torch

        check_warn(
            f"torch v{_torch.__version__} installed but no longer required by roampal. "
            f"Reclaim ~420MB: pip uninstall torch sentence-transformers"
        )
    except ImportError:
        pass  # Good — torch not installed
    try:
        import sentence_transformers as _st

        check_warn(
            f"sentence-transformers installed but no longer required by roampal. "
            f"Reclaim space: pip uninstall sentence-transformers"
        )
    except ImportError:
        pass  # Good — sentence-transformers not installed

    # Summary
    print(f"\n{BOLD}{'=' * 50}{RESET}")
    total = checks_passed + checks_failed + checks_warned
    if checks_failed == 0:
        print(f"{GREEN}All checks passed!{RESET} ({checks_passed}/{total})")
        if checks_warned > 0:
            print(f"{YELLOW}{checks_warned} warnings{RESET}")
        return 0
    else:
        print(
            f"{RED}{checks_failed} checks failed{RESET}, {checks_passed} passed, {checks_warned} warnings"
        )
        print(f"\nRun {BLUE}roampal init{RESET} to fix configuration issues.")
        return 1


def cmd_reembed(args):
    """Re-embed stored vectors after an embedder model change (v0.5.9 Item 2a)."""
    import asyncio
    from pathlib import Path
    from roampal.backend.modules.memory import UnifiedMemorySystem
    from roampal.profile_manager import active_profile_name, resolve_data_path
    import roampal.backend.modules.memory.embedding_service as es

    profile_flag = getattr(args, "profile", None)
    if profile_flag:
        os.environ["ROAMPAL_PROFILE"] = profile_flag
    # This process is the intended migration runner: suppress the UMS
    # auto-scheduler so initialize() doesn't spawn a background task that
    # races this call for the single-runner lock (post-review fix 2026-08-27).
    os.environ["ROAMPAL_REEMBED_DISABLE"] = "1"
    profile_name = active_profile_name()
    data_path = Path(resolve_data_path(profile_name))

    async def do_reembed():
        mem = UnifiedMemorySystem(data_path=data_path)
        await mem.initialize()
        from roampal.backend.modules.memory.embedding_migrator import migrate_profile
        n = await migrate_profile(
            mem, data_path,
            model=es.HF_REPO, onnx_file=es.ONNX_FILE,
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            only_collection=getattr(args, "collection", None),
        )
        return n

    n = asyncio.run(do_reembed())
    if getattr(args, "dry_run", False):
        print(f"{YELLOW}Dry run complete for profile '{profile_name}': "
              f"{n} record(s) would be re-embedded.{RESET}")
    else:
        print(f"{GREEN}Re-embed complete for profile '{profile_name}': "
              f"{n} record(s) updated.{RESET}")
    return 0
