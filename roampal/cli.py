"""
Roampal CLI - One command install for AI coding tools

Usage:
    pip install roampal
    roampal init          # Configure Claude Code / Cursor
    roampal start         # Start the memory server
    roampal status        # Check server status
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Port configuration - DEV and PROD use different ports to avoid collision
PROD_PORT = 27182
DEV_PORT = 27183


def get_data_dir(dev: bool = False) -> Path:
    """Get the data directory path. DEV uses separate directory to avoid collision with PROD."""
    if dev:
        # DEV mode uses separate directory
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', str(Path.home()))
            return Path(appdata) / "Roampal_DEV" / "data"
        elif sys.platform == 'darwin':  # macOS
            return Path.home() / "Library" / "Application Support" / "Roampal_DEV" / "data"
        else:  # Linux
            return Path.home() / ".local" / "share" / "roampal_dev" / "data"
    else:
        # PROD mode
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', str(Path.home()))
            return Path(appdata) / "Roampal" / "data"
        elif sys.platform == 'darwin':  # macOS
            return Path.home() / "Library" / "Application Support" / "Roampal" / "data"
        else:  # Linux
            return Path.home() / ".local" / "share" / "roampal" / "data"


def print_banner():
    """Print Roampal banner."""
    print(f"""
{BLUE}{BOLD}+---------------------------------------------------+
|                   ROAMPAL                         |
|     Persistent Memory for AI Coding Tools         |
+---------------------------------------------------+{RESET}
""")


def check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    try:
        import urllib.request
        from roampal import __version__

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            # Compare versions (simple tuple comparison works for semver)
            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            return (update_available, __version__, latest)
    except Exception:
        # Fail silently - don't block CLI on network issues
        try:
            from roampal import __version__
            return (False, __version__, __version__)
        except Exception:
            return (False, "unknown", "unknown")


def print_update_notice():
    """Print update notice if newer version available. Non-blocking."""
    update_available, current, latest = check_for_updates()
    if update_available:
        print(f"{YELLOW}⚠️  Update available: {latest} (you have {current}){RESET}")
        print(f"    Run: pip install --upgrade roampal\n")


def cmd_init(args):
    """Initialize Roampal for the current environment."""
    print_banner()
    print_update_notice()
    print(f"{BOLD}Initializing Roampal...{RESET}\n")

    # Detect environment
    home = Path.home()
    claude_code_dir = home / ".claude"
    cursor_dir = home / ".cursor"

    detected = []
    if claude_code_dir.exists():
        detected.append("claude-code")
    if cursor_dir.exists():
        detected.append("cursor")

    if not detected:
        print(f"{YELLOW}No AI coding tools detected.{RESET}")
        print("Roampal works with:")
        print("  - Claude Code (https://claude.com/claude-code)")
        print("  - Cursor (https://cursor.sh)")
        print("\nInstall one of these tools first, then run 'roampal init' again.")
        return

    print(f"{GREEN}Detected: {', '.join(detected)}{RESET}\n")

    # Configure each detected tool
    for tool in detected:
        if tool == "claude-code":
            configure_claude_code(claude_code_dir)
        elif tool == "cursor":
            configure_cursor(cursor_dir)

    # Create data directory
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"{GREEN}Created data directory: {data_dir}{RESET}")

    print(f"""
{GREEN}{BOLD}Roampal initialized successfully!{RESET}

{BOLD}Next step:{RESET}
  {BLUE}Restart Claude Code{RESET} and start chatting!
  The MCP server auto-starts - no manual server needed.

{BOLD}How it works:{RESET}
  - Hooks inject relevant memories into your context automatically
  - The AI learns what works and what doesn't via outcome scoring
  - You see your original message; the AI sees your message + context + scoring prompt

{BOLD}Optional commands:{RESET}
  - {BLUE}roampal ingest myfile.pdf{RESET} - Add documents to memory
  - {BLUE}roampal stats{RESET} - Show memory statistics
  - {BLUE}roampal status{RESET} - Check server status
""")


def configure_claude_code(claude_dir: Path):
    """Configure Claude Code hooks, MCP, and permissions."""
    print(f"{BOLD}Configuring Claude Code...{RESET}")

    # Create settings.json with hooks and permissions
    settings_path = claude_dir / "settings.json"

    # Load existing settings or create new
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except:
            pass

    # Configure hooks - Claude Code expects nested format with type/command
    python_exe = sys.executable.replace("\\", "\\\\")  # Escape backslashes for JSON
    settings["hooks"] = {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python_exe} -m roampal.hooks.user_prompt_submit_hook"
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python_exe} -m roampal.hooks.stop_hook"
                    }
                ]
            }
        ]
    }

    # Configure permissions to auto-allow roampal MCP tools
    # This prevents the user from being spammed with permission prompts
    if "permissions" not in settings:
        settings["permissions"] = {}
    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []

    # Add roampal MCP tools to allow list (using roampal-core server name)
    roampal_perms = [
        "mcp__roampal-core__search_memory",
        "mcp__roampal-core__add_to_memory_bank",
        "mcp__roampal-core__update_memory",
        "mcp__roampal-core__archive_memory",
        "mcp__roampal-core__get_context_insights",
        "mcp__roampal-core__record_response",
        "mcp__roampal-core__score_response"
    ]

    for perm in roampal_perms:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"  {GREEN}Created settings: {settings_path}{RESET}")
    print(f"  {GREEN}  - UserPromptSubmit hook (injects scoring + memories){RESET}")
    print(f"  {GREEN}  - Stop hook (enforces record_response){RESET}")
    print(f"  {GREEN}  - Auto-allowed MCP permissions{RESET}")

    # Create MCP configuration (server name matches permission prefix)
    mcp_config_path = claude_dir / ".mcp.json"
    mcp_config = {
        "mcpServers": {
            "roampal-core": {
                "command": sys.executable,
                "args": ["-m", "roampal.mcp.server"],
                "env": {}
            }
        }
    }

    # Merge with existing if present
    if mcp_config_path.exists():
        try:
            existing = json.loads(mcp_config_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["roampal-core"] = mcp_config["mcpServers"]["roampal-core"]
            else:
                existing["mcpServers"] = mcp_config["mcpServers"]
            mcp_config = existing
        except:
            pass

    mcp_config_path.write_text(json.dumps(mcp_config, indent=2))
    print(f"  {GREEN}Created MCP config: {mcp_config_path}{RESET}")

    # Also create local .mcp.json in current working directory
    # Some Claude Code setups look for project-level config
    local_mcp_path = Path.cwd() / ".mcp.json"
    local_mcp_config = {
        "mcpServers": {
            "roampal-core": {
                "command": sys.executable,
                "args": ["-m", "roampal.mcp.server"],
                "env": {}
            }
        }
    }

    # Merge with existing local config if present
    if local_mcp_path.exists():
        try:
            existing = json.loads(local_mcp_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["roampal-core"] = local_mcp_config["mcpServers"]["roampal-core"]
            else:
                existing["mcpServers"] = local_mcp_config["mcpServers"]
            local_mcp_config = existing
        except:
            pass

    local_mcp_path.write_text(json.dumps(local_mcp_config, indent=2))
    print(f"  {GREEN}Created local MCP config: {local_mcp_path}{RESET}")

    print(f"  {GREEN}Claude Code configured!{RESET}\n")


def configure_cursor(cursor_dir: Path):
    """Configure Cursor MCP."""
    print(f"{BOLD}Configuring Cursor...{RESET}")

    # Cursor uses a different MCP config location
    mcp_config_path = cursor_dir / "mcp.json"
    mcp_config = {
        "mcpServers": {
            "roampal": {
                "command": "python",
                "args": ["-m", "roampal.mcp.server"]
            }
        }
    }

    # Merge with existing if present
    if mcp_config_path.exists():
        try:
            existing = json.loads(mcp_config_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["roampal"] = mcp_config["mcpServers"]["roampal"]
            else:
                existing.update(mcp_config)
            mcp_config = existing
        except:
            pass

    mcp_config_path.write_text(json.dumps(mcp_config, indent=2))
    print(f"  {GREEN}Created MCP config: {mcp_config_path}{RESET}")

    print(f"  {GREEN}Cursor configured!{RESET}\n")
    print(f"  {YELLOW}Note: Cursor hooks coming in future version.{RESET}")
    print(f"  {YELLOW}For now, MCP tools provide memory access.{RESET}\n")


def cmd_start(args):
    """Start the Roampal server."""
    print_banner()

    # Determine port based on mode (DEV=27183, PROD=27182)
    # User can override with --port
    is_dev = args.dev
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port != PROD_PORT else default_port  # Use default unless explicitly overridden

    # Handle dev mode - uses Roampal_DEV folder
    if is_dev:
        os.environ["ROAMPAL_DEV"] = "1"
        data_path = get_data_dir(dev=True)
        print(f"{YELLOW}DEV MODE{RESET} - Isolated from production")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port} (PROD uses {PROD_PORT})\n")
    else:
        data_path = get_data_dir(dev=False)
        print(f"{GREEN}PROD MODE{RESET}")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port}\n")

    print(f"{BOLD}Starting Roampal server...{RESET}\n")

    host = args.host or "127.0.0.1"

    print(f"Server: http://{host}:{port}")
    print(f"Hooks endpoint: http://{host}:{port}/api/hooks/get-context")
    print(f"Health check: http://{host}:{port}/api/health")
    print(f"\nPress Ctrl+C to stop.\n")

    # Import and start server
    from roampal.server.main import start_server
    start_server(host=host, port=port)


def cmd_status(args):
    """Check Roampal server status."""
    print_banner()
    print_update_notice()

    import httpx

    host = args.host or "127.0.0.1"
    # Use dev port if --dev flag, otherwise prod port (unless explicitly overridden)
    is_dev = getattr(args, 'dev', False)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port

    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    url = f"http://{host}:{port}/api/health"

    try:
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            print(f"Server Status ({mode_str}): {GREEN}RUNNING{RESET}")
            print(f"  Port: {port}")
            print(f"  Memory initialized: {data.get('memory_initialized', False)}")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
        else:
            print(f"{RED}Server returned error: {response.status_code}{RESET}")
    except httpx.ConnectError:
        print(f"Server Status ({mode_str}): {YELLOW}NOT RUNNING{RESET}")
        start_cmd = "roampal start --dev" if is_dev else "roampal start"
        print(f"\nStart with: {start_cmd}")
    except Exception as e:
        print(f"{RED}Error checking status: {e}{RESET}")


def cmd_stats(args):
    """Show memory statistics."""
    print_banner()
    print_update_notice()

    import httpx

    host = args.host or "127.0.0.1"
    # Use dev port if --dev flag
    is_dev = getattr(args, 'dev', False)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port

    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    url = f"http://{host}:{port}/api/stats"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"{BOLD}Memory Statistics ({mode_str}):{RESET}\n")
            print(f"Data path: {data.get('data_path', 'N/A')}")
            print(f"Port: {port}")
            print(f"\nCollections:")
            for name, info in data.get("collections", {}).items():
                count = info.get("count", 0)
                print(f"  {name}: {count} items")
        else:
            print(f"{RED}Error getting stats: {response.status_code}{RESET}")
    except httpx.ConnectError:
        start_cmd = "roampal start --dev" if is_dev else "roampal start"
        print(f"{YELLOW}Server not running. Start with: {start_cmd}{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


def cmd_ingest(args):
    """Ingest a document into the books collection."""
    import asyncio
    import httpx

    print_banner()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"{RED}File not found: {file_path}{RESET}")
        return

    # Read file content
    print(f"{BOLD}Ingesting:{RESET} {file_path.name}")

    try:
        # Detect file type and read content
        suffix = file_path.suffix.lower()
        content = None
        title = args.title or file_path.stem

        if suffix == '.txt':
            content = file_path.read_text(encoding='utf-8')
        elif suffix == '.md':
            content = file_path.read_text(encoding='utf-8')
        elif suffix == '.pdf':
            # Try to use pypdf if available
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                print(f"  Extracted {len(reader.pages)} pages from PDF")
            except ImportError:
                print(f"{RED}PDF support requires pypdf: pip install pypdf{RESET}")
                return
        else:
            # Try to read as text
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                print(f"{RED}Cannot read file as text. Supported: .txt, .md, .pdf{RESET}")
                return

        if not content or len(content.strip()) == 0:
            print(f"{YELLOW}File is empty or could not be read{RESET}")
            return

        print(f"  Content length: {len(content):,} characters")

        # Handle dev mode - use correct port and data path
        is_dev = args.dev
        data_path = None
        if is_dev:
            data_path = str(get_data_dir(dev=True))
            print(f"  {YELLOW}DEV MODE{RESET} - Using: {data_path}")

        # Try to use running server first (data immediately searchable)
        # Use dev port if --dev flag
        host = "127.0.0.1"
        port = DEV_PORT if is_dev else PROD_PORT
        server_url = f"http://{host}:{port}/api/ingest"

        try:
            response = httpx.post(
                server_url,
                json={
                    "content": content,
                    "title": title,
                    "source": str(file_path),
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap
                },
                timeout=300.0  # 5 min timeout for very large files
            )

            if response.status_code == 200:
                data = response.json()
                print(f"\n{GREEN}Success!{RESET} Stored '{title}' in {data['chunks']} chunks")
                print(f"  (via running server - immediately searchable)")
                print(f"\nThe document is now searchable via:")
                print(f"  - search_memory(query, collections=['books'])")
                print(f"  - Automatic context injection via hooks")
                return
            else:
                print(f"  {YELLOW}Server error, falling back to direct storage...{RESET}")

        except httpx.ConnectError:
            print(f"  {YELLOW}Server not running, using direct storage...{RESET}")
            print(f"  {YELLOW}(Restart server for immediate searchability){RESET}")

        # Fallback: Store directly (requires server restart to be searchable)
        async def do_ingest():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            mem = UnifiedMemorySystem(data_path=data_path)
            await mem.initialize()

            doc_ids = await mem.store_book(
                content=content,
                title=title,
                source=str(file_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )

            return doc_ids

        doc_ids = asyncio.run(do_ingest())

        print(f"\n{GREEN}Success!{RESET} Stored '{title}' in {len(doc_ids)} chunks")
        print(f"\nThe document is now searchable via:")
        print(f"  - search_memory(query, collections=['books'])")
        print(f"  - Automatic context injection via hooks")
        print(f"\n{YELLOW}Note: Restart 'roampal start' for immediate searchability{RESET}")

    except Exception as e:
        print(f"{RED}Error ingesting file: {e}{RESET}")
        raise


def cmd_remove(args):
    """Remove a book from the books collection."""
    import asyncio
    import httpx

    print_banner()
    title = args.title
    print(f"{BOLD}Removing book:{RESET} {title}\n")

    # Try running server first
    host = "127.0.0.1"
    port = 27182
    server_url = f"http://{host}:{port}/api/remove-book"

    try:
        response = httpx.post(
            server_url,
            json={"title": title},
            timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("removed", 0) > 0:
                print(f"{GREEN}Success!{RESET} Removed '{title}' ({data['removed']} chunks)")
                if data.get("cleaned_kg_refs", 0) > 0:
                    print(f"  Cleaned {data['cleaned_kg_refs']} Action KG references")
            else:
                print(f"{YELLOW}No book found with title '{title}'{RESET}")
            return
        else:
            print(f"  {YELLOW}Server error, falling back to direct removal...{RESET}")

    except httpx.ConnectError:
        print(f"  {YELLOW}Server not running, using direct removal...{RESET}")

    # Fallback: Remove directly
    async def do_remove():
        from roampal.backend.modules.memory import UnifiedMemorySystem

        mem = UnifiedMemorySystem()
        await mem.initialize()
        return await mem.remove_book(title)

    result = asyncio.run(do_remove())

    if result.get("removed", 0) > 0:
        print(f"\n{GREEN}Success!{RESET} Removed '{title}' ({result['removed']} chunks)")
        if result.get("cleaned_kg_refs", 0) > 0:
            print(f"  Cleaned {result['cleaned_kg_refs']} Action KG references")
    else:
        print(f"{YELLOW}No book found with title '{title}'{RESET}")


def cmd_books(args):
    """List all books in the books collection."""
    import asyncio
    import httpx

    print_banner()
    print(f"{BOLD}Books in memory:{RESET}\n")

    # Try running server first
    host = "127.0.0.1"
    port = 27182
    server_url = f"http://{host}:{port}/api/books"

    books = None

    try:
        response = httpx.get(server_url, timeout=10.0)
        if response.status_code == 200:
            books = response.json().get("books", [])
    except httpx.ConnectError:
        pass

    # Fallback: List directly
    if books is None:
        async def do_list():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            mem = UnifiedMemorySystem()
            await mem.initialize()
            return await mem.list_books()

        books = asyncio.run(do_list())

    if not books:
        print(f"{YELLOW}No books found.{RESET}")
        print(f"\nAdd books with: roampal ingest <file>")
        return

    for book in books:
        print(f"  {GREEN}{book['title']}{RESET}")
        print(f"    Source: {book.get('source', 'unknown')}")
        print(f"    Chunks: {book.get('chunk_count', 0)}")
        if book.get('created_at'):
            print(f"    Added: {book['created_at'][:10]}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Roampal - Persistent Memory for AI Coding Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize Roampal for Claude Code / Cursor")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the memory server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    start_parser.add_argument("--port", type=int, default=27182, help="Server port")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode - use separate data directory")

    # status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    status_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    status_parser.add_argument("--dev", action="store_true", help="Check dev server status")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    stats_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    stats_parser.add_argument("--dev", action="store_true", help="Show dev server stats")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into the books collection")
    ingest_parser.add_argument("file", help="File to ingest (.txt, .md, .pdf)")
    ingest_parser.add_argument("--title", help="Document title (defaults to filename)")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    ingest_parser.add_argument("--dev", action="store_true", help="Dev mode - use separate data directory")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a book from the books collection")
    remove_parser.add_argument("title", help="Title of the book to remove")

    # books command
    books_parser = subparsers.add_parser("books", help="List all books in memory")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "books":
        cmd_books(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
