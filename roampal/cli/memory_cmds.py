"""Memory group (v0.6.0 Task 7): ingest/remove/books/summarize/context/retag.

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 7
split. The duplicate first `cmd_context` definition (a truncated stub whose
name was shadowed by the later definition) is NOT carried — it never
executed in the monolith; see the task-doc note. Cross-impl deps
(is_dev_mode, get_data_dir, get_port, _check_sidecar_configured)
import module-level from the impl / _common — safe: the package __init__
loads the impl before group modules. `_parse_last_exchange`/
`_find_latest_transcript` moved to roampal/cli/scoring.py in Task 8 (removed
with `roampal score` in the v0.6.0 smoke review);
`_check_sidecar_configured` moved to roampal/cli/_common.py in Task 9
(G3).
"""

import json
import os
import sys

from pathlib import Path

# httpx stays function-local inside each function as in the monolith
# (module-level import retained for patch targets / parity with impl).
import httpx  # noqa: F401

from roampal.cli._common import (
    BLUE,
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    _check_sidecar_configured,
    is_dev_mode,
    get_data_dir,
    get_port,
    profile_headers,
    _is_interactive,
    PROD_PORT,
    DEV_PORT,
)


def _impl():
    import importlib

    return importlib.import_module("roampal.cli._monolith_impl")


def cmd_ingest(args):
    """Ingest a document into the books collection."""
    import asyncio
    import httpx

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

        if suffix == ".txt":
            content = file_path.read_text(encoding="utf-8")
        elif suffix == ".md":
            content = file_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
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
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(
                    f"{RED}Cannot read file as text. Supported: .txt, .md, .pdf{RESET}"
                )
                return

        if not content or len(content.strip()) == 0:
            print(f"{YELLOW}File is empty or could not be read{RESET}")
            return

        print(f"  Content length: {len(content):,} characters")

        is_dev = is_dev_mode(args)
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
                    "chunk_overlap": args.chunk_overlap,
                },
                headers=profile_headers(),
                timeout=300.0,  # 5 min timeout for very large files
            )

            if response.status_code == 200:
                data = response.json()
                print(
                    f"\n{GREEN}Success!{RESET} Stored '{title}' in {data['chunks']} chunks"
                )
                print(f"  (via running server - immediately searchable)")
                print(f"\nThe document is now searchable via:")
                print(f"  - search_memory(query, collections=['books'])")
                print(f"  - Automatic context injection via hooks")
                return
            else:
                print(
                    f"  {YELLOW}Server error, falling back to direct storage...{RESET}"
                )

        except httpx.ConnectError:
            print(f"  {YELLOW}Server not running, using direct storage...{RESET}")
            print(f"  {YELLOW}(Restart server for immediate searchability){RESET}")

        # Fallback: Store directly (requires server restart to be searchable)
        async def do_ingest():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            # Same profile the server path would have used (dev keeps its
            # explicit dev data dir). Pre-fix this ignored the profile and
            # wrote to the system default data dir.
            from roampal.profile_manager import profile_header_value

            mem = UnifiedMemorySystem(
                data_path=data_path,
                profile_name=None if data_path else profile_header_value(),
            )
            await mem.initialize()

            doc_ids = await mem.store_book(
                content=content,
                title=title,
                source=str(file_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

            return doc_ids

        doc_ids = asyncio.run(do_ingest())

        print(f"\n{GREEN}Success!{RESET} Stored '{title}' in {len(doc_ids)} chunks")
        print(f"\nThe document is now searchable via:")
        print(f"  - search_memory(query, collections=['books'])")
        print(f"  - Automatic context injection via hooks")
        print(
            f"\n{YELLOW}Note: Restart 'roampal start' for immediate searchability{RESET}"
        )

    except Exception as e:
        print(f"{RED}Error ingesting file: {e}{RESET}")
        raise


def cmd_remove(args):
    """Remove a book from the books collection."""
    import asyncio
    import httpx

    title = args.title
    print(f"{BOLD}Removing book:{RESET} {title}\n")

    # v0.2.0: Handle --dev flag by setting env var
    if getattr(args, "dev", False):
        os.environ["ROAMPAL_DEV"] = "1"
        print(f"  {YELLOW}DEV mode{RESET}")

    # Try running server first
    host = "127.0.0.1"
    port = get_port(args)
    server_url = f"http://{host}:{port}/api/remove-book"

    try:
        response = httpx.post(
            server_url, json={"title": title}, headers=profile_headers(), timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("removed", 0) > 0:
                print(
                    f"{GREEN}Success!{RESET} Removed '{title}' ({data['removed']} chunks)"
                )
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
        print(
            f"\n{GREEN}Success!{RESET} Removed '{title}' ({result['removed']} chunks)"
        )
        if result.get("cleaned_kg_refs", 0) > 0:
            print(f"  Cleaned {result['cleaned_kg_refs']} Action KG references")
    else:
        print(f"{YELLOW}No book found with title '{title}'{RESET}")


def cmd_books(args):
    """List all books in the books collection."""
    import asyncio
    import httpx

    print(f"{BOLD}Books in memory:{RESET}\n")

    # Try running server first
    host = "127.0.0.1"
    port = get_port()
    server_url = f"http://{host}:{port}/api/books"

    books = None

    try:
        response = httpx.get(server_url, headers=profile_headers(), timeout=10.0)
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
        if book.get("created_at"):
            print(f"    Added: {book['created_at'][:10]}")
        print()


def cmd_summarize(args):
    """Summarize existing long memories using the sidecar."""
    import httpx

    if not _check_sidecar_configured():
        return

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"
    max_chars = args.max_chars
    collections_to_scan = (
        [args.collection] if args.collection else ["working", "history"]
    )
    dry_run = args.dry_run

    print(f"{BOLD}Summarizing memories over {max_chars} characters{RESET}")
    if dry_run:
        print(f"{YELLOW}DRY RUN -- no changes will be made{RESET}")

    from roampal.sidecar_service import get_backend_info

    backend = get_backend_info()
    print(f"Backend: {GREEN}{backend}{RESET}")

    if backend == "none available":
        print(f"\n{RED}No summarization backend available.{RESET}")
        print()
        print(f"  {BOLD}What was checked:{RESET}")
        print(
            f"    ROAMPAL_SUMMARIZE_MODEL  {RED}not set{RESET}  (opt-in to main model)"
        )
        print(
            f"    ROAMPAL_SIDECAR_URL      {RED}not set{RESET}  (custom OpenAI-compatible API)"
        )
        print(f"    ANTHROPIC_API_KEY         {RED}not set{RESET}  (Haiku direct)")
        print(
            f"    Zen free models           {RED}unavailable{RESET} (CLI only works inside OpenCode)"
        )
        print(
            f"    Ollama                    {RED}not running{RESET} (http://localhost:11434)"
        )
        print(
            f"    LM Studio                 {RED}not running{RESET} (http://localhost:1234)"
        )
        print(f"    claude CLI                {RED}not found{RESET}")
        print()
        print(f"  {BOLD}Options (pick one):{RESET}")
        print(
            f"    1. {GREEN}Install Ollama{RESET} (recommended, free, local, ~14s/memory)"
        )
        print(f"       ollama.com -> install -> ollama pull llama3.2:3b")
        print(f"    2. {GREEN}Set ANTHROPIC_API_KEY{RESET} (~$0.001/memory via Haiku)")
        print(f"       export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"    3. {GREEN}Set ROAMPAL_SUMMARIZE_MODEL{RESET} (use your main model)")
        print(f"       export ROAMPAL_SUMMARIZE_MODEL=claude-sonnet-4-5-20250929")
        print(f"       export ANTHROPIC_API_KEY=sk-ant-...")
        print(
            f"    4. {GREEN}Set ROAMPAL_SIDECAR_URL{RESET} (any OpenAI-compatible endpoint)"
        )
        print(f"       export ROAMPAL_SIDECAR_URL=https://api.groq.com/openai/v1")
        print(f"       export ROAMPAL_SIDECAR_KEY=your-key")
        print(f"       export ROAMPAL_SIDECAR_MODEL=llama-3.3-70b-versatile")
        print()
        print(
            f"  {YELLOW}Note:{RESET} OpenCode users don't need this -- memories auto-summarize"
        )
        print(f"  during normal use via the plugin (1 per exchange, zero config).")
        return
    elif "claude -p" in backend:
        print(
            f"{YELLOW}Warning: claude -p is slow (~30-60s/memory) and unreliable (~40% success rate).{RESET}"
        )
        print(
            f"For better results, install Ollama (ollama.com) or set ROAMPAL_SIDECAR_URL.{RESET}"
        )

    # Small model disclaimer
    if "Ollama" in backend or "LM Studio" in backend:
        print(
            f"{YELLOW}Note: Local models may struggle with very long memories (>5000 chars).{RESET}"
        )
        print(
            f"Those will be skipped if summarization fails. Use a larger model or API for best results.{RESET}"
        )

    print()

    # Check server is running
    try:
        httpx.get(f"{base_url}/api/health", timeout=5.0)
    except Exception:
        print(f"{RED}Server not running. Start with: roampal start{RESET}")
        return

    from roampal.sidecar_service import summarize_only

    # Scan all collections first to count available memories
    all_candidates = []  # list of (coll_name, mem) tuples
    for coll_name in collections_to_scan:
        try:
            resp = httpx.post(
                f"{base_url}/api/search",
                json={
                    "query": "",
                    "collections": [coll_name],
                    "limit": 500,
                    "sort_by": "recency",
                },
                headers=profile_headers(),
                timeout=30.0,
            )

            if resp.status_code != 200:
                print(f"{RED}Failed to search {coll_name}: {resp.status_code}{RESET}")
                continue

            results = resp.json().get("results", [])
            for r in results:
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                # Skip already-summarized memories (have summarized_at timestamp)
                if metadata.get("summarized_at"):
                    continue
                if len(content) > max_chars:
                    all_candidates.append((coll_name, r))

        except Exception as e:
            print(f"{RED}Error scanning {coll_name}: {e}{RESET}")

    if not all_candidates:
        print(f"{GREEN}No memories need summarization.{RESET}")
        return

    # Show count and prompt for limit
    print(f"{BOLD}{len(all_candidates)} memories available for summarization{RESET}")

    # Group by collection for display
    by_coll = {}
    for coll_name, mem in all_candidates:
        by_coll.setdefault(coll_name, []).append(mem)
    for coll_name, mems in by_coll.items():
        print(f"  {coll_name}: {len(mems)} memories over {max_chars} chars")

    # Determine batch limit
    batch_limit = args.limit
    if batch_limit is not None and batch_limit <= 0:
        print(f"Nothing to do (limit={batch_limit}).")
        return
    if batch_limit is None and not dry_run:
        if _is_interactive():
            # Interactive: ask user how many
            print()
            try:
                user_input = input(
                    f"How many to summarize? (Enter for all {len(all_candidates)}, or a number): "
                ).strip()
                if user_input:
                    batch_limit = int(user_input)
                    if batch_limit <= 0:
                        print(f"Cancelled.")
                        return
                    print(f"Summarizing {batch_limit} memories...")
                else:
                    print(f"Summarizing all {len(all_candidates)} memories...")
            except (ValueError, EOFError):
                print(f"Summarizing all {len(all_candidates)} memories...")
        else:
            # Non-interactive: summarize all
            print(f"Summarizing all {len(all_candidates)} memories...")

    print()

    total_summarized = 0
    total_skipped = 0
    batch_count = 0

    for i, (coll_name, mem) in enumerate(all_candidates):
        if batch_limit is not None and batch_count >= batch_limit:
            remaining = len(all_candidates) - i
            print(
                f"  {YELLOW}Batch limit reached ({batch_limit}). {remaining} remaining -- run again to continue.{RESET}"
            )
            break

        content = mem.get("content", "")
        doc_id = mem.get("id", mem.get("doc_id", ""))

        if dry_run:
            print(
                f"  [{i + 1}/{len(all_candidates)}] {doc_id}: {len(content)} chars -> would summarize"
            )
            if i < 3:  # Show sample previews for first 3
                print(f"    {YELLOW}(generating preview...){RESET}")
                try:
                    summary = summarize_only(content)
                    if summary:
                        print(f"    Preview: {summary[:100]}...")
                except Exception as e:
                    print(f"    {YELLOW}Preview failed: {e}{RESET}")
            total_summarized += 1
            batch_count += 1
            continue

        # Summarize
        summary = summarize_only(content)
        if not summary:
            print(
                f"  {YELLOW}[{i + 1}] Failed to summarize {doc_id} ({len(content)} chars), skipping{RESET}"
            )
            total_skipped += 1
            continue

        # v0.6.0 Task 37: never silently truncate — check against the memory_limits
        # rule and skip (don't store) if over the hard cap. The server also enforces
        # this at /api/memory/update-content, but we fail fast here with a clear message.
        from roampal.memory_limits import check_length

        length_err = check_length("summary", summary)
        if length_err:
            print(
                f"  {YELLOW}[{i + 1}] Skipped {doc_id}: summary too long ({len(summary)} chars) — {length_err}{RESET}"
            )
            total_skipped += 1
            continue

        # Extract noun_tags from summary (skip if memory already has tags)
        from roampal.sidecar_service import extract_tags

        existing_metadata = mem.get("metadata", {})
        has_tags = bool(existing_metadata.get("noun_tags"))
        noun_tags = [] if has_tags else (extract_tags(summary) or [])

        # Update the memory with the summary + noun_tags
        try:
            update_payload = {
                "doc_id": doc_id,
                "collection": coll_name,
                "new_content": summary,
            }
            if noun_tags:
                update_payload["noun_tags"] = noun_tags

            update_resp = httpx.post(
                f"{base_url}/api/memory/update-content",
                json=update_payload,
                headers=profile_headers(),
                timeout=10.0,
            )

            if update_resp.status_code == 200:
                total_summarized += 1
                batch_count += 1
                tag_info = (
                    f", tags={noun_tags}"
                    if noun_tags
                    else (", tags=existing" if has_tags else "")
                )
                print(
                    f"  [{i + 1}/{len(all_candidates)}] {doc_id}: {len(content)} -> {len(summary)} chars{tag_info}"
                )
            else:
                total_skipped += 1
                print(
                    f"  {YELLOW}[{i + 1}] Failed to update {doc_id}: {update_resp.status_code}{RESET}"
                )
        except Exception as e:
            total_skipped += 1
            print(f"  {RED}[{i + 1}] Error updating {doc_id}: {e}{RESET}")

    print()
    if dry_run:
        print(f"{YELLOW}DRY RUN: Would summarize {total_summarized} memories{RESET}")
    else:
        print(f"{GREEN}Summarized {total_summarized} memories{RESET}")
        if total_skipped > 0:
            print(
                f"{YELLOW}Skipped {total_skipped} (failed or too long for model){RESET}"
            )


def cmd_retag(args):
    """Upgrade memory tags using LLM (improves retrieval quality)."""
    import httpx

    if not _check_sidecar_configured():
        return

    print(f"{BOLD}Memory Cleanup — Retag{RESET}")
    print(f"""
  Uses your scoring model to re-extract tags on existing memories.
  Existing tags are replaced with fresh ones. Memories without tags get them added.

  Use --limit to test on a few first, --dry-run to preview without changing anything.
  """)

    # Get user confirmation
    try:
        confirm = input(f"\n{YELLOW}Continue? [y/N]: {RESET}").strip().lower()
        if confirm != "y":
            print(f"{YELLOW}Cancelled.{RESET}")
            return
    except (EOFError, KeyboardInterrupt):
        print(f"\n{YELLOW}Cancelled.{RESET}")
        return

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"

    # Prepare request
    request_data = {
        "collection": args.collection,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "model": args.model,
    }

    mode_str = (
        f"{YELLOW}DRY RUN — preview only, nothing will change{RESET}"
        if args.dry_run
        else "Live — tags will be replaced"
    )
    print(f"\n  Collection: {args.collection}")
    if args.limit:
        print(f"  Limit:      {args.limit} memories")
    print(f"  Mode:       {mode_str}")
    if args.model:
        print(f"  Model:      {args.model}")
    print(f"\n  Sending each memory to your scoring model for fresh tags...")

    try:
        import sys
        import threading

        # Spinner in background thread
        stop_spinner = threading.Event()

        def _spinner():
            chars = "|/-\\"
            i = 0
            while not stop_spinner.is_set():
                sys.stdout.write(f"\r  Extracting tags {chars[i % len(chars)]}")
                sys.stdout.flush()
                stop_spinner.wait(0.3)
                i += 1
            sys.stdout.write("\r  Done!      \n")
            sys.stdout.flush()

        spinner_thread = threading.Thread(target=_spinner, daemon=True)
        spinner_thread.start()

        resp = httpx.post(
            f"{base_url}/api/retag",
            json=request_data,
            headers=profile_headers(),
            timeout=300.0,  # 5 minute timeout for large collections
        )

        stop_spinner.set()
        spinner_thread.join(timeout=1)

        if resp.status_code == 200:
            result = resp.json()
            processed = result.get("processed", 0)
            tags_added = result.get("tags_added", 0)
            errs = result.get("errors", 0)

            print(f"\n{GREEN}Done.{RESET}")
            print(f"  Memories processed: {processed}")
            print(f"  New tags extracted: {tags_added}")
            if errs:
                print(f"  Errors: {errs}")

            # Show sample changes
            samples = result.get("sample_updates", [])
            if samples:
                print(f"\n  Examples:")
                for s in samples[:3]:
                    old = ", ".join(s.get("old_tags", [])) or "(none)"
                    new = ", ".join(s.get("new_tags", []))
                    print(f"    {old}  ->  {new}")

            if args.dry_run:
                print(f"\n  {YELLOW}This was a dry run — nothing was changed.{RESET}")
                print(f"  Run without --dry-run to apply.")
            elif processed > 0:
                print(f"\n  {GREEN}Tags updated. Retrieval should improve.{RESET}")

        else:
            print(f"{RED}Tag upgrade failed: {resp.status_code}{RESET}")
            try:
                error_data = resp.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"  Response: {resp.text[:200]}")

    except httpx.RequestError as e:
        print(f"{RED}Failed to connect to server: {e}{RESET}")
        print(f"  Make sure the server is running: {BLUE}roampal status{RESET}")
    except Exception as e:
        print(f"{RED}Unexpected error: {e}{RESET}")


def cmd_context(args):
    """Output recent exchange context for platform hooks."""
    import httpx

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"

    if args.recent_exchanges:
        try:
            # v0.6.0 Task 30: this command doubles as the SessionStart hook
            # (matchers compact/startup/clear). Claude Code pipes hook input
            # containing the session id — tell the server to drop its
            # per-conversation injection record so the memory block this
            # hook is about to print re-shows FULL text after compaction
            # (the pre-compaction turns' pointers no longer match what
            # post-compaction context holds). Best effort BOTH ways: an
            # unreadable stdin (non-hook invocation, pytest capture) must
            # never break the memory block below.
            try:
                if not sys.stdin.isatty():
                    try:
                        stdin_data = json.loads(sys.stdin.read() or "{}")
                    except ValueError:
                        stdin_data = {}
                    if stdin_data.get("session_id"):
                        try:
                            httpx.post(
                                f"{base_url}/api/hooks/session-reset",
                                json={"conversation_id": stdin_data["session_id"]},
                                headers=profile_headers(),
                                timeout=3.0,
                            )
                        except Exception:
                            pass  # server down / reset unreachable — keep going
            except Exception:
                pass  # stdin unreadable — this command still prints the block

            # Search for recent exchange summaries
            # v0.4.7: Fetch most recent exchange summaries (no semantic query — pure recency).
            # Empty query triggers _search_all path which now includes _add_recency_metadata.
            # Search all three collections since summaries can be promoted across tiers.
            # v0.6.0 Task 14: hook cwd = project dir → attach X-Roampal-Profile
            # (resolved through the one helper) so the server binds THIS
            # project. Round 2 Task 18: the client ALWAYS names its profile —
            # the helper now returns "default" verbatim instead of None, so
            # the header (explicit default included) is never omitted.
            headers = {}
            from roampal.profile_manager import profile_header_value

            hdr = profile_header_value()
            if hdr:
                headers["X-Roampal-Profile"] = hdr
            resp = httpx.post(
                f"{base_url}/api/search",
                json={
                    "query": "",
                    "collections": ["working", "history", "patterns"],
                    "limit": 4,
                    "sort_by": "recency",
                    "metadata_filters": {"memory_type": "exchange_summary"},
                },
                headers=headers,
                timeout=10.0,
            )

            if resp.status_code != 200:
                return

            results = resp.json().get("results", [])
            if not results:
                return

            # Format output matching _format_mem() from unified_memory_system.py
            print("RECENT EXCHANGES (last 4):")
            for r in results:
                content = (
                    r.get("text", "")
                    or r.get("content", "")
                    or r.get("metadata", {}).get("text", "")
                )
                metadata = r.get("metadata", {})
                doc_id = r.get("id", "")
                collection = r.get("collection", metadata.get("collection", "working"))
                recency = metadata.get("recency", "")
                wilson = r.get("wilson_score", metadata.get("wilson_score", 0)) or 0
                uses = r.get("uses", metadata.get("uses", 0)) or 0
                last_outcome = metadata.get("last_outcome", "")
                tag_parts = []
                if recency:
                    tag_parts.append(recency)
                tag_parts.append(collection)
                if uses > 0:
                    tag_parts.append(f"wilson:{wilson:.0%}")
                    tag_parts.append(f"used:{uses}x")
                    if last_outcome:
                        tag_parts.append(f"last:{last_outcome}")
                id_str = f" [id:{doc_id}]" if doc_id else ""
                # v0.6.0 Task 35(2): display cut raised 200 -> 300 to match
                # the summarizer ask; one source in memory_limits so every
                # recent-exchanges formatter agrees. Task 37 chunk B: the cut
                # is word-boundary and clipped entries end in "…" (spec table
                # row: RECENT EXCHANGES — word boundary, marked "…", ID kept).
                from roampal.memory_limits import RECENT_EXCHANGES_DISPLAY_CUT
                cut = RECENT_EXCHANGES_DISPLAY_CUT
                summary = content or "No content"
                if content and len(content) > cut:
                    clipped = content[:cut].rsplit(" ", 1)[0].rstrip()
                    summary = (clipped if clipped else content[:cut]) + "…"
                print(f"• {summary}{id_str} ({', '.join(tag_parts)})")

        except httpx.ConnectError:
            pass  # Server not running, fail silently for hooks
        except Exception:
            pass  # Fail silently for hooks


