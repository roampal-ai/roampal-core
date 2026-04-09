"""
Tag Migration — Backfill noun_tags on existing memories for v0.4.5.

Uses REGEX ONLY for extraction — fast, no sidecar dependency, works for all users.
Resumable: progress saved after every batch. Interruption-safe.

Two trigger points:
- `roampal init --force` -> runs synchronously with progress bar
- First server startup -> runs as background asyncio.create_task()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tag_service import extract_tags_regex

logger = logging.getLogger(__name__)


class TagMigration:
    """
    Backfill noun_tags metadata on all existing memories.

    Non-destructive: uses update_fragment_metadata() which updates metadata
    without re-embedding. ChromaDB vectors are untouched.
    """

    BATCH_SIZE = 20

    def __init__(
        self,
        collections: Dict[str, Any],
        data_dir: Path,
    ):
        self.collections = collections
        self.data_dir = data_dir
        self._migrations_dir = data_dir / "migrations"
        self._flag_file = self._migrations_dir / "v045_tags_complete"
        self._progress_file = self._migrations_dir / "v045_tags_progress.json"

    def needs_migration(self) -> bool:
        """Check if tag backfill has been completed."""
        return not self._flag_file.exists()

    async def run_migration(self, show_progress: bool = True) -> Dict[str, int]:
        """
        Backfill noun_tags on all memories without them.

        Non-blocking, resumable. Processes in batches of BATCH_SIZE.
        Writes progress after each batch so interruptions can resume.

        Args:
            show_progress: Print progress to stdout (for CLI)

        Returns:
            {"total_processed": int, "total_tagged": int, "errors": int}
        """
        self._migrations_dir.mkdir(parents=True, exist_ok=True)
        progress = self._load_progress()
        stats = {"total_processed": 0, "total_tagged": 0, "errors": 0}

        for coll_name, adapter in self.collections.items():
            completed_ids = set(progress.get(coll_name, []))

            try:
                collection_obj = adapter.collection
                if not collection_obj:
                    continue

                items = collection_obj.get(
                    limit=10000,
                    include=["metadatas", "documents"]
                )
            except Exception as e:
                logger.warning(f"Failed to read {coll_name} for migration: {e}")
                continue

            ids = items.get("ids", [])
            metadatas = items.get("metadatas", [])
            documents = items.get("documents", [])

            # Filter to memories needing tags
            needs_tags = []
            for i, doc_id in enumerate(ids):
                if doc_id in completed_ids:
                    continue
                meta = metadatas[i] if i < len(metadatas) else {}
                if meta and meta.get("noun_tags"):
                    completed_ids.add(doc_id)
                    continue
                content = documents[i] if i < len(documents) else ""
                needs_tags.append((doc_id, content, meta))

            if show_progress and needs_tags:
                print(f"  {coll_name}: {len(needs_tags)} memories to tag")

            # Process in batches
            for batch_start in range(0, len(needs_tags), self.BATCH_SIZE):
                batch = needs_tags[batch_start:batch_start + self.BATCH_SIZE]

                for doc_id, content, meta in batch:
                    try:
                        text = content or (meta or {}).get("text", "") or (meta or {}).get("content", "")
                        tags = extract_tags_regex(text)

                        if tags:
                            adapter.update_fragment_metadata(
                                doc_id, {"noun_tags": json.dumps(tags)}
                            )
                            stats["total_tagged"] += 1

                        stats["total_processed"] += 1
                        completed_ids.add(doc_id)
                    except Exception as e:
                        stats["errors"] += 1
                        logger.warning(f"Tag extraction failed for {doc_id}: {e}")

                # Save progress after each batch
                progress[coll_name] = list(completed_ids)
                self._save_progress(progress)

                if show_progress:
                    total_done = sum(len(v) for v in progress.values())
                    print(f"    Progress: {total_done} memories processed")

        # Mark migration complete
        self._flag_file.write_text(datetime.now().isoformat())

        # Clean up progress file
        if self._progress_file.exists():
            try:
                self._progress_file.unlink()
            except Exception:
                pass

        if show_progress:
            print(f"  Tagged {stats['total_tagged']} memories ({stats['errors']} errors)")

        logger.info(
            f"Tag migration complete: {stats['total_tagged']} tagged, "
            f"{stats['total_processed']} processed, {stats['errors']} errors"
        )
        return stats

    def _load_progress(self) -> Dict[str, List[str]]:
        """Load resume state from progress file."""
        if self._progress_file.exists():
            try:
                return json.loads(self._progress_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_progress(self, progress: Dict[str, List[str]]):
        """Save progress for resume support."""
        try:
            self._progress_file.write_text(json.dumps(progress))
        except OSError as e:
            logger.warning(f"Failed to save migration progress: {e}")
