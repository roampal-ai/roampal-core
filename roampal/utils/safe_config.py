"""Safe JSON config read/write (v0.6.0 Task 40).

The rule, in the user's words: we never wipe people's settings, ever — we
only adjust Roampal's own entries. Every config file `roampal init` or the
sidecar path touches must be read with an encoding-independent reader that
ABORTS on a parse error (leaving the file byte-for-byte identical), and
written through a crash-safe writer that backs the previous bytes up first.
`_safe_write_opencode_config` (cli/sidecar) delegates here so OpenCode keeps
its old call surface.

The reader uses ``utf-8-sig``: locale codecs (cp1252 on Windows) turned any
non-ASCII byte into a parse failure — and until 0.5.9 the failure path then
OVERWROTE the user's file with a fresh config. The writer prunes its own
backups (`<exact filename>.bak-<14 digits>`, newest 3 kept); old backup piles
from 0.5.3-0.5.9 grow unbounded (17 on the dev machine) and are trimmed on
the next write.
"""

import datetime
import shutil
from pathlib import Path

from roampal.utils.atomic_json import write_json_atomic


class ConfigReadError(Exception):
    """A config file exists but cannot be used as JSON config data.

    Raised only for files that EXIST — a missing file just means "empty
    config" and must never block init.

    Attributes:
        path: the file that failed
        reason: human-readable why (read error / decode error / parse error /
            JSON scalar or list instead of an object)
    """

    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")


def read_json_config(path: Path) -> dict:
    """Read a JSON config file, refusing to guess.

    Returns ``{}`` if the file does not exist. Reads as UTF-8 and accepts a
    BOM. Raises :class:`ConfigReadError` — and changes nothing on disk — when
    the file exists but cannot be read (locked, permissions), is not valid
    JSON, is not valid text in its encoding, or is not a JSON object (a list
    or scalar is not a config).

    Callers must treat this as "stop, change nothing, explain": no backup is
    created, the file stays byte-identical.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ConfigReadError(path, f"not valid UTF-8 text: {e}") from e
    except OSError as e:
        raise ConfigReadError(path, f"cannot read file: {e}") from e

    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigReadError(path, f"invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ConfigReadError(path, f"not a JSON object (got {type(data).__name__})")
    return data


def write_json_config(path: Path, data: dict, *, max_backups: int = 3) -> None:
    """Write a JSON config file atomically, with a timestamped backup first
    and old backups of THAT file pruned to the newest ``max_backups``.

    Backup: `<name>.bak-YYYYmmddHHMMSS` next to the file, a byte copy of what
    was on disk before this write (only when the file already exists).
    Write: atomic via :func:`roampal.utils.atomic_json.write_json_atomic`
    (temp file + ``os.replace``, UTF-8). Prune: only files named
    ``<exact filename>.bak-<14 digits>`` in the same folder, ordered by the
    timestamp in the name (never by mtime), oldest pruned first; prune errors
    are ignored so a cleanup hiccup never fails the write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = path.parent / f"{path.name}.bak-{ts}"
        try:
            shutil.copy2(str(path), str(backup_path))
        except OSError:
            backup_path = None
        _prune_backups(path, max_backups)
    else:
        backup_path = None

    try:
        write_json_atomic(path, data, indent=2)
    except Exception:
        # Leave no backup that misrepresents a failed write: the original
        # file is untouched, so the backup is redundant clutter at best.
        if backup_path is not None:
            try:
                backup_path.unlink()
            except OSError:
                pass
        raise


def _prune_backups(path: Path, max_backups: int) -> None:
    """Delete the oldest ``*.bak-<14 digits>`` copies of `path` beyond
    `max_backups`. Matches only `<exact filename>.bak-<14 digits>` in the
    same folder — never a user's manual ``.bak-manual`` or any other file.
    """
    if max_backups < 0:
        return
    prefix = f"{path.name}.bak-"
    pattern_len = len("YYYYmmddHHMMSS")  # 14 digits
    try:
        candidates = []
        for entry in path.parent.iterdir():
            if not entry.name.startswith(prefix):
                continue
            stamped = entry.name[len(prefix):]
            if len(stamped) != pattern_len or not stamped.isdigit():
                continue
            if not entry.is_file():
                continue
            candidates.append(entry)
        # The just-created backup sets the floor nothing real can beat (it
        # has NOW's timestamp). Sort by timestamp, keep the newest N; delete
        # the rest oldest-first.
        candidates.sort(key=lambda p: p.name[len(prefix):])
        for entry in candidates[: max(0, len(candidates) - max_backups)]:
            try:
                entry.unlink()
            except OSError:
                pass
    except OSError:
        pass
