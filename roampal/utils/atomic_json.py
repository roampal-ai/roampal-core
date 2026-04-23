"""Atomic JSON write with temp-file + rename dance. Crash-safe
for any filesystem that supports os.replace() atomicity (Linux and
Windows NTFS/ReFS both qualify).
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes to a sibling .tmp file first, then os.replace()s into place.
    If any exception is raised during the write, the temp file is
    removed and the original `path` is left untouched.

    Args:
        path: destination path
        data: JSON-serializable object
        indent: passed through to json.dump (default 2 for readability)
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        dump_kwargs = {"indent": indent}
        if indent is None:
            dump_kwargs["separators"] = (",", ":")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp_path, path)  # atomic on POSIX and NTFS
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
