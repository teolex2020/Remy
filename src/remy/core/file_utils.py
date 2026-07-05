"""
File utilities — atomic writes and safe I/O for long-running server deployments.

os.replace() is atomic on POSIX and near-atomic on Windows (NTFS MoveFileEx).
"""

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path | str, content: str, encoding: str = "utf-8"):
    """Write content to file atomically using temp file + os.replace().

    Guarantees the target file is either the old version or the new version,
    never a partial write — even if the process is killed mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory (same filesystem = atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
