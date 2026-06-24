from __future__ import annotations

import os
from typing import IO


def drop_file_cache(handle: IO[object]) -> None:
    """Tell Linux we do not need this file's cached pages after streaming it.

    This is a best-effort hint for container memory accounting. It is a no-op on
    non-POSIX platforms and never affects program correctness.
    """
    if os.name != "posix" or not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except Exception:
        pass
