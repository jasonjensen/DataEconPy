# SPDX-License-Identifier: MIT
"""Locating and loading the ``libdaec`` shared library.

All ``.daec`` reading and writing goes through the reference C library, so this
module is what stands between the package and a usable file. The library is
looked for in this order:

1. the path in the ``DATAECON_LIBRARY`` environment variable, if set;
2. the copy built into this package (``dataecon/_lib/``), which is what a wheel
   ships and what :mod:`dataecon.build` produces;
3. the platform's own search path, under the usual library names.

If none of those work, :class:`DELibraryNotFoundError` explains how to build it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

from .build import bundled_library_path, library_filename
from .errors import DELibraryNotFoundError

__all__ = ["bundled_library_path", "library_filename", "load_library"]

#: Version of the C library this package is written against.
REQUIRED_VERSION = "0.4.0"


def _candidates() -> list[Path]:
    found: list[Path] = []

    env = os.environ.get("DATAECON_LIBRARY")
    if env:
        found.append(Path(env))

    found.append(bundled_library_path())

    for name in ("daec", "libdaec"):
        located = ctypes.util.find_library(name)
        if located:
            found.append(Path(located))

    found.append(Path(library_filename()))
    return found


def load_library() -> ctypes.CDLL:
    """Load ``libdaec`` and return the :class:`ctypes.CDLL` handle.

    Raises
    ------
    DELibraryNotFoundError
        If the library cannot be found or cannot be loaded.
    """
    attempts: list[str] = []
    for candidate in _candidates():
        if candidate.is_absolute() and not candidate.exists():
            attempts.append(f"{candidate} (does not exist)")
            continue
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as err:
            attempts.append(f"{candidate} ({err})")

    tried = "\n  ".join(attempts) or "(no candidates)"
    msg = (
        "Could not load the DataEcon C library (libdaec), which DataEconPy uses "
        "for all .daec file access.\n"
        f"Tried:\n  {tried}\n\n"
        "To build it from the sources vendored with this package, run:\n"
        "    python -m dataecon.build\n"
        "Or point DATAECON_LIBRARY at an existing copy of the library."
    )
    raise DELibraryNotFoundError(msg)
