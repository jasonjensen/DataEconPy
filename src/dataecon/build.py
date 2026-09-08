# SPDX-License-Identifier: MIT
"""Build the ``libdaec`` shared library from the vendored C sources.

A wheel normally ships the compiled library, so this module exists for the cases
where it does not: an editable install, an sdist install on a platform without a
wheel, or a developer checkout. Run it as::

    python -m dataecon.build

It compiles the sources under ``vendor/dataecon-c`` and writes the shared
library into ``dataecon/_lib/``, where :mod:`dataecon._loader` looks for it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path

__all__ = ["build_library", "bundled_library_path", "library_filename", "vendor_root"]

# N.B. This module imports nothing beyond the standard library, and nothing from
# the rest of the package. The wheel build hook loads it straight off disk, in an
# isolated environment that has only the build backend installed -- importing
# `dataecon` there would fail on numpy.


def library_filename() -> str:
    """Return the platform's file name for the DataEcon shared library."""
    if sys.platform == "win32":
        return "daec.dll"
    if sys.platform == "darwin":
        return "libdaec.dylib"
    return "libdaec.so"


def bundled_library_path() -> Path:
    """Return the path the library occupies when bundled with this package."""
    return Path(__file__).resolve().parent / "_lib" / library_filename()


def vendor_root() -> Path | None:
    """Return the directory holding the vendored C sources, if it is present.

    Wheels do not ship the C sources -- only the compiled library -- so this
    returns ``None`` in a normal installed environment.
    """
    package_dir = Path(__file__).resolve().parent
    for candidate in (
        package_dir / "_csrc",  # sdist layout: sources copied into the package
        package_dir.parent.parent / "vendor" / "dataecon-c",  # source checkout
    ):
        if (candidate / "include" / "daec.h").exists():
            return candidate
    return None


def _source_files(root: Path) -> list[Path]:
    sources = sorted((root / "src" / "libdaec").glob("*.c"))
    sources.append(root / "src" / "sqlite3" / "sqlite3.c")
    return sources


def _compiler() -> list[str]:
    """Return the C compiler command to use."""
    env = os.environ.get("CC")
    if env:
        return env.split()
    configured = sysconfig.get_config_var("CC")
    if configured and shutil.which(configured.split()[0]):
        return configured.split()
    for name in ("cc", "gcc", "clang"):
        if shutil.which(name):
            return [name]
    msg = "No C compiler found. Set CC to the compiler you want to use."
    raise RuntimeError(msg)


def _platform_flags() -> tuple[list[str], list[str]]:
    """Return ``(compile_flags, link_flags)`` for the current platform."""
    if sys.platform == "win32":
        return ["-O2"], ["-shared"]
    if sys.platform == "darwin":
        return ["-O3", "-fPIC"], ["-shared", "-lpthread", "-ldl", "-lm"]
    return ["-O3", "-fPIC"], ["-shared", "-lpthread", "-ldl", "-lm"]


def build_library(
    output: Path | None = None,
    *,
    root: Path | None = None,
    verbose: bool = False,
    extra_cflags: Sequence[str] = (),
) -> Path:
    """Compile the vendored sources and return the path of the built library.

    Parameters
    ----------
    output
        Where to write the library. Defaults to the bundled location inside the
        installed package.
    root
        Directory holding the vendored sources. Defaults to :func:`vendor_root`.
    verbose
        Echo each compiler invocation.
    extra_cflags
        Additional flags appended to every compile step.
    """
    root = root or vendor_root()
    if root is None:
        msg = (
            "The vendored DataEcon C sources are not available in this "
            "installation, so the library cannot be built here. Install from a "
            "source checkout, or set DATAECON_LIBRARY to a prebuilt libdaec."
        )
        raise RuntimeError(msg)

    output = output or bundled_library_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    cc = _compiler()
    cflags, ldflags = _platform_flags()
    cflags = [*cflags, *extra_cflags]
    includes = [
        f"-I{root / 'include'}",
        f"-I{root / 'src' / 'libdaec'}",
        f"-I{root / 'src' / 'sqlite3'}",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        objects: list[str] = []
        for source in _source_files(root):
            obj = str(Path(tmp) / (source.stem + ".o"))
            cmd = [*cc, "-std=c99", *cflags, *includes, "-c", str(source), "-o", obj]
            if verbose:
                print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            objects.append(obj)

        cmd = [*cc, *cflags, *ldflags[:1], *objects, "-o", str(output), *ldflags[1:]]
        if verbose:
            print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m dataecon.build``."""
    parser = argparse.ArgumentParser(
        prog="python -m dataecon.build",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="output path")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress compiler output")
    args = parser.parse_args(argv)

    path = build_library(args.output, verbose=not args.quiet)
    print(f"Built {library_filename()} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
