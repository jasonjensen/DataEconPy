"""Shared fixtures.

The cross-connector tests need a driver compiled against ``libdaec``. Building
it needs a C compiler and the vendored sources, so the fixture skips rather than
fails when either is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import dataecon as de
from dataecon._loader import bundled_library_path

REPO_ROOT = Path(__file__).resolve().parent.parent
C_DRIVER_SOURCE = Path(__file__).resolve().parent / "cross" / "c_reference.c"
VENDOR_INCLUDE = REPO_ROOT / "vendor" / "dataecon-c" / "include"


@pytest.fixture(scope="session")
def c_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the reference C driver and return the path to the executable."""
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler available to build the reference driver")
    if not VENDOR_INCLUDE.exists():
        pytest.skip("vendored DataEcon headers are not present in this installation")

    library = bundled_library_path()
    if not library.exists():
        pytest.skip(f"libdaec has not been built at {library}")

    out = tmp_path_factory.mktemp("cdriver") / "c_reference"
    cmd = [
        compiler,
        "-std=c99",
        "-O2",
        f"-I{VENDOR_INCLUDE}",
        str(C_DRIVER_SOURCE),
        "-o",
        str(out),
        f"-L{library.parent}",
        "-ldaec",
        f"-Wl,-rpath,{library.parent}",
        "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"could not build the reference C driver:\n{result.stderr}")

    if sys.platform == "win32":
        # Windows has no rpath, so the `-Wl,-rpath` above buys nothing there and
        # the driver would die at startup with STATUS_DLL_NOT_FOUND. The DLL
        # search does begin in the executable's own directory, so put a copy of
        # libdaec beside it.
        shutil.copy2(library, out.parent / library.name)

    return out


@pytest.fixture
def memfile() -> de.DEFile:
    """An empty in-memory ``.daec`` database."""
    with de.opendaecmem() as f:
        yield f


@pytest.fixture
def daec_path(tmp_path: Path) -> Path:
    """Path for a scratch ``.daec`` file that does not exist yet."""
    return tmp_path / "scratch.daec"


def has_tsecon() -> bool:
    """Whether TimeSeriesEconPy is importable."""
    from dataecon.interop import _tsecon

    return _tsecon.available()


requires_tsecon = pytest.mark.skipif(not has_tsecon(), reason="TimeSeriesEconPy is not installed")
