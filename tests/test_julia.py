"""Cross-validation against the Julia connector.

These tests drive ``TimeSeriesEcon.jl``'s ``DataEcon`` module through the two
scripts in ``cross/``:

* ``julia_write.jl`` writes a database from Julia, which Python then verifies.
* ``julia_read.jl`` verifies, from Julia, a database that Python wrote.

They are skipped unless a Julia with ``TimeSeriesEcon`` installed is available.
Point ``JULIA`` at the executable if it is not on ``PATH``, and
``JULIA_PROJECT`` at an environment that has ``TimeSeriesEcon`` in it. See
``tests/README.md``.

The same ground is covered without Julia by ``test_c_parity.py``, which drives
the very C library that the Julia connector calls into.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dataecon as de

from .conftest import requires_tsecon

pytestmark = pytest.mark.julia

CROSS = Path(__file__).resolve().parent / "cross"
JULIA_WRITE = CROSS / "julia_write.jl"
JULIA_READ = CROSS / "julia_read.jl"


def _julia_executable() -> str | None:
    return os.environ.get("JULIA") or shutil.which("julia")


def _julia_has_timeseriesecon(julia: str) -> bool:
    probe = subprocess.run(
        [julia, "--startup-file=no", "-e", "using TimeSeriesEcon; print(1)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def julia() -> str:
    """Path to a Julia that has TimeSeriesEcon available, or skip."""
    executable = _julia_executable()
    if executable is None:
        pytest.skip("no Julia on PATH; set JULIA to enable the Julia cross-checks")
    if not _julia_has_timeseriesecon(executable):
        pytest.skip(
            "Julia is available but TimeSeriesEcon is not installed in its "
            "active project; set JULIA_PROJECT to an environment that has it"
        )
    return executable


def run_julia(julia: str, script: Path, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [julia, "--startup-file=no", str(script), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def julia_written(julia: str, tmp_path: Path) -> Path:
    """A database written by the Julia connector."""
    path = tmp_path / "julia_written.daec"
    result = run_julia(julia, JULIA_WRITE, path)
    if result.returncode != 0:
        pytest.fail(f"julia_write.jl failed:\n{result.stdout}\n{result.stderr}")
    return path


def write_julia_fixture(path: Path) -> None:
    """Write, from Python, the database that ``julia_read.jl`` expects.

    Keep this in step with the checks in ``cross/julia_read.jl``.
    """
    import tsecon as ts

    data = {
        "i64": -1234567890123,
        "f64": 3.141592653589793,
        "c128": 1.5 + 2.25j,
        "text": "hello .daec",
        "dates": {
            "q2020Q1": ts.qq(2020, 1),
            "m1999M12": ts.mm(1999, 12),
            "y2024": ts.yy(2024),
            "d20240229": ts.daily(dt.date(2024, 2, 29)),
            "b20240612": ts.bdaily(dt.date(2024, 6, 12)),
        },
        "vectors": {
            "vec_f64": np.array([1.5, -2.5, 0.0, 1e10, -3.25]),
            "vec_i32": np.array([1, -2, 3, -4], dtype=np.int32),
            "vec_str": np.array(["alpha", "beta", "gamma"], dtype=object),
            "rng_dates": ts.MITRange(ts.qq(2020, 1), ts.qq(2021, 4)),
        },
        "series": {
            "ts_q": ts.TSeries(ts.qq(2020, 1), np.arange(8.0)),
            "ts_d": ts.TSeries(ts.daily(dt.date(2024, 1, 15)), np.array([10.5, 11.5, 12.5])),
            "ts_m": ts.TSeries(ts.mm(2023, 11), np.array([-1.0, -2.0, -3.0, -4.0])),
        },
        "arrays": {
            "mat": np.arange(1.0, 7.0).reshape(2, 3, order="F"),
            "mvts": ts.MVTSeries(
                ts.qq(2021, 1), ["a", "b", "c"], np.arange(1.0, 13.0).reshape(4, 3, order="F")
            ),
            "nd": np.arange(24.0).reshape(2, 3, 4, order="F"),
        },
        "nested": {"inner": {"leaf": 42.0}},
    }
    de.writedb(path, data)
    with de.opendaec(path, write=True) as f:
        f.set_attribute("/series/ts_q", "units", "billions")


@requires_tsecon
class TestPythonReadsJulia:
    """Everything the Julia connector writes must read correctly in Python."""

    def test_scalars(self, julia_written: Path) -> None:
        data = de.readdb(julia_written)
        assert data.i64 == -1234567890123
        assert data.f64 == 3.141592653589793
        assert data.c128 == 1.5 + 2.25j
        assert data.text == "hello .daec"

    def test_dates(self, julia_written: Path) -> None:
        import tsecon as ts

        dates = de.readdb(julia_written).dates
        assert dates.q2020Q1 == ts.qq(2020, 1)
        assert dates.m1999M12 == ts.mm(1999, 12)
        assert dates.y2024 == ts.yy(2024)
        assert dates.d20240229 == ts.daily(dt.date(2024, 2, 29))
        assert dates.b20240612 == ts.bdaily(dt.date(2024, 6, 12))

    def test_vectors(self, julia_written: Path) -> None:
        vectors = de.readdb(julia_written).vectors
        assert np.array_equal(vectors.vec_f64, [1.5, -2.5, 0.0, 1e10, -3.25])
        assert np.array_equal(vectors.vec_i32, [1, -2, 3, -4])
        assert list(vectors.vec_str) == ["alpha", "beta", "gamma"]
        assert len(vectors.vec_empty) == 0

    def test_time_series(self, julia_written: Path) -> None:
        import tsecon as ts

        series = de.readdb(julia_written).series
        assert series.ts_q.firstdate == ts.qq(2020, 1)
        assert np.array_equal(series.ts_q.values, np.arange(8.0))
        assert series.ts_d.firstdate == ts.daily(dt.date(2024, 1, 15))
        assert series.ts_m.firstdate == ts.mm(2023, 11)

    def test_matrix_orientation(self, julia_written: Path) -> None:
        """A row/column-major mix-up shows up here as a transposed matrix."""
        arrays = de.readdb(julia_written).arrays
        assert arrays.mat.shape == (2, 3)
        assert np.array_equal(arrays.mat, np.arange(1.0, 7.0).reshape(2, 3, order="F"))

    def test_mvtseries(self, julia_written: Path) -> None:
        import tsecon as ts

        mvts = de.readdb(julia_written).arrays.mvts
        assert mvts.firstdate == ts.qq(2021, 1)
        assert list(mvts.column_names) == ["a", "b", "c"]
        assert np.array_equal(mvts.values, np.arange(1.0, 13.0).reshape(4, 3, order="F"))

    def test_tensor(self, julia_written: Path) -> None:
        nd = de.readdb(julia_written).arrays.nd
        assert nd.shape == (2, 3, 4)
        assert np.array_equal(nd, np.arange(24.0).reshape(2, 3, 4, order="F"))

    def test_nesting_and_attributes(self, julia_written: Path) -> None:
        assert de.readdb(julia_written).nested.inner.leaf == 42.0
        with de.opendaec(julia_written) as f:
            assert f.get_attribute("/series/ts_q", "units") == "billions"

    def test_julia_bool_attribute_is_honoured(self, julia_written: Path) -> None:
        """Julia records a Bool with a ``jtype`` attribute; we must honour it."""
        assert de.readdb(julia_written).flag is True


@requires_tsecon
class TestJuliaReadsPython:
    """Everything Python writes must read correctly in the Julia connector."""

    def test_julia_verifies_a_python_written_file(self, julia: str, tmp_path: Path) -> None:
        path = tmp_path / "python_written.daec"
        write_julia_fixture(path)
        result = run_julia(julia, JULIA_READ, path)
        assert result.returncode == 0, (
            f"Julia rejected the Python-written file:\n{result.stdout}\n{result.stderr}"
        )
