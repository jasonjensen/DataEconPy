"""Cross-validate DataEconPy against the DataEcon C library, both directions.

The C driver in ``cross/c_reference.c`` links straight against ``libdaec`` and
uses the same conventions as the Julia connector. Two comparisons pin the format
down from both sides:

* **Python reads what C writes** -- the C driver writes a database, and the
  Python dumper and the C dumper must describe it identically.
* **Python writes what C reads** -- ``cross/pywrite.py`` writes the same
  database through DataEconPy, and the C driver's dump of it must match the C
  driver's dump of its own file.

Because both sides emit the same textual format, a disagreement about element
width, byte order, column-major layout, axis kind or attributes shows up as a
diff on the offending object rather than as an opaque failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import dataecon as de
from tests.cross.pydump import dump_file
from tests.cross.pywrite import write_reference


def c_dump(driver: Path, path: Path) -> list[str]:
    """Run the C driver's ``dump`` command and return its lines."""
    result = subprocess.run(
        [str(driver), "dump", str(path)], capture_output=True, text=True, check=True
    )
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def c_write(driver: Path, path: Path) -> None:
    """Run the C driver's ``write`` command."""
    subprocess.run([str(driver), "write", str(path)], capture_output=True, check=True)


@pytest.fixture
def c_written(c_driver: Path, tmp_path: Path) -> Path:
    """A database written by the reference C driver."""
    path = tmp_path / "c_written.daec"
    c_write(c_driver, path)
    return path


@pytest.fixture
def py_written(tmp_path: Path) -> Path:
    """The same database, written by DataEconPy's low-level API."""
    path = tmp_path / "py_written.daec"
    write_reference(str(path))
    return path


def test_library_version_is_the_expected_one() -> None:
    from dataecon._loader import REQUIRED_VERSION

    assert de.library_version() == REQUIRED_VERSION


def test_python_reads_what_c_writes(c_driver: Path, c_written: Path) -> None:
    """The Python and C dumps of a C-written file must agree line for line."""
    assert sorted(dump_file(str(c_written))) == c_dump(c_driver, c_written)


def test_c_reads_what_python_writes(c_driver: Path, c_written: Path, py_written: Path) -> None:
    """The C driver must see the Python-written file as its own."""
    assert c_dump(c_driver, py_written) == c_dump(c_driver, c_written)


def test_python_reads_what_python_writes(c_driver: Path, py_written: Path) -> None:
    """Closing the triangle: Python's own dump of its own file matches C's."""
    assert sorted(dump_file(str(py_written))) == c_dump(c_driver, py_written)


def test_reference_covers_every_storage_class(c_driver: Path, c_written: Path) -> None:
    """Guard against the reference database quietly losing coverage."""
    kinds = {line.split(" ", 1)[0] for line in c_dump(c_driver, c_written)}
    assert kinds == {"CATALOG", "SCALAR", "VECTOR", "MATRIX", "TENSOR"}


class TestJuliaFixtureIsValidWithoutJulia:
    """Keep the Julia cross-check fixture honest even where Julia is missing.

    ``tests/test_julia.py`` can only run where a Julia with TimeSeriesEcon is
    installed. These checks run everywhere and would catch the fixture rotting
    into something the C library -- and therefore the Julia connector -- could
    not read.
    """

    @pytest.fixture
    def fixture_path(self, tmp_path: Path) -> Path:
        from .test_julia import write_julia_fixture

        pytest.importorskip("tsecon")
        path = tmp_path / "julia_fixture.daec"
        write_julia_fixture(path)
        return path

    def test_the_c_library_can_read_every_object(self, c_driver: Path, fixture_path: Path) -> None:
        lines = c_dump(c_driver, fixture_path)
        paths = {line.split(" ", 2)[1] for line in lines}
        assert "/series/ts_q" in paths
        assert "/arrays/mvts" in paths
        assert "/arrays/nd" in paths

    def test_python_and_c_agree_on_the_fixture(self, c_driver: Path, fixture_path: Path) -> None:
        assert sorted(dump_file(str(fixture_path))) == c_dump(c_driver, fixture_path)

    def test_the_fixture_uses_the_object_types_julia_expects(
        self, c_driver: Path, fixture_path: Path
    ) -> None:
        """Julia refuses a tseries whose axis is not a date range, and so on."""
        by_path = {line.split(" ", 2)[1]: line for line in c_dump(c_driver, fixture_path)}
        assert "type=12" in by_path["/series/ts_q"]  # type_tseries
        assert "axis=range:" in by_path["/series/ts_q"]
        assert "type=21" in by_path["/arrays/mvts"]  # type_mvtseries
        assert "axis1=range:" in by_path["/arrays/mvts"]
        assert "axis2=names:" in by_path["/arrays/mvts"]
        assert "type=20" in by_path["/arrays/mat"]  # type_matrix
        assert "axis1=plain:" in by_path["/arrays/mat"]
        assert "type=30" in by_path["/arrays/nd"]  # type_tensor
        assert "type=11" in by_path["/vectors/rng_dates"]  # type_range

    def test_the_fixture_round_trips_in_python(self, fixture_path: Path) -> None:
        import numpy as np
        import tsecon as ts

        data = de.readdb(fixture_path)
        assert data.series.ts_q.firstdate == ts.qq(2020, 1)
        assert list(data.arrays.mvts.column_names) == ["a", "b", "c"]
        assert np.array_equal(data.arrays.mat, np.arange(1.0, 7.0).reshape(2, 3, order="F"))
        assert data.nested.inner.leaf == 42.0
