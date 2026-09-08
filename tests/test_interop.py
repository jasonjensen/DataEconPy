"""pandas and polars interop."""

from __future__ import annotations

import numpy as np
import pytest

import dataecon as de

from .conftest import requires_tsecon

pd = pytest.importorskip("pandas")


class TestPandas:
    @requires_tsecon
    def test_time_indexed_series_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        from dataecon.interop.pandas import read_pandas, write_pandas

        series = ts.to_pandas(ts.TSeries(ts.qq(2020, 1), np.arange(8.0)))
        write_pandas(memfile, "s", series)
        got = read_pandas(memfile, "s")
        pd.testing.assert_series_equal(got, series)

    @requires_tsecon
    def test_time_indexed_frame_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        from dataecon.interop.pandas import read_pandas, write_pandas

        frame = ts.to_pandas(
            ts.MVTSeries(ts.mm(2020, 1), ["a", "b"], np.arange(24.0).reshape(12, 2))
        )
        write_pandas(memfile, "f", frame)
        pd.testing.assert_frame_equal(read_pandas(memfile, "f"), frame)

    @requires_tsecon
    def test_a_pandas_object_can_be_written_directly(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        frame = ts.to_pandas(ts.MVTSeries(ts.qq(2020, 1), ["x"], np.arange(4.0).reshape(4, 1)))
        memfile.write("f", frame)
        assert isinstance(memfile.read("f"), ts.MVTSeries)

    def test_a_plain_indexed_frame_becomes_a_matrix(self, memfile: de.DEFile) -> None:
        from dataecon.interop.pandas import from_pandas

        frame = pd.DataFrame(np.arange(6.0).reshape(3, 2))
        memfile.write("m", from_pandas(frame))
        assert np.array_equal(memfile.read("m"), frame.to_numpy())

    def test_a_non_time_index_is_refused_rather_than_dropped(self) -> None:
        from dataecon.interop.pandas import from_pandas

        frame = pd.DataFrame({"v": [1.0, 2.0]}, index=["alpha", "beta"])
        with pytest.raises(de.DEUnsupportedError, match="non-time index"):
            from_pandas(frame)

    def test_to_pandas_of_an_array(self) -> None:
        from dataecon.interop.pandas import to_pandas

        assert isinstance(to_pandas(np.arange(3.0)), pd.Series)
        assert isinstance(to_pandas(np.arange(6.0).reshape(3, 2)), pd.DataFrame)

    def test_to_pandas_of_a_3d_array_is_refused(self) -> None:
        from dataecon.interop.pandas import to_pandas

        with pytest.raises(de.DEUnsupportedError, match="no natural pandas form"):
            to_pandas(np.zeros((2, 2, 2)))

    @requires_tsecon
    def test_to_pandas_of_a_catalog_gives_a_dict_of_frames(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        from dataecon.interop.pandas import to_pandas

        memfile.write("w", {"a": ts.TSeries(ts.qq(2020, 1), np.arange(4.0))})
        got = to_pandas(memfile.read("w"))
        assert isinstance(got["a"], pd.Series)


class TestPolars:
    @requires_tsecon
    def test_time_column_frame_round_trips(self, memfile: de.DEFile) -> None:
        pl = pytest.importorskip("polars")
        import tsecon as ts

        from dataecon.interop.polars import read_polars, write_polars

        original = ts.MVTSeries(ts.qq(2020, 1), ["a", "b"], np.arange(8.0).reshape(4, 2))
        frame = ts.to_polars(original)
        # polars has no period dtype, so the time column is plain dates and the
        # frequency has to be supplied; tsecon requires this and so do we.
        write_polars(memfile, "f", frame, freq=ts.Quarterly())
        got = read_polars(memfile, "f")
        assert isinstance(got, pl.DataFrame)
        assert got.shape == frame.shape
        assert memfile.read("f").firstdate == original.firstdate

    @requires_tsecon
    def test_a_time_column_without_a_frequency_is_reported(self, memfile: de.DEFile) -> None:
        pytest.importorskip("polars")
        import tsecon as ts

        from dataecon.interop.polars import from_polars

        frame = ts.to_polars(ts.TSeries(ts.qq(2020, 1), np.arange(4.0)))
        with pytest.raises(ValueError, match="freq"):
            from_polars(frame)

    def test_a_frame_without_a_time_column_becomes_a_matrix(self, memfile: de.DEFile) -> None:
        pl = pytest.importorskip("polars")

        from dataecon.interop.polars import from_polars

        frame = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        memfile.write("m", from_polars(frame))
        assert np.array_equal(memfile.read("m"), frame.to_numpy())

    def test_to_polars_of_an_array(self) -> None:
        pl = pytest.importorskip("polars")

        from dataecon.interop.polars import to_polars

        assert isinstance(to_polars(np.arange(3.0)), pl.Series)
        assert isinstance(to_polars(np.arange(6.0).reshape(3, 2)), pl.DataFrame)


class TestWithoutOptionalBackends:
    """The package must import and work without pandas, polars or tsecon."""

    def test_core_import_does_not_pull_in_backends(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys\n"
            "for name in ('pandas', 'polars', 'tsecon'):\n"
            "    sys.modules[name] = None\n"
            "import dataecon\n"
            "with dataecon.opendaecmem() as f:\n"
            "    f.write('x', 1.5)\n"
            "    assert f.read('x') == 1.5\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
