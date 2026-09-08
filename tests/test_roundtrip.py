"""Value round trips: what goes in must come out unchanged."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import dataecon as de

from .conftest import requires_tsecon


class TestScalars:
    @pytest.mark.parametrize(
        "value",
        [
            0,
            1,
            -1,
            42,
            -1234567890123,
            2**62,
            -(2**62),
            0.0,
            -0.0,
            3.141592653589793,
            1e308,
            -1e-308,
            float("inf"),
            -float("inf"),
            1 + 2j,
            -1.5 - 0.25j,
            "",
            "hello",
            "béta λ ∑",
            "line\nbreak",
            "tab\there",
            True,
            False,
            None,
            b"",
            b"\x00\x01\x02binary",
        ],
    )
    def test_round_trip(self, memfile: de.DEFile, value: object) -> None:
        memfile.write("x", value)
        got = memfile.read("x")
        assert got == value
        assert type(got) is type(value)

    def test_nan_round_trips(self, memfile: de.DEFile) -> None:
        memfile.write("x", float("nan"))
        assert np.isnan(memfile.read("x"))

    def test_bool_stays_bool_and_is_not_an_int(self, memfile: de.DEFile) -> None:
        memfile.write("t", True)
        memfile.write("n", 1)
        assert memfile.read("t") is True
        number = memfile.read("n")
        assert number == 1
        assert not isinstance(number, bool)

    def test_date_round_trips(self, memfile: de.DEFile) -> None:
        value = dt.date(2024, 6, 12)
        memfile.write("d", value)
        assert memfile.read("d") == value

    def test_datetime_round_trips_to_the_second(self, memfile: de.DEFile) -> None:
        value = dt.datetime(2024, 6, 12, 15, 30, 45)
        memfile.write("d", value)
        assert memfile.read("d") == value

    @pytest.mark.parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_numpy_scalars_round_trip(self, memfile: de.DEFile, dtype: type) -> None:
        value = dtype(7)
        memfile.write("x", value)
        assert memfile.read("x") == value

    def test_very_large_int_uses_the_wide_form(self, memfile: de.DEFile) -> None:
        value = 2**100
        memfile.write("x", value)
        assert memfile.read("x") == value


class TestArrays:
    @pytest.mark.parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_vector_round_trips(self, memfile: de.DEFile, dtype: type) -> None:
        value = np.arange(6).astype(dtype)
        memfile.write("v", value)
        got = memfile.read("v")
        assert np.array_equal(got, value)
        assert got.dtype == value.dtype

    def test_bool_array_stays_bool(self, memfile: de.DEFile) -> None:
        value = np.array([True, False, True, True])
        memfile.write("v", value)
        got = memfile.read("v")
        assert got.dtype == np.bool_
        assert np.array_equal(got, value)

    def test_string_array_round_trips(self, memfile: de.DEFile) -> None:
        value = np.array(["alpha", "", "λ", "with space"], dtype=object)
        memfile.write("v", value)
        assert list(memfile.read("v")) == list(value)

    def test_list_of_numbers_becomes_an_array(self, memfile: de.DEFile) -> None:
        memfile.write("v", [1.0, 2.0, 3.0])
        assert np.array_equal(memfile.read("v"), np.array([1.0, 2.0, 3.0]))

    def test_list_of_strings_round_trips(self, memfile: de.DEFile) -> None:
        memfile.write("v", ["a", "bb", "ccc"])
        assert list(memfile.read("v")) == ["a", "bb", "ccc"]

    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 5), (5, 1), (4, 4)])
    def test_matrix_round_trips(self, memfile: de.DEFile, shape: tuple[int, int]) -> None:
        value = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        memfile.write("m", value)
        got = memfile.read("m")
        assert got.shape == shape
        assert np.array_equal(got, value)

    @pytest.mark.parametrize("shape", [(2, 3, 4), (2, 2, 2, 2), (2, 2, 2, 2, 2)])
    def test_tensor_round_trips(self, memfile: de.DEFile, shape: tuple[int, ...]) -> None:
        value = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        memfile.write("t", value)
        got = memfile.read("t")
        assert got.shape == shape
        assert np.array_equal(got, value)

    @pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
    def test_empty_arrays_round_trip(self, memfile: de.DEFile, shape: tuple[int, ...]) -> None:
        value = np.zeros(shape, dtype=np.float64)
        memfile.write("e", value)
        got = memfile.read("e")
        assert got.shape == shape
        assert got.dtype == np.float64

    def test_six_dimensions_is_rejected(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DERangeError, match="at most 5 dimensions"):
            memfile.write("t", np.zeros((2, 2, 2, 2, 2, 2)))


class TestRanges:
    def test_plain_range_round_trips(self, memfile: de.DEFile) -> None:
        memfile.write("r", range(5))
        assert memfile.read("r") == range(0, 5)

    def test_offset_range_round_trips(self, memfile: de.DEFile) -> None:
        memfile.write("r", range(3, 9))
        assert memfile.read("r") == range(3, 9)

    def test_stepped_range_is_rejected(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEUnsupportedError, match="unit-step"):
            memfile.write("r", range(0, 10, 2))


class TestCatalogs:
    def test_nested_dicts_become_nested_catalogs(self, memfile: de.DEFile) -> None:
        memfile.write("top", {"a": 1.0, "sub": {"b": 2.0, "deeper": {"c": 3.0}}})
        assert memfile.read("top/a") == 1.0
        assert memfile.read("top/sub/b") == 2.0
        assert memfile.read("top/sub/deeper/c") == 3.0

    def test_reading_a_catalog_gives_back_the_structure(self, memfile: de.DEFile) -> None:
        memfile.write("top", {"a": 1.0, "sub": {"b": 2.0}})
        got = memfile.read("top")
        assert got["a"] == 1.0
        assert got["sub"]["b"] == 2.0

    def test_intermediate_catalogs_are_created_on_write(self, memfile: de.DEFile) -> None:
        memfile.write("a/b/c/d", 1.0)
        assert memfile.read("a/b/c/d") == 1.0
        assert sorted(memfile.walk()) == ["/a/b/c/d"]

    def test_deleting_a_catalog_removes_its_contents(self, memfile: de.DEFile) -> None:
        memfile.write("top", {"a": 1.0, "sub": {"b": 2.0}})
        memfile.delete_object("top")
        assert "top" not in memfile
        assert memfile.keys() == []


@requires_tsecon
class TestTimeSeriesEconPy:
    def test_tseries_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.TSeries(ts.qq(2020, 1), np.arange(8.0))
        memfile.write("s", value)
        got = memfile.read("s")
        assert isinstance(got, ts.TSeries)
        assert got.firstdate == value.firstdate
        assert got.frequency == value.frequency
        assert np.array_equal(got.values, value.values)

    @pytest.mark.parametrize(
        "first",
        [
            "qq",
            "mm",
            "yy",
            "daily",
            "bdaily",
            "weekly",
            "halfyearly",
        ],
    )
    def test_every_frequency_round_trips(self, memfile: de.DEFile, first: str) -> None:
        import tsecon as ts

        starts = {
            "qq": ts.qq(2020, 1),
            "mm": ts.mm(2020, 3),
            "yy": ts.yy(2020),
            "daily": ts.daily("2024-01-15"),
            "bdaily": ts.bdaily("2024-06-12"),
            "weekly": ts.weekly("2024-06-12"),
            "halfyearly": ts.MIT.from_yp(ts.HalfYearly(), 2021, 2),
        }
        value = ts.TSeries(starts[first], np.arange(5.0))
        memfile.write("s", value)
        got = memfile.read("s")
        assert got.firstdate == value.firstdate
        assert got.frequency == value.frequency
        assert np.array_equal(got.values, value.values)

    def test_integer_valued_tseries_keeps_its_dtype(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.TSeries(ts.qq(2020, 1), np.arange(4, dtype=np.int32))
        memfile.write("s", value)
        got = memfile.read("s")
        assert got.values.dtype == np.int32
        assert np.array_equal(got.values, value.values)

    def test_mvtseries_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.MVTSeries(ts.mm(2019, 1), ["a", "b", "c"], np.arange(36.0).reshape(12, 3))
        memfile.write("m", value)
        got = memfile.read("m")
        assert isinstance(got, ts.MVTSeries)
        assert got.firstdate == value.firstdate
        assert list(got.column_names) == ["a", "b", "c"]
        assert np.array_equal(got.values, value.values)

    def test_mvtseries_column_order_is_preserved(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        names = ["zeta", "alpha", "mu"]
        value = ts.MVTSeries(ts.qq(2020, 1), names, np.arange(12.0).reshape(4, 3))
        memfile.write("m", value)
        got = memfile.read("m")
        assert list(got.column_names) == names
        assert np.array_equal(got.values, value.values)

    def test_mit_scalar_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.qq(2020, 3)
        memfile.write("d", value)
        got = memfile.read("d")
        assert got == value
        assert got.frequency == value.frequency

    def test_duration_scalar_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.qq(2021, 1) - ts.qq(2020, 1)
        memfile.write("d", value)
        got = memfile.read("d")
        assert int(got) == int(value)
        assert got.frequency == value.frequency

    def test_mitrange_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.MITRange(ts.qq(2020, 1), ts.qq(2021, 4))
        memfile.write("r", value)
        got = memfile.read("r")
        assert got.first() == value.first()
        assert got.last() == value.last()
        assert len(got) == len(value)

    def test_mit_array_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = np.array([ts.qq(2020, 1), ts.qq(2020, 2), ts.qq(2020, 3)], dtype=object)
        memfile.write("v", value)
        got = memfile.read("v")
        assert list(got) == list(value)

    def test_workspace_round_trips(self, memfile: de.DEFile) -> None:
        import tsecon as ts

        value = ts.Workspace(
            gdp=ts.TSeries(ts.qq(2020, 1), np.arange(4.0)),
            params=ts.Workspace(beta=0.99),
        )
        memfile.write("w", value)
        got = memfile.read("w")
        assert isinstance(got, ts.Workspace)
        assert got.params.beta == 0.99
        assert np.array_equal(got.gdp.values, value.gdp.values)


class TestWholeFileHelpers:
    def test_writedb_readdb(self, daec_path) -> None:
        data = {"a": 1.0, "b": {"c": "text"}}
        de.writedb(daec_path, data)
        got = de.readdb(daec_path, as_dict=True)
        assert got == data

    def test_readdb_of_one_catalog(self, daec_path) -> None:
        de.writedb(daec_path, {"a": {"x": 1.0}, "b": {"y": 2.0}})
        assert de.readdb(daec_path, "/a", as_dict=True) == {"x": 1.0}

    def test_writedb_append_keeps_existing_objects(self, daec_path) -> None:
        de.writedb(daec_path, {"a": 1.0})
        de.writedb(daec_path, {"b": 2.0})
        assert de.readdb(daec_path, as_dict=True) == {"a": 1.0, "b": 2.0}

    def test_writedb_without_append_empties_the_file(self, daec_path) -> None:
        de.writedb(daec_path, {"a": 1.0})
        de.writedb(daec_path, {"b": 2.0}, append=False)
        assert de.readdb(daec_path, as_dict=True) == {"b": 2.0}

    def test_writedb_rejects_a_non_mapping(self, daec_path) -> None:
        with pytest.raises(TypeError, match="mapping"):
            de.writedb(daec_path, [1, 2, 3])
