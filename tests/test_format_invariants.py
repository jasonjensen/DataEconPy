"""Byte-level invariants of the stored format.

Some things the format depends on are invisible to a value-level round trip
because both sides make the same assumption. They are checked here directly
against the stored bytes, so that a regression is caught here rather than in
another connector.
"""

from __future__ import annotations

import numpy as np
import pytest

import dataecon as de
from dataecon._codec import pack_strings, unpack_strings
from dataecon._consts import Frequency, Type


class TestStringTermination:
    """String values must be NUL-terminated.

    The C library hands a bare pointer back to its caller, and the Julia
    connector turns it into a string with ``unsafe_string``, which reads until a
    NUL. A string stored without its terminator would read past the end of the
    buffer there, so the terminator is part of the contract even though a
    Python-only round trip would not notice it missing.
    """

    def test_scalar_string_is_nul_terminated(self, memfile: de.DEFile) -> None:
        memfile.write("s", "hello")
        _info, _freq, blob = memfile.load_scalar("s")
        assert blob == b"hello\x00"

    def test_empty_scalar_string_is_a_single_nul(self, memfile: de.DEFile) -> None:
        memfile.write("s", "")
        _info, _freq, blob = memfile.load_scalar("s")
        assert blob == b"\x00"

    def test_unicode_scalar_string_is_utf8_then_nul(self, memfile: de.DEFile) -> None:
        memfile.write("s", "béta λ")
        _info, _freq, blob = memfile.load_scalar("s")
        assert blob == "béta λ".encode() + b"\x00"

    def test_string_array_elements_are_each_nul_terminated(self, memfile: de.DEFile) -> None:
        memfile.write("v", np.array(["a", "bb", ""], dtype=object))
        _info, _eltype, _elfreq, _axis, blob = memfile.load_tseries("v")
        assert blob == b"a\x00bb\x00\x00"

    def test_pack_unpack_strings_round_trip(self) -> None:
        values = ["", "a", "longer string", "béta"]
        assert unpack_strings(pack_strings(values), len(values)) == values


class TestColumnMajorLayout:
    """Arrays are stored column-major, as the C and Julia connectors expect."""

    def test_matrix_blob_is_column_major(self, memfile: de.DEFile) -> None:
        value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        memfile.write("m", value)
        _i, _e, _f, _a1, _a2, blob = memfile.load_mvtseries("m")
        assert np.frombuffer(blob, np.float64).tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    def test_tensor_blob_is_column_major(self, memfile: de.DEFile) -> None:
        value = np.arange(24.0).reshape(2, 3, 4)
        memfile.write("t", value)
        _i, _e, _f, _axes, blob = memfile.load_ndtseries("t")
        expected = value.ravel(order="F").tolist()
        assert np.frombuffer(blob, np.float64).tolist() == expected

    def test_non_contiguous_input_is_still_stored_column_major(self, memfile: de.DEFile) -> None:
        base = np.arange(12.0).reshape(3, 4)
        view = base[:, ::2]  # a strided, non-contiguous view
        memfile.write("v", view)
        assert np.array_equal(memfile.read("v"), view)


class TestElementWidths:
    """The element type is recovered from the type code and the blob size."""

    @pytest.mark.parametrize(
        ("dtype", "eltype"),
        [
            (np.int8, Type.SIGNED),
            (np.int16, Type.SIGNED),
            (np.int32, Type.SIGNED),
            (np.int64, Type.SIGNED),
            (np.uint8, Type.UNSIGNED),
            (np.uint64, Type.UNSIGNED),
            (np.float32, Type.FLOAT),
            (np.float64, Type.FLOAT),
            (np.complex64, Type.COMPLEX),
            (np.complex128, Type.COMPLEX),
        ],
    )
    def test_width_matches_dtype(self, memfile: de.DEFile, dtype: type, eltype: Type) -> None:
        value = np.arange(4).astype(dtype)
        memfile.write("v", value)
        _info, stored_eltype, _elfreq, axis, blob = memfile.load_tseries("v")
        assert stored_eltype == eltype
        assert len(blob) == axis.length * np.dtype(dtype).itemsize
        assert np.array_equal(memfile.read("v"), value)


class TestRootCatalog:
    """The root catalog carries the format version, as the C library writes it."""

    def test_new_file_records_the_format_version(self, daec_path) -> None:
        with de.opendaec(daec_path, write=True) as f:
            assert f.get_attribute(de.ROOT_ID, "DE_VERSION") == de.DE_VERSION

    def test_root_catalog_starts_empty(self, memfile: de.DEFile) -> None:
        assert memfile.catalog_size(de.ROOT_ID) == 0
        assert len(memfile) == 0


class TestAxisDeduplication:
    """Identical axes are shared rather than duplicated, as the C library does."""

    def test_identical_plain_axes_share_an_id(self, memfile: de.DEFile) -> None:
        assert memfile.axis_plain(7) == memfile.axis_plain(7)

    def test_different_lengths_get_different_ids(self, memfile: de.DEFile) -> None:
        assert memfile.axis_plain(7) != memfile.axis_plain(8)

    def test_identical_range_axes_share_an_id(self, memfile: de.DEFile) -> None:
        first = memfile.axis_range(4, Frequency.QUARTERLY_MAR, 8080)
        assert memfile.axis_range(4, Frequency.QUARTERLY_MAR, 8080) == first
        assert memfile.axis_range(4, Frequency.QUARTERLY_MAR, 8081) != first

    def test_identical_name_axes_share_an_id(self, memfile: de.DEFile) -> None:
        first = memfile.axis_names(["a", "b"])
        assert memfile.axis_names(["a", "b"]) == first
        assert memfile.axis_names(["a", "c"]) != first
