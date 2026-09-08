"""Write the reference database from Python, mirroring ``c_reference.c``.

This is deliberately written against DataEconPy's *low-level* API, the one that
maps one-to-one onto ``daec.h``, so that it makes exactly the same library calls
with exactly the same arguments as the C driver does. If the two files dump
identically under the C driver, DataEconPy writes the format correctly.

Keep this in step with ``write_*`` in ``c_reference.c``.
"""

from __future__ import annotations

import ctypes

import numpy as np

import dataecon as de
from dataecon import _clib as C
from dataecon._codec import pack_strings
from dataecon._consts import Frequency as F
from dataecon._consts import Type as T

__all__ = ["write_reference"]


def _pack_yp(freq: int, year: int, period: int) -> int:
    """Pack a year-period date through the C library."""
    out = ctypes.c_int64()
    C.check(C.lib.de_pack_year_period_date(int(freq), year, period, ctypes.byref(out)))
    return out.value


def _pack_cal(freq: int, year: int, month: int, day: int) -> int:
    """Pack a calendar date through the C library."""
    out = ctypes.c_int64()
    C.check(C.lib.de_pack_calendar_date(int(freq), year, month, day, ctypes.byref(out)))
    return out.value


def _scalar(f: de.DEFile, pid: int, name: str, obj_type: int, value: np.generic) -> int:
    return f.store_scalar(pid, name, obj_type, F.NONE, np.asarray(value).tobytes())


def _write_scalars(f: de.DEFile, pid: int) -> None:
    _scalar(f, pid, "i64", T.SIGNED, np.int64(-1234567890123))
    _scalar(f, pid, "i32", T.SIGNED, np.int32(-2000000000))
    _scalar(f, pid, "i16", T.SIGNED, np.int16(-32000))
    _scalar(f, pid, "i8", T.SIGNED, np.int8(-128))
    _scalar(f, pid, "u64", T.UNSIGNED, np.uint64(18000000000000000000))
    _scalar(f, pid, "u16", T.UNSIGNED, np.uint16(65000))
    _scalar(f, pid, "f64", T.FLOAT, np.float64(3.141592653589793))
    _scalar(f, pid, "f32", T.FLOAT, np.float32(2.5))
    _scalar(f, pid, "c128", T.COMPLEX, np.complex128(1.5 + 2.25j))
    _scalar(f, pid, "c64", T.COMPLEX, np.complex64(0.5 - 1.25j))

    # A C string literal is stored with its NUL terminator counted.
    f.store_scalar(pid, "text", T.STRING, F.NONE, b"hello .daec\x00")

    obj_id = _scalar(f, pid, "flag", T.SIGNED, np.int8(1))
    f.set_attribute(obj_id, "jtype", "Bool")


def _write_dates(f: de.DEFile, pid: int) -> None:
    def date(name: str, freq: int, code: int) -> None:
        f.store_scalar(pid, name, T.DATE, freq, np.int64(code).tobytes())

    date("q2020Q1", F.QUARTERLY_MAR, _pack_yp(F.QUARTERLY_MAR, 2020, 1))
    date("m1999M12", F.MONTHLY, _pack_yp(F.MONTHLY, 1999, 12))
    date("y2024", F.YEARLY_DEC, _pack_yp(F.YEARLY_DEC, 2024, 1))
    date("h2021H2", F.HALFYEARLY_JUN, _pack_yp(F.HALFYEARLY_JUN, 2021, 2))
    date("d20240229", F.DAILY, _pack_cal(F.DAILY, 2024, 2, 29))
    date("b20240612", F.BDAILY, _pack_cal(F.BDAILY, 2024, 6, 12))
    date("w20240612", F.WEEKLY_SUN, _pack_cal(F.WEEKLY_SUN, 2024, 6, 12))

    f.store_scalar(pid, "dur", T.SIGNED, F.QUARTERLY_MAR, np.int64(12).tobytes())


def _write_vectors(f: de.DEFile, pid: int) -> None:
    vf = np.array([1.5, -2.5, 0.0, 1e10, -3.25], dtype=np.float64)
    f.store_tseries(
        pid, "vec_f64", T.VECTOR, T.FLOAT, F.NONE, f.axis_plain(5), vf.tobytes(order="F")
    )

    vi = np.array([1, -2, 3, -4], dtype=np.int32)
    f.store_tseries(
        pid, "vec_i32", T.VECTOR, T.SIGNED, F.NONE, f.axis_plain(4), vi.tobytes(order="F")
    )

    packed = pack_strings(["alpha", "beta", "gamma"])
    f.store_tseries(pid, "vec_str", T.VECTOR, T.STRING, F.NONE, f.axis_plain(3), packed)

    f.store_tseries(pid, "vec_empty", T.VECTOR, T.FLOAT, F.NONE, f.axis_plain(0), None)

    f.store_tseries(pid, "rng_plain", T.RANGE, T.NONE, F.NONE, f.axis_plain(6), None)

    first = _pack_yp(F.QUARTERLY_MAR, 2020, 1)
    axis = f.axis_range(8, F.QUARTERLY_MAR, first)
    f.store_tseries(pid, "rng_dates", T.RANGE, T.NONE, F.NONE, axis, None)


def _write_tseries(f: de.DEFile, pid: int) -> None:
    q = np.arange(8.0)
    axis = f.axis_range(8, F.QUARTERLY_MAR, _pack_yp(F.QUARTERLY_MAR, 2020, 1))
    f.store_tseries(pid, "ts_q", T.TSERIES, T.FLOAT, F.NONE, axis, q.tobytes(order="F"))

    d = np.array([10.5, 11.5, 12.5])
    axis = f.axis_range(3, F.DAILY, _pack_cal(F.DAILY, 2024, 1, 15))
    f.store_tseries(pid, "ts_d", T.TSERIES, T.FLOAT, F.NONE, axis, d.tobytes(order="F"))

    m = np.array([-1.0, -2.0, -3.0, -4.0])
    axis = f.axis_range(4, F.MONTHLY, _pack_yp(F.MONTHLY, 2023, 11))
    f.store_tseries(pid, "ts_m", T.TSERIES, T.FLOAT, F.NONE, axis, m.tobytes(order="F"))

    qi = np.array([5, 6, 7, 8], dtype=np.int32)
    axis = f.axis_range(4, F.QUARTERLY_MAR, _pack_yp(F.QUARTERLY_MAR, 2020, 1))
    f.store_tseries(pid, "ts_i32", T.TSERIES, T.SIGNED, F.NONE, axis, qi.tobytes(order="F"))


def _write_matrices(f: de.DEFile, pid: int) -> None:
    # Column-major, so this is [[1, 3, 5], [2, 4, 6]].
    mat = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    f.store_mvtseries(
        pid,
        "mat",
        T.MATRIX,
        T.FLOAT,
        F.NONE,
        f.axis_plain(2),
        f.axis_plain(3),
        mat.tobytes(order="F"),
    )

    mv = np.arange(1.0, 13.0).reshape(4, 3, order="F")
    axis1 = f.axis_range(4, F.QUARTERLY_MAR, _pack_yp(F.QUARTERLY_MAR, 2021, 1))
    axis2 = f.axis_names(["a", "b", "c"])
    f.store_mvtseries(
        pid, "mvts", T.MVTSERIES, T.FLOAT, F.NONE, axis1, axis2, mv.tobytes(order="F")
    )


def _write_tensors(f: de.DEFile, pid: int) -> None:
    nd = np.arange(24.0).reshape(2, 3, 4, order="F")
    axes = [f.axis_plain(2), f.axis_plain(3), f.axis_plain(4)]
    f.store_ndtseries(pid, "nd", T.TENSOR, T.FLOAT, F.NONE, axes, nd.tobytes(order="F"))


def write_reference(path: str) -> None:
    """Write the reference database to ``path``, replacing anything there."""
    with de.opendaec(path, write=True, append=False) as f:
        _write_scalars(f, de.ROOT_ID)
        _write_dates(f, f.new_catalog(de.ROOT_ID, "dates"))
        _write_vectors(f, f.new_catalog(de.ROOT_ID, "vectors"))
        _write_tseries(f, f.new_catalog(de.ROOT_ID, "series"))

        arrays = f.new_catalog(de.ROOT_ID, "arrays")
        _write_matrices(f, arrays)
        _write_tensors(f, arrays)

        inner = f.new_catalog(f.new_catalog(de.ROOT_ID, "nested"), "inner")
        f.store_scalar(inner, "leaf", T.FLOAT, F.NONE, np.float64(42.0).tobytes())

        f.set_attribute("/series/ts_q", "units", "billions")
        f.set_attribute("/series/ts_q", "source", "reference C driver")
