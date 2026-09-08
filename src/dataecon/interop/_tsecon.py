# SPDX-License-Identifier: MIT
"""Bridge between DataEcon and TimeSeriesEconPy.

TimeSeriesEconPy is the primary target of this connector, but it is an optional
dependency: every entry point here degrades gracefully when ``tsecon`` is not
installed, so ``dataecon`` remains usable with plain numpy arrays alone.

Two things are mapped here:

**Frequencies.** ``tsecon``'s ``Frequency`` objects map onto DataEcon's integer
frequency codes. The correspondence is the same one the Julia connector uses, so
that a quarterly series written from Python is quarterly when read from Julia.

**Dates.** A ``tsecon`` ``MIT`` and a DataEcon date are both integers counting
periods from the same epoch, and for every frequency the two encodings agree.
Rather than assume that, :func:`verify_frequency_encoding` checks it against the
C library once per frequency, so a divergence would surface as a clear error
instead of silently shifting a series in time.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from .._consts import Frequency, freq_has_ppy
from ..errors import DEBadFrequencyError, DEError

if TYPE_CHECKING:  # pragma: no cover
    from .._codec import ArrayValues, ScalarData

__all__ = [
    "available",
    "freq_from_code",
    "freq_to_code",
    "mit_from_code",
    "mit_to_code",
    "verify_frequency_encoding",
]

try:  # pragma: no cover - trivial import guard
    import tsecon as _ts

    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _ts = None  # type: ignore[assignment]
    _AVAILABLE = False


def available() -> bool:
    """Return whether TimeSeriesEconPy is importable."""
    return _AVAILABLE


def _require() -> Any:
    if not _AVAILABLE:
        msg = (
            "This operation needs TimeSeriesEconPy. Install it with "
            "`pip install DataEconPy[tsecon]`."
        )
        raise DEError(msg)
    return _ts


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------


def _mod1(value: int, modulus: int) -> int:
    """Julia's ``mod1``: like ``%`` but returning ``modulus`` instead of 0."""
    return (value - 1) % modulus + 1


def freq_to_code(frequency: Any) -> int:
    """Return the DataEcon frequency code for a ``tsecon`` frequency."""
    ts = _require()
    if isinstance(frequency, int):
        return frequency
    if isinstance(frequency, ts.Unit):
        return int(Frequency.UNIT)
    if isinstance(frequency, ts.Daily):
        return int(Frequency.DAILY)
    if isinstance(frequency, ts.BDaily):
        return int(Frequency.BDAILY)
    if isinstance(frequency, ts.Weekly):
        return int(Frequency.WEEKLY) + _mod1(frequency.end_day, 7)
    if isinstance(frequency, ts.Monthly):
        return int(Frequency.MONTHLY)
    if isinstance(frequency, ts.Quarterly):
        return int(Frequency.QUARTERLY) + _mod1(frequency.end_month, 3)
    if isinstance(frequency, ts.HalfYearly):
        return int(Frequency.HALFYEARLY) + _mod1(frequency.end_month, 6)
    if isinstance(frequency, ts.Yearly):
        return int(Frequency.YEARLY) + _mod1(frequency.end_month, 12)

    msg = f"Cannot store a series of frequency {frequency!r} in a .daec file."
    raise DEBadFrequencyError(msg)


def freq_from_code(code: int) -> Any:
    """Return the ``tsecon`` frequency for a DataEcon frequency code.

    The family is recovered by a bitwise test in the same order the Julia
    connector uses, and the low bits give the end period. A family code with no
    end period (``freq_weekly`` = 16, say) means the family default, which is
    exactly the frequency the C library treats it as.
    """
    ts = _require()
    code = int(code)
    if code == Frequency.UNIT:
        return ts.Unit()
    if code == Frequency.DAILY:
        return ts.Daily()
    if code == Frequency.BDAILY:
        return ts.BDaily()
    if code in (Frequency.MONTHLY, Frequency.MONTHLY + 1):
        return ts.Monthly()

    for family, factory, modulus in (
        (Frequency.WEEKLY, ts.Weekly, 7),
        (Frequency.QUARTERLY, ts.Quarterly, 3),
        (Frequency.HALFYEARLY, ts.HalfYearly, 6),
        (Frequency.YEARLY, ts.Yearly, 12),
    ):
        if code & family:
            end = code - family
            return factory(modulus if end == 0 else end)

    msg = f"Unsupported frequency code in .daec file: {code}"
    raise DEBadFrequencyError(msg)


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------


def _c_pack_year_period(code: int, year: int, period: int) -> int:
    from .. import _clib as C

    out = ctypes.c_int64()
    C.check(C.lib.de_pack_year_period_date(code, year, period, ctypes.byref(out)))
    return out.value


def _c_unpack_year_period(code: int, date: int) -> tuple[int, int]:
    from .. import _clib as C

    year = ctypes.c_int32()
    period = ctypes.c_uint32()
    C.check(C.lib.de_unpack_year_period_date(code, date, ctypes.byref(year), ctypes.byref(period)))
    return year.value, period.value


def _c_pack_calendar(code: int, year: int, month: int, day: int) -> int:
    from .. import _clib as C

    out = ctypes.c_int64()
    C.check(C.lib.de_pack_calendar_date(code, year, month, day, ctypes.byref(out)))
    return out.value


def _c_unpack_calendar(code: int, date: int) -> tuple[int, int, int]:
    from .. import _clib as C

    year = ctypes.c_int32()
    month = ctypes.c_uint32()
    day = ctypes.c_uint32()
    C.check(
        C.lib.de_unpack_calendar_date(
            code, date, ctypes.byref(year), ctypes.byref(month), ctypes.byref(day)
        )
    )
    return year.value, month.value, day.value


@cache
def verify_frequency_encoding(freq_code: int) -> None:
    """Check that ``tsecon`` and DataEcon agree on this frequency's date codes.

    ``MIT`` values and DataEcon dates are both integers, and this connector
    relies on them being the *same* integers so that whole arrays of dates can
    be written without a per-element conversion. This asks the C library to pack
    a handful of dates and compares them with what ``tsecon`` produces.

    The result is cached per frequency code, so the cost is a few C calls the
    first time a frequency is used.

    Raises
    ------
    DEError
        If the two encodings disagree, naming the date that differs.
    """
    ts = _require()
    frequency = freq_from_code(freq_code)

    if freq_has_ppy(freq_code):
        samples = [(y, p) for y in (1900, 1970, 2000, 2024) for p in (1, 2)]
        for year, period in samples:
            try:
                mit = ts.MIT.from_yp(frequency, year, period)
            except (AttributeError, ValueError, TypeError):
                return  # tsecon cannot express this sample; nothing to compare
            expected = _c_pack_year_period(freq_code, year, period)
            if int(mit) != expected:
                _raise_mismatch(frequency, f"{year}-{period}", int(mit), expected)
        return

    for date in (_dt.date(1970, 1, 5), _dt.date(2000, 3, 1), _dt.date(2024, 6, 12)):
        expected = _c_pack_calendar(freq_code, date.year, date.month, date.day)
        if freq_code == Frequency.DAILY:
            mit = ts.daily(date)
        elif freq_code == Frequency.BDAILY:
            mit = ts.bdaily(date)
        else:
            mit = ts.weekly(date, freq_code % 16 or 7)
        if int(mit) != expected:
            _raise_mismatch(frequency, str(date), int(mit), expected)


def _raise_mismatch(frequency: Any, what: str, got: int, expected: int) -> None:
    msg = (
        f"TimeSeriesEconPy and DataEcon disagree on the integer code for "
        f"{what} at frequency {frequency!r}: tsecon says {got}, the DataEcon C "
        f"library says {expected}. Refusing to write data that would be read "
        f"back shifted in time."
    )
    raise DEError(msg)


def mit_to_code(mit: Any) -> tuple[int, int]:
    """Return ``(date_code, frequency_code)`` for a ``tsecon`` ``MIT``."""
    freq_code = freq_to_code(mit.frequency)
    verify_frequency_encoding(freq_code)
    return int(mit), freq_code


def mit_from_code(code: int, freq_code: int) -> Any:
    """Build a ``tsecon`` ``MIT`` from a stored date code."""
    ts = _require()
    verify_frequency_encoding(freq_code)
    return ts.MIT(freq_from_code(freq_code), int(code))


def duration_from_code(code: int, freq_code: int) -> Any:
    """Build a ``tsecon`` ``Duration`` from a stored integer."""
    ts = _require()
    return ts.Duration(freq_from_code(freq_code), int(code))


def is_mit(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``MIT``."""
    return _AVAILABLE and isinstance(value, _ts.MIT)


def is_duration(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``Duration``."""
    return _AVAILABLE and isinstance(value, _ts.Duration)


# ---------------------------------------------------------------------------
# codec hooks -- called from dataecon._codec
# ---------------------------------------------------------------------------


def encode_scalar(value: Any) -> ScalarData | None:
    """Encode an ``MIT`` or ``Duration``; return ``None`` for anything else."""
    from .._codec import ScalarData
    from .._consts import Type

    if is_mit(value):
        code, freq_code = mit_to_code(value)
        return ScalarData(Type.DATE, freq_code, np.int64(code).tobytes(), None)
    if is_duration(value):
        freq_code = freq_to_code(value.frequency)
        return ScalarData(Type.SIGNED, freq_code, np.int64(int(value)).tobytes(), None)
    return None


def encode_mit_array(array: np.ndarray) -> ArrayValues | None:
    """Encode an object array of ``MIT`` values; return ``None`` if it is not one."""
    from .._codec import ArrayValues
    from .._consts import Type

    if not _AVAILABLE or array.size == 0:
        return None
    flat = array.ravel(order="F")
    if not all(isinstance(v, _ts.MIT) for v in flat):
        return None

    frequencies = {v.frequency for v in flat}
    if len(frequencies) > 1:
        msg = (
            "Cannot store an array of MIT values with mixed frequencies: "
            f"{sorted(map(repr, frequencies))}"
        )
        raise DEError(msg)

    freq_code = freq_to_code(next(iter(frequencies)))
    verify_frequency_encoding(freq_code)
    codes = np.fromiter((int(v) for v in flat), dtype=np.int64, count=flat.size)
    return ArrayValues(Type.DATE, freq_code, codes.tobytes(), None)


def mit_array_from_codes(codes: np.ndarray, freq_code: int) -> np.ndarray:
    """Build an object array of ``MIT`` values from stored integer codes."""
    ts = _require()
    verify_frequency_encoding(freq_code)
    frequency = freq_from_code(freq_code)
    out = np.empty(codes.size, dtype=object)
    for i, code in enumerate(codes):
        out[i] = ts.MIT(frequency, int(code))
    return out


# ---------------------------------------------------------------------------
# container types
# ---------------------------------------------------------------------------


def is_tseries(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``TSeries``."""
    return _AVAILABLE and isinstance(value, _ts.TSeries)


def is_mvtseries(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``MVTSeries``."""
    return _AVAILABLE and isinstance(value, _ts.MVTSeries)


def is_mitrange(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``MITRange``."""
    return _AVAILABLE and isinstance(value, _ts.MITRange)


def is_workspace(value: Any) -> bool:
    """Return whether ``value`` is a ``tsecon`` ``Workspace``."""
    return _AVAILABLE and isinstance(value, _ts.Workspace)


def make_tseries(first_code: int, freq_code: int, values: np.ndarray) -> Any:
    """Build a ``TSeries`` from a first-date code and its values."""
    ts = _require()
    return ts.TSeries(mit_from_code(first_code, freq_code), values)


def make_mvtseries(
    first_code: int, freq_code: int, names: Sequence[str], values: np.ndarray
) -> Any:
    """Build an ``MVTSeries`` from a first-date code, column names and values."""
    ts = _require()
    return ts.MVTSeries(mit_from_code(first_code, freq_code), list(names), values)


def make_mitrange(first_code: int, freq_code: int, length: int) -> Any:
    """Build an ``MITRange`` of ``length`` periods starting at ``first_code``."""
    ts = _require()
    first = mit_from_code(first_code, freq_code)
    return ts.MITRange(first, first + (length - 1))


def make_workspace(items: dict[str, Any]) -> Any:
    """Build a ``Workspace`` from a plain dictionary."""
    ts = _require()
    return ts.Workspace(items)
