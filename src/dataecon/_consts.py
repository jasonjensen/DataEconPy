# SPDX-License-Identifier: MIT
"""Constants of the DataEcon (``.daec``) on-disk format.

The numeric values in this module are part of the file format. They mirror the
enumerations in ``include/daec.h`` of the reference C implementation
(`bankofcanada/DataEcon <https://github.com/bankofcanada/DataEcon>`_) and must
not be changed independently of it.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "DE_MAX_AXES",
    "DE_VERSION",
    "AxisType",
    "Class",
    "Frequency",
    "Type",
    "freq_has_ppy",
    "freq_is_calendar",
    "freq_ppy",
]

#: Version of the format specification this connector writes.
#:
#: Stored as the ``DE_VERSION`` attribute of the root catalog, exactly as the C
#: library does, so that ``.daec`` files written from Python are indistinguishable
#: from files written by the C or Julia connectors.
DE_VERSION = "0.4.0"

#: Maximum number of axes (dimensions) an N-d array object may have.
DE_MAX_AXES = 5


class Class(IntEnum):
    """Storage class of an object -- the ``objects.class`` column."""

    CATALOG = 0
    SCALAR = 1
    #: 1-d array. ``VECTOR`` and ``TSERIES`` are the same class.
    VECTOR = 2
    TSERIES = 2
    #: 2-d array. ``MATRIX`` and ``MVTSERIES`` are the same class.
    MATRIX = 3
    MVTSERIES = 3
    #: N-d array. ``TENSOR`` and ``NDTSERIES`` are the same class.
    TENSOR = 4
    NDTSERIES = 4

    #: Wildcard accepted by :func:`~dataecon.file.DEFile.search_catalog`.
    ANY = -1


class Type(IntEnum):
    """Type of an object -- the ``objects.type`` column.

    Values below 10 are also used as *element* types of array objects (the
    ``eltype`` column of the ``tseries``/``mvtseries``/``ndtseries`` tables).
    """

    NONE = 0

    # --- scalar types, stored in the `scalars` table -----------------------
    INTEGER = 1
    SIGNED = 1
    UNSIGNED = 2
    DATE = 3
    FLOAT = 4
    COMPLEX = 5
    STRING = 6
    OTHER_SCALAR = 7

    # --- 1-d types, stored in the `tseries` table --------------------------
    VECTOR = 10
    RANGE = 11
    TSERIES = 12
    OTHER_1D = 13

    # --- 2-d types, stored in the `mvtseries` table ------------------------
    MATRIX = 20
    MVTSERIES = 21
    OTHER_2D = 22

    # --- N-d types, stored in the `ndtseries` table ------------------------
    TENSOR = 30
    NDTSERIES = 31
    OTHER_ND = 32

    #: Wildcard accepted by :func:`~dataecon.file.DEFile.search_catalog`.
    ANY = -1


class AxisType(IntEnum):
    """Kind of an axis -- the ``axes.ax_type`` column."""

    #: A plain ``0 .. length-1`` index range. ``axes.data`` is NULL.
    PLAIN = 0
    #: A range of dates. ``axes.data`` holds the integer code of the first date.
    RANGE = 1
    #: An ordered list of names. ``axes.data`` holds them joined by ``"\\n"``.
    NAMES = 2


class Frequency(IntEnum):
    """Frequency codes.

    The codes are structured so that the *family* can be recovered with a
    bitwise AND and the end-period with a modulo -- see :func:`freq_ppy` and
    :func:`endperiod_of`.
    """

    NONE = 0
    UNIT = 11
    DAILY = 12
    BDAILY = 13

    WEEKLY = 16
    WEEKLY_SUN0 = 16
    WEEKLY_MON = 17
    WEEKLY_TUE = 18
    WEEKLY_WED = 19
    WEEKLY_THU = 20
    WEEKLY_FRI = 21
    WEEKLY_SAT = 22
    WEEKLY_SUN = 23

    MONTHLY = 32

    QUARTERLY = 64
    QUARTERLY_JAN = 65
    QUARTERLY_FEB = 66
    QUARTERLY_MAR = 67

    HALFYEARLY = 128
    HALFYEARLY_JAN = 129
    HALFYEARLY_FEB = 130
    HALFYEARLY_MAR = 131
    HALFYEARLY_APR = 132
    HALFYEARLY_MAY = 133
    HALFYEARLY_JUN = 134

    YEARLY = 256
    YEARLY_JAN = 257
    YEARLY_FEB = 258
    YEARLY_MAR = 259
    YEARLY_APR = 260
    YEARLY_MAY = 261
    YEARLY_JUN = 262
    YEARLY_JUL = 263
    YEARLY_AUG = 264
    YEARLY_SEP = 265
    YEARLY_OCT = 266
    YEARLY_NOV = 267
    YEARLY_DEC = 268


#: Bit mask selecting the four year-period frequency families.
_YP_FREQS = Frequency.MONTHLY | Frequency.QUARTERLY | Frequency.HALFYEARLY | Frequency.YEARLY

_PPY_BY_FAMILY = {
    int(Frequency.MONTHLY): 12,
    int(Frequency.QUARTERLY): 4,
    int(Frequency.HALFYEARLY): 2,
    int(Frequency.YEARLY): 1,
}


def freq_has_ppy(freq: int) -> bool:
    """Return whether ``freq`` is a year-period frequency."""
    return bool(int(freq) & _YP_FREQS)


def freq_ppy(freq: int) -> int:
    """Return the number of periods per year of a year-period frequency.

    Raises
    ------
    ValueError
        If ``freq`` is not one of the year-period frequencies.
    """
    family = int(freq) & _YP_FREQS
    try:
        return _PPY_BY_FAMILY[family]
    except KeyError:
        msg = f"Not a year-period frequency: {freq}"
        raise ValueError(msg) from None


def freq_is_calendar(freq: int) -> bool:
    """Return whether ``freq`` is daily, business-daily or weekly."""
    f = int(freq)
    return f in (Frequency.DAILY, Frequency.BDAILY) or bool(f & Frequency.WEEKLY)


def endperiod_of(freq: int) -> int:
    """Return the end-period encoded in the low bits of a frequency code.

    For weekly frequencies this is the last day of the week (1 = Monday ...
    7 = Sunday); for year-period frequencies it is the last month of the final
    period of the year. ``0`` means the family default.
    """
    return int(freq) % 16
