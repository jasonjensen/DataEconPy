# SPDX-License-Identifier: MIT
"""pandas interop.

TimeSeriesEconPy already knows how to move between its own types and pandas, so
this module routes through it rather than reimplementing frequency inference:
``TSeries``/``MVTSeries``/``Workspace`` conversions are delegated to
``tsecon.to_pandas`` and ``tsecon.from_pandas``.

What this module adds is the parts specific to ``.daec`` files: reading an
object straight into a pandas object, writing a pandas object straight into a
file, and handling the frame shapes that carry no time index (which become plain
arrays rather than time series).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import DEError, DEUnsupportedError
from . import _tsecon

if TYPE_CHECKING:  # pragma: no cover
    from ..file import DEFile

__all__ = ["from_pandas", "read_pandas", "to_pandas", "write_pandas"]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as err:  # pragma: no cover - depends on the environment
        msg = "pandas is not installed. Install it with `pip install DataEconPy[pandas]`."
        raise DEError(msg) from err
    return pd


def _has_time_index(obj: Any) -> bool:
    """Return whether the index carries dates that map onto a frequency."""
    pd = _require_pandas()
    return isinstance(obj.index, (pd.PeriodIndex, pd.DatetimeIndex))


def from_pandas(obj: Any, **kwargs: Any) -> Any:
    """Convert a pandas ``Series`` or ``DataFrame`` into a storable value.

    A time-indexed object becomes a ``TSeries`` or ``MVTSeries``; anything else
    becomes a plain numpy array, since the format has no way to record an
    arbitrary index.

    Parameters
    ----------
    obj
        The pandas object to convert.
    **kwargs
        Passed through to ``tsecon.from_pandas`` for time-indexed input.
    """
    pd = _require_pandas()
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        msg = f"Not a pandas Series or DataFrame: {type(obj).__name__}"
        raise DEUnsupportedError(msg)

    if _has_time_index(obj):
        if not _tsecon.available():
            msg = (
                "Storing a time-indexed pandas object needs TimeSeriesEconPy, "
                "which supplies the frequency model. Install it with "
                "`pip install DataEconPy[tsecon]`, or pass `obj.to_numpy()` to "
                "store the values without their index."
            )
            raise DEError(msg)
        import tsecon

        return tsecon.from_pandas(obj, **kwargs)

    if isinstance(obj, pd.DataFrame) and not _is_trivial_index(obj.index):
        msg = (
            "This DataFrame has a non-time index, which the .daec format cannot "
            "record. Reset it with `df.reset_index()`, give it a PeriodIndex or "
            "DatetimeIndex, or pass `df.to_numpy()` to store just the values."
        )
        raise DEUnsupportedError(msg)

    return np.asarray(obj.to_numpy())


def _is_trivial_index(index: Any) -> bool:
    """Return whether the index is a plain ``0..n-1`` range, i.e. carries nothing."""
    pd = _require_pandas()
    if isinstance(index, pd.RangeIndex):
        return index.start == 0 and index.step == 1
    return bool(np.array_equal(np.asarray(index), np.arange(len(index))))


def to_pandas(value: Any, **kwargs: Any) -> Any:
    """Convert a value read from a ``.daec`` file into a pandas object.

    ``TSeries`` becomes a ``Series``, ``MVTSeries`` and 2-d arrays become a
    ``DataFrame``, 1-d arrays become a ``Series``, and a ``Workspace`` or dict
    becomes a dict of pandas objects.
    """
    pd = _require_pandas()

    if _tsecon.is_tseries(value) or _tsecon.is_mvtseries(value):
        import tsecon

        return tsecon.to_pandas(value, **kwargs)

    if _tsecon.is_workspace(value):
        return {key: to_pandas(item, **kwargs) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: to_pandas(item, **kwargs) for key, item in value.items()}

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return pd.Series(value)
        if value.ndim == 2:
            return pd.DataFrame(value)
        msg = (
            f"A {value.ndim}-dimensional array has no natural pandas form; "
            "use the numpy array directly."
        )
        raise DEUnsupportedError(msg)

    return value


def read_pandas(de: DEFile, path: str, **kwargs: Any) -> Any:
    """Read the object at ``path`` and return it as a pandas object."""
    return to_pandas(de.read(path), **kwargs)


def write_pandas(de: DEFile, path: str, obj: Any, **kwargs: Any) -> int:
    """Write a pandas object at ``path``, returning the new object's id."""
    return de.write(path, from_pandas(obj, **kwargs))
