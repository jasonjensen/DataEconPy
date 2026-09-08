# SPDX-License-Identifier: MIT
"""polars interop.

The shape of this module mirrors :mod:`dataecon.interop.pandas`. polars has no
index, so a frame carries its dates in a column; ``tsecon.from_polars`` and
``tsecon.to_polars`` handle that convention (``time_col="time"`` by default) and
this module delegates to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import DEError, DEUnsupportedError
from . import _tsecon

if TYPE_CHECKING:  # pragma: no cover
    from ..file import DEFile

__all__ = ["from_polars", "read_polars", "to_polars", "write_polars"]

#: Column name that ``tsecon`` uses for the time axis of a polars frame.
DEFAULT_TIME_COLUMN = "time"


def _require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as err:  # pragma: no cover - depends on the environment
        msg = "polars is not installed. Install it with `pip install DataEconPy[polars]`."
        raise DEError(msg) from err
    return pl


def from_polars(obj: Any, *, time_col: str = DEFAULT_TIME_COLUMN, **kwargs: Any) -> Any:
    """Convert a polars ``DataFrame`` or ``Series`` into a storable value.

    A frame with a time column becomes a ``TSeries`` or ``MVTSeries``; a frame
    without one becomes a plain numpy array, since the format cannot record
    column names outside a multivariate time series.

    polars has no period dtype, so a time column is a plain ``Date`` and its
    frequency is ambiguous. ``tsecon.from_polars`` therefore needs ``freq=``,
    which this function passes through::

        from_polars(frame, freq=tsecon.Quarterly())
    """
    pl = _require_polars()

    if isinstance(obj, pl.Series):
        return obj.to_numpy()

    if not isinstance(obj, pl.DataFrame):
        msg = f"Not a polars DataFrame or Series: {type(obj).__name__}"
        raise DEUnsupportedError(msg)

    if time_col in obj.columns:
        if not _tsecon.available():
            msg = (
                "Storing a time-indexed polars frame needs TimeSeriesEconPy, "
                "which supplies the frequency model. Install it with "
                "`pip install DataEconPy[tsecon]`, or drop the time column to "
                "store just the values."
            )
            raise DEError(msg)
        import tsecon

        return tsecon.from_polars(obj, time_col=time_col, **kwargs)

    return np.asarray(obj.to_numpy())


def to_polars(value: Any, **kwargs: Any) -> Any:
    """Convert a value read from a ``.daec`` file into a polars object."""
    pl = _require_polars()

    if _tsecon.is_tseries(value) or _tsecon.is_mvtseries(value):
        import tsecon

        return tsecon.to_polars(value, **kwargs)

    if _tsecon.is_workspace(value):
        return {key: to_polars(item, **kwargs) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: to_polars(item, **kwargs) for key, item in value.items()}

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return pl.Series(value)
        if value.ndim == 2:
            return pl.DataFrame(value)
        msg = (
            f"A {value.ndim}-dimensional array has no natural polars form; "
            "use the numpy array directly."
        )
        raise DEUnsupportedError(msg)

    return value


def read_polars(de: DEFile, path: str, **kwargs: Any) -> Any:
    """Read the object at ``path`` and return it as a polars object."""
    return to_polars(de.read(path), **kwargs)


def write_polars(de: DEFile, path: str, obj: Any, **kwargs: Any) -> int:
    """Write a polars object at ``path``, returning the new object's id."""
    return de.write(path, from_polars(obj, **kwargs))
