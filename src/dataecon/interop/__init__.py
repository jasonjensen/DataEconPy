# SPDX-License-Identifier: MIT
"""Conversions between ``.daec`` contents and third-party data structures.

The connector's native vocabulary is TimeSeriesEconPy (``TSeries``,
``MVTSeries``, ``Workspace``) and numpy. This subpackage adds conversions to and
from the two dataframe libraries people most often want next.

Every backend is optional; importing this module never requires any of them.
"""

from __future__ import annotations

__all__ = [
    "from_pandas",
    "from_polars",
    "to_pandas",
    "to_polars",
]


def __getattr__(name: str) -> object:
    if name in ("to_pandas", "from_pandas"):
        from . import pandas as _pandas

        return getattr(_pandas, name)
    if name in ("to_polars", "from_polars"):
        from . import polars as _polars

        return getattr(_polars, name)
    raise AttributeError(name)
