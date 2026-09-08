# SPDX-License-Identifier: MIT
"""DataEconPy -- read and write DataEcon (``.daec``) files from Python.

``.daec`` is the time-series container used by the Bank of Canada's
`TimeSeriesEcon.jl <https://github.com/bankofcanada/TimeSeriesEcon.jl>`_. This
package is the Python connector for it, built on the same
`DataEcon <https://github.com/bankofcanada/DataEcon>`_ C library that the Julia
and MATLAB connectors use, so files written from any of them are the same files.

Its primary vocabulary is `TimeSeriesEconPy
<https://github.com/Nic2020/TimeSeriesEconPy>`_ -- ``TSeries``, ``MVTSeries``
and ``Workspace`` map onto the format's time series, multivariate time series
and catalogs. pandas and polars are reachable through :mod:`dataecon.interop`.

Quick start
-----------

.. code-block:: python

    import dataecon as de
    import tsecon as ts
    import numpy as np

    gdp = ts.TSeries(ts.qq(2020, 1), np.arange(8.0))

    de.writedb("model.daec", {"gdp": gdp, "params": {"beta": 0.99}})

    data = de.readdb("model.daec")
    data.gdp          # the TSeries, back again
    data.params.beta  # 0.99

Or work with the file handle directly:

.. code-block:: python

    with de.opendaec("model.daec", write=True) as f:
        f["gdp"] = gdp
        f["params/beta"] = 0.99
        print(f.read("gdp"))
"""

from __future__ import annotations

from ._clib import library_version
from ._consts import (
    DE_MAX_AXES,
    DE_VERSION,
    AxisType,
    Class,
    Frequency,
    Type,
)
from ._highlevel import readdb, writedb
from ._loader import library_filename
from .errors import (
    DEArgumentError,
    DEBadClassError,
    DEBadFrequencyError,
    DEBadNameError,
    DEBadTypeError,
    DEError,
    DEExistsError,
    DEInexactError,
    DELibraryNotFoundError,
    DEMissingAttributeError,
    DEObjectDoesNotExistError,
    DERangeError,
    DEReadOnlyError,
    DESQLiteError,
    DEUnsupportedError,
)
from .file import ROOT_ID, Axis, DEFile, ObjectInfo, opendaec, opendaecmem

__version__ = "0.1.0"

__all__ = [
    "DE_MAX_AXES",
    "DE_VERSION",
    "ROOT_ID",
    "Axis",
    "AxisType",
    "Class",
    "DEArgumentError",
    "DEBadClassError",
    "DEBadFrequencyError",
    "DEBadNameError",
    "DEBadTypeError",
    "DEError",
    "DEExistsError",
    "DEFile",
    "DEInexactError",
    "DELibraryNotFoundError",
    "DEMissingAttributeError",
    "DEObjectDoesNotExistError",
    "DERangeError",
    "DEReadOnlyError",
    "DESQLiteError",
    "DEUnsupportedError",
    "Frequency",
    "ObjectInfo",
    "Type",
    "__version__",
    "library_filename",
    "library_version",
    "opendaec",
    "opendaecmem",
    "readdb",
    "writedb",
]
