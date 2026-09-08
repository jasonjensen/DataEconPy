# DataEconPy

A Python connector for the **DataEcon** (`.daec`) time-series format.

`.daec` is the container that the Bank of Canada's
[TimeSeriesEcon.jl](https://github.com/bankofcanada/TimeSeriesEcon.jl) uses to
store macroeconomic data. This package lets Python read and write those files,
speaking natively to
[TimeSeriesEconPy](https://github.com/Nic2020/TimeSeriesEconPy) and, through it,
to pandas and polars.

All file access goes through the reference
[DataEcon](https://github.com/bankofcanada/DataEcon) C library — the same
`libdaec` that the Julia and MATLAB connectors call — so a file written from
Python is the same file the other connectors write, byte for byte. The C
sources are vendored in `vendor/dataecon-c/` and compiled at build time.

## Install

```bash
pip install DataEconPy                # core: numpy only
pip install "DataEconPy[tsecon]"      # + TimeSeriesEconPy (recommended)
pip install "DataEconPy[all]"         # + pandas and polars
```

Installing from source needs a C compiler; the build compiles `libdaec` into the
package. If that step is skipped or fails, build it afterwards with:

```bash
python -m dataecon.build
```

or point `DATAECON_LIBRARY` at a copy you already have (the one from
[`DataEcon_jll`](https://github.com/JuliaBinaryWrappers/DataEcon_jll.jl/releases)
works).

## Usage

### Whole files at once

```python
import dataecon as de
import tsecon as ts
import numpy as np

gdp = ts.TSeries(ts.qq(2020, 1), np.arange(8.0))

de.writedb(
    "model.daec",
    {
        "gdp": gdp,
        "params": {"beta": 0.99, "label": "baseline"},
    },
)

data = de.readdb("model.daec")
data.gdp  # the TSeries, unchanged
data.params.beta  # 0.99
```

`readdb` returns a `tsecon.Workspace` (nested catalogs become nested
Workspaces). Pass `as_dict=True` for plain dictionaries, which is also what you
get when TimeSeriesEconPy is not installed.

### One object at a time

```python
with de.opendaec("model.daec", write=True) as f:
    f["gdp"] = gdp  # catalogs along the path are created
    f["shocks/monetary"] = shock
    f.set_attribute("gdp", "units", "billions of chained 2012 dollars")

with de.opendaec("model.daec") as f:
    f.keys()  # ['gdp', 'shocks']
    list(f.walk())  # ['/gdp', '/shocks/monetary']
    f.read("shocks/monetary")
    f.get_attribute("gdp", "units")
```

`opendaec` opens read-only by default, mirroring the Julia connector. Pass
`write=True` to modify a file, `overwrite=True` to let a store replace an
existing name, and `append=False` to empty the file first.

### pandas and polars

```python
from dataecon.interop.pandas import read_pandas, write_pandas

with de.opendaec("model.daec", write=True) as f:
    write_pandas(f, "cpi", df)  # a PeriodIndex/DatetimeIndex frame
    read_pandas(f, "cpi")  # back as a DataFrame
```

Frequency inference is delegated to TimeSeriesEconPy's own
`from_pandas`/`to_pandas`, so there is one source of truth for what a quarterly
index means. A frame with no time index is stored as a plain matrix; one with a
non-time index is refused rather than silently losing it.

## What maps to what

| Python | Stored as |
|---|---|
| `Workspace`, `dict` | catalog (recursively) |
| `TSeries` | `class_tseries` / `type_tseries`, date-range axis |
| `MVTSeries` | `class_mvtseries` / `type_mvtseries`, date-range + names axes |
| `MITRange`, `range` | `type_range` |
| 1-d array | `class_tseries` / `type_vector` |
| 2-d array | `class_mvtseries` / `type_matrix` |
| 3- to 5-d array | `class_ndtseries` / `type_tensor` |
| `int`, `float`, `complex`, `str`, `bytes`, `bool`, `datetime`, `MIT`, `Duration` | `class_scalar` |

Arrays are stored column-major, as the C and Julia connectors do.

Types the format cannot describe on their own — `bool`, `datetime` — are stored
in the nearest representable form plus a `pytype`/`pyeltype` attribute naming
the original. Reading also honours the `jtype`/`jeltype` attributes written by
the Julia connector, so values written there come back as the closest Python
equivalent.

## Compatibility

Written against DataEcon `0.4.0`. The loaded library's version is checked
against that on demand via `dataecon.library_version()`.

The test suite includes a parity harness (`tests/test_c_parity.py`) that writes
files with a C driver linked directly against `libdaec` and reads them back in
Python, and vice versa, covering every storage class, element type and
frequency. See [`tests/README.md`](tests/README.md) for how to run the
cross-connector checks, including against Julia.

## Licence

MIT — see [LICENSE](LICENSE). Bundles the DataEcon C library (BSD 3-Clause,
Bank of Canada) and SQLite (public domain); see `vendor/dataecon-c/`.
