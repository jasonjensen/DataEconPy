# Tests

```bash
pip install -e ".[all]" && pip install pytest
pytest
```

The suite needs the compiled `libdaec`; `pip install -e .` builds it. If it is
missing, run `python -m dataecon.build` or set `DATAECON_LIBRARY`.

## What each module covers

| Module | Covers |
|---|---|
| `test_dates.py` | Date packing per frequency, and that `tsecon` MIT integers equal DataEcon date codes |
| `test_format_invariants.py` | Byte-level guarantees: NUL-terminated strings, column-major layout, element widths, axis sharing |
| `test_roundtrip.py` | Every value kind in and back out again |
| `test_api.py` | Opening, the mapping protocol, attributes, searching, error behaviour |
| `test_interop.py` | pandas and polars, and that the core works without them |
| `test_c_parity.py` | Cross-validation against the DataEcon C library, both directions |
| `test_julia.py` | Cross-validation against the Julia connector (skipped without Julia) |

## Cross-connector validation

Compatibility is the whole point of this package, so it is checked against other
implementations rather than only against itself.

### Against the C library (runs everywhere)

`cross/c_reference.c` links directly against `libdaec` and writes a database
using the same conventions as the Julia connector — the same axis kind per
object type, the same `jtype`/`jeltype` attributes, column-major element order.
It can also dump any database in a canonical text format. `cross/pywrite.py`
writes the same database through DataEconPy's low-level API, and
`cross/pydump.py` dumps in the same format.

`test_c_parity.py` closes the triangle:

* the C dump and the Python dump of a **C-written** file must be identical
  (Python reads what C writes);
* the C dump of a **Python-written** file must equal the C dump of the C-written
  file (Python writes what C reads).

Because both sides emit the same text, a disagreement about element width, byte
order, orientation, axis kind or attributes shows up as a diff on the offending
object.

The driver is compiled by a session fixture; the tests skip if no C compiler or
no built library is available.

### Against the Julia connector (needs Julia)

`test_julia.py` drives `cross/julia_write.jl` and `cross/julia_read.jl`, which
use `TimeSeriesEcon.jl`'s `DataEcon` module. Julia writes a database that Python
verifies, and Python writes one that Julia verifies.

These are skipped unless Julia is available with `TimeSeriesEcon` installed:

```bash
julia -e 'using Pkg; Pkg.add("TimeSeriesEcon")'
pytest -m julia
```

Set `JULIA` if Julia is not on `PATH`, and `JULIA_PROJECT` to point at an
environment that has `TimeSeriesEcon`:

```bash
JULIA=/opt/julia/bin/julia JULIA_PROJECT=~/julia-envs/tse pytest -m julia
```

By construction, the Julia connector is a thin binding over the very `libdaec`
that `test_c_parity.py` exercises, so the C parity tests cover the same format
questions on machines without a Julia toolchain. The Julia tests add the layer
above that: the connector's own conventions for `jtype`, axis choice and
`Workspace` structure.

When Julia is not available, `test_c_parity.py::TestJuliaFixtureIsValidWithoutJulia`
still checks that the fixture `test_julia.py` writes uses the object types the
Julia connector requires (a `type_tseries` with a date-range axis, a
`type_mvtseries` with range and names axes, and so on), so the fixture cannot
rot unnoticed.

## Known cross-connector asymmetries

These are properties of the format, not defects, and the tests assert them:

* **Julia `Bool` scalars read back as integers.** The Julia connector writes
  `jtype` only when the stored value's type differs from the original, and a
  `Bool` is stored as itself (a 1-byte signed integer). Nothing in the file
  records that it was boolean. Python's own `bool` round-trips, because we write
  `pytype` for it — but Julia will read *that* as an integer, symmetrically.
* **Plain ranges lose their offset.** A plain axis stores only a length, so
  Julia writes `1:6` and reads `1:6` (1-based), while Python writes `range(0, 6)`
  and reads `range(0, 6)` (0-based). Python additionally records a `pystart`
  attribute so a non-zero start survives a Python-to-Python round trip.
* **Python never writes `jtype`.** The Julia connector evaluates that attribute
  with `Meta.parse`, so a Python type name in it would raise there. Python
  writes `pytype`/`pyeltype`, which Julia ignores, and reads Julia's
  `jtype`/`jeltype` for the types it can map.

## Markers

* `julia` — needs a Julia toolchain. Run with `pytest -m julia`, or exclude with
  `pytest -m "not julia"`.
