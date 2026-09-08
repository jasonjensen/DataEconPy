# Vendored `DataEcon` C sources

This directory contains a verbatim copy of the C sources of the reference
DataEcon implementation, which `DataEconPy` compiles into the `libdaec` shared
library it drives through `ctypes`.

| | |
|---|---|
| Upstream | <https://github.com/bankofcanada/DataEcon> |
| Commit | `1a108688a044380f808bebf64079e32dbb9cd1a4` |
| Library version | `0.4.0` (`DE_VERSION` in `include/daec.h`) |
| Licence | BSD 3-Clause -- see [`LICENSE.md`](LICENSE.md) |

## What was and was not copied

Copied: `include/daec.h`, `src/libdaec/` (the library itself) and
`src/sqlite3/` (the SQLite amalgamation the library is built against).

Omitted: `src/test.c`, `src/prof.c` and `src/utils/` (the upstream test,
profiling and command-line-tool sources), which are not needed to build the
library.

`src/sqlite3/` is the standard SQLite amalgamation, version 3.50.2, which is
in the public domain. See `src/sqlite3/README.md`.

## Updating

Copy the files listed above from a newer upstream checkout, update the commit
hash and version in the table, then re-run the test suite -- in particular
`tests/test_c_parity.py`, which asserts that the loaded library reports the
version this package expects.
