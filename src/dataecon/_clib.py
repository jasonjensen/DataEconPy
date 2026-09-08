# SPDX-License-Identifier: MIT
"""``ctypes`` bindings for the DataEcon C library.

This module is a direct, mechanical translation of ``include/daec.h``; it is the
Python counterpart of ``C.jl`` in the Julia connector. It adds no behaviour of
its own beyond argument/return typing and turning non-zero result codes into
exceptions -- everything above it in the package goes through these functions,
so that a ``.daec`` file is only ever touched by the reference implementation.

The library handle is created lazily on first use, so importing :mod:`dataecon`
never fails just because the shared library is missing; the failure surfaces at
the point where a file is actually opened.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import (
    POINTER,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_void_p,
)
from typing import Any

from ._consts import DE_MAX_AXES
from ._loader import REQUIRED_VERSION, load_library
from .errors import DEError, exception_for_code

__all__ = [
    "MAX_AXES",
    "AxisT",
    "MvtseriesT",
    "NdtseriesT",
    "ObjectT",
    "ScalarT",
    "TseriesT",
    "check",
    "lib",
    "library_version",
]

MAX_AXES = DE_MAX_AXES

# --- C types ---------------------------------------------------------------

ObjIdT = c_int64
AxisIdT = c_int64
DateT = c_int64
DeFile = c_void_p
DeSearch = c_void_p
# `class_t`, `type_t`, `axis_type_t` and `frequency_t` are all C enums, i.e. int.
ClassT = c_int
TypeT = c_int
AxisTypeT = c_int
FrequencyT = c_int


class ObjectT(ctypes.Structure):
    """Mirror of ``object_t`` -- one row of the ``objects`` table."""

    _fields_ = (
        ("id", ObjIdT),
        ("pid", ObjIdT),
        ("obj_class", ClassT),
        ("obj_type", TypeT),
        ("name", c_char_p),
    )


class ScalarT(ctypes.Structure):
    """Mirror of ``scalar_t``."""

    _fields_ = (
        ("object", ObjectT),
        ("frequency", FrequencyT),
        ("nbytes", c_int64),
        ("value", c_void_p),
    )


class AxisT(ctypes.Structure):
    """Mirror of ``axis_t``."""

    _fields_ = (
        ("id", AxisIdT),
        ("ax_type", AxisTypeT),
        ("length", c_int64),
        ("frequency", FrequencyT),
        ("first", c_int64),
        ("names", c_char_p),
    )


class TseriesT(ctypes.Structure):
    """Mirror of ``tseries_t`` (also used for plain vectors and ranges)."""

    _fields_ = (
        ("object", ObjectT),
        ("eltype", TypeT),
        ("elfreq", FrequencyT),
        ("axis", AxisT),
        ("nbytes", c_int64),
        ("value", c_void_p),
    )


class MvtseriesT(ctypes.Structure):
    """Mirror of ``mvtseries_t`` (also used for plain matrices)."""

    _fields_ = (
        ("object", ObjectT),
        ("eltype", TypeT),
        ("elfreq", FrequencyT),
        ("axis1", AxisT),
        ("axis2", AxisT),
        ("nbytes", c_int64),
        ("value", c_void_p),
    )


class NdtseriesT(ctypes.Structure):
    """Mirror of ``ndtseries_t`` (also used for plain N-d arrays)."""

    _fields_ = (
        ("object", ObjectT),
        ("eltype", TypeT),
        ("elfreq", FrequencyT),
        ("naxes", c_int64),
        ("axis", AxisT * MAX_AXES),
        ("nbytes", c_int64),
        ("value", c_void_p),
    )


# --- prototypes ------------------------------------------------------------

_PROTOTYPES: dict[str, tuple[list[Any], Any]] = {
    # misc
    "de_version": ([], c_char_p),
    "de_pack_strings": ([POINTER(c_char_p), c_int64, c_char_p, POINTER(c_int64)], c_int),
    "de_unpack_strings": ([c_char_p, c_int64, POINTER(c_char_p), c_int64], c_int),
    # error
    "de_error": ([c_char_p, c_size_t], c_int),
    "de_error_source": ([c_char_p, c_size_t], c_int),
    "de_clear_error": ([], c_int),
    # file
    "de_open": ([c_char_p, POINTER(DeFile)], c_int),
    "de_open_readonly": ([c_char_p, POINTER(DeFile)], c_int),
    "de_open_memory": ([POINTER(DeFile)], c_int),
    "de_close": ([DeFile], c_int),
    "de_truncate": ([DeFile], c_int),
    # object
    "de_find_object": ([DeFile, ObjIdT, c_char_p, POINTER(ObjIdT)], c_int),
    "de_load_object": ([DeFile, ObjIdT, POINTER(ObjectT)], c_int),
    "de_delete_object": ([DeFile, ObjIdT], c_int),
    "de_set_attribute": ([DeFile, ObjIdT, c_char_p, c_char_p], c_int),
    "de_get_attribute": ([DeFile, ObjIdT, c_char_p, POINTER(c_char_p)], c_int),
    "de_get_all_attributes": (
        [DeFile, ObjIdT, c_char_p, POINTER(c_int64), POINTER(c_char_p), POINTER(c_char_p)],
        c_int,
    ),
    "de_get_object_info": (
        [DeFile, ObjIdT, POINTER(c_char_p), POINTER(c_int64), POINTER(c_int64)],
        c_int,
    ),
    "de_find_fullpath": ([DeFile, c_char_p, POINTER(ObjIdT)], c_int),
    "de_catalog_size": ([DeFile, ObjIdT, POINTER(c_int64)], c_int),
    # catalog
    "de_new_catalog": ([DeFile, ObjIdT, c_char_p, POINTER(ObjIdT)], c_int),
    # date
    "de_pack_year_period_date": ([FrequencyT, c_int32, c_uint32, POINTER(DateT)], c_int),
    "de_unpack_year_period_date": (
        [FrequencyT, DateT, POINTER(c_int32), POINTER(c_uint32)],
        c_int,
    ),
    "de_pack_calendar_date": (
        [FrequencyT, c_int32, c_uint32, c_uint32, POINTER(DateT)],
        c_int,
    ),
    "de_unpack_calendar_date": (
        [FrequencyT, DateT, POINTER(c_int32), POINTER(c_uint32), POINTER(c_uint32)],
        c_int,
    ),
    # scalar
    "de_store_scalar": (
        [DeFile, ObjIdT, c_char_p, TypeT, FrequencyT, c_int64, c_void_p, POINTER(ObjIdT)],
        c_int,
    ),
    "de_load_scalar": ([DeFile, ObjIdT, POINTER(ScalarT)], c_int),
    # axis
    "de_axis_plain": ([DeFile, c_int64, POINTER(AxisIdT)], c_int),
    "de_axis_range": ([DeFile, c_int64, FrequencyT, c_int64, POINTER(AxisIdT)], c_int),
    "de_axis_names": ([DeFile, c_int64, c_char_p, POINTER(AxisIdT)], c_int),
    "de_load_axis": ([DeFile, AxisIdT, POINTER(AxisT)], c_int),
    # tseries / mvtseries / ndtseries
    "de_store_tseries": (
        [
            DeFile,
            ObjIdT,
            c_char_p,
            TypeT,
            TypeT,
            FrequencyT,
            AxisIdT,
            c_int64,
            c_void_p,
            POINTER(ObjIdT),
        ],
        c_int,
    ),
    "de_load_tseries": ([DeFile, ObjIdT, POINTER(TseriesT)], c_int),
    "de_store_mvtseries": (
        [
            DeFile,
            ObjIdT,
            c_char_p,
            TypeT,
            TypeT,
            FrequencyT,
            AxisIdT,
            AxisIdT,
            c_int64,
            c_void_p,
            POINTER(ObjIdT),
        ],
        c_int,
    ),
    "de_load_mvtseries": ([DeFile, ObjIdT, POINTER(MvtseriesT)], c_int),
    "de_store_ndtseries": (
        [
            DeFile,
            ObjIdT,
            c_char_p,
            TypeT,
            TypeT,
            FrequencyT,
            c_int64,
            POINTER(AxisIdT),
            c_int64,
            c_void_p,
            POINTER(ObjIdT),
        ],
        c_int,
    ),
    "de_load_ndtseries": ([DeFile, ObjIdT, POINTER(NdtseriesT)], c_int),
    # search
    "de_list_catalog": ([DeFile, ObjIdT, POINTER(DeSearch)], c_int),
    "de_search_catalog": ([DeFile, ObjIdT, c_char_p, TypeT, ClassT, POINTER(DeSearch)], c_int),
    "de_next_object": ([DeSearch, POINTER(ObjectT)], c_int),
    "de_finalize_search": ([DeSearch], c_int),
}


class _Lib:
    """Lazily loaded proxy for the shared library.

    Attribute access loads the library on first use and returns the prepared
    function, so ``lib.de_open(...)`` reads exactly like a direct call.
    """

    def __init__(self) -> None:
        self._dll: ctypes.CDLL | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> ctypes.CDLL:
        dll = self._dll
        if dll is not None:
            return dll
        with self._lock:
            if self._dll is None:
                dll = load_library()
                for name, (argtypes, restype) in _PROTOTYPES.items():
                    func = getattr(dll, name)
                    func.argtypes = argtypes
                    func.restype = restype
                self._dll = dll
            return self._dll

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure(), name)

    @property
    def loaded(self) -> bool:
        """Whether the shared library has been loaded yet."""
        return self._dll is not None

    @property
    def path(self) -> str:
        """Filesystem path of the loaded shared library."""
        return self._ensure()._name


#: The shared library. Loaded on first attribute access.
lib = _Lib()


# --- error handling --------------------------------------------------------

_ERROR_BUFFER_SIZE = 1024
#: Set to true to have error messages include their C source location.
DEBUG_SOURCE = False


def last_error(rc: int) -> DEError:
    """Build the exception for the most recent library error and clear it."""
    buf = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    reader = lib.de_error_source if DEBUG_SOURCE else lib.de_error
    reported = reader(buf, _ERROR_BUFFER_SIZE)
    message = buf.value.decode("utf-8", errors="replace")
    # `rc` is what the failing call returned; `reported` is what the library's
    # error slot holds. They normally agree; prefer `rc` when they do not.
    return exception_for_code(rc or reported, message)


def check(rc: int) -> int:
    """Raise the matching exception if ``rc`` is not ``DE_SUCCESS``."""
    if rc != 0:
        raise last_error(rc)
    return rc


def clear_error() -> None:
    """Reset the library's error tracking."""
    lib.de_clear_error()


def library_version() -> str:
    """Return the version string reported by the loaded shared library."""
    return lib.de_version().decode("ascii")


def check_version() -> None:
    """Warn-free compatibility check between this package and the library.

    Raises
    ------
    DEError
        If the loaded library has a different major/minor version than the
        header this package was written against.
    """
    have = library_version()
    want = REQUIRED_VERSION
    if have.split(".")[:2] != want.split(".")[:2]:
        msg = (
            f"Loaded libdaec version {have} is not compatible with DataEconPy, "
            f"which requires {want}. Library path: {lib.path}"
        )
        raise DEError(msg)
