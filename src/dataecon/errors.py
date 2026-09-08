# SPDX-License-Identifier: MIT
"""Exceptions raised by :mod:`dataecon`.

The class hierarchy mirrors the error codes of the reference C library so that
code ported from C or Julia can map ``DE_*`` result codes onto exceptions. Every
exception carries the corresponding C code in :attr:`DEError.rc`.
"""

from __future__ import annotations

__all__ = [
    "DEAxisDoesNotExistError",
    "DEBadClassError",
    "DEBadNameError",
    "DEBadTypeError",
    "DEError",
    "DEExistsError",
    "DEInexactError",
    "DELibraryNotFoundError",
    "DEMissingAttributeError",
    "DEObjectDoesNotExistError",
    "DERangeError",
    "DEReadOnlyError",
    "DESQLiteError",
    "DEUnsupportedError",
    "exception_for_code",
]


class DEError(Exception):
    """Base class for all errors raised by this package."""

    #: Result code of the equivalent error in the reference C library.
    rc: int = -978  # DE_INTERNAL

    def __init__(self, message: str = "") -> None:
        super().__init__(message or type(self).__doc__ or "")


class DEBadAxisTypeError(DEError):
    """Invalid axis type code."""

    rc = -999


class DEBadNumAxesError(DEError):
    """Invalid number of axes."""

    rc = -998


class DEBadClassError(DEError):
    """The storage class of the object does not match the requested one."""

    rc = -997


class DEBadTypeError(DEError):
    """The type of the object is not valid for its storage class."""

    rc = -996


class DEBadElTypeError(DEError):
    """The element type of an array object is not a scalar type."""

    rc = -995


class DEBadNameError(DEError):
    """Invalid object name."""

    rc = -992


class DEBadFrequencyError(DEError):
    """Unknown or unsupported frequency code."""

    rc = -991


class DEObjectDoesNotExistError(DEError, KeyError):
    """The requested object does not exist."""

    rc = -989

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return self.args[0] if self.args else ""


class DEAxisDoesNotExistError(DEError):
    """The requested axis does not exist."""

    rc = -988


class DEArgumentError(DEError):
    """Invalid combination of arguments."""

    rc = -987


class DEExistsError(DEError):
    """An object with this name already exists in the parent catalog."""

    rc = -985


class DEBadObjectError(DEError):
    """The object exists in the catalog but its data row is missing or corrupt."""

    rc = -984


class DEDeleteRootError(DEError):
    """The root catalog cannot be deleted."""

    rc = -982


class DEMissingAttributeError(DEError, KeyError):
    """The object has no attribute with this name."""

    rc = -981

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class DEInexactError(DEError):
    """The date does not exist in the requested frequency.

    Raised, for example, when a Saturday or Sunday is given as a business-daily
    date. :attr:`code` holds the code of the preceding business day.
    """

    rc = -980

    def __init__(self, message: str = "", code: int = 0) -> None:
        super().__init__(message)
        #: Code of the nearest preceding period.
        self.code = code


class DERangeError(DEError, ValueError):
    """A value is out of the range supported by the format."""

    rc = -979


class DEUnsupportedError(DEError, TypeError):
    """The Python value cannot be represented in the DataEcon format."""

    rc = -996


class DEReadOnlyError(DEError):
    """The file is open read-only and cannot be modified."""

    rc = 8  # SQLITE_READONLY


class DELibraryNotFoundError(DEError, ImportError):
    """The ``libdaec`` shared library could not be found or loaded."""

    rc = -978


class DESQLiteError(DEError):
    """The underlying SQLite database reported an error.

    Positive result codes from the C library come straight from SQLite; see
    <https://sqlite.org/rescode.html>.
    """

    rc = 1


#: Maps a C-library result code onto the exception class that represents it.
_BY_CODE: dict[int, type[DEError]] = {
    cls.rc: cls
    for cls in (
        DEBadAxisTypeError,
        DEBadNumAxesError,
        DEBadClassError,
        DEBadTypeError,
        DEBadElTypeError,
        DEBadNameError,
        DEBadFrequencyError,
        DEObjectDoesNotExistError,
        DEAxisDoesNotExistError,
        DEArgumentError,
        DEExistsError,
        DEBadObjectError,
        DEDeleteRootError,
        DEMissingAttributeError,
        DEInexactError,
        DERangeError,
    )
}


def exception_for_code(rc: int, message: str = "") -> DEError:
    """Return the exception representing C-library result code ``rc``."""
    if rc > 0:
        return DESQLiteError(message or f"SQLite error {rc}")
    cls = _BY_CODE.get(rc)
    if cls is None:
        err = DEError(message or f"DataEcon error {rc}")
        err.rc = rc
        return err
    if cls is DEInexactError:
        return DEInexactError(message)
    return cls(message)
