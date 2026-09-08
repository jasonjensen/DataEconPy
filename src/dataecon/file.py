# SPDX-License-Identifier: MIT
"""The :class:`DEFile` handle -- reading and writing ``.daec`` files.

Every operation here delegates to the DataEcon C library through
:mod:`dataecon._clib`. The class exposes two layers:

* a **low-level** layer that mirrors ``daec.h`` one call at a time
  (:meth:`~DEFile.store_scalar`, :meth:`~DEFile.load_tseries`, ...), taking and
  returning raw type codes and ``bytes``; and
* a **value** layer (:meth:`~DEFile.write`, :meth:`~DEFile.read`, and the
  mapping protocol) that converts between those raw blobs and Python objects.

Pointers handed back by the C library are only valid until the next library
call, so everything read through them is copied immediately.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, overload

from . import _clib as C
from ._consts import DE_MAX_AXES, AxisType, Class, Frequency, Type
from .errors import (
    DEArgumentError,
    DEError,
    DEExistsError,
    DEMissingAttributeError,
    DEObjectDoesNotExistError,
    DERangeError,
    DEReadOnlyError,
)

__all__ = ["ROOT_ID", "Axis", "DEFile", "ObjectInfo", "opendaec", "opendaecmem"]

#: Object id of the ``/`` catalog, which always exists.
ROOT_ID = 0

#: Default separator used by :meth:`DEFile.get_all_attributes`, matching the
#: Julia connector so that the two agree on files containing awkward values.
DEFAULT_ATTRIBUTE_DELIMITER = "‖"


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """Metadata of one object -- a row of the ``objects`` table."""

    id: int
    parent_id: int
    obj_class: Class
    obj_type: Type
    name: str


@dataclass(frozen=True, slots=True)
class Axis:
    """An axis of an array object."""

    id: int
    ax_type: AxisType
    length: int
    frequency: Frequency
    #: Integer code of the first date, for :attr:`AxisType.RANGE` axes.
    first: int | None = None
    #: Ordered element names, for :attr:`AxisType.NAMES` axes.
    names: tuple[str, ...] | None = None


def _as_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def _copy_string(pointer: Any) -> str:
    """Copy a ``char *`` returned by the library before it is invalidated."""
    if not pointer:
        return ""
    if isinstance(pointer, bytes):
        return pointer.decode("utf-8", errors="replace")
    return ctypes.string_at(pointer).decode("utf-8", errors="replace")


def _copy_blob(pointer: int | None, nbytes: int) -> bytes | None:
    """Copy an opaque value buffer before the next library call invalidates it."""
    if not pointer or nbytes <= 0:
        return None
    return ctypes.string_at(pointer, nbytes)


def _object_from_struct(obj: C.ObjectT) -> ObjectInfo:
    return ObjectInfo(
        id=obj.id,
        parent_id=obj.pid,
        obj_class=Class(obj.obj_class),
        obj_type=Type(obj.obj_type),
        name=_copy_string(obj.name),
    )


def _axis_from_struct(axis: C.AxisT) -> Axis:
    ax_type = AxisType(axis.ax_type)
    first = axis.first if ax_type is AxisType.RANGE else None
    names: tuple[str, ...] | None = None
    if ax_type is AxisType.NAMES:
        packed = _copy_string(axis.names)
        names = tuple(packed.split("\n")) if packed else ()
    return Axis(
        id=axis.id,
        ax_type=ax_type,
        length=axis.length,
        frequency=Frequency(axis.frequency),
        first=first,
        names=names,
    )


def _split_path(path: str) -> tuple[str, str]:
    """Split ``"a/b/c"`` into ``("/a/b", "c")``."""
    cleaned = path.strip("/")
    if not cleaned:
        msg = "The root catalog '/' cannot be used as an object name."
        raise DEArgumentError(msg)
    head, _, name = cleaned.rpartition("/")
    return ("/" + head if head else "/"), name


class DEFile:
    """An open ``.daec`` file.

    Instances are normally created by :func:`opendaec` or :func:`opendaecmem`
    and used as context managers::

        with dataecon.opendaec("model.daec", write=True) as de:
            de["gdp"] = series

    All file access goes through the DataEcon C library, so files written here
    are byte-for-byte ordinary ``.daec`` files, readable by the C, Julia and
    MATLAB connectors.
    """

    def __init__(
        self,
        handle: C.DeFile,
        filename: str,
        *,
        readonly: bool = True,
        overwrite: bool = False,
    ) -> None:
        self._handle: C.DeFile | None = handle
        self._filename = filename
        self._readonly = readonly
        #: When true, storing over an existing name replaces it instead of failing.
        self.overwrite = overwrite

    # -- lifecycle ----------------------------------------------------------

    @property
    def filename(self) -> str:
        """Path this file was opened from (``":memory:"`` for in-memory files)."""
        return self._filename

    @property
    def readonly(self) -> bool:
        """Whether the file was opened read-only."""
        return self._readonly

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._handle is None

    @property
    def _de(self) -> C.DeFile:
        if self._handle is None:
            msg = f"Operation on a closed .daec file: {self._filename!r}"
            raise DEError(msg)
        return self._handle

    def _writable(self) -> C.DeFile:
        """Return the handle, refusing up front if the file is read-only.

        Letting a write reach the C library on a read-only file leaves an open
        transaction that can then never be committed, which makes the file
        impossible to close cleanly. Failing here keeps the handle usable.
        """
        handle = self._de
        if self._readonly:
            msg = f"{self._filename!r} is open read-only. Reopen it with write=True to modify it."
            raise DEReadOnlyError(msg)
        return handle

    def close(self) -> None:
        """Close the file, committing any pending writes. Idempotent."""
        if self._handle is not None:
            handle, self._handle = self._handle, None
            C.check(C.lib.de_close(handle))

    def truncate(self) -> None:
        """Delete everything in the file, leaving it as if newly created."""
        C.check(C.lib.de_truncate(self._writable()))

    def __enter__(self) -> DEFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else ("read-only" if self._readonly else "read-write")
        return f"<DEFile {self._filename!r} ({state})>"

    # -- objects ------------------------------------------------------------

    def find_object(self, parent_id: int, name: str, *, missing_ok: bool = False) -> int | None:
        """Return the id of ``name`` inside catalog ``parent_id``."""
        out = C.ObjIdT()
        rc = C.lib.de_find_object(self._de, parent_id, _as_bytes(name), ctypes.byref(out))
        if rc != 0:
            err = C.last_error(rc)
            if missing_ok and isinstance(err, DEObjectDoesNotExistError):
                return None
            raise err
        return out.value

    @overload
    def find_fullpath(self, path: str) -> int: ...

    @overload
    def find_fullpath(self, path: str, *, missing_ok: Literal[False]) -> int: ...

    @overload
    def find_fullpath(self, path: str, *, missing_ok: Literal[True]) -> int | None: ...

    def find_fullpath(self, path: str, *, missing_ok: bool = False) -> int | None:
        """Return the id of the object at ``path``.

        A leading ``/`` is optional: ``"a/b"`` and ``"/a/b"`` are the same object.
        Returns ``None`` only when ``missing_ok`` is set and the object is absent.
        """
        full = path if path.startswith("/") else "/" + path
        out = C.ObjIdT()
        rc = C.lib.de_find_fullpath(self._de, _as_bytes(full), ctypes.byref(out))
        if rc != 0:
            err = C.last_error(rc)
            if missing_ok and isinstance(err, DEObjectDoesNotExistError):
                return None
            raise err
        return out.value

    @overload
    def resolve(self, target: int | str) -> int: ...

    @overload
    def resolve(self, target: int | str, *, missing_ok: Literal[False]) -> int: ...

    @overload
    def resolve(self, target: int | str, *, missing_ok: Literal[True]) -> int | None: ...

    def resolve(self, target: int | str, *, missing_ok: bool = False) -> int | None:
        """Return an object id from either an id or a full path.

        Returns ``None`` only when ``missing_ok`` is set and the object is
        absent; otherwise a missing object raises.
        """
        if isinstance(target, str):
            return self.find_fullpath(target, missing_ok=missing_ok)  # type: ignore[call-overload,no-any-return]
        return target

    def load_object(self, target: int | str) -> ObjectInfo:
        """Return the metadata of an object."""
        obj = C.ObjectT()
        C.check(C.lib.de_load_object(self._de, self.resolve(target), ctypes.byref(obj)))
        return _object_from_struct(obj)

    def delete_object(self, target: int | str) -> None:
        """Delete an object and, for a catalog, everything inside it."""
        C.check(C.lib.de_delete_object(self._writable(), self.resolve(target)))

    def get_object_info(self, target: int | str) -> tuple[str, int, int]:
        """Return ``(fullpath, depth, created)`` for an object.

        ``created`` is a Unix timestamp recorded when the object was written.
        """
        path = ctypes.c_char_p()
        depth = ctypes.c_int64()
        created = ctypes.c_int64()
        C.check(
            C.lib.de_get_object_info(
                self._de,
                self.resolve(target),
                ctypes.byref(path),
                ctypes.byref(depth),
                ctypes.byref(created),
            )
        )
        return _copy_string(path.value), depth.value, created.value

    def get_fullpath(self, target: int | str) -> str:
        """Return the full path of an object."""
        return self.get_object_info(target)[0]

    def catalog_size(self, target: int | str = ROOT_ID) -> int:
        """Return the number of objects directly inside a catalog."""
        count = ctypes.c_int64()
        C.check(C.lib.de_catalog_size(self._de, self.resolve(target), ctypes.byref(count)))
        return count.value

    def exists(self, path: str) -> bool:
        """Return whether an object exists at ``path``."""
        return self.find_fullpath(path, missing_ok=True) is not None

    # -- attributes ---------------------------------------------------------

    def set_attribute(self, target: int | str, name: str, value: str | None) -> None:
        """Set (or replace) one attribute of an object."""
        raw = None if value is None else _as_bytes(str(value))
        C.check(
            C.lib.de_set_attribute(self._writable(), self.resolve(target), _as_bytes(name), raw)
        )

    def get_attribute(
        self, target: int | str, name: str, default: Any = None, *, required: bool = False
    ) -> Any:
        """Return one attribute of an object, or ``default`` if it is not set."""
        out = ctypes.c_char_p()
        rc = C.lib.de_get_attribute(
            self._de, self.resolve(target), _as_bytes(name), ctypes.byref(out)
        )
        if rc != 0:
            err = C.last_error(rc)
            if not required and isinstance(err, DEMissingAttributeError):
                return default
            raise err
        return _copy_string(out.value)

    def get_all_attributes(
        self, target: int | str, delimiter: str = DEFAULT_ATTRIBUTE_DELIMITER
    ) -> dict[str, str]:
        """Return all attributes of an object as a dictionary.

        The C library returns every name and every value joined into two single
        strings, so ``delimiter`` must not occur inside any attribute name or
        value. The default is the same rarely-typed character the Julia
        connector uses.
        """
        count = ctypes.c_int64()
        names = ctypes.c_char_p()
        values = ctypes.c_char_p()
        C.check(
            C.lib.de_get_all_attributes(
                self._de,
                self.resolve(target),
                _as_bytes(delimiter),
                ctypes.byref(count),
                ctypes.byref(names),
                ctypes.byref(values),
            )
        )
        if count.value == 0:
            return {}
        name_list = _copy_string(names.value).split(delimiter)
        value_list = _copy_string(values.value).split(delimiter)
        if len(name_list) != len(value_list):
            msg = (
                "Could not split the attributes of "
                f"{self.get_fullpath(target)!r}: an attribute name or value "
                f"contains the delimiter {delimiter!r}, or has a NULL value. "
                "Pass a different `delimiter`."
            )
            raise DEError(msg)
        return dict(zip(name_list, value_list, strict=True))

    # -- catalogs -----------------------------------------------------------

    def new_catalog(self, parent_id: int, name: str) -> int:
        """Create a catalog inside ``parent_id`` and return its id."""
        self._maybe_overwrite(parent_id, name)
        out = C.ObjIdT()
        C.check(
            C.lib.de_new_catalog(self._writable(), parent_id, _as_bytes(name), ctypes.byref(out))
        )
        return out.value

    def makedirs(self, path: str, *, exist_ok: bool = True) -> int:
        """Create every catalog along ``path``, returning the id of the last one."""
        parent = ROOT_ID
        for part in path.strip("/").split("/"):
            if not part:
                continue
            existing = self.find_object(parent, part, missing_ok=True)
            if existing is None:
                parent = self.new_catalog(parent, part)
                continue
            if not exist_ok:
                msg = f"Object already exists: {part!r}"
                raise DEExistsError(msg)
            info = self.load_object(existing)
            if info.obj_class is not Class.CATALOG:
                msg = (
                    f"Cannot create a catalog at {path!r}: {info.name!r} already "
                    f"exists and is a {info.obj_class.name.lower()}, not a catalog."
                )
                raise DEExistsError(msg)
            parent = existing
        return parent

    def _maybe_overwrite(self, parent_id: int, name: str, overwrite: bool | None = None) -> None:
        """Delete an existing object of this name when overwriting is enabled."""
        if overwrite is None:
            overwrite = self.overwrite
        if not overwrite:
            return
        existing = self.find_object(parent_id, name, missing_ok=True)
        if existing is not None:
            self.delete_object(existing)

    # -- axes ---------------------------------------------------------------

    def axis_plain(self, length: int) -> int:
        """Find or create a plain ``0..length-1`` axis and return its id."""
        out = C.AxisIdT()
        C.check(C.lib.de_axis_plain(self._writable(), length, ctypes.byref(out)))
        return out.value

    def axis_range(self, length: int, frequency: int, first: int) -> int:
        """Find or create a date-range axis and return its id."""
        out = C.AxisIdT()
        C.check(
            C.lib.de_axis_range(
                self._writable(), length, int(frequency), int(first), ctypes.byref(out)
            )
        )
        return out.value

    def axis_names(self, names: Sequence[str]) -> int:
        """Find or create a named axis and return its id."""
        names = list(names)
        packed = "\n".join(names)
        if any("\n" in n for n in names):
            msg = "Axis names must not contain a newline; it is the separator."
            raise DEArgumentError(msg)
        out = C.AxisIdT()
        C.check(
            C.lib.de_axis_names(self._writable(), len(names), _as_bytes(packed), ctypes.byref(out))
        )
        return out.value

    def load_axis(self, axis_id: int) -> Axis:
        """Load an axis by id."""
        axis = C.AxisT()
        C.check(C.lib.de_load_axis(self._de, axis_id, ctypes.byref(axis)))
        return _axis_from_struct(axis)

    # -- scalars (low level) ------------------------------------------------

    def store_scalar(
        self,
        parent_id: int,
        name: str,
        obj_type: int,
        frequency: int,
        blob: bytes | None,
        *,
        overwrite: bool | None = None,
    ) -> int:
        """Write a scalar object from a raw blob and return its id."""
        self._maybe_overwrite(parent_id, name, overwrite)
        out = C.ObjIdT()
        buf, nbytes = _buffer(blob)
        C.check(
            C.lib.de_store_scalar(
                self._writable(),
                parent_id,
                _as_bytes(name),
                int(obj_type),
                int(frequency),
                nbytes,
                buf,
                ctypes.byref(out),
            )
        )
        return out.value

    def load_scalar(self, target: int | str) -> tuple[ObjectInfo, int, bytes | None]:
        """Load a scalar object, returning ``(metadata, frequency, blob)``."""
        scalar = C.ScalarT()
        C.check(C.lib.de_load_scalar(self._de, self.resolve(target), ctypes.byref(scalar)))
        return (
            _object_from_struct(scalar.object),
            scalar.frequency,
            _copy_blob(scalar.value, scalar.nbytes),
        )

    # -- arrays (low level) -------------------------------------------------

    def store_tseries(
        self,
        parent_id: int,
        name: str,
        obj_type: int,
        eltype: int,
        elfreq: int,
        axis_id: int,
        blob: bytes | None,
        *,
        overwrite: bool | None = None,
    ) -> int:
        """Write a 1-d array object from a raw blob and return its id."""
        self._maybe_overwrite(parent_id, name, overwrite)
        out = C.ObjIdT()
        buf, nbytes = _buffer(blob)
        C.check(
            C.lib.de_store_tseries(
                self._writable(),
                parent_id,
                _as_bytes(name),
                int(obj_type),
                int(eltype),
                int(elfreq),
                axis_id,
                nbytes,
                buf,
                ctypes.byref(out),
            )
        )
        return out.value

    def load_tseries(self, target: int | str) -> tuple[ObjectInfo, int, int, Axis, bytes | None]:
        """Load a 1-d array, returning ``(metadata, eltype, elfreq, axis, blob)``."""
        arr = C.TseriesT()
        C.check(C.lib.de_load_tseries(self._de, self.resolve(target), ctypes.byref(arr)))
        return (
            _object_from_struct(arr.object),
            arr.eltype,
            arr.elfreq,
            _axis_from_struct(arr.axis),
            _copy_blob(arr.value, arr.nbytes),
        )

    def store_mvtseries(
        self,
        parent_id: int,
        name: str,
        obj_type: int,
        eltype: int,
        elfreq: int,
        axis1_id: int,
        axis2_id: int,
        blob: bytes | None,
        *,
        overwrite: bool | None = None,
    ) -> int:
        """Write a 2-d array object from a raw blob and return its id."""
        self._maybe_overwrite(parent_id, name, overwrite)
        out = C.ObjIdT()
        buf, nbytes = _buffer(blob)
        C.check(
            C.lib.de_store_mvtseries(
                self._writable(),
                parent_id,
                _as_bytes(name),
                int(obj_type),
                int(eltype),
                int(elfreq),
                axis1_id,
                axis2_id,
                nbytes,
                buf,
                ctypes.byref(out),
            )
        )
        return out.value

    def load_mvtseries(
        self, target: int | str
    ) -> tuple[ObjectInfo, int, int, Axis, Axis, bytes | None]:
        """Load a 2-d array, returning ``(metadata, eltype, elfreq, axis1, axis2, blob)``."""
        arr = C.MvtseriesT()
        C.check(C.lib.de_load_mvtseries(self._de, self.resolve(target), ctypes.byref(arr)))
        return (
            _object_from_struct(arr.object),
            arr.eltype,
            arr.elfreq,
            _axis_from_struct(arr.axis1),
            _axis_from_struct(arr.axis2),
            _copy_blob(arr.value, arr.nbytes),
        )

    def store_ndtseries(
        self,
        parent_id: int,
        name: str,
        obj_type: int,
        eltype: int,
        elfreq: int,
        axis_ids: Sequence[int],
        blob: bytes | None,
        *,
        overwrite: bool | None = None,
    ) -> int:
        """Write an N-d array object from a raw blob and return its id."""
        if not 1 <= len(axis_ids) <= DE_MAX_AXES:
            msg = f"A .daec array may have between 1 and {DE_MAX_AXES} axes; got {len(axis_ids)}."
            raise DERangeError(msg)
        self._maybe_overwrite(parent_id, name, overwrite)
        out = C.ObjIdT()
        buf, nbytes = _buffer(blob)
        axes = (C.AxisIdT * len(axis_ids))(*axis_ids)
        C.check(
            C.lib.de_store_ndtseries(
                self._writable(),
                parent_id,
                _as_bytes(name),
                int(obj_type),
                int(eltype),
                int(elfreq),
                len(axis_ids),
                axes,
                nbytes,
                buf,
                ctypes.byref(out),
            )
        )
        return out.value

    def load_ndtseries(
        self, target: int | str
    ) -> tuple[ObjectInfo, int, int, list[Axis], bytes | None]:
        """Load an N-d array, returning ``(metadata, eltype, elfreq, axes, blob)``."""
        arr = C.NdtseriesT()
        C.check(C.lib.de_load_ndtseries(self._de, self.resolve(target), ctypes.byref(arr)))
        axes = [_axis_from_struct(arr.axis[i]) for i in range(arr.naxes)]
        return (
            _object_from_struct(arr.object),
            arr.eltype,
            arr.elfreq,
            axes,
            _copy_blob(arr.value, arr.nbytes),
        )

    # -- searching ----------------------------------------------------------

    def list_catalog(self, target: int | str = ROOT_ID) -> list[ObjectInfo]:
        """Return the objects directly inside a catalog."""
        return list(self.search_catalog(target))

    def search_catalog(
        self,
        target: int | str = ROOT_ID,
        pattern: str | None = None,
        obj_type: int = Type.ANY,
        obj_class: int = Class.ANY,
    ) -> Iterator[ObjectInfo]:
        """Iterate over the objects in a catalog, optionally filtered.

        ``pattern`` is a SQLite ``GLOB`` pattern matched against object names.
        Pass ``target=-1`` to search the whole file rather than one catalog.
        """
        search = C.DeSearch()
        wildcard = _as_bytes(pattern) if pattern is not None else None
        C.check(
            C.lib.de_search_catalog(
                self._de,
                self.resolve(target),
                wildcard,
                int(obj_type),
                int(obj_class),
                ctypes.byref(search),
            )
        )
        try:
            obj = C.ObjectT()
            while True:
                rc = C.lib.de_next_object(search, ctypes.byref(obj))
                if rc != 0:
                    err = C.last_error(rc)
                    if err.rc == -986:  # DE_NO_OBJ: the search is exhausted
                        return
                    raise err
                yield _object_from_struct(obj)
        finally:
            C.lib.de_finalize_search(search)
            C.lib.de_clear_error()

    def walk(self, target: int | str = ROOT_ID, *, max_depth: int | None = None) -> Iterator[str]:
        """Yield the full path of every non-catalog object, depth first."""
        root_depth = 0 if self.resolve(target) == ROOT_ID else self.get_object_info(target)[1]
        stack = [(self.resolve(target), root_depth)]
        while stack:
            parent, depth = stack.pop()
            if max_depth is not None and depth - root_depth >= max_depth:
                continue
            for info in sorted(self.list_catalog(parent), key=lambda o: o.name):
                if info.obj_class is Class.CATALOG:
                    stack.append((info.id, depth + 1))
                else:
                    yield self.get_fullpath(info.id)

    # -- value layer --------------------------------------------------------
    # Implemented in _values.py to keep this module focused on the C API.

    def write(
        self,
        path: str,
        value: Any,
        *,
        overwrite: bool | None = None,
        create_parents: bool = True,
    ) -> int:
        """Write ``value`` at ``path``, choosing the storage class automatically.

        Mappings become catalogs (recursively), 1-d values become ``tseries``
        objects, 2-d values ``mvtseries``, higher-rank arrays ``ndtseries``, and
        everything else a scalar.
        """
        from ._values import write_value

        return write_value(self, path, value, overwrite=overwrite, create_parents=create_parents)

    def read(self, path: int | str = ROOT_ID, *, as_dict: bool = False) -> Any:
        """Read the object at ``path`` and return it as a Python value.

        Pass ``as_dict=True`` to get catalogs back as plain nested dictionaries
        rather than ``tsecon`` ``Workspace`` objects.
        """
        from ._values import read_value

        return read_value(self, path, as_dict=as_dict)

    # -- mapping protocol ---------------------------------------------------

    def __getitem__(self, path: str) -> Any:
        return self.read(path)

    def __setitem__(self, path: str, value: Any) -> None:
        self.write(path, value, overwrite=True)

    def __delitem__(self, path: str) -> None:
        self.delete_object(path)

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and self.exists(path)

    def __len__(self) -> int:
        return self.catalog_size(ROOT_ID)

    def keys(self) -> list[str]:
        """Return the names of the objects in the root catalog."""
        return [info.name for info in self.list_catalog(ROOT_ID)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


def _buffer(blob: bytes | None) -> tuple[Any, int]:
    """Return a ``(void *, nbytes)`` pair for a value blob."""
    if not blob:
        return None, 0
    buf = ctypes.create_string_buffer(blob, len(blob))
    return ctypes.cast(buf, ctypes.c_void_p), len(blob)


# ---------------------------------------------------------------------------
# opening files
# ---------------------------------------------------------------------------


def opendaec(
    filename: str | os.PathLike[str],
    *,
    readonly: bool | None = None,
    write: bool = False,
    append: bool = True,
    truncate: bool = False,
    overwrite: bool = False,
) -> DEFile:
    """Open a ``.daec`` file.

    Parameters
    ----------
    filename
        Path to the file. It is created if it does not exist and ``write`` is
        set.
    readonly
        Open without write access. Defaults to ``True`` unless ``write`` is set.
    write
        Open for writing. Takes precedence over ``readonly``.
    append
        Keep whatever the file already contains. The default.
    truncate
        Empty the file immediately after opening. Takes precedence over
        ``append``, and requires ``write``.
    overwrite
        Make storing an object over an existing name replace it rather than
        raise :class:`~dataecon.errors.DEExistsError`.

    Returns
    -------
    DEFile
        An open file handle, usable as a context manager.
    """
    if readonly is None:
        readonly = not write
    if write:
        readonly = False
    if not append:
        truncate = True
    if truncate and readonly:
        msg = "Cannot truncate a file opened read-only; pass write=True."
        raise DEArgumentError(msg)

    path = os.fspath(filename)
    handle = C.DeFile()
    opener = C.lib.de_open_readonly if readonly else C.lib.de_open
    C.check(opener(_as_bytes(path), ctypes.byref(handle)))

    de = DEFile(handle, path, readonly=readonly, overwrite=overwrite)
    if truncate:
        de.truncate()
    return de


def opendaecmem(*, overwrite: bool = False) -> DEFile:
    """Open a new, empty in-memory ``.daec`` database.

    Useful for tests and for building a database that is never written to disk.
    """
    handle = C.DeFile()
    C.check(C.lib.de_open_memory(ctypes.byref(handle)))
    return DEFile(handle, ":memory:", readonly=False, overwrite=overwrite)
