# SPDX-License-Identifier: MIT
"""Mapping between Python values and DataEcon storage classes.

This is the Python counterpart of ``I.jl`` in the Julia connector: it decides
which storage class a value belongs in on the way out, and rebuilds the original
kind of object on the way back in.

============================  ==========================================
Python value                   Stored as
============================  ==========================================
``Mapping`` / ``Workspace``    a catalog, written recursively
``TSeries``                    ``class_tseries`` / ``type_tseries``
``MVTSeries``                  ``class_mvtseries`` / ``type_mvtseries``
``MITRange`` / ``range``       ``type_range`` (axis only, no element data)
1-d array or sequence          ``class_tseries`` / ``type_vector``
2-d array                      ``class_mvtseries`` / ``type_matrix``
3- to 5-d array                ``class_ndtseries`` / ``type_tensor``
anything else                  ``class_scalar``
============================  ==========================================

Types that the format cannot record on their own -- ``bool``, ``datetime`` --
are stored in the closest representable form plus a ``pytype``/``pyeltype``
attribute naming the original. Reading also honours the ``jtype``/``jeltype``
attributes the Julia connector writes, so files from either side round-trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._codec import (
    JELTYPE_ATTR,
    JTYPE_ATTR,
    PYELTYPE_ATTR,
    PYTYPE_ATTR,
    decode_array,
    decode_scalar,
    encode_array_values,
    encode_scalar,
)
from ._consts import DE_MAX_AXES, AxisType, Class, Frequency, Type
from .errors import DEError, DERangeError, DEUnsupportedError
from .interop import _tsecon

if TYPE_CHECKING:  # pragma: no cover
    from .file import Axis, DEFile

__all__ = ["read_value", "write_value"]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_value(
    de: DEFile,
    path: str,
    value: Any,
    *,
    overwrite: bool | None = None,
    create_parents: bool = True,
) -> int:
    """Write ``value`` at ``path`` and return the id of the object created."""
    from .file import ROOT_ID, _split_path

    parent_path, name = _split_path(path)
    if create_parents:
        parent_id = de.makedirs(parent_path)
    else:
        parent_id = ROOT_ID if parent_path == "/" else de.resolve(parent_path)
    return write_into(de, parent_id, name, value, overwrite=overwrite)


def write_into(
    de: DEFile,
    parent_id: int,
    name: str,
    value: Any,
    *,
    overwrite: bool | None = None,
) -> int:
    """Write ``value`` as ``name`` inside catalog ``parent_id``."""
    value = _normalise(value)

    if isinstance(value, Mapping):
        return _write_catalog(de, parent_id, name, value, overwrite=overwrite)
    if _tsecon.is_workspace(value):
        return _write_catalog(de, parent_id, name, dict(value.items()), overwrite=overwrite)
    if _tsecon.is_mvtseries(value):
        return _write_mvtseries(de, parent_id, name, value, overwrite=overwrite)
    if _tsecon.is_tseries(value):
        return _write_tseries(de, parent_id, name, value, overwrite=overwrite)
    if _tsecon.is_mitrange(value):
        return _write_mitrange(de, parent_id, name, value, overwrite=overwrite)
    if isinstance(value, range):
        return _write_plain_range(de, parent_id, name, value, overwrite=overwrite)
    if isinstance(value, np.ndarray):
        return _write_array(de, parent_id, name, value, overwrite=overwrite)

    return _write_scalar(de, parent_id, name, value, overwrite=overwrite)


def _normalise(value: Any) -> Any:
    """Convert types we accept but do not store directly into ones we do."""
    if isinstance(value, np.ndarray) or _is_scalar_like(value):
        return value
    if isinstance(value, (list, tuple)):
        return _sequence_to_array(value)
    converted = _from_dataframe(value)
    if converted is not None:
        return converted
    return value


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray, np.generic)) or value is None


def _sequence_to_array(value: Sequence[Any]) -> np.ndarray:
    """Turn a list or tuple into an array, keeping object dtype for strings."""
    array = np.asarray(value)
    if array.dtype.kind in ("U", "S"):
        out = np.empty(array.shape, dtype=object)
        out[...] = array.tolist()
        return out
    return array


def _from_dataframe(value: Any) -> Any:
    """Convert a pandas or polars object to a TimeSeriesEconPy value, if it is one."""
    module = type(value).__module__.split(".")[0]
    if module == "pandas":
        from .interop.pandas import from_pandas

        return from_pandas(value)
    if module == "polars":
        from .interop.polars import from_polars

        return from_polars(value)
    return None


def _write_catalog(
    de: DEFile, parent_id: int, name: str, items: Mapping[str, Any], *, overwrite: bool | None
) -> int:
    if overwrite is None:
        overwrite = de.overwrite
    existing = de.find_object(parent_id, name, missing_ok=True)
    if existing is not None and overwrite:
        de.delete_object(existing)
        existing = None
    catalog_id = existing if existing is not None else de.new_catalog(parent_id, name)
    for key, item in items.items():
        write_into(de, catalog_id, str(key), item, overwrite=overwrite)
    return catalog_id


def _write_scalar(
    de: DEFile, parent_id: int, name: str, value: Any, *, overwrite: bool | None
) -> int:
    data = encode_scalar(value)
    obj_id = de.store_scalar(
        parent_id, name, data.type, data.frequency, data.blob, overwrite=overwrite
    )
    if data.pytype is not None:
        de.set_attribute(obj_id, PYTYPE_ATTR, data.pytype)
    return obj_id


def _write_array(
    de: DEFile, parent_id: int, name: str, array: np.ndarray, *, overwrite: bool | None
) -> int:
    ndim = array.ndim
    if ndim == 0:
        return _write_scalar(de, parent_id, name, array[()], overwrite=overwrite)
    if ndim > DE_MAX_AXES:
        msg = (
            f"The .daec format supports arrays of at most {DE_MAX_AXES} dimensions; "
            f"{name!r} has {ndim}."
        )
        raise DERangeError(msg)

    values = encode_array_values(array)
    axes = [de.axis_plain(size) for size in array.shape]

    if ndim == 1:
        obj_id = de.store_tseries(
            parent_id,
            name,
            Type.VECTOR,
            values.eltype,
            values.elfreq,
            axes[0],
            values.blob,
            overwrite=overwrite,
        )
    elif ndim == 2:
        obj_id = de.store_mvtseries(
            parent_id,
            name,
            Type.MATRIX,
            values.eltype,
            values.elfreq,
            axes[0],
            axes[1],
            values.blob,
            overwrite=overwrite,
        )
    else:
        obj_id = de.store_ndtseries(
            parent_id,
            name,
            Type.TENSOR,
            values.eltype,
            values.elfreq,
            axes,
            values.blob,
            overwrite=overwrite,
        )

    if values.pyeltype is not None:
        de.set_attribute(obj_id, PYELTYPE_ATTR, values.pyeltype)
    return obj_id


def _write_tseries(
    de: DEFile, parent_id: int, name: str, series: Any, *, overwrite: bool | None
) -> int:
    array = np.asarray(series.values)
    values = encode_array_values(array)
    first_code, freq_code = _tsecon.mit_to_code(series.firstdate)
    axis = de.axis_range(len(array), freq_code, first_code)
    obj_id = de.store_tseries(
        parent_id,
        name,
        Type.TSERIES,
        values.eltype,
        values.elfreq,
        axis,
        values.blob,
        overwrite=overwrite,
    )
    if values.pyeltype is not None:
        de.set_attribute(obj_id, PYELTYPE_ATTR, values.pyeltype)
    return obj_id


def _write_mvtseries(
    de: DEFile, parent_id: int, name: str, series: Any, *, overwrite: bool | None
) -> int:
    array = np.asarray(series.values)
    values = encode_array_values(array)
    first_code, freq_code = _tsecon.mit_to_code(series.firstdate)
    axis1 = de.axis_range(array.shape[0], freq_code, first_code)
    axis2 = de.axis_names([str(c) for c in series.column_names])
    obj_id = de.store_mvtseries(
        parent_id,
        name,
        Type.MVTSERIES,
        values.eltype,
        values.elfreq,
        axis1,
        axis2,
        values.blob,
        overwrite=overwrite,
    )
    if values.pyeltype is not None:
        de.set_attribute(obj_id, PYELTYPE_ATTR, values.pyeltype)
    return obj_id


def _write_mitrange(
    de: DEFile, parent_id: int, name: str, value: Any, *, overwrite: bool | None
) -> int:
    first_code, freq_code = _tsecon.mit_to_code(value.first())
    axis = de.axis_range(len(value), freq_code, first_code)
    return de.store_tseries(
        parent_id,
        name,
        Type.RANGE,
        Type.NONE,
        Frequency.NONE,
        axis,
        None,
        overwrite=overwrite,
    )


def _write_plain_range(
    de: DEFile, parent_id: int, name: str, value: range, *, overwrite: bool | None
) -> int:
    if value.step != 1:
        msg = (
            f"Only unit-step ranges can be stored as a .daec range; {name!r} has "
            f"step {value.step}. Convert it to an array first."
        )
        raise DEUnsupportedError(msg)
    axis = de.axis_plain(len(value))
    obj_id = de.store_tseries(
        parent_id,
        name,
        Type.RANGE,
        Type.NONE,
        Frequency.NONE,
        axis,
        None,
        overwrite=overwrite,
    )
    if value.start != 0:
        # A plain axis always starts at 0, so a non-zero start needs recording.
        de.set_attribute(obj_id, "pystart", str(value.start))
    return obj_id


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_value(de: DEFile, target: int | str, *, as_dict: bool = False) -> Any:
    """Read the object at ``target`` and return it as a Python value.

    ``as_dict`` applies to catalogs, here and recursively below: pass it to get
    plain dictionaries instead of ``tsecon`` ``Workspace`` objects.
    """
    info = de.load_object(target)
    if info.obj_class is Class.CATALOG:
        return read_catalog(de, info.id, as_dict=as_dict)
    if info.obj_class is Class.SCALAR:
        return _read_scalar(de, info.id)
    if info.obj_class is Class.TSERIES:
        return _read_1d(de, info.id)
    if info.obj_class is Class.MVTSERIES:
        return _read_2d(de, info.id)
    if info.obj_class is Class.NDTSERIES:
        return _read_nd(de, info.id)

    msg = f"Unknown storage class {info.obj_class!r} for object {info.name!r}."
    raise DEError(msg)


def read_catalog(de: DEFile, target: int | str, *, as_dict: bool = False) -> Any:
    """Read a catalog into a ``Workspace``, or a plain dict if asked."""
    items: dict[str, Any] = {}
    for info in de.list_catalog(target):
        items[info.name] = read_value(de, info.id, as_dict=as_dict)
    if as_dict or not _tsecon.available():
        return items
    return _tsecon.make_workspace(items)


def _type_attrs(de: DEFile, obj_id: int) -> tuple[str | None, str | None]:
    """Return the ``(pytype, jtype)`` attributes of an object."""
    return de.get_attribute(obj_id, PYTYPE_ATTR), de.get_attribute(obj_id, JTYPE_ATTR)


def _eltype_attrs(de: DEFile, obj_id: int) -> tuple[str | None, str | None]:
    """Return the ``(pyeltype, jeltype)`` attributes of an object."""
    return de.get_attribute(obj_id, PYELTYPE_ATTR), de.get_attribute(obj_id, JELTYPE_ATTR)


def _read_scalar(de: DEFile, obj_id: int) -> Any:
    info, frequency, blob = de.load_scalar(obj_id)
    pytype, jtype = _type_attrs(de, obj_id)
    return decode_scalar(info.obj_type, frequency, blob, pytype, jtype)


def _read_1d(de: DEFile, obj_id: int) -> Any:
    info, eltype, elfreq, axis, blob = de.load_tseries(obj_id)
    pyeltype, jeltype = _eltype_attrs(de, obj_id)

    if info.obj_type == Type.RANGE:
        if axis.ax_type is AxisType.RANGE:
            _require_tsecon("a date range", info.name)
            first = _range_first(axis, info.name)
            return _tsecon.make_mitrange(first, int(axis.frequency), axis.length)
        start = int(de.get_attribute(obj_id, "pystart", 0))
        return range(start, start + axis.length)

    values = decode_array(eltype, elfreq, blob, (axis.length,), pyeltype, jeltype)

    if info.obj_type == Type.TSERIES or axis.ax_type is AxisType.RANGE:
        _require_tsecon("a time series", info.name)
        first = _range_first(axis, info.name)
        return _tsecon.make_tseries(first, int(axis.frequency), values)
    return values


def _read_2d(de: DEFile, obj_id: int) -> Any:
    info, eltype, elfreq, axis1, axis2, blob = de.load_mvtseries(obj_id)
    pyeltype, jeltype = _eltype_attrs(de, obj_id)
    shape = (axis1.length, axis2.length)
    values = decode_array(eltype, elfreq, blob, shape, pyeltype, jeltype)

    if info.obj_type == Type.MVTSERIES or axis1.ax_type is AxisType.RANGE:
        _require_tsecon("a multivariate time series", info.name)
        names = axis2.names if axis2.names is not None else _default_names(axis2.length)
        first = _range_first(axis1, info.name)
        return _tsecon.make_mvtseries(first, int(axis1.frequency), names, values)
    return values


def _read_nd(de: DEFile, obj_id: int) -> Any:
    _info, eltype, elfreq, axes, blob = de.load_ndtseries(obj_id)
    pyeltype, jeltype = _eltype_attrs(de, obj_id)
    shape = tuple(axis.length for axis in axes)
    return decode_array(eltype, elfreq, blob, shape, pyeltype, jeltype)


def _range_first(axis: Axis, name: str) -> int:
    """Return a range axis's first date, or explain why the object is unusable."""
    if axis.first is None:
        msg = (
            f"{name!r} claims a date range but its axis carries no first date; "
            "the file may be damaged."
        )
        raise DEError(msg)
    return axis.first


def _default_names(count: int) -> list[str]:
    return [f"c{i + 1}" for i in range(count)]


def _require_tsecon(what: str, name: str) -> None:
    if _tsecon.available():
        return
    msg = (
        f"{name!r} is {what}, which needs TimeSeriesEconPy to represent. "
        "Install it with `pip install DataEconPy[tsecon]`, or use "
        "DEFile.load_tseries()/load_mvtseries() to get the raw values and axes."
    )
    raise DEError(msg)
