"""Dump a ``.daec`` file in the same textual format as ``c_reference.c``.

The C driver and this module describe the same file in the same words, so
comparing their output is a direct check that DataEconPy reads the format the
way the reference implementation does. Any disagreement about element widths,
byte order, axis interpretation or attribute handling shows up as a diff.

Object order inside a catalog is not part of the format, so callers sort the
lines before comparing.
"""

from __future__ import annotations

import numpy as np

import dataecon as de
from dataecon._codec import unpack_strings
from dataecon._consts import AxisType, Class, Type

__all__ = ["dump_file", "dump_lines"]


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}g}"


def _fmt_scalar(obj_type: int, frequency: int, blob: bytes | None) -> str:
    """Format one scalar value exactly as ``print_scalar_value`` in the C driver."""
    if not blob:
        return "null"
    obj_type = int(obj_type)
    n = len(blob)

    if obj_type == Type.SIGNED:
        if n in (1, 2, 4, 8):
            return str(int.from_bytes(blob, "little", signed=True))
    elif obj_type == Type.UNSIGNED:
        if n in (1, 2, 4, 8):
            return str(int.from_bytes(blob, "little", signed=False))
    elif obj_type == Type.DATE:
        return f"{int.from_bytes(blob, 'little', signed=True)}@{int(frequency)}"
    elif obj_type == Type.FLOAT:
        if n == 4:
            return _fmt_float(float(np.frombuffer(blob, np.float32)[0]), 9)
        if n == 8:
            return _fmt_float(float(np.frombuffer(blob, np.float64)[0]), 17)
    elif obj_type == Type.COMPLEX:
        if n == 8:
            re, im = np.frombuffer(blob, np.float32)
            return f"({_fmt_float(float(re), 9)},{_fmt_float(float(im), 9)})"
        if n == 16:
            re, im = np.frombuffer(blob, np.float64)
            return f"({_fmt_float(float(re), 17)},{_fmt_float(float(im), 17)})"
    elif obj_type == Type.STRING:
        return '"' + blob.split(b"\x00", 1)[0].decode("utf-8") + '"'

    return f"<{n} bytes>"


def _fmt_elements(eltype: int, elfreq: int, count: int, blob: bytes | None) -> str:
    """Format array elements exactly as ``print_elements`` in the C driver."""
    if count == 0 or not blob:
        return "[]"
    eltype = int(eltype)
    if eltype == Type.STRING:
        parts = ['"' + s + '"' for s in unpack_strings(blob, count)]
        return "[" + " ".join(parts) + "]"

    width = len(blob) // count
    parts = [_fmt_scalar(eltype, elfreq, blob[i * width : (i + 1) * width]) for i in range(count)]
    return "[" + " ".join(parts) + "]"


def _fmt_axis(axis: de.Axis) -> str:
    if axis.ax_type is AxisType.PLAIN:
        return f"plain:{axis.length}"
    if axis.ax_type is AxisType.RANGE:
        return f"range:{axis.length}:{int(axis.frequency)}:{axis.first}"
    if axis.ax_type is AxisType.NAMES:
        return f"names:{axis.length}:" + ",".join(axis.names or ())
    return "?"


def _fmt_attributes(f: de.DEFile, obj_id: int) -> str:
    attrs = f.get_all_attributes(obj_id, delimiter="\x1f")
    if not attrs:
        return ""
    pairs = sorted(f"{k}={v}" for k, v in attrs.items())
    return " attrs={" + ",".join(pairs) + "}"


def dump_lines(f: de.DEFile, parent: int = de.ROOT_ID) -> list[str]:
    """Return one line per object in the file, in the C driver's format."""
    lines: list[str] = []
    for info in f.list_catalog(parent):
        path = f.get_fullpath(info.id)
        attrs = _fmt_attributes(f, info.id)

        if info.obj_class is Class.CATALOG:
            lines.append(f"CATALOG {path}{attrs}")
            lines.extend(dump_lines(f, info.id))

        elif info.obj_class is Class.SCALAR:
            meta, frequency, blob = f.load_scalar(info.id)
            value = _fmt_scalar(meta.obj_type, frequency, blob)
            lines.append(
                f"SCALAR {path} type={int(meta.obj_type)} "
                f"freq={int(frequency)} value={value}{attrs}"
            )

        elif info.obj_class is Class.TSERIES:
            meta, eltype, elfreq, axis, blob = f.load_tseries(info.id)
            values = (
                "[]"
                if meta.obj_type == Type.RANGE
                else _fmt_elements(eltype, elfreq, axis.length, blob)
            )
            lines.append(
                f"VECTOR {path} type={int(meta.obj_type)} eltype={int(eltype)} "
                f"elfreq={int(elfreq)} axis={_fmt_axis(axis)} values={values}{attrs}"
            )

        elif info.obj_class is Class.MVTSERIES:
            meta, eltype, elfreq, axis1, axis2, blob = f.load_mvtseries(info.id)
            count = axis1.length * axis2.length
            lines.append(
                f"MATRIX {path} type={int(meta.obj_type)} eltype={int(eltype)} "
                f"elfreq={int(elfreq)} axis1={_fmt_axis(axis1)} axis2={_fmt_axis(axis2)} "
                f"values={_fmt_elements(eltype, elfreq, count, blob)}{attrs}"
            )

        elif info.obj_class is Class.NDTSERIES:
            meta, eltype, elfreq, axes, blob = f.load_ndtseries(info.id)
            count = 1
            axis_parts = []
            for i, axis in enumerate(axes):
                axis_parts.append(f" axis{i}={_fmt_axis(axis)}")
                count *= axis.length
            lines.append(
                f"TENSOR {path} type={int(meta.obj_type)} eltype={int(eltype)} "
                f"elfreq={int(elfreq)} naxes={len(axes)}"
                + "".join(axis_parts)
                + f" values={_fmt_elements(eltype, elfreq, count, blob)}{attrs}"
            )

        else:  # pragma: no cover - the format defines no other class
            lines.append(f"UNKNOWN {path} class={int(info.obj_class)}")

    return lines


def dump_file(path: str) -> list[str]:
    """Open ``path`` read-only and return its dump lines."""
    with de.opendaec(path) as f:
        return dump_lines(f)
