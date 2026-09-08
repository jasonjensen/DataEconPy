# SPDX-License-Identifier: MIT
"""Encoding and decoding of DataEcon value blobs.

Values are stored as raw little-endian machine bytes, exactly as the C and Julia
connectors write them:

* **Scalars** occupy one blob holding a single value. Strings are UTF-8 and
  NUL-terminated (the terminator is counted in the stored size).
* **Arrays** hold their elements contiguously in **column-major (Fortran)
  order**, because the reference implementations are C code driven by Julia,
  whose arrays are column-major. Getting this wrong transposes every matrix, so
  every multi-dimensional path here passes ``order="F"``.
* **String arrays** are a concatenation of NUL-terminated UTF-8 strings, again
  in column-major element order.

The element type is not stored as a dtype: only a coarse type code (signed,
unsigned, float, complex, string, date) plus the element width in bytes, which
is recovered by dividing the blob size by the number of elements. Types that do
not survive that round trip -- ``bool`` most notably, which is written as a
1-byte signed integer -- carry an extra attribute naming the original type.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, NamedTuple

import numpy as np

from ._consts import Frequency, Type
from .errors import DERangeError, DEUnsupportedError

__all__ = [
    "ScalarData",
    "decode_array",
    "decode_scalar",
    "encode_array_values",
    "encode_scalar",
    "pack_names",
    "pack_strings",
    "unpack_names",
    "unpack_strings",
]

#: Attribute recording the original Python type when it is not recoverable
#: from the stored type code and element width alone.
PYTYPE_ATTR = "pytype"
#: Same, for the *element* type of an array object.
PYELTYPE_ATTR = "pyeltype"
#: The equivalents written by the Julia connector, honoured when reading.
JTYPE_ATTR = "jtype"
JELTYPE_ATTR = "jeltype"


# ---------------------------------------------------------------------------
# string packing
# ---------------------------------------------------------------------------


def pack_strings(values: list[str]) -> bytes:
    """Pack strings into the NUL-separated, NUL-terminated buffer format."""
    out = bytearray()
    for s in values:
        out += s.encode("utf-8")
        out += b"\x00"
    return bytes(out)


def unpack_strings(buffer: bytes, count: int) -> list[str]:
    """Unpack ``count`` NUL-terminated strings from ``buffer``."""
    out: list[str] = []
    start = 0
    for _ in range(count):
        if start >= len(buffer):
            msg = f"Buffer holds fewer than {count} strings."
            raise DERangeError(msg)
        end = buffer.find(b"\x00", start)
        if end < 0:
            end = len(buffer)
        out.append(buffer[start:end].decode("utf-8"))
        start = end + 1
    return out


def pack_names(names: list[str]) -> str:
    """Join axis names the way the reference connectors do (newline separated)."""
    return "\n".join(names)


def unpack_names(blob: str) -> list[str]:
    """Split a packed axis-name string."""
    return blob.split("\n")


# ---------------------------------------------------------------------------
# dtype <-> (type code, element width)
# ---------------------------------------------------------------------------

_KIND_TO_TYPE = {
    "b": Type.SIGNED,  # numpy bool_ is written as a 1-byte signed integer
    "i": Type.SIGNED,
    "u": Type.UNSIGNED,
    "f": Type.FLOAT,
    "c": Type.COMPLEX,
}


def dtype_to_eltype(dtype: np.dtype) -> Type:
    """Return the DataEcon element-type code for a numpy dtype."""
    if dtype.kind in ("U", "S", "O"):
        return Type.STRING
    try:
        return _KIND_TO_TYPE[dtype.kind]
    except KeyError:
        msg = f"Cannot store numpy dtype {dtype!r} in a .daec file."
        raise DEUnsupportedError(msg) from None


def eltype_to_dtype(eltype: int, nbytes_per_element: int) -> np.dtype:
    """Return the canonical numpy dtype for a type code and element width.

    ``nbytes_per_element == 0`` means "unknown" (an empty array), in which case
    the reference connectors fall back to the platform default width, i.e.
    64-bit integers and doubles.
    """
    eltype = int(eltype)
    n = nbytes_per_element
    if eltype == Type.SIGNED:
        if n == 0:
            return np.dtype(np.int64)
        if n in (1, 2, 4, 8):
            return np.dtype(f"i{n}")
    elif eltype == Type.UNSIGNED:
        if n == 0:
            return np.dtype(np.uint64)
        if n in (1, 2, 4, 8):
            return np.dtype(f"u{n}")
    elif eltype == Type.FLOAT:
        if n == 0:
            return np.dtype(np.float64)
        if n in (2, 4, 8):
            return np.dtype(f"f{n}")
    elif eltype == Type.COMPLEX:
        if n == 0:
            return np.dtype(np.complex128)
        if n in (8, 16):
            return np.dtype(f"c{n}")
        if n == 4:
            # Julia's ComplexF16 -- numpy has no complex32, widen to complex64.
            return np.dtype(np.complex64)
    elif eltype == Type.DATE:
        return np.dtype(np.int64)
    elif eltype == Type.STRING:
        return np.dtype(object)

    msg = (
        f"Unsupported element type/width combination: "
        f"type={Type(eltype).name if eltype in set(Type) else eltype}, {n} bytes per element."
    )
    raise DEUnsupportedError(msg)


def _canonical_eltype_name(eltype: int, elfreq: int, dtype: np.dtype) -> bool:
    """Return whether ``dtype`` is what :func:`eltype_to_dtype` would produce."""
    if int(elfreq) != Frequency.NONE:
        return False
    try:
        return eltype_to_dtype(eltype, dtype.itemsize) == dtype
    except DEUnsupportedError:
        return False


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------


class ScalarData(NamedTuple):
    """A scalar value ready to be written to the ``scalars`` table."""

    type: Type
    frequency: int
    blob: bytes | None
    #: Value for the ``pytype`` attribute, or ``None`` if the type round-trips.
    pytype: str | None


def _int_blob(value: int) -> bytes:
    """Encode a Python ``int``, widening to 16 bytes if it does not fit in 8."""
    try:
        return int(value).to_bytes(8, "little", signed=True)
    except OverflowError:
        pass
    try:
        return int(value).to_bytes(16, "little", signed=True)
    except OverflowError:
        msg = f"Integer too large to store in a .daec file: {value}"
        raise DERangeError(msg) from None


def encode_scalar(value: Any) -> ScalarData:
    """Encode a Python value as a DataEcon scalar."""
    # --- TimeSeriesEconPy dates, if that package is installed ---------------
    mit_data = _encode_tsecon_scalar(value)
    if mit_data is not None:
        return mit_data

    if value is None:
        return ScalarData(Type.OTHER_SCALAR, Frequency.NONE, None, "none")

    if isinstance(value, str):
        return ScalarData(Type.STRING, Frequency.NONE, value.encode("utf-8") + b"\x00", None)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return ScalarData(Type.OTHER_SCALAR, Frequency.NONE, bytes(value), "bytes")

    # bool must precede int -- Python's bool is a subclass of int
    if isinstance(value, (bool, np.bool_)):
        return ScalarData(Type.SIGNED, Frequency.NONE, b"\x01" if value else b"\x00", "bool")

    if isinstance(value, _dt.datetime):
        seconds = value.replace(tzinfo=value.tzinfo or _dt.UTC).timestamp()
        return ScalarData(
            Type.FLOAT, Frequency.NONE, np.float64(seconds).tobytes(), "datetime.datetime"
        )

    if isinstance(value, _dt.date):
        midnight = _dt.datetime(value.year, value.month, value.day, tzinfo=_dt.UTC)
        seconds = midnight.timestamp()
        return ScalarData(
            Type.FLOAT, Frequency.NONE, np.float64(seconds).tobytes(), "datetime.date"
        )

    if isinstance(value, np.generic):
        eltype = dtype_to_eltype(value.dtype)
        return ScalarData(eltype, Frequency.NONE, value.tobytes(), None)

    if isinstance(value, int):
        return ScalarData(Type.SIGNED, Frequency.NONE, _int_blob(value), None)

    if isinstance(value, float):
        return ScalarData(Type.FLOAT, Frequency.NONE, np.float64(value).tobytes(), None)

    if isinstance(value, complex):
        return ScalarData(Type.COMPLEX, Frequency.NONE, np.complex128(value).tobytes(), None)

    msg = f"Cannot store a value of type {type(value).__name__} as a .daec scalar."
    raise DEUnsupportedError(msg)


def _encode_tsecon_scalar(value: Any) -> ScalarData | None:
    """Encode a TimeSeriesEconPy ``MIT`` or ``Duration``, or return ``None``."""
    from .interop import _tsecon

    if not _tsecon.available():
        return None
    return _tsecon.encode_scalar(value)


def decode_scalar(
    obj_type: int,
    frequency: int,
    blob: bytes | None,
    pytype: str | None = None,
    jtype: str | None = None,
) -> Any:
    """Decode a scalar read back from the ``scalars`` table."""
    obj_type = int(obj_type)
    frequency = int(frequency)
    nbytes = 0 if blob is None else len(blob)

    if pytype == "none":
        return None
    if pytype == "bytes":
        return bytes(blob or b"")

    if obj_type == Type.STRING:
        raw = bytes(blob or b"")
        return raw.split(b"\x00", 1)[0].decode("utf-8")

    if obj_type == Type.OTHER_SCALAR:
        return bytes(blob or b"")

    if blob is None:
        msg = "Scalar object has no stored value."
        raise DERangeError(msg)

    if obj_type == Type.DATE:
        code = int(np.frombuffer(blob, dtype=np.int64, count=1)[0])
        return _apply_scalar_type(_date_from_code(code, frequency), pytype, jtype)

    if obj_type == Type.SIGNED and frequency != Frequency.NONE and nbytes == 8:
        code = int(np.frombuffer(blob, dtype=np.int64, count=1)[0])
        return _apply_scalar_type(_duration_from_code(code, frequency), pytype, jtype)

    if obj_type == Type.SIGNED and nbytes == 16:
        return _apply_scalar_type(int.from_bytes(blob, "little", signed=True), pytype, jtype)
    if obj_type == Type.UNSIGNED and nbytes == 16:
        return _apply_scalar_type(int.from_bytes(blob, "little", signed=False), pytype, jtype)

    dtype = eltype_to_dtype(obj_type, nbytes)
    value = np.frombuffer(blob, dtype=dtype, count=1)[0]
    return _apply_scalar_type(value, pytype, jtype)


def _date_from_code(code: int, frequency: int) -> Any:
    from .interop import _tsecon

    if _tsecon.available():
        return _tsecon.mit_from_code(code, frequency)
    return code


def _duration_from_code(code: int, frequency: int) -> Any:
    from .interop import _tsecon

    if _tsecon.available():
        return _tsecon.duration_from_code(code, frequency)
    return code


#: Julia type names understood when reading a file written by the Julia connector.
_JTYPE_ALIASES = {
    "Bool": "bool",
    "Symbol": "str",
    "String": "str",
    "Date": "datetime.date",
    "Dates.Date": "datetime.date",
    "DateTime": "datetime.datetime",
    "Dates.DateTime": "datetime.datetime",
}


def _apply_scalar_type(value: Any, pytype: str | None, jtype: str | None) -> Any:
    """Restore the original Python type recorded in an attribute, if any."""
    name = pytype or _JTYPE_ALIASES.get(jtype or "")
    if name is None:
        return value.item() if isinstance(value, np.generic) else value

    if name == "bool":
        return bool(value)
    if name == "str":
        return str(value)
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    if name == "complex":
        return complex(value)
    if name in ("datetime.date", "date"):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.UTC).date()
    if name in ("datetime.datetime", "datetime"):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.UTC).replace(tzinfo=None)

    # An unrecognised type name (a Julia type we do not model, say) is not an
    # error: return the value as stored rather than failing the whole read.
    return value.item() if isinstance(value, np.generic) else value


# ---------------------------------------------------------------------------
# arrays
# ---------------------------------------------------------------------------


class ArrayValues(NamedTuple):
    """Encoded element data of an array object."""

    eltype: Type
    elfreq: int
    blob: bytes | None
    #: Value for the ``pyeltype`` attribute, or ``None`` if the dtype round-trips.
    pyeltype: str | None


def encode_array_values(array: np.ndarray) -> ArrayValues:
    """Encode the elements of ``array`` in column-major order."""
    from .interop import _tsecon

    if array.size == 0:
        eltype, elfreq, pyeltype = _empty_eltype(array)
        return ArrayValues(eltype, elfreq, None, pyeltype)

    if array.dtype.kind in ("U", "S", "O"):
        flat = array.ravel(order="F").tolist()
        if _tsecon.available():
            mit = _tsecon.encode_mit_array(array)
            if mit is not None:
                return mit
        if all(isinstance(v, str) for v in flat):
            return ArrayValues(Type.STRING, Frequency.NONE, pack_strings(flat), None)
        if all(isinstance(v, (bytes, bytearray)) for v in flat):
            return ArrayValues(
                Type.STRING, Frequency.NONE, b"".join(bytes(v) + b"\x00" for v in flat), "bytes"
            )
        msg = (
            "Cannot store an object array whose elements are not all strings. "
            f"Element types: {sorted({type(v).__name__ for v in flat})}"
        )
        raise DEUnsupportedError(msg)

    eltype = dtype_to_eltype(array.dtype)
    if array.dtype.kind == "b":
        # numpy bool_ is one byte; stored as a 1-byte signed integer.
        blob = array.astype(np.int8, copy=False).tobytes(order="F")
        return ArrayValues(Type.SIGNED, Frequency.NONE, blob, "bool")

    return ArrayValues(eltype, Frequency.NONE, array.tobytes(order="F"), None)


def _empty_eltype(array: np.ndarray) -> tuple[Type, int, str | None]:
    """Pick an element type for an empty array from its dtype alone."""
    if array.dtype.kind in ("U", "S", "O"):
        return Type.STRING, Frequency.NONE, None
    if array.dtype.kind == "b":
        return Type.SIGNED, Frequency.NONE, "bool"
    eltype = dtype_to_eltype(array.dtype)
    canonical = _canonical_eltype_name(eltype, Frequency.NONE, array.dtype)
    pyeltype = None if canonical else array.dtype.name
    return eltype, Frequency.NONE, pyeltype


#: numpy dtype names understood from a ``jeltype`` attribute written by Julia.
_JELTYPE_ALIASES = {
    "Bool": "bool",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "ComplexF32": "complex64",
    "ComplexF64": "complex128",
    "String": "str",
    "Symbol": "str",
}


def decode_array(
    eltype: int,
    elfreq: int,
    blob: bytes | None,
    shape: tuple[int, ...],
    pyeltype: str | None = None,
    jeltype: str | None = None,
) -> np.ndarray:
    """Decode array elements into a numpy array of the given ``shape``.

    The blob is read in column-major order; the returned array is a normal
    C-contiguous numpy array with the correct logical layout.
    """
    from .interop import _tsecon

    eltype = int(eltype)
    elfreq = int(elfreq)
    count = 1
    for dim in shape:
        count *= dim

    name = pyeltype or _JELTYPE_ALIASES.get(jeltype or "")

    if count == 0:
        dtype = _dtype_from_name(name) if name else eltype_to_dtype(eltype, 0)
        return np.empty(shape, dtype=dtype)

    if blob is None:
        msg = f"Array object of shape {shape} has no stored value."
        raise DERangeError(msg)

    if eltype == Type.STRING:
        strings = np.empty(count, dtype=object)
        strings[:] = unpack_strings(blob, count)
        return _reshape_f(strings, shape)

    itemsize, remainder = divmod(len(blob), count)
    if remainder:
        msg = f"Stored value of {len(blob)} bytes does not divide evenly into {count} elements."
        raise DERangeError(msg)

    if eltype == Type.DATE and _tsecon.available():
        codes = np.frombuffer(blob, dtype=np.int64, count=count)
        return _reshape_f(_tsecon.mit_array_from_codes(codes, elfreq), shape)

    if eltype in (Type.SIGNED, Type.UNSIGNED) and itemsize == 16:
        signed = eltype == Type.SIGNED
        out = np.empty(count, dtype=object)
        for i in range(count):
            out[i] = int.from_bytes(blob[16 * i : 16 * (i + 1)], "little", signed=signed)
        return _reshape_f(out, shape)

    if eltype == Type.COMPLEX and itemsize == 4:
        halves = np.frombuffer(blob, dtype=np.float16, count=2 * count)
        values = halves[0::2].astype(np.float32) + 1j * halves[1::2].astype(np.float32)
        return _reshape_f(values, shape)

    dtype = eltype_to_dtype(eltype, itemsize)
    values = np.frombuffer(blob, dtype=dtype, count=count)

    if name:
        target = _dtype_from_name(name)
        if target is not None and target != values.dtype:
            values = values.astype(target)

    return _reshape_f(values, shape)


def _reshape_f(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reshape a flat, column-major element sequence into ``shape``."""
    if len(shape) <= 1:
        return np.array(flat, copy=True).reshape(shape)
    return np.ascontiguousarray(flat.reshape(shape, order="F"))


def _dtype_from_name(name: str) -> np.dtype | None:
    if name == "str":
        return np.dtype(object)
    try:
        return np.dtype(name)
    except TypeError:
        return None
