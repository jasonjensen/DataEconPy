# SPDX-License-Identifier: MIT
"""Whole-file conveniences: read or write a ``.daec`` file in one call."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ._values import read_catalog, write_into
from .file import ROOT_ID, DEFile, opendaec

__all__ = ["readdb", "writedb"]


def readdb(
    source: str | os.PathLike[str] | DEFile,
    path: str | int = ROOT_ID,
    *,
    as_dict: bool = False,
) -> Any:
    """Read a whole ``.daec`` file, or one catalog inside it.

    Parameters
    ----------
    source
        Path of the file to read, or an already-open :class:`~dataecon.DEFile`.
    path
        Catalog to read. Defaults to the root, i.e. the whole file.
    as_dict
        Return plain nested dictionaries instead of ``tsecon`` ``Workspace``
        objects. Has no effect if TimeSeriesEconPy is not installed, since plain
        dictionaries are then the only option.

    Returns
    -------
    Workspace or dict
        The catalog's contents, with nested catalogs read recursively.

    Examples
    --------
    >>> data = readdb("model.daec")            # doctest: +SKIP
    >>> data.gdp                               # doctest: +SKIP
    """
    if isinstance(source, DEFile):
        return read_catalog(source, path, as_dict=as_dict)
    with opendaec(source, readonly=True) as de:
        return read_catalog(de, path, as_dict=as_dict)


def writedb(
    target: str | os.PathLike[str] | DEFile,
    data: Mapping[str, Any] | Any,
    path: str | int = ROOT_ID,
    *,
    append: bool = True,
    overwrite: bool = False,
) -> None:
    """Write a mapping of named values into a ``.daec`` file.

    Parameters
    ----------
    target
        Path of the file to write, or an already-open :class:`~dataecon.DEFile`.
        A path is created if it does not exist.
    data
        A mapping (or ``Workspace``) whose entries become objects in the file.
        Nested mappings become nested catalogs.
    path
        Catalog to write into. Defaults to the root.
    append
        Keep whatever the file already contains. Pass ``False`` to empty it
        first. Ignored when ``target`` is an open file.
    overwrite
        Replace objects whose names already exist instead of raising.

    Examples
    --------
    >>> writedb("model.daec", {"gdp": gdp, "params": {"beta": 0.99}})  # doctest: +SKIP
    """
    items = _as_items(data)
    if isinstance(target, DEFile):
        _write_items(target, path, items, overwrite=overwrite)
        return
    with opendaec(target, write=True, append=append, overwrite=overwrite) as de:
        _write_items(de, path, items, overwrite=overwrite)


def _as_items(data: Any) -> Mapping[str, Any]:
    if isinstance(data, Mapping):
        return data
    if hasattr(data, "items"):  # tsecon.Workspace and friends
        return dict(data.items())
    msg = (
        "writedb needs a mapping of names to values (a dict or a Workspace); "
        f"got {type(data).__name__}. Use DEFile.write() to store a single value."
    )
    raise TypeError(msg)


def _write_items(de: DEFile, path: str | int, items: Mapping[str, Any], *, overwrite: bool) -> None:
    parent_id = ROOT_ID if path == ROOT_ID else de.makedirs(str(path))
    for name, value in items.items():
        write_into(de, parent_id, str(name), value, overwrite=overwrite)
