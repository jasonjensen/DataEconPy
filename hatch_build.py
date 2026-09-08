"""Hatchling build hook: compile the vendored DataEcon C sources into the wheel.

DataEconPy drives the reference DataEcon C library through ctypes, so a usable
wheel has to contain that library. This hook compiles `vendor/dataecon-c` into
`src/dataecon/_lib/` at wheel-build time, which is where `dataecon._loader`
looks for it.

The compile logic lives in `src/dataecon/build.py`, which is loaded here
straight off disk rather than imported as `dataecon.build`. Wheel builds run in
an isolated environment holding only the build backend, so importing the package
proper would fail on its numpy dependency.

If compilation fails, the build still succeeds and prints a warning: the wheel
is then a pure-Python install, and the user can run `python -m dataecon.build`
or point `DATAECON_LIBRARY` at an existing copy of the library.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _load_build_module(root: str) -> Any:
    """Load src/dataecon/build.py without importing the dataecon package."""
    path = os.path.join(root, "src", "dataecon", "build.py")
    spec = importlib.util.spec_from_file_location("_dataeconpy_build", path)
    if spec is None or spec.loader is None:
        msg = f"could not load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CustomBuildHook(BuildHookInterface):
    """Builds libdaec before the wheel is assembled."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get("DATAECONPY_SKIP_LIB_BUILD"):
            self.app.display_warning("DATAECONPY_SKIP_LIB_BUILD is set; not building libdaec.")
            return

        try:
            build = _load_build_module(self.root)
            output = build.build_library(verbose=False)
        except Exception as err:  # a failed library build must not fail the wheel
            self.app.display_warning(
                f"Could not build the DataEcon C library ({err}). The wheel will "
                "not contain it; run `python -m dataecon.build` after installing, "
                "or set DATAECON_LIBRARY to a prebuilt libdaec."
            )
            return

        self.app.display_info(f"Built DataEcon C library: {output}")
        # The wheel now holds a compiled object, so it is not pure Python and
        # must carry this interpreter's platform tag rather than py3-none-any.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
