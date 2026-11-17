from __future__ import annotations

from importlib.resources import files

import apdo


def test_version_is_exposed() -> None:
    assert apdo.__version__ == "0.3.0"


def test_py_typed_marker_is_installed() -> None:
    assert files("apdo").joinpath("py.typed").is_file()
