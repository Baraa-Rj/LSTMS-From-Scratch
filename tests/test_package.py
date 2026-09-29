"""The package must install its real dependencies and import as a package."""
import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pyproject_declares_numpy_and_not_rich():
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    names = [d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in deps]
    assert "numpy" in names
    assert "rich" not in names


@pytest.mark.parametrize(
    "module",
    ["lstms.Activation", "lstms.Cell", "lstms.Lstm", "lstms.Tokenizer",
     "lstms.Trainer", "lstms.Generator", "lstms.__main__"],
)
def test_modules_import_as_package(module):
    importlib.import_module(module)
