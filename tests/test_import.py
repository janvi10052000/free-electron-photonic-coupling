"""Phase 0 check: the package and all its modules can be imported."""

import importlib

import pytest

MODULES = [
    "fepc",
    "fepc.constants",
    "fepc.pinem",
    "fepc.materials",
    "fepc.slab",
    "fepc.coupling",
    "fepc.beam",
    "fepc.bayesopt",
    "fepc.inference",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
