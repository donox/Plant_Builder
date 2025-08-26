# tests/conftest.py
# Ensure FreeCAD/Part are importable BEFORE importing project modules.
import os, sys, pathlib
import pytest
from tests.fixtures.freecad_mocks import install_freecad_stubs

# 1) Put repo's src on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]  # …/PlantBuilder
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS = REPO_ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

# 2) Install FreeCAD/Part shims
@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    # Ensure mocks can be imported
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures")))
    install_freecad_stubs(sys.modules)
    # Optional sanity log
    print(">>> FreeCAD stubs installed (pytest_sessionstart)")

# 3) Optional: make utilities.get_axes read our LCS axes if your wafer code uses it.
try:
    import utilities  # from your repo
    def _mock_get_axes(lcs):
        # Prefer explicit attributes if present (our LCS), else try to infer
        if hasattr(lcs, "x_axis") and hasattr(lcs, "y_axis") and hasattr(lcs, "z_axis"):
            return lcs.x_axis, lcs.y_axis, lcs.z_axis
        # Fallbacks: add more branches if your code uses a different layout
        raise AttributeError("Mock get_axes: LCS missing x_axis/y_axis/z_axis")
    # Only monkeypatch if not present or if you want to force this behavior:
    utilities.get_axes = _mock_get_axes  # type: ignore[attr-defined]
except Exception:
    # If utilities cannot be imported yet, tests that need it will import later.
    pass
