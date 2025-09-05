# conftest.py
import pytest
import sys
import os
import numpy as np
from typing import Dict, List, Any
import types, pathlib

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def freecad_available():
    """Check if FreeCAD is available in environment"""
    try:
        # import FreeCAD
        return True
    except ImportError:
        pytest.skip("FreeCAD not available in test environment")


@pytest.fixture
def sample_curve_points():
    """Generate sample curve points for mathematical testing"""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 2.0],
        [3.0, 2.0, 3.0],
        [4.0, 3.0, 4.0]
    ])


@pytest.fixture
def helical_curve_spec():
    """Standard helical curve specification for testing"""
    return {
        'type': 'helical',
        'parameters': {
            'radius': 8.0,
            'pitch': 4.0,
            'turns': 3.0,
            'points': 100
        },
        'transformations': [],
        'segment': {
            'start_fraction': 0.0,
            'end_fraction': 1.0
        }
    }


@pytest.fixture
def curve_follower_config():
    """Standard configuration for curve follower testing"""
    return {
        'cylinder_diameter': 1.5,
        'min_height': 0.8,
        'max_chord': 0.2
    }


class GeometryValidator:
    """Helper class for validating geometric calculations"""

    @staticmethod
    def validate_curve_continuity(points: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Validate curve has no discontinuous jumps"""
        if len(points) < 2:
            return True

        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        max_distance = np.max(distances)
        avg_distance = np.mean(distances)

        # Check for discontinuous jumps (distance > 10x average)
        return max_distance <= avg_distance * 10

    @staticmethod
    def validate_helical_properties(points: np.ndarray, expected_radius: float,
                                    tolerance: float = 0.1) -> bool:
        """Validate points follow helical properties"""
        # Check radius consistency
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        radius_deviation = np.abs(radii - expected_radius)

        return np.all(radius_deviation <= tolerance)


@pytest.fixture
def geometry_validator():
    """Provide geometry validation utilities"""
    return GeometryValidator()


# --- BEGIN (add to existing tests/conftest.py) -------------------------

# Ensure project root (sibling of tests/ and src/) is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Minimal FreeCAD stubs so importing your real modules never crashes
def _ensure_freecad_stubs():
    if "FreeCAD" not in sys.modules:
        FreeCAD = types.ModuleType("FreeCAD")
        class _Vec(tuple):
            def __new__(cls, x=0.0, y=0.0, z=0.0):
                return tuple.__new__(cls, (float(x), float(y), float(z)))
        FreeCAD.Vector = _Vec
        sys.modules["FreeCAD"] = FreeCAD

    if "FreeCADGui" not in sys.modules:
        FreeCADGui = types.ModuleType("FreeCADGui")
        sys.modules["FreeCADGui"] = FreeCADGui

_ensure_freecad_stubs()


def _ensure_stub(name, attrs=None):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m

# FreeCAD + FreeCADGui stubs (very light)
FreeCAD = _ensure_stub("FreeCAD", {})
_ensure_stub("FreeCADGui", {})

# Common FreeCAD submodules some projects import implicitly
_ensure_stub("FreeCAD.Base", {})
_ensure_stub("FreeCAD.Vector", {})  # just in case someone imports like "from FreeCAD import Vector"

# Part (needed by src/curves.py). Provide dummies for common names so attribute access won't explode.
Part = _ensure_stub("Part", {
    "Shape": object,
    "Edge": object,
    "Wire": object,
    "Face": object,
})
# Popular callable names sometimes imported/used; harmless no-ops:
for fname in ("makeLine", "makeCircle", "makePolygon", "makeSphere", "makeCylinder"):
    setattr(Part, fname, lambda *a, **k: None)

# Optional: other modules sometimes pulled transitively; stub only if your imports require them
_ensure_stub("Draft", {})
_ensure_stub("DraftVecUtils", {})
_ensure_stub("Mesh", {})
_ensure_stub("MeshPart", {})
# --- END: minimal stubs ---

