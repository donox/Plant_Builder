# conftest.py
import pytest
import sys
import os
import numpy as np
from typing import Dict, List, Any

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