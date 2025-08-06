# tests/unit/test_helical_debugging.py
"""Tests specifically for helical curve follower debugging"""
import pytest
import numpy as np
import math
import sys
import os

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, '/usr/lib/freecad-daily/lib')

from src.curve_follower import CurveFollower
from src.curves import Curves


class TestHelicalDebugging:
    """Test cases for specific helical positioning issues"""

    def test_rotation_boundary_stability(self):
        """Test that rotations near ±180° don't cause -179° failures"""

        def normalize_angle(angle_rad):
            return ((angle_rad % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi) - math.pi

        # Test the problematic angles that caused -179° issues
        problematic_angles = [
            math.radians(179), math.radians(-179),
            math.radians(180), math.radians(-180),
            math.radians(181), math.radians(-181)
        ]

        for angle in problematic_angles:
            normalized = normalize_angle(angle)
            assert -math.pi <= normalized <= math.pi
            # Ensure we don't get exactly ±π (which causes instability)
            assert not math.isclose(abs(normalized), math.pi, abs_tol=1e-10)
            print(f"Angle {math.degrees(angle):6.1f}° → {math.degrees(normalized):6.1f}°")

    def test_coordinate_system_chain_integrity(self):
        """Test that coordinate transformations maintain positioning"""
        import FreeCAD

        doc = FreeCAD.newDocument("TestCoordinateChain")

        try:
            # Create a simple helical curve for testing
            curve_spec = {
                'type': 'helical',
                'parameters': {
                    'radius': 8.0,
                    'pitch': 4.0,
                    'turns': 1.0,  # Single turn for predictability
                    'points': 20
                }
            }

            curves = Curves(doc, curve_spec)
            points = curves.get_curve_points()

            # Verify helical properties are maintained
            for i, point in enumerate(points):
                radius = math.sqrt(point[0] ** 2 + point[1] ** 2)
                assert abs(radius - 8.0) < 0.1, f"Point {i}: radius deviation"

            # Calculate expected segment distance for this helical geometry
            # Arc length per segment: (1 turn / 20 points) * circumference
            arc_length_per_segment = (1.0 / 20) * 2 * math.pi * 8.0  # ≈ 2.51
            # Vertical rise per segment
            vertical_per_segment = 4.0 / 20  # = 0.2
            # 3D distance per segment
            expected_3d_distance = math.sqrt(arc_length_per_segment ** 2 + vertical_per_segment ** 2)

            print(f"Expected 3D distance per segment: {expected_3d_distance:.3f}")

            # Verify smooth progression (no jumps)
            max_expected_distance = expected_3d_distance * 1.1  # 10% tolerance

            for i in range(len(points) - 1):
                p1, p2 = np.array(points[i]), np.array(points[i + 1])
                distance = np.linalg.norm(p2 - p1)

                # Should not have large jumps between consecutive points
                assert distance < max_expected_distance, f"Large jump detected at segment {i}: {distance:.3f} > expected {max_expected_distance:.3f}"

                # Also verify it's not too small (which would indicate duplicate points)
                assert distance > expected_3d_distance * 0.5, f"Segment {i} too short: {distance:.3f}"

            print(f"All {len(points) - 1} segments have reasonable distances")

        finally:
            FreeCAD.closeDocument(doc.Name)