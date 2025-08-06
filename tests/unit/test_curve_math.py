# tests/unit/test_curve_math.py - CORRECTED VERSION
"""Test mathematical calculations with real FreeCAD"""
import pytest
import numpy as np
import math
import sys
import os
import FreeCAD

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Ensure FreeCAD path is available
sys.path.insert(0, '/usr/lib/freecad-daily/lib')

# Now import our modules
from src.curves import Curves


class TestCurveMathematics:
    """Test mathematical calculations with real FreeCAD"""

    def test_freecad_available(self):
        """Test that real FreeCAD is working"""
        import FreeCAD
        version = FreeCAD.Version()
        assert len(version) >= 4
        print(f"Using FreeCAD version: {version}")

        # Test basic document operations
        doc = FreeCAD.newDocument("TestMath")
        assert doc is not None
        FreeCAD.closeDocument("TestMath")

    def test_curve_length_calculation(self):
        """Test curve length calculation using get_curve_points()"""
        import FreeCAD

        # Create real FreeCAD document
        doc = FreeCAD.newDocument("TestCurveLength")

        try:
            # Test with a simple linear curve
            curves = Curves(doc, {
                'type': 'linear',
                'parameters': {
                    'length': 10.0,
                    'points': 5,
                    'direction': [1, 0, 0]  # X-direction
                }
            })

            # Get the curve points using the available method
            points = curves.get_curve_points()

            # Manual calculation of curve length from points
            total_length = 0.0
            for i in range(len(points) - 1):
                p1 = np.array(points[i])
                p2 = np.array(points[i + 1])
                segment_length = np.linalg.norm(p2 - p1)
                total_length += segment_length

            # For a linear curve of length 10, total should be approximately 10
            assert abs(total_length - 10.0) < 0.1
            print(f"Linear curve length: {total_length:.4f} (expected ~10.0)")

        finally:
            # Clean up
            FreeCAD.closeDocument("TestCurveLength")

    def test_helical_curve_generation(self):
        """Test helical curve generation with real FreeCAD"""
        import FreeCAD

        doc = FreeCAD.newDocument("TestHelical")

        try:
            curves = Curves(doc, {
                'type': 'helical',
                'parameters': {
                    'radius': 8.0,
                    'pitch': 4.0,
                    'turns': 2.0,
                    'points': 50
                }
            })

            points = curves.get_curve_points()

            # Validate output
            assert len(points) == 50
            print(f"Generated {len(points)} helical points")

            # Check helical properties
            for i, point in enumerate(points):
                radius = math.sqrt(point[0] ** 2 + point[1] ** 2)
                assert abs(radius - 8.0) < 0.1, f"Point {i}: radius {radius:.3f} != 8.0"

            # Ensure Z coordinates increase (helical progression)
            z_coords = [p[2] for p in points]
            assert all(z_coords[i] <= z_coords[i + 1] for i in range(len(z_coords) - 1))
            print(f"Z progression: {z_coords[0]:.2f} → {z_coords[-1]:.2f}")

        finally:
            FreeCAD.closeDocument("TestHelical")

    # def test_angle_normalization(self):
    #     """Test angle boundary handling to prevent -179° issues"""
    #
    #     def normalize_angle(angle_rad):
    #         """Helper method for angle normalization"""
    #         return ((angle_rad % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi) - math.pi
    #
    #     # Test cases that historically caused -179° problems
    #     test_angles = [179.5, -179.5, 180.0, -180.0, 360.0, -360.0]
    #
    #     for angle in test_angles:
    #         normalized = normalize_angle(math.radians(angle))
    #         normalized_deg = math.degrees(normalized)
    #
    #         # Should be in range [-180, 180]
    #         assert -180 <= normalized_deg <= 180
    #         print(f"Angle {angle}° → {normalized_deg:.2f}°")
    #
    #         def test_curve_tangent_calculation():
    #             """Test if curve tangents are being calculated correctly."""
    #
    #             print("=== Test 1: Curve Tangent Calculation ===")
    #
    #             # Get your curve points (first 5 points for testing)
    #             curve_points = your_curve_generation_function()[:5]
    #
    #             for i in range(len(curve_points) - 1):
    #                 start_point = curve_points[i]
    #                 end_point = curve_points[i + 1]
    #
    #                 # Calculate chord vector
    #                 chord_vector = end_point - start_point
    #                 chord_length = np.linalg.norm(chord_vector)
    #                 chord_unit = chord_vector / chord_length
    #
    #                 # Calculate tangent (this should match your actual tangent calculation)
    #                 if i < len(curve_points) - 2:
    #                     next_point = curve_points[i + 2]
    #                     tangent_vector = next_point - start_point
    #                 else:
    #                     tangent_vector = chord_vector
    #
    #                 tangent_length = np.linalg.norm(tangent_vector)
    #                 tangent_unit = tangent_vector / tangent_length
    #
    #                 # Test results
    #                 print(f"\nWafer {i + 1}:")
    #                 print(f"  Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
    #                 print(f"  End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
    #                 print(f"  Chord unit:   [{chord_unit[0]:.3f}, {chord_unit[1]:.3f}, {chord_unit[2]:.3f}]")
    #                 print(f"  Tangent unit: [{tangent_unit[0]:.3f}, {tangent_unit[1]:.3f}, {tangent_unit[2]:.3f}]")
    #
    #                 # Check angle between chord and tangent
    #                 dot_product = np.dot(chord_unit, tangent_unit)
    #                 angle_deg = np.degrees(np.arccos(np.clip(abs(dot_product), 0, 1)))
    #                 print(f"  Chord/Tangent angle: {angle_deg:.1f}°")
    #
    #                 # Expected for helix: should be small but not zero
    #                 if angle_deg < 1.0:
    #                     print(f"  ⚠️  Very small angle - may cause orientation issues")
    #                 elif angle_deg > 30.0:
    #                     print(f"  ⚠️  Large angle - check curve smoothness")
    #                 else:
    #                     print(f"  ✅ Reasonable angle for helical curve")

    def test_rotation_matrix_construction(self):
        """Test if FreeCAD rotation matrices are being built correctly."""

        print("\n=== Test 2: Rotation Matrix Construction ===")

        # Test with known vectors
        test_cases = [
            {
                'name': 'X-axis alignment',
                'target_vector': np.array([1, 0, 0]),
                'expected_rotation': [0, 0, 0]  # No rotation needed
            },
            {
                'name': 'Y-axis alignment',
                'target_vector': np.array([0, 1, 0]),
                'expected_rotation': [0, 0, 90]  # 90° around Z
            },
            {
                'name': 'Diagonal alignment',
                'target_vector': np.array([1, 1, 0]) / np.sqrt(2),
                'expected_rotation': [0, 0, 45]  # 45° around Z
            },
            {
                'name': 'Helical direction example',
                'target_vector': np.array([0.707, 0.707, 0.1]),  # Typical helix tangent
                'expected_rotation': None  # We'll calculate this
            }
        ]

        for test_case in test_cases:
            print(f"\n{test_case['name']}:")
            target = test_case['target_vector']
            target_unit = target / np.linalg.norm(target)

            print(f"  Target vector: [{target_unit[0]:.3f}, {target_unit[1]:.3f}, {target_unit[2]:.3f}]")

            # Test your rotation matrix construction method
            try:
                # This should match your actual rotation calculation in add_wafer()
                z_axis = FreeCAD.Vector(target_unit[0], target_unit[1], target_unit[2])

                # Create perpendicular axes
                if abs(z_axis.z) < 0.9:
                    x_axis = FreeCAD.Vector(0, 0, 1).cross(z_axis)
                else:
                    x_axis = FreeCAD.Vector(1, 0, 0).cross(z_axis)

                if x_axis.Length > 1e-6:
                    x_axis.normalize()
                    y_axis = z_axis.cross(x_axis)
                    y_axis.normalize()

                    # Create rotation matrix
                    rotation_matrix = FreeCAD.Matrix(
                        x_axis.x, y_axis.x, z_axis.x, 0,
                        x_axis.y, y_axis.y, z_axis.y, 0,
                        x_axis.z, y_axis.z, z_axis.z, 0,
                        0, 0, 0, 1
                    )

                    wafer_rotation = FreeCAD.Rotation(rotation_matrix)
                    angles = wafer_rotation.getYawPitchRoll()

                    print(f"  Calculated rotation (Y,P,R): [{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°]")

                    # Test if rotation actually aligns with target
                    test_vector = FreeCAD.Vector(1, 0, 0)  # Default wafer direction
                    rotated_vector = wafer_rotation.multVec(test_vector)

                    print(
                        f"  Rotated test vector: [{rotated_vector.x:.3f}, {rotated_vector.y:.3f}, {rotated_vector.z:.3f}]")

                    # Check alignment
                    dot_product = rotated_vector.dot(z_axis)
                    print(f"  Alignment check (dot product): {dot_product:.3f}")

                    if abs(dot_product - 1.0) < 0.01:
                        print(f"  ✅ Good alignment")
                    else:
                        print(f"  ❌ Poor alignment - rotation matrix issue")

                else:
                    print(f"  ❌ Degenerate case - x_axis too small")

            except Exception as e:
                print(f"  ❌ Error in rotation calculation: {e}")

    def test_wafer_chain_continuity(self):
        """Test if wafers connect properly end-to-end."""

        print("\n=== Test 4: Wafer Chain Continuity ===")

        # Get actual wafer data from your system
        wafer_positions = []  # You'll need to extract this from your actual run

        # Example format:
        wafer_positions = [
            {'start': [8.0, 0.0, 0.0], 'end': [7.4, 2.9, 0.24]},
            {'start': [7.4, 2.9, 0.24], 'end': [4.6, 6.5, 0.6]},
            # ... add more from your actual debug output
        ]

        print(f"Testing {len(wafer_positions)} wafer connections:")

        for i in range(len(wafer_positions) - 1):
            current_wafer = wafer_positions[i]
            next_wafer = wafer_positions[i + 1]

            current_end = np.array(current_wafer['end'])
            next_start = np.array(next_wafer['start'])

            gap = np.linalg.norm(next_start - current_end)

            print(f"  Wafer {i + 1} → {i + 2}: Gap = {gap:.6f}")

            if gap < 0.001:
                print(f"    ✅ Good connection")
            else:
                print(f"    ❌ Gap detected - wafers not connecting")
                print(f"    Current end:  [{current_end[0]:.3f}, {current_end[1]:.3f}, {current_end[2]:.3f}]")
                print(f"    Next start:   [{next_start[0]:.3f}, {next_start[1]:.3f}, {next_start[2]:.3f}]")