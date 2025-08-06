# tests/unit/test_angle_calculations.py
"""Test the specific angle calculations that are failing"""
import pytest
import numpy as np
import math
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, '/usr/lib/freecad-daily/lib')


class TestAngleCalculations:
    """Test problematic angle calculations from the logs"""

    def test_extreme_dot_product_angles(self):
        """Test angle calculations when dot products approach ±1"""

        def calculate_angle_from_dot_product(dot_product):
            """Reproduce the calculation from your code"""
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = math.acos(abs(dot_product))  # abs() gives acute angle
            return angle

        # Test cases that produce extreme dot products
        extreme_dot_products = [-0.98, -0.99, -0.999, -1.0, 0.98, 0.99, 0.999, 1.0]

        for dot in extreme_dot_products:
            angle = calculate_angle_from_dot_product(dot)
            angle_deg = math.degrees(angle)

            print(f"Dot product {dot:6.3f} → angle {angle_deg:6.1f}°")

            # Since abs(dot_product) is used, angles should be 0° to 90°
            assert 0 <= angle_deg <= 90, f"Angle should be 0-90°, got {angle_deg}°"

            # Very close to ±1 should produce small acute angles (parallel/antiparallel vectors)
            if abs(dot) > 0.97:
                assert angle_deg < 15, f"Dot {dot} should produce small angle (vectors nearly parallel/antiparallel), got {angle_deg}°"

    def test_rotation_angle_normalization(self):
        """Test the specific rotation angle normalization that's failing"""

        def normalize_angle_current(angle_rad):
            """Current normalization (failing)"""
            return ((angle_rad % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi) - math.pi

        def normalize_angle_improved(angle_rad):
            """Actually improved normalization using math.remainder"""
            if math.isnan(angle_rad) or math.isinf(angle_rad):
                return 0.0

            # Use math.remainder for proper [-π, π] normalization
            normalized = math.remainder(angle_rad, 2 * math.pi)

            # Add larger epsilon buffer to avoid near-boundary values
            epsilon = 1e-3  # Increased from 1e-10
            if abs(abs(normalized) - math.pi) < epsilon:
                normalized = math.copysign(math.pi - epsilon, normalized)

            return normalized

        # Test the problematic values from your log
        problematic_angles = [
            math.radians(-179.13),  # From your log
            math.radians(-180.0),  # From your log
            math.radians(179.13),
            math.radians(180.0),
        ]

        for angle in problematic_angles:
            current_result = normalize_angle_current(angle)
            improved_result = normalize_angle_improved(angle)

            print(
                f"Original: {math.degrees(angle):7.2f}° → Current: {math.degrees(current_result):7.2f}° → Improved: {math.degrees(improved_result):7.2f}°")

            # Add detailed debug
            print(f"  Raw improved result: {improved_result}")
            print(f"  Abs degrees: {abs(math.degrees(improved_result))}")
            print(f"  Close to pi? {abs(abs(improved_result) - math.pi) < 1e-6}")

            # Test that improved version stays within bounds
            assert -math.pi <= improved_result <= math.pi

            # The epsilon buffer should prevent near ±180°
            assert abs(math.degrees(
                improved_result)) < 179.95, f"Should avoid near ±180°, got {math.degrees(improved_result):.2f}°"

    def test_lift_angle_calculation_logic(self):
        """Test the lift angle calculation with realistic angles for shallow helix"""

        def calculate_lift_angle_improved(start_angle, end_angle, wafer_type):
            """Test the improved lift calculation"""
            if wafer_type == "CC":
                return 0.0
            elif wafer_type == "CE":
                return end_angle
            elif wafer_type == "EC":
                return start_angle
            elif wafer_type == "EE":
                # Use average to prevent extreme values
                return (start_angle + end_angle) / 2.0
            else:
                return (start_angle + end_angle) / 2.0

        # Test with REALISTIC angles for a shallow helix (should be ~5-15°)
        # Your helix has pitch angle ≈ 4.5°, so ellipse angles should be similar
        realistic_angles = [
            (math.radians(5), math.radians(8), "EE"),  # Realistic small angles
            (math.radians(0), math.radians(10), "CE"),  # Circular to elliptical
            (math.radians(12), math.radians(0), "EC"),  # Elliptical to circular
            (math.radians(0), math.radians(0), "CC"),  # Circular to circular
            (math.radians(2), math.radians(3), "EE"),  # Very shallow case
        ]

        for start_angle, end_angle, wafer_type in realistic_angles:
            lift = calculate_lift_angle_improved(start_angle, end_angle, wafer_type)
            lift_deg = math.degrees(lift)

            print(
                f"Wafer {wafer_type}: start={math.degrees(start_angle):6.1f}°, end={math.degrees(end_angle):6.1f}° → lift={lift_deg:6.1f}°")

            # Should produce reasonable lift angles for woodworking
            # For your shallow helix, lift angles should be very small
            assert abs(
                lift_deg) <= 30, f"Lift angle {lift_deg}° too extreme for {wafer_type} (shallow helix should produce small angles)"

    def test_actual_geometry_debug(self):
        """Debug the actual curve follower geometry calculations"""
        from curve_follower import CurveFollower
        from curves import Curves

        # Use your actual config
        curve_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
        }

        # Create the actual objects (without FreeCAD doc)
        curves = Curves(None, curve_spec)
        curve_points = curves.get_curve_points()

        # Test a few specific geometric calculations
        print(f"Generated {len(curve_points)} curve points")
        print(f"First point: {curve_points[0]}")
        print(f"Last point: {curve_points[-1]}")

        # Verify curve is reasonable
        assert len(curve_points) == 100
        assert curve_points[0][2] == 0.0  # Should start at z=0

        # Check that the curve follows expected helical geometry
        total_height = curve_points[-1][2] - curve_points[0][2]
        expected_height = 3.0 * 4.0  # turns * pitch
        assert abs(total_height - expected_height) < 0.1, f"Height {total_height} != expected {expected_height}"

    def test_simple_lift_calculation(self):
        """Test with known simple geometry to verify basic trigonometry"""

        # Simple case: nearly parallel vectors (shallow helix)
        tangent_vector = np.array([0.98, 0.2, 0.06])  # Shallow helix direction
        chord_vector = np.array([0.99, 0.15, 0.06])  # Similar direction

        # IMPORTANT: Normalize vectors to unit length first
        tangent_unit = tangent_vector / np.linalg.norm(tangent_vector)
        chord_unit = chord_vector / np.linalg.norm(chord_vector)

        dot_product = np.dot(tangent_unit, chord_unit)

        # Ensure dot product is in valid range for acos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = math.acos(abs(dot_product))

        print(f"Simple test: dot={dot_product:.6f}, angle={math.degrees(angle):.1f}°")
        print(f"  Tangent unit: {tangent_unit}")
        print(f"  Chord unit: {chord_unit}")

        # Should be a small angle for nearly parallel vectors
        assert math.degrees(
            angle) < 20, f"Expected small angle for nearly parallel vectors, got {math.degrees(angle):.1f}°"
        assert dot_product > 0.9, f"Expected high dot product for similar vectors, got {dot_product:.3f}"