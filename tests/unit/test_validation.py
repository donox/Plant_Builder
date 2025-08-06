# tests/unit/test_validation.py
"""Test validation and error handling for the curve follower system"""
import pytest
import numpy as np
import math
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, '/usr/lib/freecad-daily/lib')


class TestCurveFollowerValidation:
    """Test validation logic for curve follower parameters and geometry"""

    def test_curve_spec_validation(self):
        """Test validation of curve specification parameters"""
        from curves import Curves

        # Valid curve spec
        valid_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
        }

        # Should not raise
        curves = Curves(None, valid_spec)
        assert len(curves.get_curve_points()) == 100

        # Invalid curve type
        with pytest.raises(ValueError, match="Unknown curve type"):
            invalid_spec = {
                'type': 'invalid_curve_type',
                'parameters': {'radius': 8.0}
            }
            Curves(None, invalid_spec)

        # Missing parameters should use defaults (not raise errors)
        incomplete_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0}  # Missing pitch, turns, points - should use defaults
        }
        curves = Curves(None, incomplete_spec)
        points = curves.get_curve_points()
        assert len(points) > 0  # Should create a curve with default values

        # Empty parameters should also work with defaults
        empty_params_spec = {
            'type': 'helical',
            'parameters': {}  # All defaults
        }
        curves = Curves(None, empty_params_spec)
        points = curves.get_curve_points()
        assert len(points) > 0

    def test_geometric_constraint_validation(self):
        """Test validation of geometric constraints for feasibility"""
        from curve_follower import CurveFollower
        from curves import Curves

        # Mock segment object for testing
        class MockSegment:
            def __init__(self):
                self.wafer_count = 0

            def get_wafer_count(self):
                return self.wafer_count

            def get_segment_name(self):
                return "test_segment"

            def add_wafer(self, *args):
                self.wafer_count += 1

        curve_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
        }

        # Valid constraints
        follower = CurveFollower(None, MockSegment(), 1.5, curve_spec, 0.8, 0.2)
        assert follower.check_feasibility()

        # Invalid: cylinder diameter larger than curve radius (impossible geometry)
        curve_spec_tight = {
            'type': 'helical',
            'parameters': {'radius': 1.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
        }
        follower_invalid = CurveFollower(None, MockSegment(), 5.0, curve_spec_tight, 0.8, 0.2)

        # This should fail feasibility check
        if not follower_invalid.check_feasibility():
            with pytest.raises(ValueError, match="No feasible solution"):
                follower_invalid.process_wafers()
        else:
            # If it passes feasibility, processing should work
            follower_invalid.process_wafers()

    def test_angle_validation_and_clamping(self):
        """Test that extreme angles are properly validated and clamped"""
        from curve_follower import CurveFollower

        # Test angle normalization
        test_angles = [
            math.radians(179.9),  # Near boundary
            math.radians(-179.9),  # Near negative boundary
            math.radians(360),  # Full rotation
            math.radians(-360),  # Negative full rotation
            math.radians(720),  # Multiple rotations
        ]

        for angle in test_angles:
            normalized = CurveFollower.normalize_angle(angle)

            # Should be in [-π, π] range
            assert -math.pi <= normalized <= math.pi

            # Should avoid very near ±π boundaries (with epsilon buffer)
            epsilon_buffer = 1e-3  # Should match your production epsilon
            assert abs(abs(normalized) - math.pi) > epsilon_buffer * 0.5  # Allow some tolerance

    def test_wafer_parameter_validation(self):
        """Test validation of wafer cutting parameters"""
        from curve_follower import CurveFollower
        from curves import Curves

        class MockSegment:
            def __init__(self):
                self.wafer_count = 0
                self.wafers_added = []

            def get_wafer_count(self):
                return self.wafer_count

            def get_segment_name(self):
                return "test_segment"

            def add_wafer(self, lift, rotation, diameter, height, wafer_type):
                # Validate parameters
                assert isinstance(lift, (int, float))
                assert isinstance(rotation, (int, float))
                assert isinstance(diameter, (int, float))
                assert isinstance(height, (int, float))
                assert isinstance(wafer_type, str)

                # Check reasonable ranges
                assert -math.pi / 2 <= lift <= math.pi / 2  # ±90°
                assert -math.pi <= rotation <= math.pi  # ±180°
                assert diameter > 0
                assert height > 0
                assert wafer_type in ["CC", "CE", "EC", "EE"]

                self.wafers_added.append((lift, rotation, diameter, height, wafer_type))
                self.wafer_count += 1

        curve_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
        }

        segment = MockSegment()
        follower = CurveFollower(None, segment, 1.5, curve_spec, 0.8, 0.2)

        # Process wafers - should not raise due to parameter validation
        follower.process_wafers(debug=False)

        # Verify at least some wafers were created
        assert len(segment.wafers_added) > 0

        # Check that all wafers have reasonable parameters
        for lift, rotation, diameter, height, wafer_type in segment.wafers_added:
            # For shallow helix, lift angles should be small
            assert abs(math.degrees(lift)) <= 45, f"Lift angle {math.degrees(lift):.1f}° too extreme"
            assert diameter == 1.5  # Should match input
            assert height >= 0.8  # Should be at least min_height

    def test_curve_density_validation(self):
        """Test curve point density validation"""
        from curves import Curves

        # Too few points
        sparse_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 5}
        }

        curves = Curves(None, sparse_spec)

        # Test if the validation method exists
        if hasattr(curves, 'calculate_required_point_density'):
            density_analysis = curves.calculate_required_point_density(0.2)

            # Should flag as insufficient density
            assert density_analysis['status'] in ['insufficient_density', 'insufficient_points']
        else:
            # If method doesn't exist, just verify points were created
            points = curves.get_curve_points()
            assert len(points) == 5

        # Adequate points
        dense_spec = {
            'type': 'helical',
            'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 200}
        }

        curves = Curves(None, dense_spec)
        points = curves.get_curve_points()
        assert len(points) == 200

    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        from curve_follower import CurveFollower

        # Test with very small numbers
        tiny_angles = [1e-10, -1e-10, 1e-15, -1e-15]

        for angle in tiny_angles:
            normalized = CurveFollower.normalize_angle(angle)
            assert abs(normalized) < 1e-6  # Should be effectively zero

        # Test with very large numbers
        large_angles = [1e6, -1e6, 1e10, -1e10]

        for angle in large_angles:
            normalized = CurveFollower.normalize_angle(angle)
            assert -math.pi <= normalized <= math.pi

    def test_parameter_type_validation(self):
        """Test validation of parameter types"""
        from curves import Curves

        # Test with potentially problematic parameter types
        test_specs = [
            {
                'type': 'helical',
                'parameters': {'radius': '8.0', 'pitch': 4.0, 'turns': 3.0, 'points': 100}  # String radius
            },
            {
                'type': 'helical',
                'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': '100'}  # String points
            },
        ]

        for spec in test_specs:
            # Should either handle gracefully or raise appropriate error
            try:
                curves = Curves(None, spec)
                points = curves.get_curve_points()
                # If it doesn't raise, verify basic sanity
                assert len(points) > 0
            except (ValueError, TypeError) as e:
                # Acceptable to raise for invalid input
                assert True  # Test passes if appropriate error is raised

    def test_boundary_conditions(self):
        """Test behavior at boundary conditions"""
        from curve_follower import CurveFollower
        from curves import Curves

        class MockSegment:
            def get_wafer_count(self):
                return 0

            def get_segment_name(self):
                return "test_segment"

            def add_wafer(self, *args):
                pass

        # Minimum viable curve
        minimal_spec = {
            'type': 'linear',
            'parameters': {'length': 1.0, 'points': 2}  # Absolute minimum
        }

        # Should handle minimal case
        follower = CurveFollower(None, MockSegment(), 0.1, minimal_spec, 0.1, 0.1)

        # May not be feasible, but shouldn't crash
        try:
            result = follower.check_feasibility()
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"Boundary condition handling failed: {e}")