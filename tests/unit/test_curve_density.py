# tests/unit/test_curve_density.py
"""Test automatic curve point density calculation based on curvature"""
import pytest
import numpy as np
import math
from  src.curves import Curves


class TestCurveDensity:
    """Test automatic point density calculation"""

    def test_curvature_based_point_calculation(self):
        """Test point density analysis works for different curve complexities"""

        # Create different curves and test their density requirements
        test_cases = [
            {
                'type': 'helical',
                'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 30}
            },
            {
                'type': 'helical',
                'parameters': {'radius': 8.0, 'pitch': 4.0, 'turns': 3.0, 'points': 100}
            }
        ]

        for i, curve_spec in enumerate(test_cases):
            curves = Curves(None, curve_spec)  # No doc needed for testing
            analysis = curves.calculate_required_point_density(0.2)

            print(f"Case {i + 1}: {curve_spec['parameters']['points']} points")
            print(f"  Status: {analysis['status']}")
            print(f"  Message: {analysis['message']}")

            # The low-point case should need more points
            if curve_spec['parameters']['points'] == 30:
                assert analysis['status'] == 'insufficient_density'
            else:
                assert analysis['status'] == 'adequate_density'

    def test_wafer_count_estimation(self):
        """Test estimating wafer count from curve properties"""

        def estimate_wafer_count(curve_length, min_height, max_chord):
            """Estimate how many wafers will be needed"""

            # Conservative estimate: each wafer spans approximately min_height
            # with some safety margin for geometric constraints
            avg_wafer_length = min_height * 1.2  # 20% safety margin
            estimated_count = max(3, int(curve_length / avg_wafer_length))

            return estimated_count

        # Test with your current parameters
        curve_length = 151.05  # From your log
        min_height = 0.8
        max_chord = 0.2

        estimated_wafers = estimate_wafer_count(curve_length, min_height, max_chord)

        print(f"Curve length: {curve_length:.2f}")
        print(f"Min height: {min_height}, Max chord: {max_chord}")
        print(f"Estimated wafers needed: {estimated_wafers}")

        # Your log shows 50 wafers created, so estimate should be reasonable
        assert 20 <= estimated_wafers <= 200, f"Wafer estimate {estimated_wafers} unreasonable"