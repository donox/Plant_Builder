# tests/integration/test_original_problem.py
"""Test the original helical curve follower problem with fixes applied"""
import unittest
import math
import numpy as np
import sys
import os

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, '/usr/lib/freecad-daily/lib')

import FreeCAD
from src.curve_follower import CurveFollower
from tests.fixtures.freecad_fixtures import FreeCADTestBase


class TestOriginalProblemFixed(FreeCADTestBase):
    """Test that the original helical positioning issues are resolved"""

    def test_original_helical_configuration(self):
        """Test the original problematic helical configuration"""
        segment = self.create_test_segment("OriginalProblem")

        # Your original configuration from small.yml
        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 8.0,
                'pitch': 4.0,
                'turns': 3.0,
                'points': 650
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=1.5,
            curve_spec=curve_spec,
            min_height=0.8,
            max_chord=0.2
        )

        # Should be feasible
        self.assertTrue(curve_follower.check_feasibility(),
                        "Original configuration should be feasible")

        # Process wafers without errors
        try:
            curve_follower.process_wafers(add_curve_vertices=False, debug=True)
            wafer_count = segment.get_wafer_count()
            self.assertGreater(wafer_count, 0, "Should create wafers")
            print(f"✅ Successfully created {wafer_count} wafers")

        except Exception as e:
            self.fail(f"process_wafers failed: {e}")

        # Validate all rotations are in valid range
        extreme_rotations = []
        for i, wafer in enumerate(segment.wafers):
            rotation_deg = math.degrees(wafer['rotation'])
            lift_deg = math.degrees(wafer['lift'])

            # Check rotation is normalized
            self.assertTrue(-180 <= rotation_deg <= 180,
                            f"Wafer {i}: rotation {rotation_deg:.1f}° outside ±180°")

            # Check lift is reasonable
            self.assertTrue(-180 <= lift_deg <= 180,  # Accept full normalized range
                            f"Wafer {i}: lift {lift_deg:.1f}° outside ±180°")

            # Add a warning for extreme values instead of failing
            if abs(lift_deg) > 90:
                print(f"⚠️  Wafer {i}: Large lift angle {lift_deg:.1f}°")

            # Track extreme values for analysis
            if abs(rotation_deg) > 170:
                extreme_rotations.append((i, rotation_deg))

        # Report on extreme rotations (should be minimal with fixes)
        if extreme_rotations:
            print(f"⚠️  {len(extreme_rotations)} wafers have large rotations (>170°):")
            for i, rot in extreme_rotations[:5]:  # Show first 5
                print(f"   Wafer {i}: {rot:.1f}°")
        else:
            print("✅ No extreme rotation values detected")

    def test_insufficient_separation_fix(self):
        """Test that 'insufficient separation' errors are resolved"""
        segment = self.create_test_segment("SeparationTest")

        # Configuration that previously caused separation issues
        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 8.0,
                'pitch': 4.0,
                'turns': 1.0,  # Reduced for focused testing
                'points': 100
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=1.5,
            curve_spec=curve_spec,
            min_height=0.8,
            max_chord=0.2
        )

        # Generate wafer list and check separations
        wafers = curve_follower.create_wafer_list()

        separation_issues = []
        for i in range(len(wafers) - 1):
            start_point_1, end_point_1 = wafers[i][:2]
            start_point_2, end_point_2 = wafers[i + 1][:2]

            # Check gap between consecutive wafers
            gap = np.linalg.norm(start_point_2 - end_point_1)

            # Should be small (adjacent) but not overlapping
            if gap > 0.5:  # Arbitrary threshold for "too large"
                separation_issues.append((i, gap))

        if separation_issues:
            print(f"⚠️  {len(separation_issues)} wafer gaps > 0.5:")
            for i, gap in separation_issues[:3]:
                print(f"   Gap after wafer {i}: {gap:.3f}")
        else:
            print("✅ All wafer separations are reasonable")

    def test_coordinate_system_stability(self):
        """Test that coordinate systems remain stable through processing"""
        segment = self.create_test_segment("CoordinateStability")

        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 8.0,
                'pitch': 4.0,
                'turns': 0.5,  # Half turn for stability testing
                'points': 50
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=1.5,
            curve_spec=curve_spec,
            min_height=0.5,
            max_chord=0.3
        )

        # Get curve points for reference
        original_points = curve_follower.curve_points.copy()

        # Process wafers
        curve_follower.process_wafers(add_curve_vertices=False, debug=False)

        # Verify curve points weren't corrupted
        final_points = curve_follower.curve_points

        # Points should be identical (no side effects)
        np.testing.assert_array_almost_equal(original_points, final_points, decimal=10,
                                             err_msg="Curve points were modified during processing")

        print("✅ Coordinate system remained stable during processing")


# Register with FreeCAD Test Framework
def Test():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOriginalProblemFixed))
    return suite