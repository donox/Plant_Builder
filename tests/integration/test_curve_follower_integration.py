# tests/integration/test_curve_follower_integration.py
"""Integration tests for CurveFollower with real FreeCAD"""
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


class TestCurveFollowerIntegration(FreeCADTestBase):
    """Integration tests for CurveFollower"""

    def test_simple_helical_wafer_generation(self):
        """Test basic helical wafer generation without errors"""
        segment = self.create_test_segment("HelicalTest")

        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 5.0,
                'pitch': 2.0,
                'turns': 0.5,  # Half turn to keep it simple
                'points': 10  # Small number for debugging
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=1.0,
            curve_spec=curve_spec,
            min_height=0.3,
            max_chord=0.2
        )

        # Test feasibility check
        self.assertTrue(curve_follower.check_feasibility())

        # Generate wafer list
        wafers = curve_follower.create_wafer_list()
        self.assertGreater(len(wafers), 0, "Should generate at least one wafer")

        # Validate no extreme rotation values
        for i, (start_point, end_point, start_angle, end_angle, rotation, wafer_type) in enumerate(wafers):
            # Check for the -179° problem
            self.assertTrue(-math.pi <= rotation <= math.pi,
                            f"Wafer {i}: rotation {math.degrees(rotation):.1f}° outside valid range")

            # Check angles are reasonable
            self.assertTrue(-math.pi / 2 <= start_angle <= math.pi / 2,
                            f"Wafer {i}: start_angle extreme")
            self.assertTrue(-math.pi / 2 <= end_angle <= math.pi / 2,
                            f"Wafer {i}: end_angle extreme")


# Register with FreeCAD Test Framework
def Test():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCurveFollowerIntegration))
    return suite