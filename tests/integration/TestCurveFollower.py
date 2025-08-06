# tests/integration/TestCurveFollower.py
import unittest
import math
import numpy as np
from tests.fixtures.freecad_fixtures import FreeCADTestBase

try:
    import FreeCAD
    from src.curve_follower import CurveFollower
    from src.curves import Curves

    FREECAD_AVAILABLE = True
except ImportError:
    FREECAD_AVAILABLE = False


@unittest.skipUnless(FREECAD_AVAILABLE, "FreeCAD not available")
class TestCurveFollowerIntegration(FreeCADTestBase):
    """Integration tests requiring full FreeCAD environment"""

    def test_document_management(self):
        """Test proper FreeCAD document setup and cleanup"""
        self.assertIsNotNone(self.doc)
        self.assertEqual(self.doc.Name, f"Test_{self.__class__.__name__}")
        self.assertGreaterEqual(len(self.doc.Objects), 0)

    def test_helical_curve_creation_in_freecad(self):
        """Test helical curve creation with FreeCAD objects"""
        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 8.0,
                'pitch': 4.0,
                'turns': 2.0,
                'points': 50
            }
        }

        curves = Curves(self.doc, curve_spec)
        points = curves.get_curve_points()

        # Validate curve properties
        self.assertEqual(len(points), 50)

        # Check helical properties
        for point in points:
            radius = math.sqrt(point[0] ** 2 + point[1] ** 2)
            self.assertAlmostEqual(radius, 8.0, delta=0.1)

        # Ensure Z coordinates increase (helical progression)
        z_coords = [p[2] for p in points]
        self.assertTrue(all(z_coords[i] <= z_coords[i + 1] for i in range(len(z_coords) - 1)))

    def test_wafer_positioning_system(self):
        """Test wafer positioning without coordinate system failures"""
        segment = self.create_test_segment("PositionTest")

        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 5.0,
                'pitch': 2.0,
                'turns': 1.0,
                'points': 20
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=1.0,
            curve_spec=curve_spec,
            min_height=0.5,
            max_chord=0.2
        )

        # Test curve following without GUI dependencies
        self.assertTrue(curve_follower.check_feasibility())

        # Generate wafer list
        wafers = curve_follower.create_wafer_list()
        self.assertGreater(len(wafers), 0)

        # Validate wafer parameters
        for i, (start_point, end_point, start_angle, end_angle, rotation, wafer_type) in enumerate(wafers):
            # Check mathematical validity
            self.assertIsInstance(start_point, np.ndarray)
            self.assertIsInstance(end_point, np.ndarray)
            self.assertTrue(-math.pi <= start_angle <= math.pi)
            self.assertTrue(-math.pi <= end_angle <= math.pi)
            self.assertTrue(-math.pi <= rotation <= math.pi)
            self.assertIn(wafer_type, ['CC', 'CE', 'EC', 'EE'])

            # Check height constraint
            height = np.linalg.norm(end_point - start_point)
            self.assertGreaterEqual(height, curve_follower.min_height)

    def test_coordinate_system_transformations(self):
        """Test LCS coordinate system operations don't fail"""
        segment = self.create_test_segment("CoordinateTest")

        # Simple linear curve for predictable testing
        curve_spec = {
            'type': 'linear',
            'parameters': {
                'length': 10.0,
                'points': 5,
                'direction': [0, 0, 1]  # Vertical line
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=2.0,
            curve_spec=curve_spec,
            min_height=1.0,
            max_chord=0.5
        )

        # Process wafers and check for coordinate system errors
        try:
            curve_follower.process_wafers(add_curve_vertices=False, debug=False)
            wafer_count = segment.get_wafer_count()
            self.assertGreater(wafer_count, 0)

            # Validate no extreme rotation values (the -179° problem)
            for wafer in segment.wafers:
                self.assertTrue(-math.pi <= wafer['rotation'] <= math.pi,
                                f"Rotation {wafer['rotation']} outside valid range")

        except Exception as e:
            self.fail(f"Coordinate system transformation failed: {e}")

    def test_curve_validation_system(self):
        """Test curve validation catches sampling issues"""
        segment = self.create_test_segment("ValidationTest")

        # Create curve with insufficient sampling
        curve_spec = {
            'type': 'helical',
            'parameters': {
                'radius': 10.0,
                'pitch': 1.0,
                'turns': 5.0,
                'points': 5  # Way too few points
            }
        }

        curve_follower = CurveFollower(
            doc=self.doc,
            segment=segment,
            cylinder_diameter=2.0,
            curve_spec=curve_spec,
            min_height=1.0,
            max_chord=0.2
        )

        validation_result = curve_follower.validate_and_adjust_curve_sampling()

        # Should detect insufficient sampling
        self.assertIn(validation_result['status'], ['insufficient_sampling', 'insufficient_points'])
        self.assertIn('recommended_points', validation_result)
        self.assertGreater(validation_result['recommended_points'], 5)


# Register with FreeCAD Test Framework
def Test():
    """Entry point for FreeCAD Test Workbench"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCurveFollowerIntegration))
    return suite


if __name__ == '__main__':
    unittest.main()