"""Tests for reconstruction module."""

import unittest
import tempfile
from pathlib import Path


# Mock FreeCAD for testing
class MockVector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class MockPlacement:
    def __init__(self):
        self.Base = MockVector()
        self.Rotation = None


class MockLCS:
    def __init__(self):
        self.Placement = MockPlacement()


class MockDocument:
    def __init__(self):
        self.Objects = []

    def addObject(self, type_name, name):
        obj = MockLCS()
        self.Objects.append(obj)
        return obj

    def recompute(self):
        pass


try:
    import FreeCAD
except ImportError:
    FreeCAD = type('FreeCAD', (), {
        'Document': MockDocument,
        'Vector': MockVector,
        'Placement': MockPlacement
    })()

from reconstruction.cut_list_parser import CutListParser, WaferSpec


class TestCutListParser(unittest.TestCase):
    """Test cut list parsing."""

    def setUp(self):
        """Create test cut list file."""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.test_file.write("""Cut List for: Test Project

Project Bounds
	X-min: -5.00	X_max: 5.00
	Y-min: -5.00	Y_max: 5.00
	Z-min: 0.00	Z_max: 10.00

Cutting order:
Segment 'test_segment' cuts

Wafer	Type	Lift(deg)	Rotate(deg)	Outside		SawPos(deg)
1	CE	5.00		1.00		2 0/16		0.00
2	EE	5.00		1.00		2 0/16		179.00
3	EC	---		---		2 0/16		178.00

	Total Cylinder Length: 6.00
""")
        self.test_file.close()

    def tearDown(self):
        """Clean up test file."""
        Path(self.test_file.name).unlink()

    def test_parse_file(self):
        """Test parsing complete file."""
        parser = CutListParser(units_per_inch=1.0)  # Use inches
        result = parser.parse_file(self.test_file.name)

        self.assertEqual(result['project_name'], 'Test Project')
        self.assertEqual(len(result['segments']), 1)

    def test_parse_bounds(self):
        """Test parsing project bounds."""
        parser = CutListParser()
        result = parser.parse_file(self.test_file.name)

        bounds = result['bounds']
        self.assertEqual(bounds['x_min'], -5.00)
        self.assertEqual(bounds['x_max'], 5.00)

    def test_parse_segment(self):
        """Test parsing segment data."""
        parser = CutListParser(units_per_inch=1.0)
        result = parser.parse_file(self.test_file.name)

        segment = result['segments'][0]
        self.assertEqual(segment.segment_name, 'test_segment')
        self.assertEqual(len(segment.wafers), 3)
        self.assertEqual(segment.total_cylinder_length, 6.0)

    def test_parse_wafer_with_values(self):
        """Test parsing wafer with all values."""
        parser = CutListParser(units_per_inch=1.0)
        result = parser.parse_file(self.test_file.name)

        wafer = result['segments'][0].wafers[0]
        self.assertEqual(wafer.wafer_num, 1)
        self.assertEqual(wafer.wafer_type, 'CE')
        self.assertEqual(wafer.lift_angle, 5.0)
        self.assertEqual(wafer.rotation_angle, 1.0)
        self.assertEqual(wafer.outside_height, 2.0)  # 2 0/16 inches

    def test_parse_last_wafer(self):
        """Test parsing last wafer with --- values."""
        parser = CutListParser(units_per_inch=1.0)
        result = parser.parse_file(self.test_file.name)

        wafer = result['segments'][0].wafers[2]
        self.assertEqual(wafer.wafer_type, 'EC')
        self.assertIsNone(wafer.lift_angle)
        self.assertIsNone(wafer.rotation_angle)

    def test_fractional_inches(self):
        """Test parsing fractional inch measurements."""
        parser = CutListParser(units_per_inch=1.0)

        # Test various formats
        self.assertEqual(parser._parse_fractional_inches("2 0/16"), 2.0)
        self.assertEqual(parser._parse_fractional_inches("1 8/16"), 1.5)
        self.assertEqual(parser._parse_fractional_inches("3 4/16"), 3.25)


if __name__ == '__main__':
    unittest.main()