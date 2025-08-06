# tests/fixtures/freecad_fixtures.py
import FreeCAD
import unittest
from typing import Any, Dict, Optional


class FreeCADTestBase(unittest.TestCase):
    """Base class for all FreeCAD integration tests"""

    def setUp(self) -> None:
        """Create clean FreeCAD document for testing"""
        self.doc_name = f"Test_{self.__class__.__name__}"

        # Close any existing test documents
        if FreeCAD.ActiveDocument and FreeCAD.ActiveDocument.Name == self.doc_name:
            FreeCAD.closeDocument(self.doc_name)

        # Create fresh document
        self.doc = FreeCAD.newDocument(self.doc_name)
        FreeCAD.setActiveDocument(self.doc_name)

        # Store initial object count for cleanup validation
        self.initial_objects = len(self.doc.Objects)

    def tearDown(self) -> None:
        """Clean up FreeCAD document and validate cleanup"""
        if hasattr(self, 'doc') and self.doc:
            # Validate test didn't leak objects
            final_objects = len(self.doc.Objects)
            if final_objects > self.initial_objects:
                print(f"WARNING: Test leaked {final_objects - self.initial_objects} objects")

            # Force close document
            FreeCAD.closeDocument(self.doc.Name)

    def create_test_segment(self, name: str = "TestSegment") -> Any:
        """Create a test FlexSegment for wafer testing"""

        # Mock FlexSegment implementation for testing
        class TestFlexSegment:
            def __init__(self, doc):
                self.doc = doc
                self.wafers = []
                self.name = name
                self.lcs_objects = []

            def add_wafer(self, lift, rotation, diameter, height, wafer_type):
                """Mock wafer creation that tracks parameters"""
                wafer_data = {
                    'lift': lift,
                    'rotation': rotation,
                    'diameter': diameter,
                    'height': height,
                    'type': wafer_type,
                    'id': len(self.wafers)
                }
                self.wafers.append(wafer_data)
                return wafer_data

            def get_wafer_count(self):
                return len(self.wafers)

            def get_segment_name(self):
                return self.name

            def register_curve_vertices_group(self, group_name):
                """Mock registration of curve vertices"""
                self.curve_vertices_group = group_name

        return TestFlexSegment(self.doc)

    def assert_no_geometry_errors(self):
        """Validate FreeCAD document has no geometry errors"""
        for obj in self.doc.Objects:
            if hasattr(obj, 'Shape') and obj.Shape:
                self.assertTrue(obj.Shape.isValid(),
                                f"Object {obj.Name} has invalid geometry")

    def assert_coordinate_system_valid(self, lcs_objects):
        """Validate LCS objects have proper coordinate systems"""
        for lcs in lcs_objects:
            self.assertTrue(hasattr(lcs, 'Placement'))
            self.assertIsNotNone(lcs.Placement)