"""
loft_segment.py - Loft-based segment management

This module manages segments created using the loft-based wafer approach.
A segment is a collection of wafers that can be positioned, transformed,
and concatenated with other segments.
"""

import FreeCAD as App
import Part
from curve_follower_loft import CurveFollowerLoft


class LoftSegment:
    """
    Manages a segment created using loft-based wafer generation

    A segment encapsulates:
    - Wafer generation from a curved loft
    - Segment positioning and placement
    - Bounding box calculation
    - Visualization control
    """

    def __init__(self, doc, name, curve_spec, wafer_settings, segment_settings, base_placement=None):
        """
        Initialize loft segment

        Args:
            doc: FreeCAD document
            name: Segment name
            curve_spec: Dictionary with curve type and parameters
            wafer_settings: Dictionary with wafer configuration
            segment_settings: Dictionary with segment settings
            base_placement: App.Placement for segment positioning (default: origin)
        """
        self.doc = doc
        self.name = name
        self.curve_spec = curve_spec
        self.wafer_settings = wafer_settings
        self.segment_settings = segment_settings
        self.base_placement = base_placement if base_placement else App.Placement()

        # Internal state
        self.follower = None
        self.wafer_list = []

        # Bounding box (calculated after wafer generation)
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        print(f"Created LoftSegment: {name}")

    def generate_wafers(self):
        """
        Generate wafers for this segment using loft-based approach

        Returns:
            List of Wafer objects
        """
        print(f"\nGenerating wafers for segment '{self.name}'...")

        try:
            # Create follower with wafer settings
            self.follower = CurveFollowerLoft(wafer_settings=self.wafer_settings)

            # Generate loft and wafers
            self.follower.generate_loft_wafers(self.curve_spec, self.wafer_settings)

            # Get wafer list
            self.wafer_list = self.follower.get_wafer_list()

            # Calculate bounding box
            self._calculate_bounds()

            # Apply base placement transformation if needed
            if not self._is_identity_placement(self.base_placement):
                self._apply_placement_to_wafers()

            print(f"✓ Generated {len(self.wafer_list)} wafers for segment '{self.name}'")

            return self.wafer_list

        except Exception as e:
            print(f"✗ Error generating wafers for segment '{self.name}': {e}")
            raise

    def _is_identity_placement(self, placement):
        """Check if placement is identity (no transformation)"""
        identity_pos = App.Vector(0, 0, 0)
        identity_rot = App.Rotation(0, 0, 0, 1)

        pos_close = (placement.Base - identity_pos).Length < 1e-6
        rot_close = placement.Rotation.isSame(identity_rot, 1e-6)

        return pos_close and rot_close

    def _apply_placement_to_wafers(self):
        """
        Apply base placement transformation to all wafers

        This transforms the entire segment (all wafers and their LCS)
        by the base placement.
        """
        print(f"  Applying placement transformation to segment '{self.name}'...")

        for wafer in self.wafer_list:
            if wafer.wafer is not None:
                # Transform the wafer solid
                wafer.wafer.Placement = self.base_placement.multiply(wafer.wafer.Placement)

                # Transform LCS
                if wafer.lcs1:
                    wafer.lcs1 = self.base_placement.multiply(wafer.lcs1)
                if wafer.lcs2:
                    wafer.lcs2 = self.base_placement.multiply(wafer.lcs2)

                # Transform geometry centers
                if 'center1' in wafer.geometry:
                    wafer.geometry['center1'] = self.base_placement.multVec(wafer.geometry['center1'])
                if 'center2' in wafer.geometry:
                    wafer.geometry['center2'] = self.base_placement.multVec(wafer.geometry['center2'])

        # Recalculate bounds after transformation
        self._calculate_bounds()

    def _calculate_bounds(self):
        """Calculate bounding box for all wafers in segment"""
        if not self.wafer_list:
            self.x_min = self.x_max = None
            self.y_min = self.y_max = None
            self.z_min = self.z_max = None
            return

        x_coords = []
        y_coords = []
        z_coords = []

        for wafer in self.wafer_list:
            if wafer.wafer is not None:
                bbox = wafer.wafer.BoundBox
                x_coords.extend([bbox.XMin, bbox.XMax])
                y_coords.extend([bbox.YMin, bbox.YMax])
                z_coords.extend([bbox.ZMin, bbox.ZMax])

        if x_coords:
            self.x_min = min(x_coords)
            self.x_max = max(x_coords)
            self.y_min = min(y_coords)
            self.y_max = max(y_coords)
            self.z_min = min(z_coords)
            self.z_max = max(z_coords)
        else:
            self.x_min = self.x_max = None
            self.y_min = self.y_max = None
            self.z_min = self.z_max = None

    def get_bounds(self):
        """
        Get bounding box as a dictionary

        Returns:
            Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        """
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'z_min': self.z_min,
            'z_max': self.z_max
        }

    def get_end_placement(self):
        """
        Get the placement at the end of this segment

        This is used for concatenating segments - the next segment
        starts where this one ends.

        Returns:
            App.Placement at the end of the segment
        """
        if not self.wafer_list or len(self.wafer_list) == 0:
            return self.base_placement

        # Get the last wafer's second LCS (the exit face)
        last_wafer = self.wafer_list[-1]

        if last_wafer.lcs2:
            return last_wafer.lcs2
        else:
            # Fallback to base placement if LCS not available
            return self.base_placement

    def visualize(self, show_lcs=True, show_cutting_planes=True, lcs_size=None):
        """
        Visualize this segment in FreeCAD

        Args:
            show_lcs: Show local coordinate systems
            show_cutting_planes: Show cutting plane discs
            lcs_size: Size of LCS display
        """
        if not self.follower:
            print(f"Cannot visualize segment '{self.name}' - no wafers generated")
            return

        if not self.follower.generator:
            print(f"Cannot visualize segment '{self.name}' - no generator available")
            return

        print(f"Visualizing segment '{self.name}'...")

        # Use the generator's visualization (CurveFollowerLoft has a 'generator' attribute)
        self.follower.generator.visualize_in_freecad(
            self.doc,
            show_lcs=show_lcs,
            show_cutting_planes=show_cutting_planes,
            lcs_size=lcs_size
        )

    def get_wafer_count(self):
        """Get number of wafers in this segment"""
        return len(self.wafer_list)

    def get_total_volume(self):
        """Get total volume of all wafers in segment"""
        return sum(w.volume for w in self.wafer_list if w.wafer is not None)

    def get_summary(self):
        """
        Get summary information about this segment

        Returns:
            Dictionary with segment statistics
        """
        return {
            'name': self.name,
            'wafer_count': self.get_wafer_count(),
            'total_volume': self.get_total_volume(),
            'bounds': self.get_bounds(),
            'base_placement': self.base_placement
        }

    def __repr__(self):
        return f"LoftSegment(name='{self.name}', wafers={len(self.wafer_list)})"