"""
loft_segment.py - Loft-based segment management

This module manages segments created using the loft-based wafer approach.
A segment is a collection of wafers that can be positioned, transformed,
and concatenated with other segments.
"""

import FreeCAD as App
import Part
from curve_follower_loft import CurveFollowerLoft
from core.logging_setup import get_logger

logger = get_logger(__name__)


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

        logger.debug(f"Created LoftSegment: {name}")

    def generate_wafers(self):
        """
        Generate wafers for this segment using loft-based approach

        Returns:
            List of Wafer objects
        """
        logger.info(f"Generating wafers for segment '{self.name}'")

        try:
            # Create follower with wafer settings
            self.follower = CurveFollowerLoft(wafer_settings=self.wafer_settings)
            logger.debug(f"Created CurveFollowerLoft for '{self.name}'")

            # Generate loft and wafers
            self.follower.generate_loft_wafers(self.curve_spec, self.wafer_settings)

            # Get wafer list
            self.wafer_list = self.follower.get_wafer_list()

            # Calculate bounding box
            self._calculate_bounds()

            # Apply base placement transformation if needed
            if not self._is_identity_placement(self.base_placement):
                logger.debug(f"Applying placement transformation to segment '{self.name}'")
                self._apply_placement_to_wafers()

            logger.info(f"Generated {len(self.wafer_list)} wafers for segment '{self.name}'")

            return self.wafer_list

        except Exception as e:
            logger.error(f"Failed to generate wafers for segment '{self.name}': {e}", exc_info=True)
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
        logger.debug(f"Placement transformation applied to {len(self.wafer_list)} wafers")

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
            logger.coord(f"Bounds: X({self.x_min:.2f},{self.x_max:.2f}) "
                        f"Y({self.y_min:.2f},{self.y_max:.2f}) "
                        f"Z({self.z_min:.2f},{self.z_max:.2f})")
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

    def visualize(self, doc):
        """Create FreeCAD visualization of the segment"""
        logger.info(f"Visualizing segment '{self.name}'")

        # Create main group for this segment
        segment_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_Group")
        logger.debug(f"Created segment group: {self.name}_Group")

        # Create subgroups
        wafer_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_Wafers")
        reference_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_Reference")

        segment_group.addObject(wafer_group)
        segment_group.addObject(reference_group)
        logger.debug(f"Created subgroups: Wafers and Reference")

        # Add loft to reference group (if it exists)
        if hasattr(self.follower, 'generator') and hasattr(self.follower.generator, 'loft'):
            loft = self.follower.generator.loft
            if loft:
                loft_obj = doc.addObject("Part::Feature", f"Loft_{self.name}")
                loft_obj.Shape = loft
                loft_obj.ViewObject.Transparency = 70
                loft_obj.ViewObject.ShapeColor = (0.8, 0.9, 1.0)
                reference_group.addObject(loft_obj)
                logger.debug("Added loft to reference group")

        # Add spine to reference group (hidden by default)
        if hasattr(self.follower, 'generator') and hasattr(self.follower.generator, 'spine_curve'):
            spine = self.follower.generator.spine_curve
            if spine:
                spine_obj = doc.addObject("Part::Feature", f"Loft_spine_{self.name}")
                spine_obj.Shape = spine
                spine_obj.ViewObject.LineColor = (1.0, 0.0, 0.0)
                spine_obj.ViewObject.LineWidth = 2.0
                spine_obj.ViewObject.Visibility = False
                reference_group.addObject(spine_obj)
                logger.debug("Added spine to reference group (hidden)")

        # Add reference curve to reference group (hidden by default)
        if hasattr(self.follower, 'generator') and hasattr(self.follower.generator, 'sample_points'):
            points = self.follower.generator.sample_points
            if points and len(points) > 1:
                # Convert to App.Vector if needed
                vec_points = []
                for p in points:
                    if isinstance(p, App.Vector):
                        vec_points.append(p)
                    elif isinstance(p, (tuple, list)) and len(p) >= 3:
                        vec_points.append(App.Vector(p[0], p[1], p[2]))
                    else:
                        continue

                if len(vec_points) > 1:
                    edges = [Part.LineSegment(vec_points[i], vec_points[i + 1]).toShape()
                             for i in range(len(vec_points) - 1)]
                    wire = Part.Wire(edges)
                    curve_obj = doc.addObject("Part::Feature", f"Loft_reference_{self.name}")
                    curve_obj.Shape = wire
                    curve_obj.ViewObject.LineColor = (0.0, 1.0, 0.0)
                    curve_obj.ViewObject.LineWidth = 1.0
                    curve_obj.ViewObject.Visibility = False
                    reference_group.addObject(curve_obj)
                    logger.debug(f"Added reference curve ({len(vec_points)} points, hidden)")

        # Create LCS group if needed
        lcs_group = None
        if self.segment_settings.get('show_lcs', False):
            lcs_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_LCS")
            reference_group.addObject(lcs_group)
            logger.debug("Created LCS group")

        # Add wafers to wafer group
        wafer_count = 0
        for i, wafer_data in enumerate(self.wafer_list):
            if wafer_data.wafer is None:
                continue

            wafer_obj = doc.addObject("Part::Feature", f"Wafer_{self.name}_{i}")
            wafer_obj.Shape = wafer_data.wafer

            # Alternate colors
            if i % 2 == 0:
                wafer_obj.ViewObject.ShapeColor = (0.5, 1.0, 0.5)
            else:
                wafer_obj.ViewObject.ShapeColor = (0.7, 0.7, 1.0)

            wafer_obj.ViewObject.Transparency = 20
            wafer_group.addObject(wafer_obj)
            wafer_count += 1

            # Add LCS if requested (in LCS group, hidden by default)
            if lcs_group is not None:
                if hasattr(wafer_data, 'lcs1') and wafer_data.lcs1:
                    lcs_obj = doc.addObject("App::Placement", f"LCS_{self.name}_{i}_1")
                    lcs_obj.Placement = wafer_data.lcs1
                    lcs_obj.ViewObject.Visibility = False
                    lcs_group.addObject(lcs_obj)

                if hasattr(wafer_data, 'lcs2') and wafer_data.lcs2:
                    lcs_obj = doc.addObject("App::Placement", f"LCS_{self.name}_{i}_2")
                    lcs_obj.Placement = wafer_data.lcs2
                    lcs_obj.ViewObject.Visibility = False
                    lcs_group.addObject(lcs_obj)

        logger.debug(f"Added {wafer_count} wafers to visualization")

        # Add cutting planes to reference group (hidden by default)
        if hasattr(self.follower, 'cutting_planes'):
            cutting_planes = self.follower.cutting_planes
            if cutting_planes:
                plane_count = 0
                for i, plane in enumerate(cutting_planes):
                    if plane is not None:
                        plane_obj = doc.addObject("Part::Feature", f"CuttingPlane_{self.name}_{i}")
                        plane_obj.Shape = plane
                        plane_obj.ViewObject.ShapeColor = (1.0, 0.8, 0.0)
                        plane_obj.ViewObject.Transparency = 80
                        plane_obj.ViewObject.Visibility = False
                        reference_group.addObject(plane_obj)
                        plane_count += 1
                logger.debug(f"Added {plane_count} cutting planes (hidden)")

        doc.recompute()
        logger.info(f"Visualization complete for '{self.name}'")

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