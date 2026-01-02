"""
loft_segment.py - Loft-based segment management

This module manages segments created using the loft-based wafer approach.
A segment is a collection of wafers that can be positioned, transformed,
and concatenated with other segments.
"""

import FreeCAD as App
from numpy.ma.core import max_filler
from scipy.optimize import curve_fit
from sqlalchemy.sql.operators import from_

# import Part
from curve_follower_loft import CurveFollowerLoft
from curve_io import get_wire_from_label, sample_points_on_wire, transform_world_to_local
from curve_follower_loft import CurveFollowerLoft
from wafer_loft import LoftWaferGenerator
from curve_io import get_wire_from_label, sample_points_on_wire, transform_world_to_local

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

    def __init__(self, doc, name, curve_spec, wafer_settings, segment_settings, base_placement=None,
                 connection_spec=None):
        """
        Initialize loft segment

        Args:
            doc: FreeCAD document
            name: Segment name
            curve_spec: Dictionary with curve type and parameters
            wafer_settings: Dictionary with wafer configuration
            segment_settings: Dictionary with segment settings
            base_placement: App.Placement for segment positioning (default: origin)
            connection_spec: Dictionary with connection parameters (rotation_angle, etc.)
        """
        self.doc = doc
        self.name = name
        self.curve_spec = curve_spec
        self.wafer_settings = wafer_settings
        self.segment_settings = segment_settings
        self.base_placement = base_placement if base_placement else App.Placement()
        self.connection_spec = connection_spec if connection_spec else {}

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

        # logger.debug(f"Created LoftSegment: {name}")

    def generate_wafers(self):
        """Generate wafers for this segment using the loft-based approach."""
        logger.info(f"Generating wafers for segment {self.name}")
        try:
            curve_type = None
            if self.curve_spec:
                curve_type = self.curve_spec.get("type", None)

            if curve_type == "existing_curve":
                wafers = self._generate_wafers_from_existing_curve()
            else:
                wafers = self._generate_wafers_from_parametric_curve()

            self.wafer_list = wafers
            self._calculate_bounds()
            logger.info(f"Generated {len(self.wafer_list)} wafers for segment {self.name}")
            return self.wafer_list

        except Exception as e:
            logger.error(f"Failed to generate wafers for segment {self.name}: {e}", exc_info=True)
            raise


    def _generate_wafers_from_parametric_curve(self):
        """
        Generate wafers using the existing Curves + CurveFollowerLoft pipeline.
        This preserves current behavior for non-customcurve types.
        """
        self.follower = CurveFollowerLoft(wafer_settings=self.wafer_settings)
        # Existing path: CurveFollowerLoft handles Curves and loft creation. [file:1][file:7]
        self.follower.generate_loft_wafers(self.curve_spec, self.wafer_settings, base_placement=None)
        wafers = self.follower.get_wafer_list()
        return wafers

    def _generate_wafers_from_existing_curve(self):
        """
        Generate wafers from an existing FreeCAD curve object, referenced by Label
        via curve_spec:
            type: custom_curve
            from_label: SomeCurveLabel
            num_samples: 200
        """
        if not self.curve_spec:
            raise ValueError("curve_spec is required for custom_curve segments.")

        from_label = self.curve_spec.get("from_label", None)
        num_samples = int(self.curve_spec.get("num_samples", 0))

        if not from_label:
            raise ValueError("custom_curve requires 'from_label' in curve_spec.")
        if num_samples < 2:
            raise ValueError(f"custom_curve requires num_samples >= 2, got {num_samples}.")

        if self.doc is None:
            raise ValueError("LoftSegment.doc is None; cannot look up existing curve object.")

        # 1) Find the wire by label (hard fail if not unique / not continuous).
        wire = get_wire_from_label(self.doc, from_label, tol=1e-4)

        # 2) Sample world-coordinate points along the wire.
        points_world = sample_points_on_wire(wire, num_samples)

        # 3) Transform to segment-local coordinates using base_placement.
        #    This mirrors the world->local pattern used in generate_closing_curve. [file:2][file:7]
        points_local = transform_world_to_local(points_world, self.base_placement)

        # 4) Feed these local points into CurveFollowerLoft.
        self.follower = CurveFollowerLoft(wafer_settings=self.wafer_settings)
        self.follower.set_curve_points(points_local)
        self.follower.create_spine_curve()

        # Now reuse the existing loft + wafer logic in CurveFollowerLoft.
        # We only need LoftWaferGenerator to operate on the already-created spine.
        # The simplest way that reuses existing code is to call generate_loft_wafers
        # with a minimal curve_spec that tells it "points already set".
        #
        # For now, we add a dedicated method on CurveFollowerLoft would be ideal,
        # but to minimize changes, we call its internal machinery directly.

        # Create a spine for the LoftWaferGenerator just as generate_loft_wafers would.
        # This assumes LoftWaferGenerator.create_spine_from_points matches our points. [file:3]
        generator = LoftWaferGenerator(
            cylinder_radius=self.wafer_settings.get("cylinder_diameter", 1.875) / 2.0,
            wafer_settings=self.wafer_settings,
        )
        generator.create_spine_from_points(points_local)
        generator.create_loft_along_spine(self.follower)
        # Sampling strategy: reuse the limited_sampler logic inside CurveFollowerLoft. [file:1]
        target_chord = self.wafer_settings.get("max_chord", 0.5)
        max_filler_wafer_count = self.wafer_settings.get("max_wafer_count", None)

        def limited_sampler(spine_edge):
            params = self.follower.chord_distance_sampler(spine_edge, target_chord)
            if max_filler_wafer_count and len(params) > max_filler_wafer_count + 1:
                params = params[: max_filler_wafer_count + 1]
            return params

        generator.sample_points_along_loft(limited_sampler)
        generator.calculate_cutting_planes()
        wafers = generator.generate_wafers()

        # Save min/max X for this segment if available, same as CurveFollowerLoft. [file:1][file:3]
        xcoords = []
        for wafer in wafers:
            if wafer.wafer is not None:
                bbox = wafer.wafer.BoundBox
                xcoords.extend([bbox.XMin, bbox.XMax])
        if xcoords:
            self.xmin = min(xcoords)
            self.xmax = max(xcoords)
        else:
            self.xmin = None
            self.xmax = None

        return wafers



    def transform_curve_points(self, points):
        """
        Transform curve points by base_placement

        Args:
            points: List or numpy array of curve points

        Returns:
            List of App.Vector transformed points
        """
        import numpy as np

        # Convert to numpy array if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        transformed_points = []
        for point in points:
            # Create App.Vector
            if isinstance(point, App.Vector):
                vec = point
            else:
                vec = App.Vector(float(point[0]), float(point[1]), float(point[2]))

            # Transform by base_placement
            transformed_vec = self.base_placement.multVec(vec)
            transformed_points.append(transformed_vec)

        return transformed_points

    def _is_identity_placement(self, placement):
        """Check if placement is identity (no transformation)"""
        identity_pos = App.Vector(0, 0, 0)
        identity_rot = App.Rotation(0, 0, 0, 1)

        pos_close = (placement.Base - identity_pos).Length < 1e-6
        rot_close = placement.Rotation.isSame(identity_rot, 1e-6)

        return pos_close and rot_close


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

    def get_start_placement(self):
        """
        Get the placement at the START of this segment in WORLD coordinates

        For aligning segments:
        - Inverse transform by start placement to bring back to origin
        - Then transform by previous segment's end placement

        Returns:
            App.Placement in world coordinates where this segment starts
        """
        if not self.wafer_list or len(self.wafer_list) == 0:
            # No wafers - segment starts at base_placement
            return self.base_placement

        # Get first wafer's entry LCS (in local coordinates)
        first_wafer = self.wafer_list[0]

        if first_wafer.lcs1:
            # Transform local LCS to world coordinates
            local_start_lcs = first_wafer.lcs1
            world_start_lcs = self.base_placement.multiply(local_start_lcs)
            return world_start_lcs
        else:
            # No LCS available - return base_placement
            return self.base_placement

    def get_end_placement(self):
        """
        Get the placement at the END of this segment in WORLD coordinates
        This is where the next segment should start.

        Returns:
            App.Placement in world coordinates
        """
        if not self.wafer_list or len(self.wafer_list) == 0:
            # No wafers - return base_placement unchanged
            return self.base_placement

        # Get last wafer's exit LCS (in local coordinates)
        last_wafer = self.wafer_list[-1]

        if last_wafer.lcs2:
            # Transform local LCS to world coordinates
            local_end_lcs = last_wafer.lcs2
            world_end_lcs = self.base_placement.multiply(local_end_lcs)
            return world_end_lcs
        else:
            # No LCS available - return base_placement
            return self.base_placement

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
        Get the placement at the end of this segment in WORLD coordinates

        This is used for concatenating segments - the next segment
        starts where this one ends.

        Returns:
            App.Placement at the end of the segment in world coordinates
        """
        if not self.wafer_list or len(self.wafer_list) == 0:
            return self.base_placement

        # Get the last wafer's second LCS (the exit face) in LOCAL coordinates
        last_wafer = self.wafer_list[-1]

        if last_wafer.lcs2:
            # Transform local LCS to world coordinates
            local_lcs = last_wafer.lcs2
            world_lcs = self.base_placement.multiply(local_lcs)
            return world_lcs
        else:
            # Fallback to base placement if LCS not available
            return self.base_placement

    def visualize(self, doc):
        """Create FreeCAD visualization of the segment as a Part container"""
        logger.info(f"Visualizing segment '{self.name}'")

        # Create Part container
        segment_part = doc.addObject("App::Part", f"{self.name}_Part")
        segment_part.Placement = self.base_placement
        # logger.debug(f"Created Part with placement: {self.base_placement}")

        # Create groups (NOT added to Part yet)
        wafer_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_Wafers")
        reference_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_Reference")

        # Add loft to reference_group ONLY
        if hasattr(self.follower, 'generator') and hasattr(self.follower.generator, 'loft'):
            loft = self.follower.generator.loft
            if loft and self.segment_settings.get('show_loft', False):
                loft_obj = doc.addObject("Part::Feature", f"Loft_{self.name}")
                loft_obj.Shape = loft
                loft_obj.ViewObject.Transparency = 70
                loft_obj.ViewObject.ShapeColor = (0.8, 0.9, 1.0)
                reference_group.addObject(loft_obj)  # Only add to group
                # logger.debug("Added loft")

        # Add spine to reference_group ONLY
        if hasattr(self.follower, 'generator') and hasattr(self.follower.generator, 'spine_curve'):
            spine = self.follower.generator.spine_curve
            if spine:
                spine_obj = doc.addObject("Part::Feature", f"Spine_{self.name}")
                spine_obj.Shape = spine
                spine_obj.ViewObject.LineColor = (1.0, 0.0, 0.0)
                spine_obj.ViewObject.LineWidth = 2.0
                spine_obj.ViewObject.Visibility = False
                reference_group.addObject(spine_obj)  # Only add to group
                # logger.debug("Added spine")

        # Create LCS group if needed
        lcs_group = None
        if self.segment_settings.get('show_lcs', False):
            lcs_group = doc.addObject("App::DocumentObjectGroup", f"{self.name}_LCS")
            # logger.debug("Created LCS group")

        # Add wafers to wafer_group ONLY
        wafer_count = 0
        for i, wafer_data in enumerate(self.wafer_list):
            if wafer_data.wafer is None:
                continue

            wafer_obj = doc.addObject("Part::Feature", f"Wafer_{self.name}_{i}")
            wafer_obj.Shape = wafer_data.wafer

            if i % 2 == 0:
                wafer_obj.ViewObject.ShapeColor = (0.5, 1.0, 0.5)
            else:
                wafer_obj.ViewObject.ShapeColor = (0.7, 0.7, 1.0)

            wafer_obj.ViewObject.Transparency = 20
            wafer_group.addObject(wafer_obj)  # Only add to group
            wafer_count += 1

            # Add LCS to lcs_group ONLY
            if lcs_group is not None:
                if hasattr(wafer_data, 'lcs1') and wafer_data.lcs1:
                    lcs_obj = doc.addObject("PartDesign::CoordinateSystem", f"LCS_{self.name}_{i}_1")
                    lcs_obj.Placement = wafer_data.lcs1
                    lcs_group.addObject(lcs_obj)  # Only add to group

                if hasattr(wafer_data, 'lcs2') and wafer_data.lcs2:
                    lcs_obj = doc.addObject("PartDesign::CoordinateSystem", f"LCS_{self.name}_{i}_2")
                    lcs_obj.Placement = wafer_data.lcs2
                    lcs_group.addObject(lcs_obj)  # Only add to group

        # logger.debug(f"Added {wafer_count} wafers")

        # NOW add the groups to the Part (only once, at the end)
        segment_part.addObject(wafer_group)
        segment_part.addObject(reference_group)
        if lcs_group is not None:
            segment_part.addObject(lcs_group)

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