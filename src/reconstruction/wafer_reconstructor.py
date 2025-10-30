"""Reconstruct wafer geometry from cutting list specifications."""

import math
import numpy as np
from typing import Tuple, Optional, Any
import FreeCAD
import Part

try:
    from core.logging_setup import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .cut_list_parser import WaferSpec


class WaferReconstructor:
    """Reconstruct FreeCAD wafer geometry from cut list parameters."""

    def __init__(self, doc: Any, cylinder_diameter: float):
        """Initialize reconstructor.

        Args:
            doc: FreeCAD document to create objects in
            cylinder_diameter: Diameter of cylinder in model units
        """
        self.doc = doc
        self.cylinder_diameter = cylinder_diameter
        self.radius = cylinder_diameter / 2.0

    def create_wafer_from_spec(self,
                               wafer_spec: WaferSpec,
                               start_lcs: Any,
                               prev_wafer_spec: Optional[WaferSpec] = None
                              ) -> Tuple[Any, Any, Any]:
        """Create wafer geometry and end LCS from specifications.

        Args:
            wafer_spec: Parsed wafer parameters from cut list
            start_lcs: LCS at wafer start (FreeCAD PartDesign::CoordinateSystem)
            prev_wafer_spec: Previous wafer's spec (for validation)

        Returns:
            Tuple of (wafer_cylinder, end_lcs, wafer_feature)
        """
        # Calculate chord length from outside height
        chord_length = self._calculate_chord_length(
            wafer_spec.outside_height,
            wafer_spec.wafer_type,
            wafer_spec.lift_angle if wafer_spec.lift_angle else 0.0
        )

        wafer_spec.chord_length = chord_length

        # Calculate end LCS position and orientation
        end_lcs = self._create_end_lcs(
            start_lcs,
            chord_length,
            wafer_spec.lift_angle if wafer_spec.lift_angle else 0.0,
            wafer_spec.rotation_angle if wafer_spec.rotation_angle else 0.0,
            wafer_spec.wafer_type,
            wafer_spec.wafer_num
        )

        # Create wafer cylinder
        wafer_cylinder = self._create_cylinder(
            start_lcs,
            end_lcs,
            wafer_spec.wafer_type,
            wafer_spec.outside_height,
            chord_length,
            wafer_spec.wafer_num
        )

        logger.debug(f"Reconstructed wafer {wafer_spec.wafer_num}: "
                    f"chord={chord_length:.3f}, outside={wafer_spec.outside_height:.3f}")

        return wafer_cylinder, end_lcs, wafer_cylinder

    def _calculate_chord_length(self, outside_height: float,
                                wafer_type: str,
                                lift_angle: float) -> float:
        """Back-calculate chord length from outside height.

        The outside height includes extensions at elliptical cuts.
        We need to reverse the extension calculation.

        Args:
            outside_height: Total outside height in mm
            wafer_type: CE, EE, EC, or CC
            lift_angle: Lift angle in degrees

        Returns:
            Chord length (distance between LCS centers) in mm
        """
        logger.debug(f"Calculating chord: outside={outside_height:.3f}mm, "
                     f"type={wafer_type}, lift={lift_angle:.2f}°")

        if wafer_type == "CC":
            # No extensions for circular-circular
            return outside_height

        # Calculate extensions based on wafer type
        lift_rad = math.radians(lift_angle if lift_angle else 0.0)

        # Extension calculation: ext = radius * tan(lift_angle)
        # DO NOT DIVIDE BY 2!
        if abs(lift_rad) < 1e-6:
            extension = 0.0
        else:
            extension = self.radius * math.tan(abs(lift_rad))

        # Cap extension at reasonable value
        extension = min(extension, self.cylinder_diameter)

        if wafer_type == "CE" or wafer_type == "EC":
            # One elliptical end
            chord = outside_height - extension
        elif wafer_type == "EE":
            # Two elliptical ends
            chord = outside_height - 2 * extension
        else:
            chord = outside_height

        # Ensure positive chord length
        chord = max(chord, 0.1)

        logger.debug(f"  -> chord={chord:.3f}mm, extension={extension:.3f}mm")

        return chord

    def _create_end_lcs(self, start_lcs, chord_length, lift_angle, rotation_angle, wafer_type, wafer_num):
        """Create LCS with proper pitch for conical spiral."""

        start_pos = start_lcs.Placement.Base
        start_rot = start_lcs.Placement.Rotation

        rotation_rad = math.radians(rotation_angle)

        # Get previous chord direction
        prev_chord_dir = start_rot.multVec(FreeCAD.Vector(0, 0, 1))

        # Calculate current radius (distance from Z-axis)
        current_radius = math.sqrt(start_pos.x ** 2 + start_pos.y ** 2)

        # Estimate pitch angle based on conical geometry
        # For a conical spiral: as radius decreases, pitch angle increases
        # This is approximate - you could refine based on the actual curve geometry
        xy_len = math.sqrt(prev_chord_dir.x ** 2 + prev_chord_dir.y ** 2)
        current_pitch = math.atan2(prev_chord_dir.z, xy_len)

        # CRITICAL: Adjust pitch based on changing radius
        # The pitch increase rate can be estimated from the spiral geometry
        # For your spiral: radius changes from 5.0 to 2.5 over height 3.0
        # A simple linear model: pitch_rate = (max_pitch - min_pitch) / height

        # Estimate the pitch increase per unit height
        height_traveled = start_pos.z
        total_height = 3.0  # from your curve spec
        radius_start = 5.0
        radius_end = 2.5

        # Simple model: pitch increases as radius decreases
        # pitch ~ constant_vertical_velocity / (2π * radius * angular_velocity)
        # So: pitch ∝ 1/radius
        estimated_radius = radius_start - (radius_start - radius_end) * (height_traveled / total_height)

        if estimated_radius > 0.1:  # avoid division by zero
            # Adjust pitch based on estimated radius change
            pitch_scale = radius_start / estimated_radius
            adjusted_pitch = current_pitch * pitch_scale
            # Limit the adjustment to prevent runaway
            adjusted_pitch = min(adjusted_pitch, math.radians(15))  # cap at 15°
        else:
            adjusted_pitch = current_pitch

        # Rotate the XY projection
        cos_rot = math.cos(rotation_rad)
        sin_rot = math.sin(rotation_rad)
        new_x = prev_chord_dir.x * cos_rot - prev_chord_dir.y * sin_rot
        new_y = prev_chord_dir.x * sin_rot + prev_chord_dir.y * cos_rot

        # Reconstruct 3D direction with adjusted pitch
        new_xy_len = math.sqrt(new_x ** 2 + new_y ** 2)
        if new_xy_len > 1e-10:
            scale = math.cos(adjusted_pitch) / new_xy_len
            new_x *= scale
            new_y *= scale

        new_z = math.sin(adjusted_pitch)

        new_chord_dir = FreeCAD.Vector(new_x, new_y, new_z)
        new_chord_dir.normalize()

        # Calculate end position
        end_pos = start_pos + new_chord_dir * chord_length

        # Calculate orientation
        end_rot = self._calculate_interface_orientation(prev_chord_dir, new_chord_dir)

        end_lcs = self.doc.addObject('PartDesign::CoordinateSystem',
                                     f'reconstructed_w{wafer_num}_end_lcs')
        end_lcs.Placement = FreeCAD.Placement(end_pos, end_rot)
        end_lcs.Visibility = False

        logger.info(f"🔧 Reconstructed wafer {wafer_num}:")
        logger.info(f"   Current radius: {current_radius:.3f}")
        logger.info(f"   Pitch angle: {math.degrees(adjusted_pitch):.2f}°")
        logger.info(f"   New chord dir: {new_chord_dir}")

        return end_lcs

    def _calculate_interface_orientation(self, prev_chord, current_chord):
        """Calculate LCS orientation from chord directions using geometric constraints.

        This replicates the logic from curve_follower._calculate_interface_orientation.
        """
        # Z-axis points along current chord
        z_axis = FreeCAD.Vector(current_chord.x, current_chord.y, current_chord.z)
        z_axis.normalize()

        # Calculate plane normal from the two chords
        plane_normal = prev_chord.cross(current_chord)

        if plane_normal.Length < 1e-6:
            # Parallel chords - use arbitrary perpendicular
            if abs(z_axis.z) < 0.9:
                plane_normal = z_axis.cross(FreeCAD.Vector(0, 0, 1))
            else:
                plane_normal = z_axis.cross(FreeCAD.Vector(1, 0, 0))

        plane_normal.normalize()

        # X-axis perpendicular to plane normal and z-axis
        # This is what creates the 3D spiral shape!
        x_axis = plane_normal.cross(z_axis)
        x_axis.normalize()

        # Y-axis completes right-handed system
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize()

        # Create rotation matrix
        matrix = FreeCAD.Matrix(
            x_axis.x, y_axis.x, z_axis.x, 0,
            x_axis.y, y_axis.y, z_axis.y, 0,
            x_axis.z, y_axis.z, z_axis.z, 0,
            0, 0, 0, 1
        )

        return FreeCAD.Rotation(matrix)

    def _create_cylinder(self,
                        start_lcs: Any,
                        end_lcs: Any,
                        wafer_type: str,
                        outside_height: float,
                        chord_length: float,
                        wafer_num: int) -> Any:
        """Create cylinder geometry for wafer.

        Args:
            start_lcs: Start LCS (FreeCAD PartDesign::CoordinateSystem)
            end_lcs: End LCS (FreeCAD PartDesign::CoordinateSystem)
            wafer_type: Type of wafer
            outside_height: Total height including extensions
            chord_length: Chord length
            wafer_num: Wafer number for naming

        Returns:
            FreeCAD Part::Feature object
        """
        start_pos = start_lcs.Placement.Base
        end_pos = end_lcs.Placement.Base

        # Cylinder axis direction
        axis_vec = end_pos - start_pos
        axis_length = axis_vec.Length

        if axis_length < 1e-6:
            logger.warning(f"Wafer {wafer_num} has zero length!")
            axis_length = 0.1
            axis_vec = FreeCAD.Vector(0, 0, 1)
        else:
            axis_vec.normalize()

        # Calculate extensions
        extension_length = outside_height - chord_length

        # Cylinder starts before start_lcs and extends beyond end_lcs
        if wafer_type == "CE":
            start_ext = 0.001
            end_ext = extension_length
        elif wafer_type == "EC":
            start_ext = extension_length
            end_ext = 0.001
        elif wafer_type == "EE":
            start_ext = extension_length / 2.0
            end_ext = extension_length / 2.0
        else:  # CC
            start_ext = 0.001
            end_ext = 0.001

        cylinder_start = start_pos - axis_vec * start_ext
        cylinder_length = outside_height

        # Create cylinder
        cylinder = Part.makeCylinder(
            self.radius,
            cylinder_length,
            cylinder_start,
            axis_vec
        )

        # Create FreeCAD object
        wafer_obj = self.doc.addObject("Part::Feature", f"reconstructed_wafer_{wafer_num}")
        wafer_obj.Shape = cylinder
        wafer_obj.ViewObject.Transparency = 30

        return wafer_obj