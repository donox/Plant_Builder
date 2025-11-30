from core.logging_setup import get_logger, log_coord, apply_display_levels
apply_display_levels(["ERROR", "WARNING", "INFO", "DEBUG"])
# apply_display_levels(["ERROR", "WARNING", "INFO"])

logger = get_logger(__name__)
import numpy as np
import math
import FreeCAD
import Part


class Wafer(object):
    EPS = 1e-9
    def __init__(self, app, gui, parm_set, wafer_type="EE"):
        self.app = app
        self.gui = gui
        self.parm_set = parm_set        # Naming preamble for elements in a set (segment)
        self.wafer_type = wafer_type    # EE, CE, EC, CC
        self.outside_height = None
        self.lift_angle = None      # All angles in degrees
        self.rotation_angle = None
        self.lift_start_angle = None
        self.lift_end_angle = None
        self.cylinder_radius = None
        self.lcs_top = None
        self.lcs_base = None
        self.wafer_name = None
        self.wafer = None
        self.angle = None   # Angle to x-y plane of top surface
        self.chord_length = None

    def set_parameters(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE"):
        self.lift_angle = lift
        self.rotation_angle = rotation
        self.cylinder_radius = cylinder_diameter / 2
        self.outside_height = outside_height
        self.wafer_type = wafer_type

    def make_wafer_from_lcs(self, lcs1, lcs2, cylinder_diameter, wafer_name,
                            start_cut_angle=None, end_cut_angle=None,
                            start_extension=None, end_extension=None):
        """Create a wafer from two LCS objects.

        Args:
            start_cut_angle: Angle in degrees (0 for circular cut)
            end_cut_angle: Angle in degrees (0 for circular cut)
            start_extension: Extension at start of cylinder (overrides angle calculation)
            end_extension: Extension at end of cylinder (overrides angle calculation)
        """
        self.lcs_top = lcs2
        self.lcs_base = lcs1
        self.cylinder_radius = cylinder_diameter / 2
        self.wafer_name = wafer_name

        # Get positions and the wafer axis (chord)
        pos1 = lcs1.Placement.Base
        pos2 = lcs2.Placement.Base
        wafer_axis = pos2 - pos1
        chord_length = wafer_axis.Length

        if chord_length < 1e-6:
            raise ValueError("Wafer has zero length")
        self.chord_length = chord_length

        wafer_axis.normalize()

        # Use passed extensions if provided, otherwise calculate from angles
        if start_extension is not None and end_extension is not None:
            extend1 = start_extension
            extend2 = end_extension
            logger.debug(f"Using passed extensions: extend1={extend1:.3f}, extend2={extend2:.3f}")
        else:
            # Fallback to angle-based calculation
            epsilon = 0.001  # Small value to avoid numerical issues

            # Check the wafer type characters directly
            if self.wafer_type[0] == 'C':  # Circular start
                extend1 = epsilon
            else:  # Elliptical start
                if start_cut_angle is not None and abs(start_cut_angle) > 0.001:
                    extend1 = self.cylinder_radius * np.tan(np.deg2rad(abs(start_cut_angle))) + epsilon
                else:
                    extend1 = epsilon

            if self.wafer_type[1] == 'C':  # Circular end
                extend2 = epsilon
            else:  # Elliptical end
                if end_cut_angle is not None and abs(end_cut_angle) > 0.001:
                    extend2 = self.cylinder_radius * np.tan(np.deg2rad(abs(end_cut_angle))) + epsilon
                else:
                    extend2 = epsilon

            # Cap extensions to reasonable values
            max_extend = chord_length * 0.5
            extend1 = min(extend1, max_extend)
            extend2 = min(extend2, max_extend)
            logger.debug(f"Calculated extensions: extend1={extend1:.3f}, extend2={extend2:.3f}")

        # Create cylinder with extensions
        # CRITICAL: LCS positions remain at pos1 and pos2, but cylinder extends beyond them
        cylinder_start = pos1 - wafer_axis * extend1
        cylinder_length = chord_length + extend1 + extend2
        cylinder_end = pos1 + wafer_axis * (chord_length + extend2)

        # logger.debug(f"      Cylinder: chord_length={chord_length:.3f}, extend1={extend1:.3f}, extend2={extend2:.3f}")
        # logger.debug(f"      Total cylinder length: {cylinder_length:.3f}")
        # logger.debug(f"        🔍 WAFER GEOMETRY DEBUG for {wafer_name}:")
        # logger.debug(f"           LCS1 position: [{pos1.x:.3f}, {pos1.y:.3f}, {pos1.z:.3f}]")
        # logger.debug(f"           LCS2 position: [{pos2.x:.3f}, {pos2.y:.3f}, {pos2.z:.3f}]")
        # logger.debug(
        #     f"           Cylinder START: [{cylinder_start.x:.3f}, {cylinder_start.y:.3f}, {cylinder_start.z:.3f}]")
        # logger.debug(f"           Cylinder END:   [{cylinder_end.x:.3f}, {cylinder_end.y:.3f}, {cylinder_end.z:.3f}]")
        # logger.debug(f"           Extension beyond LCS1: {extend1:.3f} (backwards)")
        # logger.debug(f"           Extension beyond LCS2: {extend2:.3f} (forwards)")

        # Create the cylinder
        cylinder = Part.makeCylinder(
            self.cylinder_radius,
            cylinder_length,
            cylinder_start,
            wafer_axis
        )

        # Verify the shape is valid
        if not cylinder.isValid():
            logger.error(f"      WARNING: Created cylinder is not valid!")
            cylinder.fix(0.1, 0.1, 0.1)
            if not cylinder.isValid():
                logger.error(f"      ERROR: Could not fix invalid cylinder")

        # Create the wafer object
        self.wafer = self.app.activeDocument().addObject("Part::Feature", wafer_name)
        self.wafer.Shape = cylinder
        self.wafer.ViewObject.Transparency = 0
        self.wafer.Visibility = False

        self.lcs1 = lcs1
        self.lcs2 = lcs2

        # logger.debug(f"      Created cylinder for wafer between {lcs1.Label} and {lcs2.Label}")
        # logger.debug(f"      LCS positions unchanged - extensions are purely geometric")

    def get_lift_angle(self):
        """
        Per-step lift/tilt (radians) to go from LCS1(n) to LCS1(n+1).

        Lift is the angle between the two START-face Z axes:
            lift = acos(  z1(n) · z1(n+1)  ).

        Return 0.0 if next_wafer is missing or geometry is degenerate.
        """
        return self.lift_angle

    def get_rotation_angle(self):
        return self.rotation_angle

    def get_outside_height(self):
        return self.outside_height

    def get_wafer_type(self):
        return self.wafer_type

    def get_cylinder_diameter(self):
        return self.cylinder_radius * 2

    def get_angle(self):
        return self.angle

    def get_chord_length(self):
        """Return the chord length (distance between LCS1 and LCS2)."""
        return self.chord_length

    def get_top(self):
        return self.lcs_top

    def get_base(self):
        return self.lcs_base

    def get_wafer_name(self):
        return self.wafer_name

    def get_lcs_top(self):
        return self.lcs_top

    def get_wafer(self):
        return self.wafer


def log_lcs_info(lcs, tag, logger_level="info"):
    """
    Log detailed information about a FreeCAD LCS (Local Coordinate System) object.

    Args:
        lcs: FreeCAD LCS object (PartDesign::CoordinateSystem)
        tag: String identifier for the LCS in log output
        logger_level: Logging level ("debug", "info", "warning", "error")
    """
    try:
        if lcs is None:
            getattr(logger, logger_level)(f"{tag}: LCS is None")
            return

        # Get the logging function
        log_func = getattr(logger, logger_level)

        # Basic object info
        log_func(f"\n=== LCS INFO: {tag} ===")
        log_func(f"  Label: {getattr(lcs, 'Label', 'N/A')}")
        log_func(f"  Name: {getattr(lcs, 'Name', 'N/A')}")

        # Check if object has placement
        if not hasattr(lcs, 'Placement'):
            log_func(f"  ERROR: No Placement attribute")
            return

        placement = lcs.Placement

        # Position information
        base = placement.Base
        log_func(f"  Position: [{base.x:.6f}, {base.y:.6f}, {base.z:.6f}]")

        # Rotation information
        rotation = placement.Rotation

        # Euler angles (in degrees)
        euler = rotation.toEuler()
        log_func(f"  Euler angles: [Yaw={euler[0]:.3f}°, Pitch={euler[1]:.3f}°, Roll={euler[2]:.3f}°]")

        # Axis-angle representation
        # axis = rotation.Axis
        # angle_deg = rotation.Angle * 180.0 / 3.14159265359
        # log_func(f"  Axis-Angle: Axis=[{axis.x:.6f}, {axis.y:.6f}, {axis.z:.6f}], Angle={angle_deg:.3f}°")

        # Quaternion
        # q = rotation.Q
        # log_func(f"  Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")

        # Local coordinate axes in global coordinates
        x_axis = rotation.multVec(FreeCAD.Vector(1, 0, 0))
        y_axis = rotation.multVec(FreeCAD.Vector(0, 1, 0))
        z_axis = rotation.multVec(FreeCAD.Vector(0, 0, 1))

        log_func(f"  Local X-axis in global: [{x_axis.x:.6f}, {x_axis.y:.6f}, {x_axis.z:.6f}]")
        log_func(f"  Local Y-axis in global: [{y_axis.x:.6f}, {y_axis.y:.6f}, {y_axis.z:.6f}]")
        log_func(f"  Local Z-axis in global: [{z_axis.x:.6f}, {z_axis.y:.6f}, {z_axis.z:.6f}]")

        # Verify orthonormality
        x_len = x_axis.Length
        y_len = y_axis.Length
        z_len = z_axis.Length
        xy_dot = x_axis.dot(y_axis)
        xz_dot = x_axis.dot(z_axis)
        yz_dot = y_axis.dot(z_axis)

        # log_func(f"  Axis lengths: X={x_len:.6f}, Y={y_len:.6f}, Z={z_len:.6f}")
        # log_func(f"  Dot products: X·Y={xy_dot:.6f}, X·Z={xz_dot:.6f}, Y·Z={yz_dot:.6f}")
        #
        # # Check for orthonormality issues
        # if abs(x_len - 1.0) > 1e-6 or abs(y_len - 1.0) > 1e-6 or abs(z_len - 1.0) > 1e-6:
        #     log_func(f"  WARNING: Axes are not unit length!")
        # if abs(xy_dot) > 1e-6 or abs(xz_dot) > 1e-6 or abs(yz_dot) > 1e-6:
        #     log_func(f"  WARNING: Axes are not orthogonal!")
        #
        # # Additional placement info if available
        # if hasattr(lcs, 'Visibility'):
        #     log_func(f"  Visibility: {lcs.Visibility}")

        log_func(f"=== END LCS INFO: {tag} ===\n")

    except Exception as e:
        getattr(logger, logger_level)(f"ERROR logging LCS info for {tag}: {e}")


# Convenience wrapper functions for different log levels
def   log_lcs_debug(lcs, tag):
    """Log LCS info at DEBUG level"""
    log_lcs_info(lcs, tag, "debug")


def log_lcs_info_level(lcs, tag):
    """Log LCS info at INFO level"""
    log_lcs_info(lcs, tag, "info")


def log_lcs_warning(lcs, tag):
    """Log LCS info at WARNING level"""
    log_lcs_info(lcs, tag, "warning")