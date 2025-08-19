try:
    from core.logging_setup import get_logger
except Exception:
    try:
        from logging_setup import get_logger
    except Exception:
        import logging
        get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import numpy as np
import time
import FreeCAD
import Part
import utilities


class Wafer(object):

    def __init__(self, app, gui, parm_set, wafer_type="EE"):
        self.app = app
        self.gui = gui
        self.parm_set = parm_set        # Naming preamble for elements in a set (segment)
        self.wafer_type = wafer_type    # EE, CE, EC, CC
        self.outside_height = None
        self.lift_angle = None
        self.rotation_angle = None
        self.cylinder_radius = None
        self.lcs_top = None
        self.lcs_base = None
        self.wafer_name = None
        self.wafer = None
        self.angle = None   # Angle to x-y plane of top surface

    def set_parameters(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE"):
        self.lift_angle = lift
        self.rotation_angle = rotation
        self.cylinder_radius = cylinder_diameter / 2
        self.outside_height = outside_height
        self.wafer_type = wafer_type

    def make_wafer_from_lcs(self, lcs1, lcs2, cylinder_diameter, wafer_name,
                            start_cut_angle=None, end_cut_angle=None):
        """Create a wafer from two LCS objects.

        Args:
            start_cut_angle: Angle in radians (0 for circular cut)
            end_cut_angle: Angle in radians (0 for circular cut)
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

        wafer_axis.normalize()

        # Determine extensions based on cut angles
        epsilon = 0.001  # Small value to avoid numerical issues

        # Check the wafer type characters directly
        if self.wafer_type[0] == 'C':  # Circular start
            extend1 = epsilon
        else:  # Elliptical start
            if start_cut_angle is not None and abs(start_cut_angle) > 0.001:
                extend1 = self.cylinder_radius * np.tan(abs(start_cut_angle)) + epsilon
            else:
                extend1 = epsilon

        if self.wafer_type[1] == 'C':  # Circular end
            extend2 = epsilon
        else:  # Elliptical end
            if end_cut_angle is not None and abs(end_cut_angle) > 0.001:
                extend2 = self.cylinder_radius * np.tan(abs(end_cut_angle)) + epsilon
            else:
                extend2 = epsilon

        logger.debug(f"        Extensions based on type {self.wafer_type}: extend1={extend1:.3f}, extend2={extend2:.3f}")

        # Cap extensions to reasonable values
        max_extend = chord_length * 0.5
        extend1 = min(extend1, max_extend)
        extend2 = min(extend2, max_extend)

        # Create cylinder with extensions
        cylinder_start = pos1 - wafer_axis * extend1
        cylinder_length = chord_length + extend1 + extend2
        cylinder_end = pos1 + wafer_axis * (chord_length + extend2)

        logger.debug(f"      Cylinder: chord_length={chord_length:.3f}, extend1={extend1:.3f}, extend2={extend2:.3f}")
        logger.debug(f"      Total cylinder length: {cylinder_length:.3f}")
        # DEBUG: Print the relationship between LCS and actual geometry
        logger.debug(f"        🔍 WAFER GEOMETRY DEBUG for {wafer_name}:")
        logger.debug(f"           LCS1 position: [{pos1.x:.3f}, {pos1.y:.3f}, {pos1.z:.3f}]")
        logger.debug(f"           LCS2 position: [{pos2.x:.3f}, {pos2.y:.3f}, {pos2.z:.3f}]")
        logger.debug(f"           Cylinder START: [{cylinder_start.x:.3f}, {cylinder_start.y:.3f}, {cylinder_start.z:.3f}]")
        logger.debug(f"           Cylinder END:   [{cylinder_end.x:.3f}, {cylinder_end.y:.3f}, {cylinder_end.z:.3f}]")
        logger.debug(f"           Offset from LCS1 to cylinder start: {extend1:.3f} (backwards)")
        logger.debug(f"           Offset from LCS2 to cylinder end: {(cylinder_end - pos2).Length:.3f}")

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

        logger.debug(f"      Created cylinder for wafer between {lcs1.Label} and {lcs2.Label}")

    @staticmethod
    def _make_rectangle(long_side, short_side):
        part = Part.makePolygon([FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(long_side, 0, 0),
                                 FreeCAD.Vector(long_side, short_side, 0), FreeCAD.Vector(0, short_side, 0),
                                 FreeCAD.Vector(0, 0, 0)])
        face = Part.Face(part)
        return face

    def get_lift_angle(self):
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

    def lift_lcs(self, lcs, lcs_type):
        """Move lcs on base surface to corresponding position on top.

        This is a side-effecting operation.  Result is modified input lcs.

        There are four cases where the base is a circle (C) or ellipse (E) and similarly on top.

            For CC:  The lift angle is zero and the new placement is simply the addition of the
                outside height to the z-axis.
            For CE: The wafer is a cylinder of height outside height / 2 and an inclined section
                with a circle at the base and ellipse at the top.
            For EC: The wafer is an inverted CE.
            For EE: The wafer is an EC at the base and a CE on top.

        Calculations:
            Calculate each part separately (thus CE above is a cylinder on bottom and an ellipse on top with the
            cylinder of height outside_height / 2).  Then combine two parts appropriately to create the full case above.

            If base is an ellipse, rotate point to global zero.  Figure will be a cylinder inclined
            by the lift angle with inclination in the xz plane. Make changes and rotate back to original position.

        """
        # logger.debug(f"LIFT_LCS: {lcs_type}, angle: {convert_angle(lift_angle)}, diam: {cylinder_diameter}, oh: {outside_height}")
        parts = {"CC": ("CC2", "CC2"),
                 "CE": ("CC2", "CE2"),
                 "EC": ("EC2", "CC2"),
                 "EE": ("EC2", "CE2")}
        parts_used = parts[lcs_type]
        h2 = self.outside_height / 2
        d2 = self.cylinder_radius
        la = self.lift_angle        # Lift angle already divided for end segments
        if lcs_type == "EE":
            la = self.lift_angle / 2
        result_vec = []
        for i in range(2):
            # logger.debug(f"PARTS: {parts_used}, LCS: {lcs.Label}, i: {i}")
            if "EC2" == parts_used[i]:
                del_x = 0  # -d2 * np.sin(la)
                del_z = h2 - d2 * np.tan(la)
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"EC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}, {i}: {parts_used}")
            if "CE2" == parts_used[i]:
                del_x = 0  # -d2 * np.sin(la)
                del_z = h2 - d2 * np.tan(la)
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"CE  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}  {i}: {parts_used}")
            if "CC2" == parts_used[i]:
                del_x = 0
                del_z = 0              # removes cylindrical part
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"CC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)} res: {result_vec[i]}")
            rot = FreeCAD.Rotation(0, 0, 0)
            if parts_used[i] in ["CE2", "EC2"]:
                # Not clear if Rotation wants radians or degrees.  Testing says radians - complex_path says degrees
                rot = FreeCAD.Rotation(0, -np.rad2deg(la), 0)
            # if parts_used[i] == "EE2":          # there is no EE2
            #     rot = FreeCAD.Rotation(0, -np.rad2deg(la * 2), 0)
            new_place = FreeCAD.Placement(result_vec[i], rot)
            lcs.Placement = lcs.Placement.multiply(new_place)
            # break

    def validate_segment_join(segment1_last_wafer, segment2_first_wafer):
        # Check that segment1 ends with 'C' and segment2 starts with 'C'
        if segment1_last_wafer.wafer_type[1] != 'C':
            raise ValueError("Segment must end with circular cut for joining")
        if segment2_first_wafer.wafer_type[0] != 'C':
            raise ValueError("Segment must start with circular cut for joining")


def convert_angle(angle):
    return np.round(np.rad2deg(angle), 3)