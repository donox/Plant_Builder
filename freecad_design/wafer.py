import numpy as np
import time
import FreeCAD


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

    def make_wafer_from_lcs(self, lcs1, lcs2, cylinder_diameter, wafer_name):
        """Make a wafer by cutting a cylinder with the xy-planes of two lcs's."""
        # TODO: if xy-planes are parallel, need to handle special cases (2 - co-linear z-axis and not)
        self.lcs_top = lcs2
        self.lcs_base = lcs1
        self.cylinder_radius = cylinder_diameter / 2
        self.wafer_name = wafer_name

        # create a cylinder between lcs1 and lcs2 with the cylinder axis
        # along the path between the origin points of the lcs's
        wafer_1 = self.wafer_type[0]
        wafer_2 = self.wafer_type[1]
        if wafer_1 == 'E':
            e1 = self.app.activeDocument().addObject('Part::Ellipse', self.parm_set + "e1")
            e1.MinorRadius = self.cylinder_radius
            if self.lift_angle:
                e1.MajorRadius = self.cylinder_radius / np.cos(self.lift_angle)
            else:
                e1.MajorRadius = self.cylinder_radius
        elif wafer_1 == 'C':
            e1 = self.app.activeDocument().addObject('Part::Circle', self.parm_set + "e1")
            e1.Radius = self.cylinder_radius
        else:
            raise ValueError(f"Unrecognized Wafer Type: {wafer_1}")
        e1.Placement = lcs1.Placement
        e1.Visibility = False
        if wafer_2 == 'E':
            e2 = self.app.activeDocument().addObject('Part::Ellipse', self.parm_set + "e1")
            e2.MinorRadius = self.cylinder_radius
            if self.lift_angle:
                e2.MajorRadius = self.cylinder_radius / np.cos(self.lift_angle)
            else:
                e2.MajorRadius = self.cylinder_radius
        elif wafer_2 == 'C':
            e2 = self.app.activeDocument().addObject('Part::Circle', self.parm_set + "e1")
            e2.Radius = self.cylinder_radius
        else:
            raise ValueError(f"Unrecognized Wafer Type: {wafer_2}")
        e2.Placement = lcs2.Placement
        e2.Visibility = False
        e_edge = e2.Shape.Edges[0]
        e_normal = e_edge.normalAt(0)   # normal to edge lies in plane of the ellipse
        self.angle = 90 - np.rad2deg(e_normal.getAngle(self.app.Vector(0, 0, 1)))
        # print(f"{e_edge} at angle: {self.angle}")
        self.wafer = self.app.activeDocument().addObject('Part::Loft', wafer_name)
        self.wafer.Sections = [e1, e2]
        self.wafer.Solid = True
        self.wafer.Visibility = True

    def rotate_to_vertical(self, x_ang, y_ang):
        self.wafer.Placement.Matrix.rotateX(x_ang)
        self.wafer.Placement.Matrix.rotateY(y_ang)

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


def lift_lcs(lcs, lift_angle, cylinder_diameter, outside_height, lcs_type):
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
    # print(f"LIFT_LCS: {lcs_type}, angle: {convert_angle(lift_angle)}, diam: {cylinder_diameter}, oh: {outside_height}")
    parts = {"CC": ("CC2", "CC2"),
             "CE": ("CC2", "CE2"),
             "EC": ("EC2", "CC2"),
             "EE": ("EC2", "CE2")}
    parts_used = parts[lcs_type]
    h2 = outside_height / 2
    d2 = cylinder_diameter / 2
    la = lift_angle / 2
    result_vec = []
    for i in range(2):
        # print(f"PARTS: {parts_used}, LCS: {lcs.Label}, i: {i}")
        if "EC2" == parts_used[i]:
            del_x = -d2 * np.sin(la)
            del_z = h2 - d2 * np.tan(la)
            result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
            # print(f"EC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}, {i}: {parts_used}")
        if "CE2" == parts_used[i]:
            del_x = -d2 * np.sin(la)
            del_z = h2 - d2 * np.tan(la)
            result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
            # print(f"CE  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}  {i}: {parts_used}")
        if "CC2" == parts_used[i]:
            del_x = 0
            del_z = 0              # removes cylindrical part
            result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
            # print(f"CC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)} res: {result_vec[i]}")
        rot = FreeCAD.Rotation(0, 0, 0)
        if parts_used[i] in ["CE2", "EC2"]:
            # Not clear if Rotation wants radians or degrees.  Testing says radians - complex_path says degrees
            rot = FreeCAD.Rotation(0, -np.rad2deg(la), 0)
        # if parts_used[i] == "EE2":          # there is no EE2
        #     rot = FreeCAD.Rotation(0, -np.rad2deg(la * 2), 0)
        new_place = FreeCAD.Placement(result_vec[i], rot)
        lcs.Placement = lcs.Placement.multiply(new_place)
        # break
    # lcso = FreeCAD.activeDocument().getObject(lcs.Label)        # Debug - checking actual lift angle
    # vec = FreeCAD.Vector(0, 0, 1)
    # vec2 = lcso.getGlobalPlacement().Rotation.multVec(vec)
    # angle = vec2.getAngle(vec)
    # print(f"LIFT_LCS: Input: {np.round(np.rad2deg(lift_angle),2)}, lift_angle - {np.round(np.rad2deg(angle),2)}")


def convert_angle(angle):
    return np.round(np.rad2deg(angle), 3)