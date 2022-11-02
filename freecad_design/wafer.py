import numpy as np
import time

class Wafer(object):

    def __init__(self, app, gui, parm_set):
        self.app = app
        self.gui = gui
        self.parm_set = parm_set
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
        self.lcs_top = lcs1
        self.lcs_base = lcs2
        self.cylinder_radius = cylinder_diameter / 2
        self.wafer_name = wafer_name

        # create a cylinder between lcs1 and lcs2 with the cylinder axis
        # along the path between the origin points of the lcs's
        e1 = self.app.activeDocument().addObject('Part::Ellipse', self.parm_set + "e1")
        e1.Placement = lcs1.Placement
        e1.MinorRadius = self.cylinder_radius
        e1.MajorRadius = self.cylinder_radius
        e1.Visibility = False
        e2 = self.app.activeDocument().addObject('Part::Ellipse', self.parm_set + "e2")
        e2.Placement = lcs2.Placement
        e2.MinorRadius = self.cylinder_radius
        e2.MajorRadius = self.cylinder_radius  # TODO: Need to account for lift angle
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

