import string
import inspect
import importlib.util
import Part
import Draft
import FreeCAD
import numpy as np
import csv
import re


class DrawLines(object):
    def __init__(self, app, doc):
        self.app = app
        self.doc = doc

    def draw_line(self, line_start, line_end, angle, axis, name):
        # place is Placement Base, angle is a rotation, length in mm
        print(f"\nLine: {line_start} - {line_end}")
        ls = Part.makeLine(line_start, line_end)
        s = self.doc.addObject("Part::Feature", name)
        if angle:
            rot = self.app.Rotation(axis, angle)
            obj_base = s.Shape.Placement.Base
            new_place = self.app.Placement(obj_base, rot, obj_base)
            ls.Placement = new_place
        s.Shape = ls
        basemat = ls.Placement.Base
        basemat = [basemat.x, basemat.y, basemat.z]
        basemat = [np.round(x, 2) for x in basemat]
        rotmat = ls.Placement.Rotation.toMatrix().A
        rotmat = [np.round(xx, 2) for xx in rotmat]
        print(f"Location: {basemat}")
        print(f"Rotation X: {rotmat[0:4]}")
        print(f"Rotation Y: {rotmat[4:8]}")
        print(f"Rotation Z: {rotmat[8:12]}")
        print(f"Rotation +: {rotmat[12:16]}")
        xy_ang = np.round(np.rad2deg(np.arcsin(line_end[2]/ls.Length)), 2)
        print(f"Angle to XY: {xy_ang}")
        self.doc.recompute()
        return s

    def get_angles(self, line):
        """Determine angle between a line (or object) and the z-axis

        Extra the yaw from the rotation matrix.  We excerpt from code in
        https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix"""
        z_ang, y_ang, x_ang = line.Placement.Rotation.toEulerAngles("YawPitchRoll")
        shape = line.Shape
        line_start = np.round(shape.FirstParameter, 2)
        line_end = np.round(shape.LastParameter, 2)
        xr = np.deg2rad(x_ang)
        line_len = np.round(line.Shape.Length, 2)
        x_loc = np.round(line_len * np.sin(xr), 2)
        print(f"Edge: {line_start} - {line_end}, Len: {line_len}, XLOC: {x_loc}")
        print(f"Angles: X: {np.round(x_ang, 2)}, Y: {np.round(y_ang, 2)}, Z: {np.round(z_ang, 2)}")
        circle = self.doc.addObject("Part::Circle", "x-circ")
        circle.Radius = 1
        circle.Placement = self.app.Placement(self.app.Vector(x_loc, 0, 0), self.app.Rotation(0, 0, 0))
        box = self.doc.addObject("Part::Box", "origin")
        box.Length = 1
        box.Height = 1
        box.Width = 1
        box.Placement = self.app.Placement(self.app.Vector(0, 0, 0), self.app.Rotation(0, 0, 0))
        return x_ang, y_ang, z_ang
