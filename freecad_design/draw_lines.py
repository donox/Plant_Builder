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
        # print(f"\nLine: {line_start} - {line_end}")
        ls = Part.makeLine(line_start, line_end)
        s = self.doc.addObject("Part::Feature", name)
        if angle:
            rot = self.app.Rotation(axis, angle)
            obj_base = s.Shape.Placement.Base
            new_place = self.app.Placement(obj_base, rot, obj_base)
            ls.Placement = new_place
        s.Shape = ls
        lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', name + "_lcs")
        lcs2.Placement = ls.Placement
        lcs2.Support = [(s, 'Vertex2')]     # Move LCS to end of line by attaching and translating
        lcs2.MapMode = 'Translate'
        # Rotate LCS2 so that z-axis is in direction of line
        self.rotate_lcs_to_line(lcs2, line_start, line_end)
        lcs3 = self.doc.addObject('PartDesign::CoordinateSystem', name + "_lcs_pre")
        lcs3.Placement = ls.Placement
        lcs3.Support = [(s, 'Vertex2')]     # Move LCS to end of line by attaching and translating
        lcs3.MapMode = 'Translate'
        return s

    def rotate_lcs_to_line(self, lcs, line_start, line_end):
        """Align LCS such that z-axis is along line, positive from start to end."""
        vx = self.app.Vector(1, 0, 0)
        vy = self.app.Vector(0, 1, 0)
        vz = self.app.Vector(0, 0, 1)
        del_x = line_end.x - line_start.x
        del_y = line_end.y - line_start.y
        del_z = line_end.z - line_start.z
        line_vector = self.app.Vector(del_x, del_y, del_z)
        y_rot = np.arccos(del_x / np.sqrt(del_y ** 2 + del_z ** 2))
        x_rot = np.arccos(del_y / np.sqrt(del_x ** 2 + del_z ** 2))
        *axis, angle = DrawLines.euler_yzx_to_axis_angle(x_rot, y_rot, 0)

        lcs.Placement.Rotation = self.app.Rotation(self.app.Vector(*axis), angle)
        # lcs.Placement.Matrix.rotateX(y_rot)
        # newplace = self.app.Placement(lcs.Placement.Base, rot)
        # lcs.Placement = newplace
        print(f"Unit Vector: {axis}, Angle: {angle}")
        print(f"X, Y: {np.rad2deg(x_rot)}, {np.rad2deg(y_rot)}")
        print(f"Angles: X: {np.rad2deg(vx.getAngle(line_vector))}, Y: {np.rad2deg(vy.getAngle(line_vector))}, Z: {np.rad2deg(vz.getAngle(line_vector))}")
        return

    def get_end_of_line(self, line):
        """Determine endpoint of an arbitrary line."""
        # This may not work (not properly tested)
        start = line.Placement.Base
        z_ang, y_ang, x_ang = line.Placement.Rotation.toEulerAngles("zyx")
        line_len = line.Length
        x_pos = line_len * np.cos(np.deg2rad(x_ang)) + start.x
        y_pos = line_len * np.cos(np.deg2rad(y_ang)) + start.y
        z_pos = line_len * np.cos(np.deg2rad(z_ang)) + start.z
        app_x = self.app.Vector(x_pos, y_pos, z_pos)
        print(f"End Pos: {app_x}")
        return app_x

    def make_placement(self, place_str):
        """Read FreeCAD (printed) Placement as string and convert to Placement."""
        vectors = re.findall(r'\(.+?\)', place_str)
        if len(vectors) < 2:
            print(f"FOUND BAD PLACEMENT: {place_str}")
            return
        pos = eval("self.app.Vector" + vectors[0])
        rot = eval("self.app.Rotation" + vectors[1])
        new_place = self.app.Placement(pos, rot)
        return new_place

    @staticmethod
    def euler_yzx_to_axis_angle(x_e_r, y_e_r, z_e_r, normalize=True):
        x_e = np.deg2rad(x_e_r)
        y_e = np.deg2rad(y_e_r)
        z_e = np.deg2rad(z_e_r)
        c1 = np.cos(z_e / 2)
        s1 = np.sin(z_e / 2)
        c2 = np.cos(x_e / 2)
        s2 = np.sin(x_e / 2)
        c3 = np.cos(y_e / 2)
        s3 = np.sin(y_e / 2)
        c1c2 = c1 * c2
        s1s2 = s1 * s2
        w = c1c2 * c3 - s1s2 * s3
        x = c1c2 * s3 + s1s2 * c3
        y = s1 * c2 * c3 + c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3
        angle = 2 * np.arccos(w)
        if normalize:
            norm = x * x + y * y + z * z
            if norm < 0.001:
                # when all euler angles are zero angle =0 so
                # we can set axis to anything to avoid divide by zero
                x = 1
                y = 0
                z = 0
            else:
                norm = np.sqrt(norm)
                x /= norm
                y /= norm
                z /= norm
        return x, y, z, np.rad2deg(angle)
