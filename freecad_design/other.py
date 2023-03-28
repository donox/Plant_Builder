import Part
import Draft
import FreeCAD
from FreeCAD import Base
import numpy as np
import csv
import re
import sys
import math
from .wafer import Wafer
from .flex_segment import FlexSegment
from .utilities import convert_angle, format_vector,  print_placement_2, get_lift_and_rotate, format_float



class Other:
    def __init__(self):
        self.doc = FreeCAD.ActiveDocument
        self.xhat = FreeCAD.Vector(1, 0, 0)
        self.yhat = FreeCAD.Vector(0, 1, 0)
        self.zhat = FreeCAD.Vector(0, 0, 1)

    def do_it(self):
        lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', "lcs1")
        lx = 2
        ly = 2
        lz = 2
        # lvec = FreeCAD.Vector(-0.021587793754935802, -1.9632208867019203, 1.1776612677410379)
        lvec = FreeCAD.Vector(lx, ly, lz)
        lrot = FreeCAD.Rotation(10, 20, 30)
        lcs1.Placement = FreeCAD.Placement(lvec, lrot)
        x = 2
        y = 3
        z = 2
        # vector = lvec
        vector = FreeCAD.Vector(x, y, z)
        vector = lcs1.Placement.multVec(vector)
        l, r = get_lift_and_rotate(lcs1, vector)
        print(f"Vector: {format_vector(vector)}, Lift: {format_float(l)}, Rotate: {format_float(r)}")

    def rotate_lcs(self):
        point = FreeCAD.Vector(16, 7, -8)
        lcs_origin = FreeCAD.Vector(2, 4, 1)

        # Define the arbitrary point for the vertex
        vertex = self.doc.addObject('Part::Vertex', "Kdot")
        vertex.Placement.Base = point

        # Create lcs with arbitrary position and rotation
        lcs = self.doc.addObject('PartDesign::CoordinateSystem', "LCS")
        lcs.Placement.Base = lcs_origin
        lcs_base = self.doc.addObject('PartDesign::CoordinateSystem', "LCS_BASE")

        # Get the direction vector of the line joining the vertex and the LCS position
        line_dir = vertex.Placement.Base - lcs.Placement.Base
        line_dir.normalize()
        line = Part.LineSegment()
        line.StartPoint = lcs.Placement.Base
        line.EndPoint = vertex.Placement.Base
        obj = self.doc.addObject("Part::Feature", "Line")
        obj.Shape = line.toShape()

        direct = obj.Shape.Curve.Direction
        rot = FreeCAD.Rotation(direct.cross(self.zhat), self.zhat, direct, 'ZXY')
        lcs.Placement = FreeCAD.Placement(line.EndPoint, rot)
        lcs_base.Placement.Base = line.EndPoint
        self.doc.recompute()

        x_ang = self.xhat.getAngle(direct)
        y_ang = self.yhat.getAngle(direct)
        z_ang = self.zhat.getAngle(direct)
        print(f"X: {convert_angle(x_ang)}, Y: {convert_angle(y_ang)}, Z: {convert_angle(z_ang)}")

    @staticmethod
    def lift_and_rotate_lcs():
        doc = FreeCAD.ActiveDocument
        xhat = FreeCAD.Vector(1, 0, 0)
        yhat = FreeCAD.Vector(0, 1, 0)
        zhat = FreeCAD.Vector(0, 0, 1)

        lift_angle = np.deg2rad(40)
        rotate_angle = np.deg2rad(30)

        lcs = doc.addObject('PartDesign::CoordinateSystem', "LCS")
        lcs_base = doc.addObject('PartDesign::CoordinateSystem', "LCS_BASE")
        mat = lcs.Placement.Matrix
        mat.rotateY(lift_angle)
        mat.rotateZ(rotate_angle)
        lcs.Placement = mat

        angles = lcs.Placement.Rotation.getYawPitchRoll()
        print(f"EULER Z(rotate): {np.round(angles[0], 2)}, Y(lift): {np.round(angles[1], 2)}, X: {np.round(angles[2], 2)}")
        # print(f"{print_placement(lcs.Placement)}")

    def follow_path(self, overhand_knot_path):
        scale = 70
        point_count = 30  # actual is 1 less
        knot_rotation = 0
        path_origin = FreeCAD.Vector(8, 4, 5)
        # content: (t, place, dist) - t=point nbr, place is placement, dist is dist from prior point
        path_place_list = overhand_knot_path(scale, point_count, knot_rotation)

        # Create lcs with arbitrary position and rotation
        global_lcs = self.doc.addObject('PartDesign::CoordinateSystem', "Global_LCS")
        current_lcs = self.doc.addObject('PartDesign::CoordinateSystem', "LCS_Current")   # base for path
        current_lcs.Placement.Base = path_origin

        for t, place, dist in path_place_list:
            inv = current_lcs.Placement.inverse()

            next_lcs = self.doc.addObject('PartDesign::CoordinateSystem', f"LCS_{t}")

            line = Part.LineSegment()
            line.StartPoint = current_lcs.Placement.Base
            line.EndPoint = place.Base
            line_obj = self.doc.addObject("Part::Feature", "Line")
            line_obj.Shape = line.toShape()
            direct = line_obj.Shape.Curve.Direction
            rot = FreeCAD.Rotation(direct.cross(self.zhat), self.zhat, direct, 'ZXY')

            next_lcs.Placement.Base = line_obj.Shape.Edges[0].lastVertex().Point
            next_lcs.Placement.Rotation = rot

            pl = next_lcs.Placement
            next_lcs.Placement = inv.multiply(next_lcs.Placement)
            angles = next_lcs.Placement.Rotation.getYawPitchRoll()
            print(
                f"EULER Z(rotate): {np.round(angles[0], 2)}, Y(lift): {np.round(angles[1], 2)}, X: {np.round(angles[2], 2)}")
            next_lcs.Placement = pl

            current_lcs = next_lcs

    def get_vertex_direction(self, vertex,  lcs_start):
        # Get the direction vector of the line joining the vertex and the LCS position
        line_dir = vertex - lcs_start.Base
        line_dir.normalize()
        line = Part.LineSegment()
        line.StartPoint = lcs_start.Base
        line.EndPoint = vertex
        line_obj = self.doc.addObject("Part::Feature", "Line")
        line_obj.Shape = line.toShape()

        direct = line_obj.Shape.Curve.Direction
        # rot = FreeCAD.Rotation(direct.cross(self.zhat), self.zhat, direct, 'ZXY')
        # lcs_start.Placement = FreeCAD.Placement(line.EndPoint, rot)
        # lcs_end.Placement.Base = line.EndPoint
        # self.doc.recompute()

        x_ang = self.xhat.getAngle(direct)
        y_ang = self.yhat.getAngle(direct)
        z_ang = self.zhat.getAngle(direct)
        print(f"Angles: X: {convert_angle(x_ang)}, Y: {convert_angle(y_ang)}, Z: {convert_angle(z_ang)}")
        return line_obj, x_ang, y_ang, z_ang

    def yp(self):
        target_point = Base.Vector(10, 10, 10)

        # Create an arbitrarily positioned and rotated LCS
        lcs_base = Base.Vector(5, 5, 5)
        lcs_rotation = Base.Rotation(Base.Vector(1, 0, 0), 0)
        lcs = Base.Placement(lcs_base, lcs_rotation)

        # Calculate the Yaw and Pitch
        yaw, pitch = self.get_yaw_pitch(lcs, target_point)

        # Print the results
        print("Yaw: {:.2f}°".format(yaw))
        print("Pitch: {:.2f}°".format(pitch))

    @staticmethod
    def get_yaw_pitch(lcs, target_point):
        # Calculate the vector between the LCS origin and the target point
        direction_vector = target_point - lcs.Base

        # Transform the direction_vector to the LCS's local coordinates
        direction_vector_local = lcs.Rotation.inverted().multVec(direction_vector)

        # Normalize the vector
        direction_vector_local.normalize()

        # Calculate the Yaw and Pitch
        yaw = math.atan2(direction_vector_local.y, direction_vector_local.x)
        pitch = math.asin(direction_vector_local.z)

        return math.degrees(yaw), math.degrees(pitch)
