import Part
import Draft
import FreeCAD
import numpy as np
import csv
import re
import sys
import math
from .wafer import Wafer
from .flex_segment import FlexSegment
from .utilities import convert_angle, format_vector,  print_placement, get_lift_and_rotate


class PathFollower(object):
    def __init__(self, segment):
        self.segment = segment

        self.wafer_height = None
        self.wafer_diameter = None

        self.curve_selector = None
        self.nbr_points = None
        self.curve_rotation = None
        self.increment = None
        self.scale = None

    def set_curve_parameters(self, curve_selector, number_of_points, rotation, increment, scale):
        self.curve_selector = curve_selector
        self.nbr_points = number_of_points
        self.scale = scale
        self.increment = increment  # multiplier on point number used to determine increase in angle step to build curve
        self.curve_rotation = rotation

    def set_wafer_parameters(self, outside_height, wafer_diameter):
        self.wafer_height = outside_height
        self.wafer_diameter = wafer_diameter

    def implement_curve(self):
        curve_path = self.curve(self.curve_selector, self.scale, self.nbr_points, self.increment, self.curve_rotation,
                                add_vertex=True)

        # set initial lcs at global origin.  Move its coordinates to top of each wafer as wafer is created
        base_lcs = FreeCAD.activeDocument().addObject('PartDesign::CoordinateSystem',  "base lcs")

        wafer_type = "EE"
        for point in curve_path:
            print(f"POINT: {point}")
            point_nbr = point[0]
            point_place = point[1]      # FreeCAD Vector
            # next curve point must be higher than wafer height (not true, needs only height at center point)
            icnt = 0
            point_place_z = base_lcs.Placement.inverse().multVec(point_place).z
            print(f"Z MODS: {np.round(point_place_z, 3)} from {np.round(point_place.z, 3)}")
            if point_place_z <= self.wafer_height:
                print(f"Point {point_nbr} failed, Z-Position: {np.round(point_place_z, 3)}, outside: {self.wafer_height}")
                pass
            while base_lcs.Placement.inverse().multVec(point_place).z > self.wafer_height:
                print(f"POINT: {point}")
                icnt += 1
                print(f"CANDIDATE: {point_nbr} - {format_vector(point_place)}")
                lift, rotate_angle = get_lift_and_rotate(base_lcs, point_place)
                if np.abs(rotate_angle) > 90:
                    rotate_angle = np.sign(rotate_angle) * np.abs(180 - np.abs(rotate_angle))
                if lift > 30.0:
                    print(f"LIFT > 30 Needed: L: {np.round(lift,3)}")
                    lift = 30.0
                elif lift < 0:
                    print(f"Lift less than zero on point {point_nbr}, lift: {lift}, rotate: {rotate_angle}")
                else:
                    lift = np.round(lift / 2, 0) * 2
                rotate_angle = np.round(rotate_angle, 0)
                # rotate_angle = 0
                print(f"L/R NEEDED: L: {lift}, R: {rotate_angle}\n")
                self.segment.add_wafer(np.deg2rad(lift), np.deg2rad(rotate_angle), self.wafer_diameter,
                                       self.wafer_height, wafer_type=wafer_type)
                base_lcs = self.segment.get_lcs_top()
                l, r = get_lift_and_rotate(base_lcs, point_place)    # !!!! DEBUG
                print(f"AFTER NEW WAFER: L: {np.round(l, 0)}, R: {np.round(r, 0)}")
                if icnt > 10:
                    print(f"Exceeded 10 wafers for single point")
                    break
            if point_nbr > 1:
                break
        if self.segment.wafer_count > 0:
            self.segment.fuse_wafers()
        else:
            print(f"NO WAFERS in segment {self.segment.prefix}")

    def implement_curve_test(self):
        curve_path = self.curve(self.curve_selector, self.scale, self.nbr_points, self.increment, self.curve_rotation,
                                add_vertex=True)
        # set initial lcs at global origin.  Move its coordinates to top of each wafer as wafer is created
        base_lcs = FreeCAD.activeDocument().addObject('PartDesign::CoordinateSystem',  "base lcs")

        wafer_type = "EE"
        wafer_count = 0
        wafer_max = 2
        icnt_max = 10
        point_max = 9
        for point in curve_path:
            print(f"\nPOINT: {point}")
            point_nbr = point[0]
            point_place = point[1]      # FreeCAD Vector
            # next curve point must be higher than wafer height (not true, needs only height at center point)
            icnt = 0
            point_place_z = base_lcs.Placement.inverse().multVec(point_place).z
            # print(f"Z ROTATION: {np.round(point_place_z, 3)} from {np.round(point_place.z, 3)}")
            if point_place_z <= self.wafer_height:
                print(f"Point {point_nbr} failed, Z-Position: {np.round(point_place_z, 3)}, outside: {self.wafer_height}")
                pass
            while base_lcs.Placement.inverse().multVec(point_place).z > self.wafer_height:
                icnt += 1
                print(f"CANDIDATE: {point_nbr} - {format_vector(point_place)}")
                # print(f"PLACEMENT: {base_lcs.Placement}")
                lift, rotate_angle = get_lift_and_rotate(base_lcs, point_place)
                if lift > 30.0:
                    print(f"LIFT > 30 Needed: L: {np.round(lift,3)}")
                    lift = 30.0
                elif lift < 0:
                    print(f"Lift less than zero on point {point_nbr}, lift: {lift}, rotate: {rotate_angle}")
                else:
                    lift = np.round(lift / 2, 0) * 2
                rotate_angle = np.round(rotate_angle, 0)
                # rotate_angle = 0
                print(f"L/R NEEDED: L: {lift}, R: {rotate_angle}\n")
                self.segment.add_wafer(np.deg2rad(lift), np.deg2rad(rotate_angle), self.wafer_diameter,
                                       self.wafer_height, point_place, wafer_type=wafer_type)
                wafer_count += 1
                base_lcs = self.segment.get_lcs_top()
                l, r = get_lift_and_rotate(base_lcs, point_place)    # !!!! DEBUG
                print(f"AFTER NEW WAFER: L: {np.round(l, 0)}, R: {np.round(r, 0)}")
                if wafer_count > wafer_max:
                    print(f"Exceeded {wafer_max} wafers")
                    break
                if icnt > icnt_max:
                    print(f"Exceeded {icnt_max} wafers for single point")
                    break
            if wafer_count > wafer_max:
                break
            if point_nbr > point_max :
                print(f"Exceeded point max: {point_max}")
                break
        if self.segment.wafer_count > 0:
            self.segment.fuse_wafers()
        else:
            print(f"NO WAFERS in segment {self.segment.prefix}")

    @staticmethod
    def curve(curve_selector, scale, point_count, increment, rotation, add_vertex=True):
        """Compute path of specified curve.

        Parameters:
            curve_selector: name of curve to be created.
            scale: multiplier for all values of x, y, z.
            point_count: number of points to return.
            increment: increment parameter to determine curve input values.
            rotation: number of degrees to rotate entire curve (to cause initial points to have positive z coordinate)
            add_vertex: boolean to add vertex for each point.

        Return: list of tuples - (point nbr, FreeCAD vector)"""
        if add_vertex:
            doc = FreeCAD.activeDocument()
            point_group = doc.getObjectsByLabel("Curve_Points")
            if point_group:
                point_group = point_group[0]
            else:
                point_group = doc.addObject("App::DocumentObjectGroup", "Curve_Points")
        result = []

        if curve_selector == "overhand knot":
            for t in range(point_count):
                anglex = math.radians(t * increment) - math.pi
                x = (math.cos(anglex) + 2 * math.cos(2 * anglex)) * scale
                y = (math.sin(anglex) - 2 * math.sin(2 * anglex)) * scale
                z = (-math.sin(3 * anglex)) * scale
                if t == 0:  # set origin of knot to global origin
                    x0 = x
                    y0 = y
                    z0 = z
                vec1 = FreeCAD.Vector(x - x0, y - y0, z - z0)
                if rotation > 0:
                    rot = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), rotation)
                    vec = rot.multVec(vec1)
                else:
                    vec = vec1
                result.append((t, vec))
                if add_vertex:
                    lcs1 = FreeCAD.activeDocument().addObject('Part::Vertex', "Kdot" + str(t))
                    lcs1.X = vec.x
                    lcs1.Y = vec.y
                    lcs1.Z = vec.z
                    point_group.addObjects([lcs1])
                if anglex > 2 * np.pi:
                    break

        elif curve_selector == "test knot":
            for t in range(point_count):
                anglex = math.radians(np.mod(t * scale * 2, 360))
                angley = (math.sin(anglex) - 2 * math.sin(2 * anglex)) * scale
                anglez = math.radians(np.mod(t * scale, 360))
                print(f"\nPOINT NBR: {t}")
                print(f"ANGLE: X: {np.rad2deg(anglex)}, Y: {np.rad2deg(angley)}, Z: {np.rad2deg(anglez)}")
                x = t * (math.sin(anglex))
                y = t * (math.sin(angley))
                z = t * .8
                if t == 0:  # set origin of knot to global origin
                    x0 = x
                    y0 = y
                    z0 = z
                vec1 = FreeCAD.Vector(x - x0, y - y0, z - z0)
                print(f"ACTUAL: X - {np.round(x - x0, 2)}, Y - {np.round(y - y0, 2)}, Z - {np.round(z - z0, 2)}, ")
                if rotation > 0:
                    rot = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), rotation)
                    vec = rot.multVec(vec1)
                else:
                    vec = vec1
                result.append((t, vec))
                if add_vertex:
                    lcs1 = FreeCAD.activeDocument().addObject('Part::Vertex', "Kdot" + str(t))
                    lcs1.X = vec.x
                    lcs1.Y = vec.y
                    lcs1.Z = vec.z
                    point_group.addObjects([lcs1])
                if anglex > 2 * np.pi:
                    break
        else:
            raise ValueError(f"Invalid curve specifier: ..{curve_selector}.")
        return result




