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
from .utilities import convert_angle


class PathFollowing(object):
    def __init__(self):
        pass

    def target_practice(self):
        tf = "/home/don/Documents/Temp/a_practice.txt"
        seg = FlexSegment("X",  False, tf, True, False)     # name, show_lcs, temp_file, to_build, rotate_segment
        wafer_diameter = 2.5
        outside_height = 1.0

        nbr_points = 100
        scale = 4
        increment = 4.0     # multiplier on point number used to determine increase in angle step to build curve
        curve_rotation = 0
        curve_path = self.curve("overhand knot", scale, nbr_points, increment, curve_rotation, add_vertex=True)

        # set initial lcs at global origin.  Move its coordinates to top of each wafer as wafer is created
        base_lcs = FreeCAD.activeDocument().addObject('PartDesign::CoordinateSystem',  "base lcs")
        transform_to_global_origin = base_lcs.Placement.inverse()

        wafer_type = "CE"
        for point in curve_path:
            point_nbr = point[0]
            point_place = point[1]      # FreeCAD Vector
            # next curve point must be higher than wafer height (not true, needs only height at center point)
            icnt = 0
            if base_lcs.Placement.inverse().multVec(point_place).z <= outside_height:
                print(f"Point {point_nbr} failed, Height: {point_place.z}")
            while base_lcs.Placement.inverse().multVec(point_place).z > outside_height:
                icnt += 1
                # print(f"CANDIDATE: {point_nbr} - {format_vector(point_place)}")
                lift, rotate_angle = self.get_lift_and_rotate(base_lcs, point_place)
                if np.abs(rotate_angle) > 90:
                    rotate_angle = np.sign(rotate_angle) * np.abs(180 - np.abs(rotate_angle))
                if lift > 15.0:
                    lift = 15.0
                elif lift < 0:
                    print(f"Lift less than zero on point {point_nbr}, lift: {lift}, rotate: {rotate_angle}")
                else:
                    lift = np.round(lift / 2, 0) * 2
                rotate_angle = np.round(rotate_angle, 0)
                # rotate_angle = 0
                print(f"L: {lift}, R: {rotate_angle}")
                seg.add_wafer(np.deg2rad(lift), np.deg2rad(rotate_angle), wafer_diameter, outside_height, wafer_type=wafer_type)
                base_lcs = seg.get_lcs_top()
                if icnt > 10:
                    print(f"Exceeded 10 wafers for single point")
                    break
            if point_nbr > 190:
                break


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
    start_element = 0.0

    if curve_selector == "overhand knot":
        for t in range(point_count):
            angle = math.radians(t * increment) - math.pi
            x = (math.cos(angle) + 2 * math.cos(2 * angle)) * scale
            y = (math.sin(angle) - 2 * math.sin(2 * angle)) * scale
            z = (-math.sin(3 * angle)) * scale
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
            if angle > 2 * np.pi:
                break
    else:
        raise ValueError(f"Invalid curve specifier: {curve_selector}")
    return result


def get_lift_and_rotate(lcs, vector):
    """Determine lift and rotate needed to z-axis point from the origin to the location of a point (vector)."""
    vec = lcs.Placement.inverse().multVec(vector)
    v_x = vec.x
    v_y = vec.y
    v_z = vec.z
    rotate_angle = np.arctan2(v_y, v_x)         # - np.pi / 2
    v_len = np.sqrt(v_x ** 2 + v_y ** 2)
    lift = np.arctan2(v_len, v_z)
    print(f"LIFT/ROTATE: L: {convert_angle(lift)}, R: {convert_angle(rotate_angle)}")
    # print(f"LIFT/ROTATE: in: {format_vector(vector)}, out: {format_vector(vec)}")
    # print(f"Place: \n{print_placement(lcs.Placement)} Inv: \n{print_placement(lcs.Placement.inverse())}")
    return np.rad2deg(lift), np.rad2deg(rotate_angle)
