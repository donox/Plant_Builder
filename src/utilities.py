import Part
import FreeCAD
import numpy as np
import csv
import re
import sys
import math


def print_placement(plc):
    """Print a placement as a numpy array."""
    ary = np.array(plc.Matrix.A)  # one dimensional numpy array
    res = ""
    for i in range(4):
        for j in range(4):
            res += f"\t{np.round(ary[i*4+j], 3):>6.3f}"
        res += "\n"
    return res


def squared_distance(place1, place2):
    """Return square of distance between two Placements"""
    p1 = place1.Base
    p2 = place2.Base
    p3 = p2.sub(p1)
    return p3.x ** 2 + p3.y ** 2 + p3.z ** 2


def display_axis_angle(pref, place):
    print(f"{pref}: {np.round(place.Rotation.Angle, 2)} {np.round(place.Rotation.Axis, 2)}")


def convert_angle(angle):
    return np.round(np.rad2deg(angle), 3)


def format_vector(vec):
    x = vec.x
    y = vec.y
    z = vec.z
    return f"vector: x: {np.round(x, 3)}, y: {np.round(y, 3)}, z: {np.round(z, 3)}"


def position_to_str(x):
    inches = int(x)
    fraction = int((x - inches) * 16)
    return f'{inches:2d}" {fraction:2d}/16'
