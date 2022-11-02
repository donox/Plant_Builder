import FreeCAD
from math import sin, cos, radians

i = 0
gu = None
cube = None
cube_placement = None
r_cube_pl = None


def setup():
    global cube, cube_placement, r_cube_pl
    cube = FreeCAD.ActiveDocument.getObjectsByLabel("Cube")[0]
    # store initial placement (needed to restore initial position)
    cube_placement = cube.Placement
    # store object placements in a new variable
    r_cube_pl = cube.Placement


def reset_cube():
    global r_cube_pl
    # function to restore initial position of the objects
    cube.Placement = r_cube_pl


def update_cube():
    global i, cube
    alpha = radians(i)
    x = 150.0 * cos(alpha)
    y = 150.0 * sin(alpha)
    cube.Placement = FreeCAD.Placement(cube_placement.Base + FreeCAD.Vector(150 - x, 0, 0),
                                       cube_placement.Rotation)
    i += 1


FreeCAD.animation_reset = reset_cube
FreeCAD.animation_update = update_cube
setup()

# OLD WORKING COPY
# i = 0
# gu = None
# cube = None
# cube_placement = None
# r_cube_pl = None
#
#
# def setup():
#     global cube, cube_placement, r_cube_pl
#     cube = FreeCAD.ActiveDocument.getObjectsByLabel("Cube")[0]
#     # store initial placement (needed to restore initial position)
#     cube_placement = cube.Placement
#     # store object placements in a new variable
#     r_cube_pl = cube.Placement
#
#
# def reset_cube():
#     global r_cube_pl
#     # function to restore initial position of the objects
#     cube.Placement = r_cube_pl
#
#
# def update_cube():
#     global i, cube
#     alpha = radians(i)
#     x = 150.0 * cos(alpha)
#     y = 150.0 * sin(alpha)
#     cube.Placement = FreeCAD.Placement(cube_placement.Base + FreeCAD.Vector(150 - x, 0, 0),
#                                        cube_placement.Rotation)
#     i += 1
#
#
# FreeCAD.animation_reset = reset_cube
# FreeCAD.animation_update = update_cube
# setup()
