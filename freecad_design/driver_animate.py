import FreeCAD
from math import sin, cos, radians
import numpy as np


class RotateCube(object):
    def __init__(self, base_lcs, segment, segment_top_lcs):
        self.init_segment_placement = base_lcs.Placement
        self.i = 0
        self.base_lcs = base_lcs
        self.segment_top_lcs = segment_top_lcs
        self.segment_placement = segment.Placement
        self.segment = segment

        self.setup()
        self.set_freecad_globals()

    def setup(self):
        # n1 = self.base_lcs.Placement.Rotation.Axis
        # zhat = FreeCAD.Vector(0, 0, 1)
        # a = self.base_lcs.Placement.Base
        # rot = FreeCAD.Rotation(n1.cross(zhat), zhat, n1, 'ZXY')
        # pl = FreeCAD.Placement(a, rot)
        # self.segment.Placement = pl
        FreeCAD.animation_count = 0

    def reset_cube(self):  # QTimer requires a thunk
        def reset_cube():
            # function to restore initial position of the objects
            self.segment.Placement = self.init_segment_placement
        return reset_cube

    def update_cube(self):
        def update_cube():
            if self.i > 1.0:
                i = 0
                self.segment.Placement = self.segment_placement
            rot1 = self.base_lcs.Placement.Rotation
            rot2 = self.segment_placement.Rotation
            base1 = self.base_lcs.Placement.Base
            base2 = self.segment_placement.Base
            new_rot = rot2.slerp(rot1, self.i)
            new_base = base1 * self.i + base2 * (1-self.i)
            self.segment.Placement = FreeCAD.Placement(new_base, new_rot)
            print(f"UPDATE: i={self.i}, {np.round(self.segment.Placement.Rotation.toEulerAngles('ZYX'))}")
            print(f"AXIS: i={self.i}, {np.round(self.segment.Placement.Rotation.Axis)}")
            print(f"ANGLE: i={self.i}, {np.round(self.segment.Placement.Rotation.Angle)}")
            print(f"BASE: {np.round(self.segment.Placement.Base / 25.4, 3)}, i: {self.i}")
            self.i += 0.05
        return update_cube

    def set_freecad_globals(self):
        FreeCAD.animation_reset = self.reset_cube()
        FreeCAD.animation_update = self.update_cube()
        FreeCAD.animation_count = 0

# OLD WORKING COPY
# i = 0
# gu = None
# cube = None
# segment_placement = None
# r_cube_pl = None
#
#
# def setup():
#     global cube, segment_placement, r_cube_pl
#     cube = FreeCAD.ActiveDocument.getObjectsByLabel("Cube")[0]
#     # store initial placement (needed to restore initial position)
#     segment_placement = cube.Placement
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
#     cube.Placement = FreeCAD.Placement(segment_placement.Base + FreeCAD.Vector(150 - x, 0, 0),
#                                        segment_placement.Rotation)
#     i += 1
#
#
# FreeCAD.animation_reset = reset_cube
# FreeCAD.animation_update = update_cube
# setup()
