import utilities
import numpy as np


class MakeHelix(object):
    def __init__(self, segment):
        self.segment = segment

        self.outside_height = None
        self.wafer_diameter = None
        self.lift_angle = None
        self.rotation_angle = None
        self.wafer_count = 0

    def create_helix(self, wafer_count, wafer_diameter, outside_height, lift_angle, rotation_angle, label_prefix):
        self.wafer_diameter = wafer_diameter
        self.outside_height = outside_height
        self.wafer_count = wafer_count
        self.lift_angle = lift_angle
        self.rotation_angle = rotation_angle
        lift = np.deg2rad(self.lift_angle)
        rotate = np.deg2rad(self.rotation_angle)
        if wafer_count < 3:
            raise ValueError(f"Wafer count of {wafer_count} is insufficient to build a helix.  Must be 3 or more.")
        self.segment.add_wafer(lift / 2, rotate, self.wafer_diameter, self.outside_height, wafer_type="CE")
        for _ in range(wafer_count - 2):
            self.segment.add_wafer(lift, rotate, self.wafer_diameter, self.outside_height, wafer_type="EE")
        self.segment.add_wafer(lift / 2, rotate, self.wafer_diameter, self.outside_height, wafer_type="EC")

        if self.segment.wafer_count > 0:
            self.segment.fuse_wafers()


