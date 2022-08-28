import numpy as np
import csv
from .wafer import Wafer


class Segment(object):
    def __init__(self, lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count):
        self.lift_angle = lift_angle
        self.rotation_angle = rotation_angle
        self.outside_height = outside_height
        self.cylinder_diameter = cylinder_diameter
        self.wafer_count = wafer_count
        self.helix_radius = None
        self.inside_height = None
        self.get_wafer_parameters()

    def get_wafer_parameters(self):
        if self.lift_angle != 0:  # leave as simple cylinder if zero
            # Assume origin at center of ellipse, x-axis along major axis, positive to outside.
            self.helix_radius = np.math.cos(self.lift_angle) / np.math.sin(self.lift_angle) * self.outside_height
            self.inside_height = (self.helix_radius - self.cylinder_diameter) * np.math.tan(self.lift_angle)