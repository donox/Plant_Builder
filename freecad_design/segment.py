import numpy as np
import csv
from .wafer import Wafer
from .structurehelix import StructureHelix
import FreeCAD
import FreeCADGui



class Segment(object):
    def __init__(self, prefix, lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count, show_lcs,
                 temp_file):
        self.prefix = prefix + "_"
        self.lift_angle = np.deg2rad(float(lift_angle))
        self.rotation_angle = np.deg2rad(float(rotation_angle))
        self.outside_height = float(outside_height)
        self.cylinder_diameter = float(cylinder_diameter)
        self.wafer_count = int(wafer_count)
        self.helix_radius = None
        self.inside_height = None
        self.show_lcs = show_lcs
        self.temp_file = temp_file
        self.helix = None
        self.segment_type = None

        self.get_wafer_parameters()
        self.remove_prior_version()

    def get_wafer_count(self):
        return self.wafer_count

    def get_lift_angle(self):
        return self.lift_angle

    def get_rotation_angle(self):
        return self.rotation_angle

    def get_outside_height(self):
        return self.outside_height

    def get_helix_radius(self):
        return self.helix_radius

    def get_inside_height(self):
        return self.inside_height

    def get_cylinder_diameter(self):
        return self.cylinder_diameter

    def get_wafer_parameters(self):
        if self.lift_angle != 0:  # leave as simple cylinder if zero
            # Assume origin at center of ellipse, x-axis along major axis, positive to outside.
            self.helix_radius = np.math.cos(self.lift_angle) / np.math.sin(self.lift_angle) * self.outside_height
            # print(f"SET RADIUS: {self.helix_radius}, Lift: {self.lift_angle}, Height: {self.outside_height}")
            self.inside_height = (self.helix_radius - self.cylinder_diameter) * np.math.tan(self.lift_angle)

    def get_transform_to_top(self):
        if self.segment_type == 'helix' and self.helix:
            return self.helix.transform_to_top
        else:
            raise ValueError("Segment has no valid transform to top")

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        if self.segment_type == 'helix' and self.helix:
            self.helix.move_content(transform)
        else:
            raise ValueError("Segment has no valid content to transform")

    def remove_prior_version(self):
        # TODO: not do so if making cut list???
        name = self.prefix + ".+"
        doc = FreeCAD.activeDocument()
        doc_list = doc.findObjects(Name=name)  # remove prior occurrence of set being built
        for item in doc_list:
            if item.Label != 'Parms_Master':
                doc.removeObject(item.Label)

    def build_helix(self):
        position_offset = 0         # TODO: Remove???
        minor_radius = (self.cylinder_diameter / 2)
        major_radius = (self.cylinder_diameter / 2) / np.cos(self.lift_angle)
        # print(f"Major: {major_radius}, Minor: {minor_radius}, Lift: {np.rad2deg(self.lift_angle)}")
        helix = StructureHelix(FreeCAD, FreeCADGui, self, self.prefix, self.temp_file, position_offset)
        # !!!! helix.add_segment(outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count)
        helix.write_instructions()
        fused_result, last_loc = helix.create_structure(major_radius, minor_radius, self.temp_file, self.show_lcs)
        self.helix = helix
        self.segment_type = "helix"
        return helix


