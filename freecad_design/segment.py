import numpy as np
import csv
from .wafer import Wafer
from .structurehelix import StructureHelix
import FreeCAD
import FreeCADGui


class Segment(object):
    def __init__(self, prefix, lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count, show_lcs,
                 temp_file, to_build, trace=None):
        self.prefix = prefix + "_"
        self.trace = trace
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
        self.to_build = to_build    # if False, use existing structure

        self.segment_object = None
        self.lcs_base = None
        self.lcs_top = None
        self.transform_to_top = None

        self.get_wafer_parameters()
        if self.to_build:
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
        if (self.segment_type == 'helix' and self.helix) or not self.to_build:
            return self.transform_to_top
        else:
            raise ValueError("Segment has no valid transform to top")

    def move_content(self, transform):
        # return
        pl = self.segment_object.Placement
        self.segment_object.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.lcs_top.Placement
        self.lcs_top.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.lcs_base.Placement
        self.lcs_base.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        # print(f"LABELS: {self.segment_object.Label}, {self.lcs_base.Label}, {self.lcs_top.Label}")
        self.trace("MOVE", self.prefix, self.lcs_top.Label, self.lcs_top.Placement)

    def move_content_to_zero(self, transform):
        """Relocate to a zero base corresponding to a new build.  Transform is lcs_base.inverse()"""
        self.move_content(transform)

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        if (self.segment_type == 'helix' and self.helix) or not self.to_build:
            # print(f"BEFORE: {self.segment_object.Label}, {self.segment_object.Placement}")
            self.move_content(transform)
            # print(f"AFTER: {self.segment_object.Label}, {self.segment_object.Placement}")
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
        fuse, base, top, transform = helix.get_segment_objects()
        self.segment_object = fuse
        self.lcs_base = base
        self.lcs_top = top
        self.transform_to_top = transform
        if self.trace:
            self.trace("BUILD", self.prefix, "base", self.lcs_base.Placement, "top", self.lcs_top.Placement)
        return helix

    def reconstruct_helix(self):
        """Reconstruct existing helix from model tree."""
        try:
            self.segment_type = "existing"
            doc = FreeCAD.activeDocument()
            self.lcs_top = doc.getObjectsByLabel(self.prefix + "lcs_top")[0]
            self.lcs_base = doc.getObjectsByLabel(self.prefix + "lcs_base")[0]
            self.segment_object = doc.getObjectsByLabel(self.prefix + "FusedResult")[0]
            self.transform_to_top = Segment.make_transform_align(self.lcs_base, self.lcs_top)
            self.move_content_to_zero(self.lcs_base.Placement.inverse())
            if self.trace:
                self.trace("RE-BUILD", self.prefix, "base", self.lcs_base.Placement, "top", self.lcs_top.Placement)

        except Exception as e:
            raise ValueError(f"Failed to Reconstruct Segment: {e.args}")

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        return tr




