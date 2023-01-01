import numpy as np
import csv
from .wafer import Wafer
from .structurehelix import StructureHelix
import FreeCAD
import FreeCADGui


def position_to_str(x):
    inches = int(x)
    fraction = int((x - inches) * 16)
    return f'{inches:2d}" {fraction:2d}/16'


class Segment(object):
    def __init__(self, prefix, lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count, show_lcs,
                 temp_file, to_build, rotate_segment, trace=None):
        self.prefix = prefix + "_"
        self.trace = trace
        self.lift_angle = np.deg2rad(float(lift_angle))
        self.rotation_angle = np.deg2rad(float(rotation_angle))
        self.outside_height = float(outside_height)
        self.cylinder_diameter = float(cylinder_diameter)
        self.rotate_segment = float(rotate_segment)
        self.wafer_count = int(wafer_count)
        self.wafer_list = []            # This is empty for segments that were reconstructed from the model tree
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
        self.transform_to_top = None  # transform from base of segment t0 top of segment

        self.get_wafer_parameters()
        if self.to_build:
            self.remove_prior_version()

    def get_segment_object(self):
        return self.segment_object

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

    def get_lcs_top(self):
        print(f"LCS TOP: {self.lcs_top.Label}")
        return self.lcs_top

    def get_lcs_base(self):
        if not self.lcs_base:
            raise ValueError(f"lcs_base not set")
        return self.lcs_base

    def get_transform_to_top(self):
        if (self.segment_type == 'helix' and self.helix) or not self.to_build:
            return self.transform_to_top
        else:
            raise ValueError("Segment has no valid transform to top")

    def get_segment_rotation(self):
        return self.rotate_segment

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
                doc.removeObject(item.Name)

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
        self.wafer_list = helix.get_wafer_list()
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
            remove_string = f"{self.prefix}e.+|{self.prefix}we.+"
            doc_list = doc.findObjects(Name=remove_string)  # obj names start with l,e,L,E,f,K
            for item in doc_list:
                try:
                    doc.removeObject(item.Label)
                except:
                    pass

        except Exception as e:
            raise ValueError(f"Failed to Reconstruct Segment: {e.args}")

    def make_cut_list(self, segment_no, cuts_file):
        parm_str = f"\n\nCut list for segment: {segment_no}\n"
        parm_str += f"Lift Angle: {np.round(np.rad2deg(self.lift_angle), 2)} degrees\n"
        parm_str += f"Rotation Angle: {np.rad2deg(self.rotation_angle)} degrees\n"
        parm_str += f"Outside Wafer Height: {np.round(self.outside_height / 25.4, 2)} in\n"
        parm_str += f"Inside Wafer Height: {np.round(self.inside_height / 25.4, 2)} in\n"
        parm_str += f"Cylinder Diameter: {np.round(self.cylinder_diameter / 25.4, 2)} in\n"
        parm_str += f"Helix Radius: \t{np.round(self.helix_radius / 25.4, 2)} in\n"
        cuts_file.write(parm_str)
        cuts_file.write(f"Wafer Count: {self.wafer_count}\n\n")
        try:
            nbr_rotations = int(360 / np.rad2deg(self.rotation_angle))
            step_size = nbr_rotations / 2 - 1
        except Exception as e:
            nbr_rotations = None
            step_size = 0

        current_position = 0  # Angular position of index (0-360) when rotating cylinder for cutting
        s1 = "Step: "
        s2 = " at position: "
        for i in range(self.wafer_count):
            str1 = str(i)
            if len(str1) == 1:
                str1 = s1 + ' ' + str1
            else:
                str1 = s1 + str1
            str2 = str(current_position)
            if len(str2) == 1:
                str2 = s2 + ' ' + str2
            else:
                str2 = s2 + str2
            cuts_file.write(f"{str1} {str2};    Done: _____\n")
            if nbr_rotations:
                current_position = int((current_position + step_size) % nbr_rotations)

    def print_construction_list(self, segment_no, cons_file, global_placement):
        parm_str = f"\nConstruction list for segment: {segment_no}\n"
        parm_str += f"Lift Angle: {np.round(np.rad2deg(self.lift_angle), 2)} degrees\n"
        parm_str += f"Rotation Angle: {np.rad2deg(self.rotation_angle)} degrees\n"
        parm_str += f"Outside Wafer Height: {position_to_str(self.outside_height)} in\n"
        parm_str += f"Inside Wafer Height: {position_to_str(self.inside_height)} in\n"
        parm_str += f"Cylinder Diameter:{position_to_str(self.cylinder_diameter)} in\n"
        parm_str += f"Helix Radius: \t{position_to_str(self.helix_radius)} in\n"
        parm_str += f"Segment Rotation: \t{position_to_str(self.rotate_segment)} in\n"
        cons_file.write(parm_str)
        cons_file.write(f"Wafer Count: {self.wafer_count}\n\n")

        if not self.wafer_list:
            cons_file.write(f"This segment was reconstructed thus there is no wafer list")
            return None
        for wafer_num, wafer in enumerate(self.wafer_list):
            top_lcs_place = wafer.get_top().Placement
            global_loc = global_placement.multiply(top_lcs_place)
            num_str = str(wafer_num)
            local_x = position_to_str(top_lcs_place.Base.x)
            global_x = position_to_str(global_loc.Base.x)
            local_y = position_to_str(top_lcs_place.Base.y)
            global_y = position_to_str(global_loc.Base.y)
            local_z = position_to_str(top_lcs_place.Base.z)
            global_z = position_to_str(global_loc.Base.z)
            str1 = f"Wafer: {num_str}\t at Local: [{local_x:{5}}, {local_y}, {local_z}],"
            str1 += f" \tat Global: [{global_x}, {global_y}, {global_z}]\n"
            cons_file.write(str1)
        return global_loc

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        return tr




