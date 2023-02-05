import numpy as np
import csv
from .wafer import Wafer
from .structurehelix import StructureHelix
import FreeCAD
import FreeCADGui
from .utilities import position_to_str


class FlexSegment(object):
    def __init__(self, prefix,  show_lcs, temp_file, to_build, rotate_segment, trace=None):
        self.doc = FreeCAD.ActiveDocument
        self.gui = FreeCADGui
        self.prefix = prefix + "_"
        self.rotate_segment = rotate_segment
        self.trace = trace
        self.wafer_count = 0
        self.wafer_list = []            # This is empty for segments that were reconstructed from the model tree
        self.show_lcs = show_lcs
        self.temp_file = temp_file
        self.segment_type = None
        self.to_build = to_build    # if False, use existing structure
        self.segment_object = None      # THIS WAS HELIX - DOES NOT EXIST HERE?
        self.transform_to_top = None  # transform from base of segment t0 top of segment
        if self.to_build:
            self.remove_prior_version()
        self.lcs_base = self.doc.addObject('PartDesign::CoordinateSystem', self.prefix + "_lcs_base")
        self.lcs_top = self.doc.addObject('PartDesign::CoordinateSystem', self.prefix + "_lcs_top")
        self.lcs_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "_lcs_group")

    def add_wafer(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE"):
        # Make wafer at base and move after creation.  Creating at the target location seems to confuse OCC
        # causing some wafer to be constructed by lofting to the wrong side of the target ellipse.
        self.wafer_count += 1
        name_base = self.prefix + str(self.wafer_count)
        wafer_name = name_base + "_w"
        wafer = Wafer(FreeCAD, self.gui, self.prefix, wafer_type=wafer_type)
        wafer.set_parameters(lift, rotation, cylinder_diameter, outside_height, wafer_type="EE")
        lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_1lcs")
        lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_2lcs")
        self.lcs_group.addObjects([lcs1, lcs2])
        wafer.lift_lcs(lcs2, wafer_type)
        matrix = lcs2.Placement.toMatrix()
        matrix.rotateZ(-rotation)
        lcs2.Placement = FreeCAD.Placement(matrix)
        if not self.show_lcs:
            lcs1.Visibility = False
            lcs2.Visibility = False
        wafer.make_wafer_from_lcs(lcs1, lcs2, cylinder_diameter, wafer_name)
        # print(f"Wafer {wafer_name} angle (top ellipse) to X-Y plane: {np.round(wafer.get_angle(), 3)}")

        lcs1.Placement = self.lcs_top.Placement
        lcs2.Placement = self.lcs_top.Placement.multiply(lcs2.Placement)
        wafer_loft = wafer.get_wafer()
        wafer_loft.Placement = lcs1.Placement
        self.wafer_list.append(wafer)
        self.lcs_top.Placement = lcs2.Placement

    def get_segment_name(self):
        return self.prefix[:-1]    # strip trailing underscore

    def get_segment_object(self):
        return self.segment_object

    def get_wafer_count(self):
        return self.wafer_count

    def get_wafer_parameters(self):
        raise NotImplementedError(f"need to identify specific wafer and get from there")
        if self.lift_angle != 0:  # leave as simple cylinder if zero
            la = self.lift_angle / 2
            oh = self.outside_height / 2
            # Assume origin at center of ellipse, x-axis along major axis, positive to outside.
            self.helix_radius = oh / np.math.tan(la)
            # print(f"SET RADIUS: {self.helix_radius}, Lift: {self.lift_angle}, Height: {self.outside_height}")
            self.inside_height = (self.helix_radius - self.cylinder_diameter) * np.math.tan(la) * 2

    def get_lcs_top(self):
        # print(f"LCS TOP: {self.lcs_top.Label}")
        return self.lcs_top

    def get_lcs_base(self):
        if not self.lcs_base:
            raise ValueError(f"lcs_base not set")
        return self.lcs_base

    def get_transform_to_top(self):    # Does this need changing
        if self.to_build:
            if not self.transform_to_top:
                # print(f"TO TOP: Base: {self.lcs_base.Placement}. Top: {self.lcs_top.Placement}")
                self.transform_to_top = self.make_transform_align(self.lcs_base, self.lcs_top)
            return self.transform_to_top
        else:
            print(f"NO TRANSFORM: {self.get_segment_name()}, BUILD? {self.to_build}")
            raise ValueError("Segment has no valid transform to top as it was created in prior run.")

    def fuse_wafers(self):
        name = self.get_segment_name()
        if len(self.wafer_list) > 1:
            fuse = self.doc.addObject("Part::MultiFuse", name + "FusedResult")
            fuse.Shapes = [x.wafer for x in self.wafer_list]
        elif len(self.wafer_list) == 1:
            fuse = self.wafer_list[0].wafer
            fuse.Label = name + "FusedResult"
        else:
            raise ValueError("Zero Length Wafer List when building helix")
        fuse.Visibility = True
        fuse.ViewObject.DisplayMode = "Shaded"
        fuse.Placement = self.lcs_base.Placement
        self.segment_object = fuse

    def get_segment_rotation(self):
        return self.rotate_segment

    def move_content(self, transform):
        pl = self.segment_object.Placement
        self.segment_object.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.lcs_top.Placement
        self.lcs_top.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.lcs_base.Placement
        self.lcs_base.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        # print(f"LABELS: {self.segment_object.Label}, {self.lcs_base.Label}, {self.lcs_top.Label}")
        # self.trace("MOVE", self.prefix, self.lcs_top.Label, self.lcs_top.Placement)

    def move_content_to_zero(self, transform):
        """Relocate to a zero base corresponding to a new build.  Transform is lcs_base.inverse()"""
        self.move_content(transform)

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        self.move_content(transform)

    def remove_prior_version(self):
        # TODO: not do so if making cut list???
        name = self.prefix + ".+"
        doc = FreeCAD.activeDocument()
        doc_list = doc.findObjects(Name=name)  # remove prior occurrence of set being built
        for item in doc_list:
            if item.Label != 'Parms_Master':
                doc.removeObject(item.Name)

    # def build_helix(self):
    #     position_offset = 0         # TODO: Remove???
    #     minor_radius = (self.cylinder_diameter / 2)
    #     major_radius = (self.cylinder_diameter / 2) / np.cos(self.lift_angle)
    #     # print(f"Major: {major_radius}, Minor: {minor_radius}, Lift: {np.rad2deg(self.lift_angle)}")
    #     helix = StructureHelix(FreeCAD, FreeCADGui, self, self.prefix, self.temp_file, position_offset)
    #     # !!!! helix.add_segment(outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count)
    #     helix.write_instructions()
    #     fused_result, last_loc = helix.create_structure(major_radius, minor_radius, self.temp_file, self.show_lcs)
    #     self.helix = helix
    #     self.segment_type = "helix"
    #     fuse, base, top, transform = helix.get_segment_objects()
    #     self.segment_object = fuse
    #     self.lcs_base = base
    #     self.lcs_top = top
    #     self.lcs_top.Visibility = False
    #     self.transform_to_top = transform
    #     self.wafer_list = helix.get_wafer_list()
    #     if self.trace:
    #         self.trace("BUILD", self.prefix, "base", self.lcs_base.Placement, "top", self.lcs_top.Placement)
    #     return helix
    #
    # def reconstruct_helix(self):
    #     """Reconstruct existing helix from model tree."""
    #     try:
    #         self.segment_type = "existing"
    #         doc = FreeCAD.activeDocument()
    #         self.lcs_top = doc.getObjectsByLabel(self.prefix + "lcs_top")[0]
    #         self.lcs_base = doc.getObjectsByLabel(self.prefix + "lcs_base")[0]
    #         self.segment_object = doc.getObjectsByLabel(self.prefix + "FusedResult")[0]
    #         self.transform_to_top = Segment.make_transform_align(self.lcs_base, self.lcs_top)
    #         self.move_content_to_zero(self.lcs_base.Placement.inverse())
    #         if self.trace:
    #             self.trace("RE-BUILD", self.prefix, "base", self.lcs_base.Placement, "top", self.lcs_top.Placement)
    #         remove_string = f"{self.prefix}e.+|{self.prefix}we.+"
    #         doc_list = doc.findObjects(Name=remove_string)  # obj names start with l,e,L,E,f,K
    #         for item in doc_list:
    #             try:
    #                 doc.removeObject(item.Label)
    #             except:
    #                 pass
    #
    #     except Exception as e:
    #         raise ValueError(f"Failed to Reconstruct Segment: {e.args}")

    def make_cut_list(self, segment_no, cuts_file):
        parm_str = f"\n\nCut list for segment: {segment_no}\n"
        # parm_str += f"Lift Angle: {np.round(np.rad2deg(self.lift_angle), 2)} degrees\n"
        # parm_str += f"Rotation Angle: {np.rad2deg(self.rotation_angle)} degrees\n"
        # parm_str += f"Outside Wafer Height: {np.round(self.outside_height, 2)} in\n"
        # if self.inside_height:
        #     parm_str += f"Inside Wafer Height: {np.round(self.inside_height, 2)} in\n"
        # else:
        #     parm_str += f"Inside Wafer Height: NONE\n"
        # parm_str += f"Cylinder Diameter: {np.round(self.cylinder_diameter, 2)} in\n"
        # if self.helix_radius:
        #     parm_str += f"Helix Radius: \t{np.round(self.helix_radius, 2)} in\n"
        # else:
        #     parm_str += f"Helix Radius: NONE\n"
        # cuts_file.write(parm_str)
        # cuts_file.write(f"Wafer Count: {self.wafer_count}\n\n")
        # try:
        #     step_size = np.rad2deg(self.rotation_angle)
        # except Exception as e:
        #     nbr_rotations = None
        #     step_size = 0

        current_position = 0  # Angular position of index (0-360) when rotating cylinder for cutting
        s1 = "Step: "
        s2 = " at position: "
        for i, wafer in enumerate(self.wafer_list):
            ra = wafer.get_rotation_angle()
            if ra is not None:
                ra = np.round(np.rad2deg(ra), 2)
            else:
                ra = 0
            la = wafer.get_lift_angle()
            if la:
                la = np.round(np.rad2deg(la), 2)
            else:
                la = "NA"
            oh = wafer.get_outside_height()
            if oh:
                oh = np.round(oh, 2)
            else:
                oh = "NA"
            str1 = str(i)
            if len(str1) == 1:
                str1 = s1 + ' ' + str1      # account for number of steps (max <= 99)
            else:
                str1 = s1 + str1
            str2 = str(current_position)
            if len(str2) == 1:
                str2 = s2 + ' ' + str2
            else:
                str2 = s2 + str2
            str3 = f"\tLift: {la}\tRotate: {ra}\tHeight: {oh}"
            cuts_file.write(f"{str1} {str2} {str3};    Done: _____\n")
            current_position = int((current_position - ra + 180) % 360)

    def print_construction_list(self, segment_no, cons_file, global_placement, find_min_max):
        parm_str = f"\nConstruction list for segment: {segment_no}\n"
        # if self.inside_height:
        #     parm_str += f"Inside Wafer Height: {position_to_str(self.inside_height)} in\n"
        # else:
        #     parm_str += f"Inside Wafer Height: NONE\n"
        # parm_str += f"Cylinder Diameter:{position_to_str(self.cylinder_diameter)} in\n"
        parm_str += f"Segment Rotation: \t{position_to_str(self.rotate_segment)} in\n"
        parm_str += f"\n\tNote 'angle' below is angle of top wafer surface to X-Y plane.\n"
        cons_file.write(parm_str)
        cons_file.write(f"Wafer Count: {self.wafer_count}\n\n")

        if not self.wafer_list:
            cons_file.write(f"This segment was reconstructed thus there is no wafer list")
            return None
        for wafer_num, wafer in enumerate(self.wafer_list):
            # if wafer_num:
            #     prior_top = top_lcs_place
            top_lcs_place = wafer.get_top().Placement
            global_loc = global_placement.multiply(top_lcs_place)
            num_str = str(wafer_num + 1)       # Make one-based for conformance with  in shop
            local_x = position_to_str(top_lcs_place.Base.x)
            global_x = position_to_str(global_loc.Base.x)
            local_y = position_to_str(top_lcs_place.Base.y)
            global_y = position_to_str(global_loc.Base.y)
            local_z = position_to_str(top_lcs_place.Base.z)
            global_z = position_to_str(global_loc.Base.z)
            find_min_max(global_loc.Base)
            # Seems to be a long way around to get the lcs z-axis's vector
            lcso = FreeCAD.activeDocument().getObject(wafer.get_lcs_top().Label)
            vec = FreeCAD.Vector(0, 0, 1)
            vec2 = lcso.getGlobalPlacement().Rotation.multVec(vec)
            angle = vec2.getAngle(vec)
            str1 = f"Wafer: {num_str}\tat Local:  [{local_x:{5}}, {local_y}, {local_z}],"
            str1 += f" with Angle: {np.round(np.rad2deg(angle), 1)}\n"
            str1 += f"\t\tat Global: [{global_x}, {global_y}, {global_z}]\n"
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




