import numpy as np
import csv
# from .wafer import lift_lcs
from .wafer import Wafer
import re
import Part
import FreeCAD
from . import utilities

class StructureHelix(object):
    def __init__(self, App, Gui, parent_segment, parm_set, lcs_file_name, position_offset, trace=None):
        self.gui = Gui
        self.doc = App.ActiveDocument
        self.parent_segment = parent_segment
        self.trace = trace
        self.parm_set = parm_set
        self.position_offset = position_offset      # displacement on x axis of all elements
        self.lcs_file = open(lcs_file_name, "w+")
        self.lcs_writer = csv.writer(self.lcs_file)
        self.lcs_group = self.doc.addObject("App::DocumentObjectGroup", parm_set + "All_LCS")

        ps = self.parent_segment
        self.wafer_count = ps.get_wafer_count()
        self.lift_angle = ps.get_lift_angle()
        self.rotation_angle = ps.get_rotation_angle()
        self.helix_radius = ps.get_helix_radius()
        self.cylinder_diameter = ps.get_cylinder_diameter()
        self.outside_height = ps.get_outside_height()
        # The assumption is that a single Structure creates a single fused result such as a helix
        # There is also an LCS that is the last location of the structure (position of the top ellipse
        # for a helix).
        self.result = None
        self.wafer_list = []
        self.result_LCS_base = None
        self.result_LCS_top = None
        self.named_result_LCS_top = None        # LCS with segment name appended
        self.named_result_LCS_base = None
        self.transform_to_top = None   # transform that will move LCS at base to conform to LCS at top

    def write_instructions(self):
        # In principle, this should not be necessary, but combining with the actual construction seems to
        # have some sort of interaction at a lower level of structure.  This separates the specification of the
        # placements from the construction of the wafers themselves.
        lcs_temp = self.doc.addObject("PartDesign::CoordinateSystem", self.parm_set + "LCS_Global")  # Used to write file of positions
        lcs_temp.Visibility = False
        self.lcs_group.addObjects([lcs_temp])
        total_wafer_count = -1
        wc = self.parent_segment.get_wafer_count()
        for j in range(wc):
            # print(f"LA: {np.rad2deg(segment.lift_angle)}, RA: {np.rad2deg(segment.rotation_angle)}")
            # for wafers with twist between wafers rather than within wafer, change position of rotation below.
            total_wafer_count += 1
            e_name = 'e' + str(total_wafer_count)
            self.lcs_writer.writerow([e_name, lcs_temp.Placement])
            e_name += '_top'
            # print(f"LIFT in WRITE: lift angle - {np.round(np.rad2deg(self.lift_angle), 2)}, j - {j}")
            if j == 0:
                # lift_lcs(lcs_temp, 0, self.helix_radius, self.outside_height / 2)   # Presume cylinder for half
                lift_lcs(lcs_temp, self.lift_angle, self.cylinder_diameter, self.outside_height, "CE")
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle)    # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "CE"])
            elif j == wc - 1:
                lift_lcs(lcs_temp, self.lift_angle, self.cylinder_diameter, self.outside_height, "EC")
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle)  # This makes rotation occur within a wafer
                # lift_lcs(lcs_temp, 0, self.helix_radius, self.outside_height / 2)
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "EC"])
            else:
                lift_lcs(lcs_temp, self.lift_angle, self.cylinder_diameter, self.outside_height, "EE")
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle)  # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "EE"])
            # lcso = self.doc.getObject(lcs_temp.Label)        # Debug - checking actual lift angle
            # vec = FreeCAD.Vector(0, 0, 1)
            # vec2 = lcso.getGlobalPlacement().Rotation.multVec(vec)
            # angle = vec2.getAngle(vec)
            # print(f"WRITE_LCS: Input: {np.round(np.rad2deg(self.lift_angle),2)}, lift_angle - {np.round(np.rad2deg(angle),2)}, radius - {np.round(self.helix_radius, 2)}")

        self.lcs_writer.writerow(["Done", "Done"])
        self.lcs_file.close()

    def create_structure(self, major_radius, minor_radius, csv_file, show_lcs):
        """Create structure from I/O stream or file"""
        with open(csv_file, newline='') as infile:
            reader = csv.reader(infile)
            last_location = None
            first_location = None
            while True:
                p1, p1_place = next(reader)
                if p1 == "Done":
                    break
                place1 = self.make_placement(p1_place)
                p2, p2_place, p2_type = next(reader)
                place2 = self.make_placement(p2_place)
                lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', self.parm_set + p2 + "lcs")
                lcs2.Placement = place2
                lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', self.parm_set + p1 + "lcs")
                self.lcs_group.addObjects([lcs1, lcs2])
                if not first_location:
                    first_location = lcs1
                lcs1.Placement = place1
                if not show_lcs:
                    lcs1.Visibility = False
                    lcs2.Visibility = False
                wafer_name = self.parm_set + "w" + p1
                wafer = Wafer(FreeCAD, self.gui, self.parm_set, wafer_type=p2_type)
                wafer.make_wafer_from_lcs(lcs1, lcs2, minor_radius, wafer_name)
                # print(f"Wafer {wafer_name} angle to X-Y plane: {wafer.get_angle()}")
                self.wafer_list.append(wafer)
            if not first_location:
                raise ValueError(f"File {csv_file} apparently missing content")
            if len(self.wafer_list) > 1:
                fuse = self.doc.addObject("Part::MultiFuse", self.parm_set + "FusedResult")
                fuse.Shapes = [x.wafer for x in self.wafer_list]
            elif len(self.wafer_list) == 1:
                fuse = self.wafer_list[0].wafer
                fuse.Label = self.parm_set + "FusedResult"
            else:
                raise ValueError("Zero Length Wafer List when building helix")
            fuse.Visibility = True
            fuse.ViewObject.DisplayMode = "Shaded"
            fuse.Placement = first_location.Placement  # Places segment at location of first wafer
            FreeCAD.activeDocument().recompute()
            self.result = fuse
            self.named_result_LCS_top = self.doc.addObject('PartDesign::CoordinateSystem', self.parm_set + "lcs_top")
            self.named_result_LCS_top.Placement = lcs2.Placement
            self.named_result_LCS_top.Visibility = False
            self.result_LCS_top = lcs2
            self.result_LCS_base = first_location
            new_name = self.parm_set + "lcs_base"
            self.named_result_LCS_base = self.doc.addObject('PartDesign::CoordinateSystem', new_name)
            self.named_result_LCS_base.Placement = first_location.Placement
            self.named_result_LCS_base.Visibility = False
            self.transform_to_top = StructureHelix.make_transform_align(self.named_result_LCS_base, self.named_result_LCS_top)
            return fuse, last_location

    def get_segment_objects(self):
        return self.result, self.named_result_LCS_base, self.named_result_LCS_top, self.transform_to_top

    def get_wafer_list(self):
        return self.wafer_list

    def make_placement(self, place_str):
        """Create a placement as read from file."""
        vectors = re.findall(r'\(.+?\)', place_str)
        if len(vectors) < 2:
            print(f"FOUND BAD PLACEMENT: {place_str}")
            return
        pos = eval("FreeCAD.Vector" + vectors[0])
        pos[0] += self.position_offset
        rot = eval("FreeCAD.Rotation" + vectors[1])
        newplace = FreeCAD.Placement(pos, rot)
        return newplace

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        # make available to console
        return tr

