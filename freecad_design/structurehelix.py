import numpy as np
import csv
from .wafer import Wafer
from .segmentold import SegmentOLD
from .wafer import Wafer
import re
import Part
import FreeCAD



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

        from .driver import Driver

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
            if j == 0:
                self.lift_lcs(lcs_temp, self.lift_angle/2, self.helix_radius, self.outside_height)
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle/2)    # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "CE"])
            elif j == wc - 1:
                self.lift_lcs(lcs_temp, self.lift_angle/2, self.helix_radius, self.outside_height)
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle/2)  # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "EC"])
            else:
                self.lift_lcs(lcs_temp, self.lift_angle, self.helix_radius, self.outside_height)
                lcs_temp.Placement.Matrix.rotateZ(self.rotation_angle)  # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement, "EE"])

        self.lcs_writer.writerow(["Done", "Done"])
        self.lcs_file.close()

    def lift_lcs(self, lcs, lift_angle, helix_radius, outside_height):
        # print(f"LIFT ANGLE: {lift_angle}, {helix_radius}, {outside_height}")
        if lift_angle == 0:
            lcs.Placement.Base = lcs.Placement.Base + FreeCAD.Vector(0, 0, outside_height)
            return
        translate_vector = FreeCAD.Vector(-helix_radius, 0, 0)
        lcs.Placement.Base = lcs.Placement.Base + translate_vector
        pm = lcs.Placement.toMatrix()
        pm.rotateY(lift_angle)
        lcs.Placement = FreeCAD.Placement(pm)
        lcs.Placement.Base = lcs.Placement.Base - translate_vector

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
            self.named_result_LCS_top.Visibility = True             # !!!!!!!!!!!!!!!!!!!!!
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

    def move_content(self, transform):
        pl = self.result.Placement
        self.result.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.named_result_LCS_top.Placement
        self.named_result_LCS_top.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
        pl = self.named_result_LCS_base.Placement
        self.named_result_LCS_base.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)

    def make_cut_list(self, cuts_file):
        # TODO: THIS HAS TO BE BASED ON SEGMENTS
        print(f"Cuts Made")
        cuts_file.write("Cutting order:\n\n\n")
        parm_str = f"Lift Angle: {np.rad2deg(self.lift_angle)} degrees\n"
        parm_str += f"Rotation Angle: {np.rad2deg(self.rotation_angle)} degrees\n"
        parm_str += f"Outside Wafer Height: {np.round(self.outside_height / 25.4, 2)} in\n"
        parm_str += f"Inside Wafer Height: {np.round(self.inside_height / 25.4, 2)} in\n"
        parm_str += f"Cylinder Diameter: {np.round(self.cylinder_diameter / 25.4, 2)} in\n"
        parm_str += f"Helix Radius: \t{np.round(self.helix_radius / 25.4, 2)} in\n"
        cuts_file.write(parm_str)
        cuts_file.write(f"Wafer Count: {self.wafer_count}\n\n")
        nbr_rotations = int(360 / np.rad2deg(self.rotation_angle))
        step_size = nbr_rotations / 2 - 1
        current_position = 0
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
            self.cuts_file.write(f"{str1} {str2};    Done: _____\n")
            current_position = int((current_position + step_size) % nbr_rotations)
        self.cuts_file.close()

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

    def make_ellipse(self, name, major_radius, minor_radius, lcs, trace_file, show_lcs):
        """
        Make ellipse inclined and rotated compared to another ellipse (e_prior).

        :param App: FreeCAD application
        :param name: str -> name of created ellipse
        :param major_radius: float -> ellipse major axis / 2
        :param minor_radius:  float -> ellipse minor axis / 2
        :param lcs: properly oriented lcs at center of ellipse
        :param trace_file: file_ptr -> open file pointer for tracing or None if no tracing
        :param show_lcs:  boolean, add an LCS on top surface of ellipse
        :return: resultant ellipse
        """
        e2 = FreeCAD.activeDocument().addObject('Part::Ellipse', self.parm_set + name)
        e2.MajorRadius = major_radius
        e2.MinorRadius = minor_radius
        e2.Placement = lcs.Placement
        e2.Visibility = False
        if show_lcs and not name.endswith('top'):
            e2_ctr = FreeCAD.activeDocument().addObject('PartDesign::CoordinateSystem',  self.parm_set + name + "lcs")
            e2_ctr.Placement = e2.Placement
        return e2

    def rotate_vertically(self):
        # Set top location before rotation, but relocate to account for position of base moved to 0,0,0
        # print_placement("LCS_TOP (before)", self.result_LCS_top)
        box_x = self.result_LCS_top.Placement.Base.x - self.result_LCS_base.Placement.Base.x
        box_y = self.result_LCS_top.Placement.Base.y - self.result_LCS_base.Placement.Base.y
        box_z = self.result_LCS_top.Placement.Base.z - self.result_LCS_base.Placement.Base.z

        # Rotate result to place vertically on each axis.  This requires only two rotations in x and y.
        # Unclear why np.pi is different for x and y axes.
        x_ang = np.pi / 2 - np.arcsin(box_z / np.sqrt(box_z ** 2 + box_y ** 2))
        # self.result.Placement.Matrix.rotateX(x_ang)
        self.result_LCS_top.Placement.Matrix.rotateX(x_ang)
        self.named_result_LCS_top.Placement.Matrix.rotateX(x_ang)
        self.result_LCS_base.Placement.Matrix.rotateX(x_ang)

        box_x = self.result_LCS_top.Placement.Base.x - self.result_LCS_base.Placement.Base.x
        box_z = self.result_LCS_top.Placement.Base.z - self.result_LCS_base.Placement.Base.z
        y_ang = np.arcsin(box_z / np.sqrt(box_z ** 2 + box_x ** 2)) - np.pi / 2
        # self.result.Placement.Matrix.rotateY(y_ang)
        self.result_LCS_top.Placement.Matrix.rotateY(y_ang)
        self.named_result_LCS_top.Placement.Matrix.rotateY(y_ang)
        self.result_LCS_base.Placement.Matrix.rotateY(y_ang)

        # print(f"In Structure: {self.result.Placement}")
        print(f"Rotate X: {np.rad2deg(x_ang)}, Y: {np.rad2deg(y_ang)}")

        for wafer in self.wafer_list:
            wafer.rotate_to_vertical(x_ang, y_ang)
        FreeCAD.ang = (np.rad2deg(x_ang), np.rad2deg(y_ang))

    def rotate_about_axis(self, obj, v1, v2, angle, rot_center):
        """Rotate an object about an axis defined by two vectors by a specified angle. """
        axis = FreeCAD.Vector(v2 - v1)
        line = Part.LineSegment()
        line.StartPoint = v1
        line.EndPoint = v2
        objln = self.doc.addObject("Part::Feature", self.parm_set + "Line")
        objln.Shape = line.toShape()
        objln.ViewObject.LineColor = (204.0, 170.0, 34.0)
        objln2 = self.doc.addObject("Part::Feature", self.parm_set + "Line")
        objln2.Shape = line.toShape()
        objln2.ViewObject.LineColor = (104.0, 100.0, 134.0)

        rot = FreeCAD.Rotation(axis, angle)
        obj_base = obj.Placement.Base
        print(f"Obj: {obj_base}, CTR: {rot_center}")
        new_place = FreeCAD.Placement(obj_base, rot, rot_center)
        obj.Placement = new_place

        objln.Placement = new_place
        rot = FreeCAD.Rotation(axis, 15)
        obj_base = obj.Placement.Base
        print(f"Obj: {obj_base}, CTR2: {rot_center}")
        new_place = FreeCAD.Placement(obj_base, rot, rot_center)
        objln2.Placement = new_place

        self.doc.recompute()

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        # make available to console
        return tr

