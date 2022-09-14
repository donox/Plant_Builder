import numpy as np
import csv
from .wafer import Wafer
from .segment import Segment
from .wafer import Wafer
import re
import Part


def structure_dummy():
    # print(f"structure_dummy called")
    return


class Structure_Helix(object):
    def __init__(self, App, lcs_file_name):
        self.app = App
        self.doc = App.ActiveDocument
        self.lcs_file = open(lcs_file_name, "w+")
        self.lcs_writer = csv.writer(self.lcs_file)
        self.segment_list = []
        self.lcs_group = self.doc.addObject("App::DocumentObjectGroup", "All LCS")
        # The assumption is that a single Structure creates a single fused result such as a helix
        # There is also an LCS that is the last location of the structure (position of the top ellipse
        # for a helix).
        self.result = None
        self.wafer_list = []
        self.result_LCS_base = None
        self.result_LCS_top = None

    def add_segment(self, outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count):
        seg = Segment(lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count)
        self.segment_list.append(seg)

    def write_instructions(self):
        lcs_temp = self.doc.addObject('PartDesign::CoordinateSystem', 'LCS')  # Used to write file of positions
        lcs_temp.Visibility = False
        self.lcs_group.addObjects([lcs_temp])
        total_wafer_count = -1

        for segment in self.segment_list:
            for j in range(segment.wafer_count):
                # print(f"LA: {np.rad2deg(segment.lift_angle)}, RA: {np.rad2deg(segment.rotation_angle)}")
                # for wafers with twist between wafers rather than within wafer, change position of rotation below.
                total_wafer_count += 1
                e_name = 'e' + str(total_wafer_count)
                self.lcs_writer.writerow([e_name, lcs_temp.Placement])
                e_name += '_top'
                self.lift_lcs(lcs_temp, segment.lift_angle, segment.helix_radius, segment.outside_height)
                lcs_temp.Placement.Matrix.rotateZ(segment.rotation_angle)    # This makes rotation occur within a wafer
                self.lcs_writer.writerow([e_name, lcs_temp.Placement])
        self.lcs_writer.writerow(["Done", "Done"])
        self.lcs_file.close()

    def lift_lcs(self, lcs, lift_angle, helix_radius, outside_height):
        if lift_angle == 0:
            lcs.Placement.Base = lcs.Placement.Base + self.app.Vector(0, 0, outside_height)
            return
        translate_vector = self.app.Vector(-helix_radius, 0, 0)
        lcs.Placement.Base = lcs.Placement.Base + translate_vector
        pm = lcs.Placement.toMatrix()
        pm.rotateY(lift_angle)
        lcs.Placement = self.app.Placement(pm)
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
                p2, p2_place = next(reader)
                place2 = self.make_placement(p2_place)
                lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', p2 + "lcs")
                lcs2.Placement = place2
                lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', p1 + "lcs")
                self.lcs_group.addObjects([lcs1, lcs2])
                if not first_location:
                    first_location = lcs1
                lcs1.Placement = place1
                if not show_lcs:
                    lcs1.Visibility = False
                    lcs2.Visibility = False
                wafer_name = "w" + p1
                wafer = Wafer(self.app)
                wafer.make_wafer_from_lcs(lcs1, lcs2, minor_radius, wafer_name)
                print(f"Wafer {wafer_name} angle to X-Y plane: {wafer.get_angle()}")
                self.wafer_list.append(wafer)
            fuse = self.doc.addObject("Part::MultiFuse", "FusedResult")
            fuse.Shapes = [x.wafer for x in self.wafer_list]
            fuse.Visibility = True
            fuse.ViewObject.DisplayMode = "Shaded"
            self.app.activeDocument().recompute()
            self.result = fuse
            self.result_LCS_top = lcs2
            self.result_LCS_base = first_location
            return fuse, last_location

    def make_cut_list(self, cuts_file):
        # TODO: THIS HAS TO BE BASED ON SEGMENTS
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
        vectors = re.findall(r'\(.+?\)', place_str)
        if len(vectors) < 2:
            print(f"FOUND BAD PLACEMENT: {place_str}")
            return
        pos = eval("self.app.Vector" + vectors[0])
        rot = eval("self.app.Rotation" + vectors[1])
        newplace = self.app.Placement(pos, rot)
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
        :return: resultant ellipse
        """
        e2 = self.app.activeDocument().addObject('Part::Ellipse', name)
        e2.MajorRadius = major_radius
        e2.MinorRadius = minor_radius
        e2.Placement = lcs.Placement
        e2.Visibility = False
        if show_lcs and not name.endswith('top'):
            e2_ctr = self.app.activeDocument().addObject('PartDesign::CoordinateSystem',  name + "lcs")
            e2_ctr.Placement = e2.Placement
        return e2

    def rotate_vertically(self):

        # Set top location before rotation, but relocate to account for position of base moved to 0,0,0
        # print_placement("LCS_TOP (before)", self.result_LCS_top)
        box_x = self.result_LCS_top.Placement.Base.x - self.result_LCS_base.Placement.Base.x
        box_y = self.result_LCS_top.Placement.Base.y - self.result_LCS_base.Placement.Base.y
        box_z = self.result_LCS_top.Placement.Base.z - self.result_LCS_base.Placement.Base.z

        # Rotate result by specified amount
        v1 = self.result_LCS_base.Placement.Base
        v2 = self.result_LCS_top.Placement.Base
        rot_center = v1
        # self.rotate_about_axis(self.result_LCS_base, v1, v2, 45, rot_center)
        # self.rotate_about_axis(self.result_LCS_top, v1, v2, 45, rot_center)
        # self.rotate_about_axis(self.result_LCS_base, v1, v2, 45, rot_center)  # Why duplicate above ???????

        # Rotate result to place vertically on each axis.  This requires only two rotations in x and y.
        # This does not seem to work....  is box_diag correct (it's ignoring negative portion of values)
        x_ang = np.arcsin(box_z / np.sqrt(box_z ** 2 + box_y ** 2))
        self.result.Placement.Matrix.rotateX(x_ang)
        self.result_LCS_top.Placement.Matrix.rotateX(x_ang)
        # self.result_LCS_base.Placement.Matrix.rotateX(x_ang)
        y_ang = np.arcsin(box_z / np.sqrt(box_z ** 2 + box_x ** 2))
        # self.result.Placement.Matrix.rotateY(y_ang)
        self.result_LCS_top.Placement.Matrix.rotateY(y_ang)
        print(f"Rotate X: {np.rad2deg(x_ang)}, Y: {np.rad2deg(y_ang)}")
        # self.result_LCS_base.Placement.Matrix.rotateY(y_ang)

        for wafer in self.wafer_list:
            wafer.rotate_to_vertical(x_ang, y_ang)

    def rotate_about_axis(self, obj, v1, v2, angle, rot_center):
        """Rotate an object about an axis defined by two vectors by a specified angle. """
        axis = self.app.Vector(v2 - v1)
        line = Part.LineSegment()
        line.StartPoint = v1
        line.EndPoint = v2
        objln = self.doc.addObject("Part::Feature", "Line")
        objln.Shape = line.toShape()
        objln.ViewObject.LineColor = (204.0, 170.0, 34.0)
        objln2 = self.doc.addObject("Part::Feature", "Line")
        objln2.Shape = line.toShape()
        objln2.ViewObject.LineColor = (104.0, 100.0, 134.0)

        rot = self.app.Rotation(axis, angle)
        obj_base = obj.Placement.Base
        print(f"Obj: {obj_base}, CTR: {rot_center}")
        new_place = self.app.Placement(obj_base, rot, rot_center)
        obj.Placement = new_place

        objln.Placement = new_place
        rot = self.app.Rotation(axis, 15)
        obj_base = obj.Placement.Base
        print(f"Obj: {obj_base}, CTR2: {rot_center}")
        new_place = self.app.Placement(obj_base, rot, rot_center)
        objln2.Placement = new_place

        
        self.doc.recompute()