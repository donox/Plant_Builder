import numpy as np
import csv
from .wafer import Wafer
from .segment import Segment
import re


def structure_dummy():
    # print(f"structure_dummy called")
    return


class Structure(object):
    def __init__(self, App, lcs_file_name):
        self.app = App
        self.doc = App.ActiveDocument
        self.lcs_file = open(lcs_file_name, "w+")
        self.lcs_writer = csv.writer(self.lcs_file)
        self.segment_list = []
        self.lcs_group = self.doc.addObject("App::DocumentObjectGroup", "All LCS")

    def add_segment(self, outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count):
        seg = Segment(lift_angle, rotation_angle, outside_height, cylinder_diameter, wafer_count)
        self.segment_list.append(seg)

    def write_instructions(self):
        lcs_temp = self.doc.addObject('PartDesign::CoordinateSystem', 'LCS')  # Used to write file of positions
        lcs_temp.Visibility = False
        lcs_top = self.doc.addObject('PartDesign::CoordinateSystem', "lcs_top")
        lcs_base = self.doc.addObject('PartDesign::CoordinateSystem', "lcs_base")   # Should be same as fuse placement
        self.lcs_group.addObjects([lcs_temp])
        total_wafer_count = -1

        for segment in self.segment_list:
            for j in range(segment.wafer_count):
                # print(f"LA: {np.rad2deg(segment.lift_angle)}, RA: {np.rad2deg(segment.rotation_angle)}")
                # for wafers with twist between wafers rather than within wafer, change position of rotation below.
                total_wafer_count += 1
                e_name = 'e' + str(total_wafer_count)
                self.lcs_writer.writerow([e_name, lcs_temp.Placement])
                if j == 0:
                    lcs_base.Placement = lcs_temp.Placement
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
            wafer_list = []
            last_location = None
            while True:
                p1, p1_place = next(reader)
                if p1 == "Done":
                    break
                place = self.make_placement(p1_place)
                loft_name = "l" + p1
                lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', p1 + "lcs")
                lcs1.Placement = place
                if not show_lcs:
                    lcs1.Visibility = False
                e1 = self.make_ellipse(p1, major_radius, minor_radius, lcs1, False, False)
                p2, p2_place = next(reader)
                place = self.make_placement(p2_place)
                lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', p2 + "lcs")
                lcs2.Placement = place
                self.lcs_group.addObjects([lcs1, lcs2])
                if not show_lcs:
                    lcs2.Visibility = False
                e2 = self.make_ellipse(p1, major_radius, minor_radius, lcs2, False, False)
                last_location = e2.Placement
                wafer = self.doc.addObject('Part::Loft', loft_name)
                wafer.Sections = [e1, e2]
                wafer.Solid = True
                wafer.Visibility = True
                wafer_list.append(wafer)
            fuse = self.doc.addObject("Part::MultiFuse", "FusedResult")
            fuse.Shapes = wafer_list
            fuse.Visibility = True
            fuse.ViewObject.DisplayMode = "Shaded"
            self.app.activeDocument().recompute()
            return fuse, last_location

    def make_cut_list(self, cuts_file):
        # THIS HAS TO BE BASED ON SEGMENTS
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