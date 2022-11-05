import string
import inspect
import importlib.util
import Part
import Draft
import FreeCAD
import numpy as np
import csv
import re
import time
import importlib.util
import sys
import tokenize
from .structurehelix import StructureHelix
from .wafer import Wafer


# Now on github as plant-builder
class Driver(object):

    def __init__(self, App, Gui, assembly_name, master_spreadsheet):
        self.App = App
        self.Gui = Gui
        self.doc = App.activeDocument()
        self.parent_assembly = App.listDocuments()[assembly_name]
        if not self.parent_assembly:
            raise ValueError(f"Assembly {assembly_name} not found.")
        self.parent_parms = self.parent_assembly.getObjectsByLabel(master_spreadsheet)
        if self.parent_parms:
            self.parent_parms = self.parent_parms[0]
            self.get_parm, self.set_parm = self.handle_spreadsheet(self.parent_parms)   # Functions to handle parameters
        else:
            raise ValueError(f"Spreadsheet {master_spreadsheet} not found")
        # trace parms
        self.trace_file_name = None
        self.trace_file = None
        self.do_trace = None
        self._set_up_trace()
        self.get_object_by_label = self._gobj()
        FreeCAD.gobj = self.get_object_by_label         # simplify getting things in console
        self.parm_set = self.get_parm("set_name")
        print(f"Parameter Set: {self.parm_set}")
        self.position_offset = self.get_parm("position_offset")
        # sequence_list is a string member of the property list associated with the master spreadsheet.
        # The syntax of the sequence list is:
        # "xx_yy_... where xx_ is of the form "sd_" for d as a digit(s) with the order of the
        # elements giving the prepended string naming the segment or LCS (top or bottom) in the ordered
        # set of segments defining the resultant structure.  The last element is the name for the current sequence.
        self.sequence_list = self.get_sequence_property()
        if self.sequence_list:
            self.current_sequence_name = self.sequence_list[-1]
        else:
            self.current_sequence_name = None

    def get_sequence_property(self):
        """Get list of items giving sequence of segments in overall design."""
        try:
            seq_string = self.get_parm("seg_sequence")
            if seq_string is not None:
                seq_list = seq_string.split("_")[0:-1]
            else:
                seq_list = []
            return seq_list
        except Exception as e:
            print(f"Failed to get/find Sequence Item: {e.args}")

    def add_sequence_property(self, element):
        """Add an element to existing sequence list and update spreadsheet."""
        self.sequence_list.append(element + "_")
        seq_content = "_".join(self.sequence_list)
        self.set_parm("seg_sequence", seq_content)

    def _gobj(self):
        """Function to get an object by label in FreeCAD"""
        def gobj(name):
            obj = self.App.ActiveDocument.getObjectsByLabel(name)
            if obj:
                obj = obj[0]
            else:
                obj = None
            return obj
        return gobj

    def _get_workflow(self):
        return self.get_parm("workflow")

    def workflow(self):
        case = self._get_workflow()

        # remove existing objects
        remove_existing = Driver.make_tf("remove_existing", self.parent_parms)
        do_cuts = Driver.make_tf("print_cuts", self.parent_parms)
        if not do_cuts and remove_existing:  # if do_cuts, there is no generated display
            doc_list = self.doc.findObjects(Name="l.+|e.+|L.+|E.+|f.+")  # obj names start with l,e,L,E
            for item in doc_list:
                if item.Label != 'Parms_Master':
                    try:
                        self.doc.removeObject(item.Label)
                    except:
                        pass
        if not do_cuts and (case == "helix" or case == "animate"):
            doc_list = self.doc.findObjects(Name=self.parm_set + ".+")  # remove prior occurrence of set being built
            for item in doc_list:
                if item.Label != 'Parms_Master':
                    self.doc.removeObject(item.Label)

        if case == "helix":
            self.make_helix()

        if case == "animate":
            self.make_helix()
            h1 = self.get_object_by_label("s1_FusedResult")
            h2 = self.get_object_by_label("s2_FusedResult")
            l1 = self.get_object_by_label("s1_lcs_top")
            l2 = self.get_object_by_label("s2_lcs_base")
            tr = self.make_transform_align(l2, l1)
            print(f"H1: {h1.Placement}\nH2: {h2.Placement}\nTR: {tr}")
            FreeCAD.tr = tr
            h2.Placement = tr.multiply(h1.Placement)
            # self.animate()

        if case == "make_wafer":
            # make single wafer
            v1 = self.App.Vector(0, 0, 10)
            v2 = self.App.Vector(15, 40, 50)
            v3 = self.App.Vector(35, 60, 80)
            lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', self.parm_set + "pos1_lcs")
            lcs1.Placement.Base = v1
            lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', self.parm_set + "pos2_lcs")
            lcs2.Placement.Base = v2
            wafer = Wafer(self.App, self.Gui, self.parm_set)
            wafer.make_wafer_from_lcs(lcs1, lcs2, self.get_parm("cylinder_diameter"), self.parm_set + "foo")
            wafer.wafer.Visibility = False
            lcs1.Visibility = False
            lcs2.Visibility = False
            drawer = DrawLines(self.App, self.doc)
            drawer.draw_line(lcs2.Placement.Base, v3, None, None, self.parm_set + "L1")

        if case == "other":
            lcs = self.doc.getObjectsByLabel("LCS")[0]
            cube = self.doc.getObjectsByLabel("Cube")[0]

            def animate_OLD():
                print("X")
                cube.Placement.Base.x += 1
                cube.ViewObject.show()
                self.Gui.updateGui()
            lcs_axis = lcs.Placement.Rotation.Axis
            cube_axis = cube.Placement.Rotation.Axis
            angle = lcs_axis.getAngle(cube_axis)
            print(f"LCS: {np.round(lcs_axis, decimals=3)}, cube: {np.round(cube_axis, decimals=3)}, angle: {np.round(np.rad2deg(angle), decimals=3)}")
            cube.ViewObject.Visibility = True
            cube.ViewObject.show()

            zhat = self.App.Vector(0, 0, 1)
            # a = shp.Vertexes[1].Point
            a = lcs.Placement.Base
            a = self.App.Vector(2, 3, 4)
            norm_axis = cube_axis.cross(lcs_axis)
            print(f"NORM: {np.round(np.rad2deg(norm_axis), decimals=3)}")
            rot = self.App.Rotation(norm_axis, zhat, lcs_axis, 'ZYX')
            cube.Placement = self.App.Placement(a, rot)

    def make_helix(self):
        do_cuts = Driver.make_tf("print_cuts", self.parent_parms)
        print(f"Working parameter set: {self.parm_set}")
        helix = self.build_helix(self.parm_set)
        if do_cuts:
            cuts_file_name = self.get_parm("cuts_file")
            cuts_file = open(cuts_file_name, "w+")
            helix.make_cut_list(cuts_file)
        # helix.rotate_vertically()
        # self.align_segments()


    def make_transform_align(self, helix_1, helix_2):
        l1 = helix_1.Placement
        l2 = helix_2.Placement
        tr = l1.inverse().multiply(l2)
        FreeCAD.align = tr
        return tr

    def animate(self):
        helix = self.get_object_by_label("s2_FusedResult")
        lcs1 = self.get_object_by_label("s1_lcs_top")
        lcs2 = self.get_object_by_label("s2_lcs_top")
        if not helix or not lcs1 or not lcs2:
            print("Failed to find object")
            return
        # else:
        #     print(f"{helix.Placement, lcs1.Placement, lcs2.Placement}")

        # driver_animate implements the animation increments and is problem specific
        # animate implements the timer and calls to update in QTCore
        #    This occurs because of the access to FreeCAD's environment and the inability of
        #    the QT timer to deal with a procedure name embedded in another procedure

        file_path = '/home/don/FreecadProjects/Macros/PyMacros/Plant_Builder/freecad_design/driver_animate.py'
        module_name = 'driver_animate'
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        drive_animation = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = drive_animation
        spec.loader.exec_module(drive_animation)
        rotate_cube = drive_animation.RotateCube(lcs1, helix, lcs2)

        file_path = '/home/don/FreecadProjects/Macros/PyMacros/Plant_Builder/freecad_design/animate.py'
        module_name = 'animate'
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    def align_segments(self):
        """Align existing segments end-to-end."""
        return                  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        segments = []
        segment_tops = []
        for item in self.sequence_list:
            segments.append(self.get_object_by_label(item + "_FusedResult"))
            segment_tops.append(self.get_object_by_label(item + "_lcs_top"))
        if len(segment_tops) > 1:
            for i in range(len(segment_tops)-2):
                # align segment "si_" and "si+1_"
                top = segment_tops[i]
                print(f"Segments: {top.Label}, {segment_tops[i+1].Label}")
                self.move_and_rotate_segment(top, segments[i+1], segment_tops[i+1])

    def move_and_rotate_segment(self, base_lcs, segment, segment_top_lcs):
        """Rotate a segment and its top_lcs such that it is aligned with the base_lcs (top from prior segment"""
        return

    def _set_up_trace(self):
        self.trace_file_name = self.parent_parms.get("trace_file")
        self.do_trace = Driver.make_tf("do_trace", self.parent_parms)
        if self.do_trace:
            self.trace_file = open(self.trace_file_name, "w+")
            self.trace_file.write("Start Trace\n")
        else:
            self.trace_file = None

    def build_helix(self, parm_set):
        lcs_file_name = self.get_parm("lcs_file")
        outside_height = self.get_parm("outside_height")
        cylinder_diameter = self.get_parm("cylinder_diameter")
        lift_angle = np.deg2rad(self.get_parm("lift_angle"))
        rotation_angle = np.deg2rad(self.get_parm("rotation_angle"))
        minor_radius = (cylinder_diameter / 2)
        major_radius = (cylinder_diameter / 2) / np.cos(lift_angle)
        print(f"Major: {major_radius}, Minor: {minor_radius}, Lift: {np.rad2deg(lift_angle)}")
        wafer_count = self.get_parm("wafer_count")
        show_lcs = Driver.make_tf("show_lcs", self.parent_parms)
        helix = StructureHelix(self.App, self.Gui, parm_set, lcs_file_name, self.position_offset)
        helix.add_segment(outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count)
        helix.write_instructions()
        fused_result, last_loc = helix.create_structure(major_radius, minor_radius, lcs_file_name, show_lcs)
        return helix

    def handle_spreadsheet(self, sheet):
        def get_parm(parm_name):
            try:
                parm_value = sheet.get(parm_name)
                if parm_value == "None":
                    parm_value = None
                if self.do_trace:
                    self.trace_file.write(f"Parameter: {parm_name} fetched with value: {parm_value}\n")
                return parm_value
            except Exception as e:
                print(f"Exception {e} reading from spreadsheet for value: {parm_name}")
                raise e

        def set_parm(parm_name, new_value):
            try:
                parm_value = sheet.set(parm_name, new_value)
                sheet.recompute()
                if self.do_trace:
                    self.trace_file.write(f"Parameter: {parm_name} set with value: {new_value}\n")
                return parm_value
            except Exception as e:
                print(f"Exception {e} writing to spreadsheet for value: {new_name}")
                raise e
        return get_parm, set_parm

    @staticmethod
    def make_tf(variable_name, parent_parms):
        try:
            if parent_parms.get(variable_name) == "True":
                print(f"{variable_name} = True")
                return True
            else:
                print(f"{variable_name} = False")
                return False
        except Exception as e:
            print(f"Exception: {e} on reference to {variable_name}")
            raise e
