import string
import inspect
import importlib.util
import Part
import Draft
import FreeCAD
import numpy as np
import csv
import re
from .structure_helix import Structure_Helix
from .wafer import Wafer
from .draw_lines import DrawLines


# Now on github as plant-builder
class Driver(object):

    def __init__(self, App, assembly_name, master_spreadsheet):
        self.App = App
        self.doc = App.activeDocument()
        self.parent_assembly = App.listDocuments()[assembly_name]
        if not self.parent_assembly:
            raise ValueError(f"Assembly {assembly_name} not found.")
        self.parent_parms = self.parent_assembly.getObjectsByLabel(master_spreadsheet)
        if self.parent_parms:
            self.parent_parms = self.parent_parms[0]
            self.get_parm = self.handle_spreadsheet(self.parent_parms)   # Function to retrieve a specific parameter
        else:
            raise ValueError(f"Spreadsheet {master_spreadsheet} not found")

        # trace parms
        self.trace_file_name = None
        self.trace_file = None
        self.do_trace = None
        self._set_up_trace()

    def _gobj(self):     # Can't reference App in FreeCAD
        def gobj(name):
            objs = App.activeDocument.getObjectsByLabel(name)[0]
            return objs

    def _get_workflow(self):
        return self.get_parm("workflow")

    def workflow(self):
        case = self._get_workflow()

        if case == "helix":
            # Remove existing objects
            do_cuts = Driver.make_tf("print_cuts", self.parent_parms)
            remove_existing = Driver.make_tf("remove_existing", self.parent_parms)
            if not do_cuts and remove_existing:  # if do_cuts, there is no generated display
                doc_list = self.doc.findObjects(Name="l.+|e.+|L.+|E.+")  # obj names start with l,e,L,E
                for item in doc_list:
                    if item.Label != 'Parms_Master':
                        self.doc.removeObject(item.Label)
            helix = self.build_helix()
            if do_cuts:
                cuts_file_name = self.get_parm("cuts_file")
                cuts_file = open(cuts_file_name, "w+")
                helix.make_cut_list(cuts_file)
            helix.rotate_vertically()

        if case == "draw_lines":
            # Draw some lines
            drawer = DrawLines(self.App, self.doc)
            v1 = self.App.Vector(0, 0, 0)
            v2 = self.App.Vector(10, 20, 90)
            angle = 30
            axis = self.App.Vector(0, 0, 1)
            ln = drawer.draw_line(v1, v2, angle, axis, "foo")
            angle = 75
            ln2 = drawer.draw_line(v1, v2, angle, axis, "bar")
            drawer.get_angles(ln)

        if case == "make_wafer":
            # make single wafer
            wafer = Wafer(self.App)
            wafer.make_wafer_from_lcs(helix.result_LCS_base, helix.result_LCS_top,
                                      self.get_parm("cylinder_diameter"), "foo")

    def _set_up_trace(self):
        self.trace_file_name = self.parent_parms.get("trace_file")
        self.do_trace = Driver.make_tf("do_trace", self.parent_parms)
        if self.do_trace:
            self.trace_file = open(self.trace_file_name, "w+")
            self.trace_file.write("Start Trace\n")
        else:
            self.trace_file = None

    def build_helix(self):
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
        helix = Structure_Helix(self.App, lcs_file_name)
        helix.add_segment(outside_height, cylinder_diameter, lift_angle, rotation_angle, wafer_count)
        helix.write_instructions()
        fused_result, last_loc = helix.create_structure(major_radius, minor_radius, lcs_file_name, show_lcs)
        print(f"Before: {fused_result.Placement.toMatrix()}")
        fused_result.Placement.Matrix.rotateX(np.deg2rad(30))
        print(f"After: {fused_result.Placement.toMatrix()}")
        return helix

    def handle_spreadsheet(self, sheet):
        def get_parm(parm_name):
            try:
                parm_value = sheet.get(parm_name)
                if self.do_trace:
                    self.trace_file.write(f"Parameter: {parm_name} fetched with value: {parm_value}\n")
                return parm_value
            except Exception as e:
                print(f"Exception {e} reading from spreadsheet for value: {parm_name}")
                raise e
        return get_parm

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
