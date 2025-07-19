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
import math
from .wafer import Wafer
from .flex_segment import FlexSegment
from .path_following import PathFollower
from .make_helix import MakeHelix
from .make_rectangle import MakeRectangle
from . import utilities
import pydevd_pycharm

# pip install pydevd-pycharm~=241.15989.155

# Now on github as plant-builder
class Driver(object):
    # trace = None

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
            self.get_parm, self.set_parm = self.handle_spreadsheet(self.parent_parms)  # Functions to handle parameters
        else:
            raise ValueError(f"Spreadsheet {master_spreadsheet} not found")
        self.segment_list = []
        # trace parms
        self.trace_file_name = None
        self.trace_file = None
        self.do_trace = None
        self.relocate_segments_tf = None
        self._set_up_trace()
        self.get_object_by_label = self._gobj()
        FreeCAD.gobj = self.get_object_by_label  # simplify getting things in console
        self.compound_transform = None  # transform from base of first segment to top of last
        self.handle_arrows = None  # holder for arrow command that must run after segment relocation
        self.path_place_list = None  # list of triples - point nbr, point place, distance to next point
        self.first_segment = True  # used to initialize segment list when relocating
        # Support for remote debugging to FreeCAD
        pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

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
        """Determine workflow as specified in Parms_Master.

        This is called directly from the initiating macro."""
        case = self._get_workflow()   # Workflow is specified in the parameter spreadsheet as "workflow"

        # remove existing objects if not building from description file where removed are specified explicitly.
        if case != "segments" and case != "relative_places":
            remove_existing = Driver.make_tf("remove_existing", self.parent_parms)
            do_cuts = Driver.make_tf("print_cuts", self.parent_parms)
            remove_string = "K.+|L+.|N+.|base_lcs.*"
            if not do_cuts and remove_existing:  # if do_cuts, there is no generated display
                remove_string += "|.+|e.+|E.+|f.+|A.+"
            self.remove_objects_re(remove_string)

        self.relocate_segments_tf = Driver.make_tf("relocate_segments", self.parent_parms)
        print(f"Case to execiute: {case}")
        if case == "segments":
            # This case reads the descriptor file and builds multiple segments
            self.build_from_file()
            self.process_arrow_command()
            if Driver.make_tf("print_cuts", self.parent_parms):
                self.build_cut_list()
            if Driver.make_tf("print_place", self.parent_parms):
                self.build_place_list()

        if case == "relative_places":
            self.build_from_file()
            start_seg = self.get_parm("relative_start")
            end_seg = self.get_parm("relative_end")
            self.build_relative_place_list(start_seg, end_seg)

        if case == "animate":
            # THIS IS STALE
            self.make_helix()
            h1 = self.get_object_by_label("s1_FusedResult")
            h2 = self.get_object_by_label("s2_FusedResult")
            l1 = self.get_object_by_label("s1_lcs_top")
            l2 = self.get_object_by_label("s2_lcs_base")
            tr = self.make_transform_align(l2, l1)
            print(f"H1: {h1.Placement}\nH2: {h2.Placement}\nTR: {tr}")
            FreeCAD.tr = tr
            h2.Placement = tr.multiply(h1.Placement)
            self.animate()

        if case == "other":
            pass

        if self.do_trace:
            self.trace_file.close()

    def relocate_segment(self):
        """Relocate segments end to end."""
        if not self.relocate_segments_tf:
            return
        segment = self.segment_list[-1]
        # Rotate entire segment around its own base, if needed
        # This occurs if the segment had a rotation specifier
        angle = segment.get_segment_rotation()
        segment_rotation = FreeCAD.Placement(FreeCAD.Vector(0, 0, 0),
                                             FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), angle))
        if self.first_segment:
            self.first_segment = False
            self.compound_transform = segment_rotation  # for first segment, we rotate only
        else:
            self.compound_transform = self.compound_transform.multiply(segment_rotation)
        segment.move_to_top(self.compound_transform)
        self.compound_transform = self.compound_transform.multiply(segment.get_transform_to_top())

    def build_cut_list(self):
        print(f"BUILD CUT LIST")
        cuts_file_name = self.get_parm("cuts_file")
        cuts_file = open(cuts_file_name, "w+")
        cuts_file.write("Cutting order:\n")

        for nbr, segment in enumerate(self.segment_list):
            segment.make_cut_list(nbr, cuts_file)

        cuts_file.close()

    def build_place_list(self):
        print(f"BUILD PLACE LIST")
        min_max = [[0, 0], [0, 0], [0, 0]]

        def find_min_max(base):
            for i in range(3):
                if base[i] < min_max[i][0]:
                    min_max[i][0] = np.round(base[i], 3)
                if base[i] > min_max[i][1]:
                    min_max[i][1] = np.round(base[i], 3)

        cuts_file_name = self.get_parm("place_file")
        cuts_file = open(cuts_file_name, "w+")
        cuts_file.write("Wafer Placement:\n\n\n")
        global_placement = FreeCAD.Placement(FreeCAD.Vector(0, 0, 0), FreeCAD.Rotation(0, 0, 0))

        for nbr, segment in enumerate(self.segment_list):
            print(f"SEGMENT: {segment.get_segment_name()}")
            global_placement = segment.print_construction_list(nbr, cuts_file, global_placement, find_min_max)
        min_max_str = f"\nGlobal Min Max:\n\tX: {min_max[0][0]} - {min_max[0][1]}, "
        min_max_str += f"Y: {min_max[1][0]} - {min_max[1][1]}, Z: {min_max[2][0]} - {min_max[2][1]}"
        cuts_file.write(f"{min_max_str}")

        cuts_file.close()

    def build_relative_place_list(self, seg_start, seg_end):
        # Construct placement list assuming seg_start is at 0,0,0 and continue through seg_end
        # This is used to build placement data for a series of segments that may occur in the midst of
        # a larger structure.
        print(f"BUILD RELATIVE PLACE LIST from {seg_start} to {seg_end}")
        cuts_file_name = self.get_parm("place_file")
        cuts_file = open(cuts_file_name, "w+")
        cuts_file.write("Wafer Placement:\n\n\n")
        global_placement = FreeCAD.Placement(FreeCAD.Vector(0, 0, 0), FreeCAD.Rotation(0, 0, 0))
        start_seg = None
        end_seg = None
        for nbr, segment in enumerate(self.segment_list):
            if segment.get_segment_name() == seg_start:
                start_seg = nbr
                break
        for nbr, segment in enumerate(self.segment_list):
            if segment.get_segment_name() == seg_end:
                end_seg = nbr
                break
        if not start_seg or not end_seg:
            raise ValueError(f"Invalid start or ending segment name: start {seg_start}, end {seg_end}")
        for nbr, segment in enumerate(self.segment_list[start_seg:end_seg + 1]):
            global_placement = segment.print_construction_list(nbr + start_seg, cuts_file, global_placement)

        cuts_file.close()

    def remove_objects_re(self, remove_string):
        """Remove objects containing 'name' as a part of a label."""
        doc_list = self.doc.findObjects(Name=remove_string)  # obj names start with l,e,L,E,f,K
        # print(f"DOC LIST LEN: {len(doc_list)}")
        for item in doc_list:
            if item.Label != 'Parms_Master':
                try:
                    self.doc.removeObject(item.Label)
                except Exception as e:
                    print(f"Remove object Exception: {e.args}")     # TODO: replace with actual exception
                    pass

    def _operations_reader(self, parm_name):
        """Create csv reader to handle structure definition commands."""
        # Continuation lines are designated with a "+" operator in the first position which is removed before
        # returning the line to the calling code. It is an error to find a continuation line at the operator level.
        print(f"_operations_reader: parm_name: {parm_name}")
        csv_file = open(self.get_parm(parm_name), "r")
        reader = csv.reader(csv_file)
        new_line = []
        do_read = True
        operator = None
        icnt = 0

        def fill_buffer():
            nonlocal do_read, new_line, operator, icnt
            if icnt > 40:
                foo = 3/0
            icnt += 1
            if do_read:
                do_read = False
                try:
                    line = next(reader)
                    print(f"Operation: {line}")
                except StopIteration:
                    csv_file.close()
                    new_line = ["stop"]
                new_line = []
                for item in line:  # space in 'name' was an issue, get all spaces to be sure
                    new_line.append(item.strip())
                if new_line:
                    operator = new_line[0]
                else:
                    do_read = True
                    operator = ""

        def get_operator_line():
            nonlocal new_line, operator, do_read
            fill_buffer()
            if len(operator) < 2:
                do_read = True
                return "blank", "blank"
            if operator == "+":
                raise ValueError(f"Call for continuation at top level")
            else:
                operator = new_line[0]
            do_read = True
            return operator, new_line

        def get_continuation_line():
            nonlocal new_line, operator, do_read
            fill_buffer()
            if operator != "+":
                raise ValueError(f"Call for continuation but not found")
            else:
                new_line = new_line[1:]
            do_read = True
            return operator, new_line

        def test_continuation():
            nonlocal operator, do_read
            if do_read:
                fill_buffer()
            if operator == '+':
                return True
            else:
                return False

        return get_operator_line, get_continuation_line, test_continuation

    def build_from_file(self):
        """Read file and build multiple segments"""
        get_operator_line, get_continuation_line, test_continuation = self._operations_reader("description_file")
        while True:
            operator, new_line = get_operator_line()
            print(f"op: {operator}, line: {new_line}")
            if operator == 'comment' or operator == '#' or operator == "blank":
                pass
            elif operator == 'remove':
                print(f"Remove: {new_line}")
                names = new_line[1:]
                for name in names:
                    self.remove_objects_re(f"{name}.+")
            elif operator == 'set_position':
                # set the location and orientation of the next segment to be built.
                positions = new_line[1:]   # contains 3 x,y,z values and optional roll,pitch,yaw
                print(f"POSITION: {new_line[1:]}")
                positions = [float(x) for x in positions]
                pos = FreeCAD.Vector(positions[0], positions[1], positions[2])
                if len(positions) == 6:
                    rot = FreeCAD.Rotation(positions[3], positions[4], positions[5])
                else:
                    rot = FreeCAD.Rotation(0, 0, 0)
                self.compound_transform = FreeCAD.Placement(pos, rot)
                self.first_segment = False
                print(f"PLACEMENT: {self.compound_transform}")
            elif operator == 'helix':
                print(f"HELIX: {new_line[1:]}")
                tf = self.get_parm("flex_temp")
                segment_type, lift_angle, rotate_angle, wafer_count, name, outside_height, cylinder_diameter, \
                    show_lcs, build_segment, rotate_segment = new_line
                self.remove_objects_re(f"{name}.+")
                lift_angle = float(lift_angle)
                wafer_count = int(wafer_count)
                outside_height = float(outside_height)
                cylinder_diameter = float(cylinder_diameter)
                rotate_angle = float(rotate_angle)
                rotate_segment = float(rotate_segment)
                show_lcs = bool(show_lcs == "True")
                new_segment = FlexSegment(name,  show_lcs, tf, build_segment, rotate_segment)
                self.segment_list.append(new_segment)
                helix = MakeHelix(new_segment)
                helix.create_helix(wafer_count, cylinder_diameter, outside_height, lift_angle, rotate_angle, name)
                self.relocate_segment()

            elif operator == 'rectangle':
                tf = self.get_parm("flex_temp")
                segment_type, lift_angle, rotate_angle, wafer_count, name, outside_height, cylinder_diameter, \
                    show_lcs, build_segment, rotate_segment = new_line
                self.remove_objects_re(f"{name}.+")
                lift_angle = float(lift_angle)
                wafer_count = int(wafer_count)
                outside_height = float(outside_height)
                cylinder_diameter = float(cylinder_diameter)
                long_side = cylinder_diameter
                short_side = cylinder_diameter
                rotate_angle = float(rotate_angle)
                rotate_segment = float(rotate_segment)
                show_lcs = bool(show_lcs == "True")
                new_segment = FlexSegment(name,  show_lcs, tf, build_segment, rotate_segment)
                self.segment_list.append(new_segment)
                box = MakeRectangle(new_segment)
                box.create_boxes(wafer_count, cylinder_diameter, cylinder_diameter,
                                 outside_height, lift_angle, rotate_angle, name)
                self.relocate_segment()

            elif operator == "curve":
                tf = self.get_parm("flex_temp")     # Temp file for path following
                segment_type, segment_name, curve_type, nbr_points, increment, scale, curve_rotation = new_line
                if test_continuation():
                    _, ln = get_continuation_line()
                    wafer_height, outside_diameter = ln
                else:
                    raise ValueError("Missing wafer descriptor continuation")
                if test_continuation():
                    _, ln = get_continuation_line()
                    show_lcs, build_segment, rotate_segment = ln
                    show_lcs = bool(show_lcs == "True")
                else:
                    raise ValueError("Missing segment controls continuation")
                self.remove_objects_re(f"{segment_name}.+|Curve.+")
                nbr_points = int(nbr_points)
                increment = float(increment)
                scale = float(scale)
                curve_rotation = float(curve_rotation)
                wafer_height = float(wafer_height)
                outside_diameter = float(outside_diameter)
                rotate_segment = float(rotate_segment)

                segment = FlexSegment(segment_name, show_lcs, tf, build_segment, rotate_segment)
                self.segment_list.append(segment)
                follower = PathFollower(segment)

                follower.set_curve_parameters(curve_type, nbr_points, curve_rotation, increment, scale)
                follower.set_wafer_parameters(wafer_height, outside_diameter)
                follower.implement_curve()
                if segment.get_segment_object():
                    print(f"Segment {segment_name} has segment object {segment.get_segment_object()}")
                    # self.relocate_segment()
                else:
                    print(f"Curve {segment_name} did not produce a segment object.")

            elif operator == 'arrows':
                # add lines from last wafer to specified point and normal to last wafer
                # this must run after segment relocation so temp hold for now.  Note if multiple
                # arrow specs in file, the last will prevail
                self.handle_arrows = new_line

            elif operator == 'path':
                # create target path to follow
                segment_type, scale, point_count, knot_rotation = new_line
                scale = float(scale)
                point_count = int(point_count)
                self.path_place_list = Driver.overhand_knot_path(scale, point_count, int(knot_rotation))

            elif operator == 'stop':
                return
            else:
                print(f"NOT RECOGNIZED: {operator}")
                break

    def process_arrow_command(self):
        if not self.handle_arrows:
            return
        # input string from description file (delayed input to allow for segment relocation)
        _, size, point_nbr = self.handle_arrows
        size = float(size)
        point_nbr = int(point_nbr)
        segment_list_top = self.segment_list[0]
        lcs_top = segment_list_top.get_lcs_base()
        if not self.compound_transform:
            raise ValueError(f"TOP MISSING")
        new_place = lcs_top.Placement.multiply(self.compound_transform)  # Don't change LCS Base, so new Placement
        # arrow('Normal', new_place, size)

        # Now add line to specified point
        if len(self.path_place_list) <= point_nbr:
            raise ValueError(f"No path point: {point_nbr}")
        nbr, place, dist = self.path_place_list[point_nbr]
        line = Part.LineSegment()
        line.StartPoint = new_place.Base
        line.EndPoint = place.Base
        obj = self.doc.addObject("Part::Feature", "Line")
        obj.Shape = line.toShape()

        # find angle between normal to face and line to point on curve
        p = FreeCAD.Placement(FreeCAD.Vector(0, 0, 3), FreeCAD.Rotation(0, 0, 0))
        line2 = Part.LineSegment()
        line2.StartPoint = new_place.Base
        line2.EndPoint = new_place.multiply(p).Base
        line_tan = line.tangent(0)[0]
        line2_tan = line2.tangent(0)[0]
        obj = self.doc.addObject("Part::Feature", "Line")
        obj.Shape = line2.toShape()
        angle = np.round(np.rad2deg(line2_tan.getAngle(line_tan)), 1)
        print(f" ANGLE: {angle}")

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE: {object_1.Label}, {object_2.Label}, {l1}")
        FreeCAD.align = tr  # make available to console
        return tr

    def animate(self):
        # THIS CODE IS STALE
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

    def _set_up_trace(self):
        self.trace_file_name = self.parent_parms.get("trace_file")
        self.do_trace = Driver.make_tf("do_trace", self.parent_parms)
        if self.do_trace:
            self.trace_file = open(self.trace_file_name, "w")
            self.trace_file.write("Start Trace\n")
            self.trace_file.flush()
        else:
            self.trace_file = None

    def trace(self, *args):
        if self.do_trace:
            if self.trace_file.closed:
                print("FILE WAS CLOSED")
                self.trace_file = open(self.trace_file_name, "a")
            trace_string = ''
            for arg in args:
                trace_string += "  " + repr(arg) + "\n"
            self.trace_file.write(trace_string)
            self.trace_file.flush()
            print(trace_string)

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
                print(f"Exception {e} writing to spreadsheet for value: {parm_name}")
                raise e

        return get_parm, set_parm

    @staticmethod
    def make_tf(variable_name, parent_parms):
        print(f"make_tf:  Variable: {variable_name}, parent: {parent_parms}")
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

    @staticmethod
    def convert_tf(value):
        if value.strip() == 'True':
            return True
        else:
            return False

    @staticmethod
    def overhand_knot_path(scale, point_count, rotation, add_vertex=True):
        """Compute path of overhand knot.

        Return: list of
                t = position number along curve ,
                3D Placement of t
                distance to next t"""
        # NOTE:  if the first non-zero point has a zero y value, the candidate selected will have no lift
        # angle as all candidates have a zero yaw value.  To fix, add some arbitrary rotation to the whole path.
        point_list = []
        if add_vertex:
            knot_group = FreeCAD.activeDocument().addObject("App::DocumentObjectGroup", "Knot Group")
        for t in range(point_count):
            angle = math.radians(t * 5) - math.pi
            x = (math.cos(angle) + 2 * math.cos(2 * angle)) * scale
            y = (math.sin(angle) - 2 * math.sin(2 * angle)) * scale
            z = (-math.sin(3 * angle)) * scale
            if t == 0:  # set origin of knot to global origin
                x0 = x
                y0 = y
                z0 = z
            vec1 = FreeCAD.Vector(x - x0, y - y0, z - z0)
            rot = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), rotation)
            vec = rot.multVec(vec1)
            # print(f"Knot: t: {t}, x: {np.round(x, 3)}, y: {np.round(y, 3)}, z:  {np.round(z, 3)}")
            point_list.append((t, vec))
        p_prior = point_list[0]
        place_list = []
        for p_next in point_list[1:]:
            t = p_prior[0]
            v1 = p_prior[1]
            v2 = p_next[1]
            angle = v1.getAngle(v2)
            if math.isnan(angle):
                angle = 0.0
            rot = FreeCAD.Rotation(v1, angle)
            place = FreeCAD.Placement(v1, rot)
            dist = v1.distanceToPoint(v2)
            place_list.append((t, place, dist))
            # arrow("A" + str(t), place)
            p_prior = p_next
            # print(f"\t t: {t}, A: {np.round(angle, 2)} place: {place}")
            if add_vertex:
                lcs1 = FreeCAD.activeDocument().addObject('Part::Vertex', "Kdot" + str(t))
                lcs1.X = v1.x
                lcs1.Y = v1.y
                lcs1.Z = v1.z
                knot_group.addObjects([lcs1])
        return place_list

def arrow(name, placement, size):
    """Simple arrow to show location and direction."""
    n = []
    v = FreeCAD.Vector(0, 0, 0)
    n.append(v)
    vpoint = FreeCAD.Vector(0, 0, size * 6)
    n.append(vpoint)
    v = FreeCAD.Vector(size, 0, size * 5)
    n.append(v)
    n.append(vpoint)
    v = FreeCAD.Vector(-size, 0, size * 5)
    n.append(v)
    p = FreeCAD.activeDocument().addObject("Part::Polygon", name)
    p.Nodes = n
    rot = placement.Rotation
    loc = placement.Base
    p.Placement.Rotation = rot
    p.Placement.Base = loc
    return p


def calculate_distance_and_rotation(placement, vec):
    # Calculate the distance between a placement and a vector in mm and radians
    distance = np.sqrt(
        (vec.x - placement.Base.x) ** 2 + (vec.y - placement.Base.y) ** 2 + (vec.z - placement.Base.z) ** 2)
    # Get the rotation matrix of the placement
    rot_matrix = placement.Rotation.toMatrix()

    # Rotate the vector around the z-axis of the placement
    rotated_vec = rot_matrix.multiply(vec)

    # Calculate the angle to rotate around the z-axis of the placement
    angle = np.arctan2(rotated_vec.y, rotated_vec.x)

    # Calculate the component distances and angles
    x_distance = rotated_vec.x
    y_distance = rotated_vec.y
    z_distance = rotated_vec.z
    x_angle = np.arctan2(rotated_vec.z, rotated_vec.x)
    y_angle = np.arctan2(rotated_vec.z, rotated_vec.y)

    return distance, angle, x_distance, y_distance, z_distance, x_angle, y_angle
