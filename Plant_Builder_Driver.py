import string
import inspect
import importlib.util
import Part
import Draft
# import pytransform3d
import FreeCAD
import numpy as np
import csv
import re


def import_modules(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def driver(App, assembly_name, master_spreadsheet):

    doc = App.activeDocument()
    parent_assembly = App.listDocuments()[assembly_name]
    if not parent_assembly:
        raise ValueError(f"Assembly {assembly_name} not found.")
    parent_parms = parent_assembly.getObjectsByLabel(master_spreadsheet)
    if parent_parms:
        parent_parms = parent_parms[0]
    else:
        raise ValueError(f"Spreadsheet {master_spreadsheet} not found")

    trace_file_name = parent_parms.get("trace_file")
    do_trace = make_tf("do_trace", parent_parms)
    if do_trace:
        trace_file = open(trace_file_name, "w+")
        trace_file.write("Start Trace\n")
    else:
        trace_file = None

    testing = make_tf("testing", parent_parms)
    do_cuts = make_tf("print_cuts", parent_parms)

    # Write record of lcs Placements to file
    lcs_file_name = parent_parms.get("lcs_file")
    do_lcs = make_tf("do_lcs", parent_parms)
    if do_lcs and not testing:
        lcs_file = open(lcs_file_name, "w+")
        lcs_writer = csv.writer(lcs_file)
    else:
        lcs_file = None

    lift_angle = np.deg2rad(parent_parms.get("lift_angle"))
    rotation_angle = np.deg2rad(parent_parms.get("rotation_angle"))
    wafer_count = parent_parms.get("wafer_count") + 1   # +1 to account for startup
    segment_list = []

    # use_file imports a list of lift, rotate, count for building a multi-shaped model
    if make_tf("use_file", parent_parms):
        with open(parent_parms.get("description_file"), 'r') as infile:
            csvfile = csv.reader(infile)
            for la, ra, wc in csvfile:
                if len(la) < 4:
                    segment_list.append((np.deg2rad(float(la)), np.deg2rad(float(ra)), int(wc)))
                    print(segment_list[-1])
    else:
        segment_list.append((lift_angle, rotation_angle, wafer_count))

    outside_height = parent_parms.get("outside_height").Value   # take value as height has units
    cylinder_diameter = parent_parms.get("cylinder_diameter").Value
    minor_radius = (cylinder_diameter/2) * np.cos(lift_angle)
    major_radius = (cylinder_diameter/2)
    # Helix Radius is to outside edge
    helix_radius, inside_height = get_wafer_parms(lift_angle, rotation_angle, outside_height, cylinder_diameter)

    show_lcs = make_tf("show_lcs", parent_parms)

    # Remove existing objects
    if not do_cuts and make_tf("remove_existing", parent_parms):   # if do_cuts, there is no generated display
        doc_list = doc.findObjects(Name="l.+|e.+|L.+|E.+")  # obj names start with l,e,L,E
        for item in doc_list:
            if item.Label != 'Parms_Master':
                doc.removeObject(item.Label)

    if do_trace:
        trace_file.write(f"lift_angle: \t\t{np.rad2deg(lift_angle)}\n")
        trace_file.write(f"rotation_angle: \t{np.rad2deg(rotation_angle)}\n")
        trace_file.write(f"outside_height: \t{outside_height / 25.4} in\n")
        trace_file.write(f"inside_height: \t{inside_height / 25.4} in\n")
        trace_file.write(f"cylinder_diameter: \t{cylinder_diameter / 25.4} in\n")
        trace_file.write(f"helix_radius: \t{helix_radius / 25.4} in\n")
        trace_file.write(f"wafer_count: \t\t{wafer_count}\n")

    parm_str = f"Lift Angle: {np.rad2deg(lift_angle)} degrees\n"
    parm_str += f"Rotation Angle: {np.rad2deg(rotation_angle)} degrees\n"
    parm_str += f"Outside Wafer Height: {np.round(outside_height / 25.4, 2)} in\n"
    parm_str += f"Inside Wafer Height: {np.round(inside_height / 25.4, 2)} in\n"
    parm_str += f"Cylinder Diameter: {np.round(cylinder_diameter / 25.4, 2)} in\n"
    parm_str += f"Helix Radius: \t{np.round(helix_radius / 25.4, 2)} in\n"

    if do_cuts:
        cuts_file_name = parent_parms.get("cuts_file")
        cuts_file = open(cuts_file_name, "w+")
        cuts_file.write("Cutting order:\n\n\n")
        cuts_file.write(parm_str)
        cuts_file.write(f"Wafer Count: {wafer_count}\n\n")
    else:
        cuts_file = None

    # For Testing
    if testing:
        wafer, top_loc = wafer_from_csv(App, major_radius, minor_radius, lcs_file_name, show_lcs)
        # sweep = make_sweep(App, 3, 50, .05, 100, 15)
        # doc.recompute()

    elif do_cuts:

        nbr_rotations = int(360 / np.rad2deg(rotation_angle))
        step_size = nbr_rotations / 2 - 1
        current_position = 0
        s1 = "Step: "
        s2 = " at position: "
        for i in range(wafer_count):
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
            current_position = int((current_position + step_size) % nbr_rotations)
        cuts_file.close()

    else:
        # Main Process Flow
        lcs_group = doc.addObject("App::DocumentObjectGroup", "All LCS")
        lcs_temp = doc.addObject('PartDesign::CoordinateSystem', 'LCS')  # Used to write file of positions
        lcs_temp.Visibility = False
        lcs_top = doc.addObject('PartDesign::CoordinateSystem', "lcs_top")
        lcs_base = doc.addObject('PartDesign::CoordinateSystem', "lcs_base")   # Should be same as fuse placement
        lcs_group.addObjects([lcs_temp])
        total_wafer_count = -1
        for lift_angle, rotation_angle, wafer_count in segment_list:
            helix_radius, inside_height = get_wafer_parms(lift_angle, rotation_angle, outside_height, cylinder_diameter)
            for j in range(wafer_count):
                # for wafers with twist between wafers rather than within wafer, change position of rotation below.
                total_wafer_count += 1
                e_name = 'e' + str(total_wafer_count)
                lcs_writer.writerow([e_name, lcs_temp.Placement])
                if j == 0:
                    lcs_base.Placement = lcs_temp.Placement
                e_name += '_top'
                lift_lcs(App, lcs_temp, lift_angle, helix_radius, outside_height)
                lcs_temp.Placement.Matrix.rotateZ(rotation_angle)    # This makes rotation occur on wafer, not between
                lcs_writer.writerow([e_name, lcs_temp.Placement])
        lcs_writer.writerow(["Done", "Done"])
        lcs_file.close()
        # top_location is location of topmost ellipse
        fuse, top_location = wafer_from_csv(App, major_radius, minor_radius
                                            , lcs_file_name, show_lcs, lcs_group)
        lcs_top.Placement = top_location

        # Set top location before rotation, but relocate to account for position of base moved to 0,0,0
        print_placement("LCS_TOP (before)", lcs_top)
        box_x = lcs_top.Placement.Base.x - lcs_base.Placement.Base.x
        box_y = lcs_top.Placement.Base.y - lcs_base.Placement.Base.y
        box_z = lcs_top.Placement.Base.z - lcs_base.Placement.Base.z

        # Rotate result by specified amount
        if False:
            v1 = lcs_base.Placement.Base
            v2 = lcs_top.Placement.Base
            rot_center = v1
            rotate_about_axis(fuse, v1, v2, 180, rot_center)
            rotate_about_axis(lcs_top, v1, v2, 180, rot_center)
            rotate_about_axis(lcs_base, v1, v2, 180, rot_center)

        # Rotate result to place vertically on each axis.  This requires only two rotations in x and y.
        # This does not seem to work....  is box_diag correct (it's ignoring negative portion of values)
        x_ang = np.arccos(box_z / np.sqrt(box_z ** 2 + box_y ** 2))
        fuse.Placement.Matrix.rotateX(x_ang)
        lcs_top.Placement.Matrix.rotateX(x_ang)
        lcs_base.Placement.Matrix.rotateX(x_ang)
        y_ang = np.arccos(box_z / np.sqrt(box_z ** 2 + box_x ** 2))
        fuse.Placement.Matrix.rotateY(y_ang)
        lcs_top.Placement.Matrix.rotateY(y_ang)
        lcs_base.Placement.Matrix.rotateY(y_ang)

        parm_str += f"Total Wafer Count: {total_wafer_count}"
        print(parm_str)



def rotate_about_axis(obj, v1, v2, angle, rot_center):
    axis = FreeCAD.Vector(v2 - v1)
    rot = FreeCAD.Rotation(axis, angle)
    obj_base = obj.Placement.Base
    print(f"Obj: {obj_base}, CTR: {rot_center}")
    new_place = FreeCAD.Placement(obj_base, rot, rot_center)
    obj.Placement = new_place


def show_angle(angle):
    return np.round(np.rad2deg(angle), 2)


def to_inch(dist):
    return np.round(dist / 25.4, 2)


def print_placement(text, obj):
    print(f"{text}: x-{to_inch(obj.Placement.Base.x)}, y-{to_inch(obj.Placement.Base.y)}, z-{to_inch(obj.Placement.Base.z)}")


def wafer_from_csv(App, major_radius, minor_radius, csv_file, show_lcs, lcs_group):
    """Create wafer and placement of top ellipse."""
    doc = App.activeDocument()
    with open(csv_file, newline='') as infile:
        reader = csv.reader(infile)
        wafer_list = []
        last_location = None
        while True:
            p1, p1_place = next(reader)
            # print(p1,p1_place)
            if p1 == "Done":
                break
            place = make_placement(App, p1_place)
            loft_name = "l" + p1
            lcs1 = doc.addObject('PartDesign::CoordinateSystem',  p1 + "lcs")
            lcs1.Placement = place
            if not show_lcs:
                lcs1.Visibility = False
            # print("lcs1.Placement = Freecad." + p1_place)
            e1 = make_ellipse(App, p1, major_radius, minor_radius, lcs1, False, False)
            # e1.Placement = place
            p2, p2_place = next(reader)
            # print(p2, p2_place)
            place = make_placement(App, p2_place)
            lcs2 = doc.addObject('PartDesign::CoordinateSystem', p2 + "lcs")
            lcs2.Placement = place
            lcs_group.addObjects([lcs1, lcs2])
            if not show_lcs:
                lcs2.Visibility = False
            e2 = make_ellipse(App, p1, major_radius, minor_radius, lcs2, False, False)
            last_location = e2.Placement
            wafer = doc.addObject('Part::Loft', loft_name)
            wafer.Sections = [e1, e2]
            wafer.Solid = True
            wafer.Visibility = True
            wafer_list.append(wafer)
        fuse = doc.addObject("Part::MultiFuse", "FusedResult")
        fuse.Shapes = wafer_list
        fuse.Visibility = True
        fuse.ViewObject.DisplayMode = "Shaded"
        App.activeDocument().recompute()
        return fuse, last_location


def lift_lcs(App, lcs, lift_angle, helix_radius, outside_height):
    if lift_angle == 0:
        lcs.Placement.Base = lcs.Placement.Base + FreeCAD.Vector(0, 0, outside_height)
        return
    translate_vector = App.Vector(-helix_radius, 0, 0)
    lcs.Placement.Base = lcs.Placement.Base + translate_vector
    pm = lcs.Placement.toMatrix()
    pm.rotateY(lift_angle)
    lcs.Placement = FreeCAD.Placement(pm)
    lcs.Placement.Base = lcs.Placement.Base - translate_vector


def display_matrix(matrix, label, trace_file):
    if not trace_file:
        return
    ux = np.round(matrix.A[0:3], 3) # matrix.A is all matrix elements (4 x 4)
    uy = np.round(matrix.A[4:7], 3)
    uz = np.round(matrix.A[8:11], 3)
    pos = np.round((matrix.A14, matrix.A24, matrix.A34), 3)
    trace_file.write(f"Matrix: {label}\n\t")
    trace_file.write(f"ux: {ux}\t")
    trace_file.write(f"uy: {uy}\t")
    trace_file.write(f"uz: {uz}\t")
    trace_file.write(f"pos: {pos}\n")


def get_wafer_parms(lift_angle, rotation_angle, outside_height, cylinder_diameter):
    if lift_angle == 0:     # leave as simple cylinder
        return 0, outside_height
    # Assume origin at center of ellipse, x-axis along major axis, positive to outside.
    helix_radius = np.math.cos(lift_angle)/np.math.sin(lift_angle) * outside_height
    inside_height = (helix_radius - cylinder_diameter) * np.math.tan(lift_angle)
    return helix_radius, inside_height


def make_ellipse(App, name, major_radius, minor_radius, lcs, trace_file, show_lcs):
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
    e2 = App.activeDocument().addObject('Part::Ellipse', name)
    e2.MajorRadius = major_radius
    e2.MinorRadius = minor_radius
    e2.Placement = lcs.Placement
    e2.Visibility = False
    if show_lcs and not name.endswith('top'):
        e2_ctr = App.activeDocument().addObject('PartDesign::CoordinateSystem',  name + "lcs")
        e2_ctr.Placement = e2.Placement
    return e2


def make_tf(variable_name, parent_parms):
    if parent_parms.get(variable_name) == "True":
        print(f"{variable_name} = True" )
        return True
    else:
        print(f"{variable_name} = False")
        return False


def make_helix(App, a, b, t, n):
    step = t / n
    eval_points = []
    cum_step = 0
    while cum_step <= t:
        eval_points.append(App.Vector(a * np.cos(cum_step) - .5, a * np.sin(cum_step), cum_step + 7))
        cum_step += step
    return eval_points


def make_sweep(App, sweep_nbr, a, b, t, n):
    doc = App.ActiveDocument
    helix_points = make_helix(App, a, b, t, n)
    points = [App.Vector(0, 0, 0), App.Vector(0, 0, 2), App.Vector(0, 0, 4)]
    points += helix_points
    spline1 = Draft.make_bspline(points, closed=False)
    circle_name = 'Circle' + str(sweep_nbr)
    circle = doc.addObject("Part::Circle", circle_name)
    circle.Radius = 15
    circle.Placement = App.Placement(points[0], App.Rotation(45, 0, 0))
    s_name = 'Sweep' + str(sweep_nbr)
    sweep = doc.addObject('Part::Sweep', s_name)
    sweep.Sections = [circle]
    sweep.Spine = (spline1, ['Edge1'])
    sweep.Solid = True
    sweep.Frenet = False
    return sweep


def make_placement(App, place_str):
    vectors = re.findall(r'\(.+?\)', place_str)
    if len(vectors) < 2:
        print(f"FOUND BAD PLACEMENT: {place_str}")
        return
    pos = eval("FreeCAD.Vector" + vectors[0])
    rot = eval("FreeCAD.Rotation" + vectors[1])
    newplace = FreeCAD.Placement(pos, rot)
    return newplace

