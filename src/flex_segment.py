try:
    from core.logging_setup import get_logger
except Exception:
    try:
        from logging_setup import get_logger
    except Exception:
        import logging
        get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import sys

import numpy as np
import csv
from wafer import Wafer
import FreeCAD
import FreeCADGui
from utilities import position_to_str


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
        # Create groups for organization
        self.main_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "segment_group")
        self.visualization_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "visualization")
        self.wafer_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "wafers")

        self.main_group.addObject(self.visualization_group)
        self.main_group.addObject(self.lcs_group)
        self.main_group.addObject(self.wafer_group)

        self.transform_callbacks = []
        self._setup_transform_properties()
        self.already_relocated = False  # Track relocation status
        self.stopper = False

    def add_wafer(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE",
                  start_pos=None, end_pos=None, curve_tangent=None):
        """Add a wafer with optional 3D positioning data.

        Args:
            lift: float, lift angle in radians
            rotation: float, rotation angle in radians
            cylinder_diameter: float, diameter of cylinder
            outside_height: float, outside height of wafer
            wafer_type: str, type of wafer (CE, EE, EC, CC)
            start_pos: numpy array [x, y, z] or None
            end_pos: numpy array [x, y, z] or None
            curve_tangent: tangent to curve, if provided
            """

        # Initialize wafer_count if not exists
        if not hasattr(self, 'wafer_count'):
            self.wafer_count = 0
            self._prev_wafer_axis = None

        # Only increment wafer_count ONCE
        self.wafer_count += 1

        logger.debug(f"\n=== Creating Wafer {self.wafer_count} ===")
        logger.debug(f"  Type: {wafer_type}")
        logger.debug(f"  Lift: {np.rad2deg(lift):.2f}°")
        logger.debug(f"  Rotation: {np.rad2deg(rotation):.2f}°")

        # Create wafer objects
        name_base = self.prefix + str(self.wafer_count)
        wafer_name = name_base + "_w"
        wafer = Wafer(FreeCAD, self.gui, self.prefix, wafer_type=wafer_type)
        wafer.set_parameters(lift, rotation, cylinder_diameter, outside_height, wafer_type=wafer_type)

        lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_1lcs")
        lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_2lcs")
        self.lcs_group.addObjects([lcs1, lcs2])
        wafer.lcs1 = lcs1
        wafer.lcs2 = lcs2
        wafer.segment = self

        if not self.show_lcs:
            lcs1.Visibility = False
            lcs2.Visibility = False

        # CYLINDER INTERSECTION PLANE APPROACH
        if start_pos is not None and end_pos is not None:
            # Ensure we're working with numpy arrays
            if not isinstance(start_pos, np.ndarray):
                start_pos = np.array(start_pos)
            if not isinstance(end_pos, np.ndarray):
                end_pos = np.array(end_pos)

            # Current wafer axis (cylinder axis)
            wafer_vector = end_pos - start_pos  # This now works with numpy arrays
            wafer_length = np.linalg.norm(wafer_vector)

            if wafer_length > 1e-6:
                wafer_direction = wafer_vector / wafer_length
                current_axis = FreeCAD.Vector(wafer_direction[0], wafer_direction[1], wafer_direction[2])

                # For lcs1 (start of wafer)
                if self.wafer_count == 1:
                    logger.debug(f"    📍 First wafer - creating initial LCS")
                    # Calculate rotation matrix based on wafer direction
                    z_axis = current_axis  # Along the cylinder
                    logger.debug(f"    📍 Z-axis (current_axis): {z_axis}")

                    # Create consistent orientation
                    if abs(z_axis.dot(FreeCAD.Vector(0, 0, 1))) > 0.99:
                        # Nearly vertical - use standard axes
                        x_axis = FreeCAD.Vector(1, 0, 0)
                        y_axis = FreeCAD.Vector(0, 1, 0)
                        logger.debug("    📍 Using standard orientation for vertical cylinder")
                    else:
                        # Non-vertical - create perpendicular frame
                        ref_vec = FreeCAD.Vector(0, 0, 1)
                        x_axis = z_axis.cross(ref_vec).normalize()
                        y_axis = z_axis.cross(x_axis).normalize()
                        logger.debug(f"    📍 Created perpendicular frame for non-vertical cylinder")
                        logger.debug(f"       X-axis: {x_axis}")
                        logger.debug(f"       Y-axis: {y_axis}")

                    # Create rotation matrix
                    rotation_matrix = FreeCAD.Matrix(
                        x_axis.x, y_axis.x, z_axis.x, 0,
                        x_axis.y, y_axis.y, z_axis.y, 0,
                        x_axis.z, y_axis.z, z_axis.z, 0,
                        0, 0, 0, 1
                    )

                    lcs1.Placement = FreeCAD.Placement(
                        FreeCAD.Vector(start_pos[0], start_pos[1], start_pos[2]),
                        FreeCAD.Rotation(rotation_matrix)
                    )
                    logger.debug(f"    📍 Set LCS1 placement with rotation")
                else:
                    # Use stored placement from previous wafer in same segment
                    if hasattr(self, '_last_lcs2_placement'):
                        lcs1.Placement = self._last_lcs2_placement
                    else:
                        logger.error("ERROR: No previous LCS2 placement found!")

                # For lcs2 (end of wafer)
                # We need to know the next wafer's direction to calculate the bisecting plane
                # The lift angle tells us the angle between current and next wafer

                if wafer_type[1] == 'C':  # Circular end (perpendicular cut)
                    z_axis = current_axis

                    if abs(z_axis.x) < 0.001 and abs(z_axis.y) < 0.001 and abs(abs(z_axis.z) - 1.0) < 0.001:
                        # Vertical cylinder - use standard orientation
                        x_axis = FreeCAD.Vector(1, 0, 0)
                        y_axis = FreeCAD.Vector(0, 1, 0)
                        logger.debug("    📌 Using standard orientation for vertical cylinder LCS2")
                    else:
                        # Non-vertical - existing cross product logic
                        if abs(z_axis.z) < 0.9:
                            x_axis = FreeCAD.Vector(0, 0, 1).cross(z_axis)
                        else:
                            x_axis = FreeCAD.Vector(1, 0, 0).cross(z_axis)

                        x_axis.normalize()
                        y_axis = z_axis.cross(x_axis)
                        y_axis.normalize()

                    rotation_matrix = FreeCAD.Matrix(
                        x_axis.x, y_axis.x, z_axis.x, 0,
                        x_axis.y, y_axis.y, z_axis.y, 0,
                        x_axis.z, y_axis.z, z_axis.z, 0,
                        0, 0, 0, 1
                    )

                    test_rot = FreeCAD.Rotation(rotation_matrix)
                    euler = test_rot.toEuler()
                    logger.debug(
                        f"       Rotation will create: Yaw={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Roll={euler[2]:.1f}°")
                    if abs(euler[0]) > 1.0 or abs(euler[1]) > 1.0 or abs(euler[2]) > 1.0:
                        logger.info(f"       ⚠️ WARNING: Non-identity rotation detected!")

                else:  # Elliptical end (angled cut)
                    # The bisecting plane normal can be calculated from:
                    # 1. Current cylinder axis
                    # 2. The lift angle (which is half the bend angle)

                    # The bisecting plane contains the current axis
                    # and makes an angle with it based on the lift

                    # Start with a perpendicular frame
                    if abs(current_axis.z) < 0.9:
                        temp_x = FreeCAD.Vector(0, 0, 1).cross(current_axis)
                    else:
                        temp_x = FreeCAD.Vector(1, 0, 0).cross(current_axis)

                    temp_x.normalize()
                    temp_y = current_axis.cross(temp_x)
                    temp_y.normalize()

                    # The bisecting plane normal is tilted from current_axis by the lift angle
                    # Rotate around temp_x by the lift angle
                    tilt_matrix = FreeCAD.Matrix()
                    tilt_matrix.rotateX(lift)

                    # Apply tilt to get the bisecting plane normal
                    base_matrix = FreeCAD.Matrix(
                        temp_x.x, temp_y.x, current_axis.x, 0,
                        temp_x.y, temp_y.y, current_axis.y, 0,
                        temp_x.z, temp_y.z, current_axis.z, 0,
                        0, 0, 0, 1
                    )

                    tilted_matrix = base_matrix.multiply(tilt_matrix)

                    # Extract the z-axis (bisecting plane normal)
                    z_axis = FreeCAD.Vector(tilted_matrix.A13, tilted_matrix.A23, tilted_matrix.A33)

                    # The x-axis (major axis of ellipse) lies in the plane of the two cylinders
                    # This is perpendicular to both the bisecting plane normal and
                    # the cross product of the two cylinder axes

                    # For now, we'll keep the x-axis in a consistent orientation
                    x_axis = FreeCAD.Vector(tilted_matrix.A11, tilted_matrix.A21, tilted_matrix.A31)
                    y_axis = FreeCAD.Vector(tilted_matrix.A12, tilted_matrix.A22, tilted_matrix.A32)

                    # Apply rotation for EE wafers
                    if wafer_type == "EE" and abs(rotation) > 1e-6:
                        rotation_matrix = FreeCAD.Matrix()
                        rotation_matrix.rotateZ(rotation)
                        tilted_matrix = tilted_matrix.multiply(rotation_matrix)
                        logger.debug(f"  Applied EE rotation of {np.rad2deg(rotation):.2f}°")

                    rotation_matrix = tilted_matrix

                lcs2.Placement = FreeCAD.Placement(
                    FreeCAD.Vector(end_pos[0], end_pos[1], end_pos[2]),
                    FreeCAD.Rotation(rotation_matrix)
                )

                # Store for next wafer
                self._last_lcs2_placement = lcs2.Placement
                self._prev_wafer_axis = current_axis

                # Update lcs_top
                self.lcs_top.Placement = lcs2.Placement

                logger.debug(
                    f"  LCS1 at: [{lcs1.Placement.Base.x:.3f}, {lcs1.Placement.Base.y:.3f}, {lcs1.Placement.Base.z:.3f}]")
                logger.debug(
                    f"  LCS2 at: [{lcs2.Placement.Base.x:.3f}, {lcs2.Placement.Base.y:.3f}, {lcs2.Placement.Base.z:.3f}]")

        else:
            raise ValueError(f"Missing start or end position")

        # Create the wafer geometry
        try:
            # Calculate cut angles based on wafer type and lift
            if wafer_type[0] == 'C':  # Circular start
                start_cut_angle = 0.0
            else:  # Elliptical start
                start_cut_angle = lift  # Or calculate based on geometry

            if wafer_type[1] == 'C':  # Circular end
                end_cut_angle = 0.0
            else:  # Elliptical end
                end_cut_angle = lift  # Or calculate based on geometry

            wafer.make_wafer_from_lcs(lcs1, lcs2, cylinder_diameter, wafer_name,
                                      start_cut_angle, end_cut_angle)
            self._last_wafer = wafer
            self.wafer_list.append(wafer)

            # Ensure wafer is placed under the segment's wafers subgroup
            try:
                # Create wafers group if missing
                if not hasattr(self, 'wafer_group') or self.wafer_group is None:
                    self.wafer_group = self.doc.addObject('App::DocumentObjectGroup', self.prefix + 'wafers')
                    if hasattr(self, 'main_group') and self.main_group is not None:
                        try:
                            self.main_group.addObject(self.wafer_group)
                        except Exception:
                            pass
                # Move the created wafer object under wafers group
                if hasattr(wafer, 'wafer') and wafer.wafer is not None:
                    try:
                        self.wafer_group.addObject(wafer.wafer)
                    except Exception:
                        try:
                            current = list(getattr(self.wafer_group, 'Group', []))
                            current.append(wafer.wafer)
                            self.wafer_group.Group = current
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception as e:
            logger.error(f"ERROR creating wafer: {e}")
            import traceback
            traceback.print_exc()

        logger.debug(f"  ✅ Wafer {self.wafer_count} completed")

    def get_segment_name(self):
        return self.prefix[:-1]    # strip trailing underscore

    def get_segment_object(self):
        return self.segment_object

    def get_wafer_count(self):
        return self.wafer_count

    def get_lcs_top(self):
        # logger.debug(f"LCS TOP: {self.lcs_top.Label}")
        return self.lcs_top

    def get_lcs_base(self):
        if not self.lcs_base:
            raise ValueError(f"lcs_base not set")
        return self.lcs_base

    def fuse_wafers(self):
        """Fuse wafers using sequential binary fusion for robustness."""
        name = self.get_segment_name()

        if len(self.wafer_list) == 0:
            raise ValueError("Zero Length Wafer List when building helix")
        elif len(self.wafer_list) == 1:
            fuse = self.wafer_list[0].wafer
            fuse.Label = name + "FusedResult"
        else:
            # Start with first wafer
            result = self.wafer_list[0].wafer.Shape.copy()

            # Fuse each subsequent wafer
            for i in range(1, len(self.wafer_list)):
                try:
                    wafer_shape = self.wafer_list[i].wafer.Shape

                    # Check if shapes are valid before fusion
                    if not result.isValid():
                        logger.error(f"  Warning: Result shape invalid before fusing wafer {i + 1}")
                        result.fix(0.1, 0.1, 0.1)

                    if not wafer_shape.isValid():
                        logger.error(f"  Warning: Wafer {i + 1} shape invalid")
                        wafer_shape.fix(0.1, 0.1, 0.1)

                    # Perform fusion
                    logger.debug(f"  Fusing wafer {i + 1}/{len(self.wafer_list)}...")
                    new_result = result.fuse(wafer_shape)

                    # Check result
                    if new_result.isValid():
                        result = new_result
                    else:
                        logger.error(f"  Warning: Fusion result invalid at wafer {i + 1}, trying to fix...")
                        new_result.fix(0.1, 0.1, 0.1)
                        if new_result.isValid():
                            result = new_result
                        else:
                            logger.error(f"  Could not fix fusion result, skipping wafer {i + 1}")

                except Exception as e:
                    logger.error(f"  Error fusing wafer {i + 1}: {e}")
                    continue

            # Create final fused object
            fuse = self.doc.addObject("Part::Feature", name + "FusedResult")
            fuse.Shape = result

        # Update base LCS to match first wafer FIRST
        if self.wafer_list and hasattr(self.wafer_list[0], 'lcs1'):
            first_wafer_lcs1 = self.wafer_list[0].lcs1
            if first_wafer_lcs1.Placement.Rotation.toEuler() != self.lcs_base.Placement.Rotation.toEuler():
                logger.debug(f"  Updating base LCS rotation to match first wafer")
                self.lcs_base.Placement = FreeCAD.Placement(
                    self.lcs_base.Placement.Base,  # Keep position
                    first_wafer_lcs1.Placement.Rotation  # Update rotation
                )

        fuse.Visibility = True
        fuse.ViewObject.DisplayMode = "Shaded"
        # fuse.Placement = FreeCAD.Placement
        self.segment_object = fuse

        # DEBUG: Check bounds of fused object
        if fuse.Shape.BoundBox.isValid():
            bb = fuse.Shape.BoundBox
            logger.debug(f"\n🔍 FUSED GEOMETRY BOUNDS for {self.get_segment_name()}:")
            logger.debug(f"   Min: [{bb.XMin:.3f}, {bb.YMin:.3f}, {bb.ZMin:.3f}]")
            logger.debug(f"   Max: [{bb.XMax:.3f}, {bb.YMax:.3f}, {bb.ZMax:.3f}]")
            logger.debug(f"   Base LCS: {self.lcs_base.Placement.Base}")
            logger.debug(f"   Top LCS: {self.lcs_top.Placement.Base}")

        # Add the fused result to the main group
        if self.segment_object:
            self.main_group.addObject(self.segment_object)

        self.doc.recompute()
        logger.info(f"Fusion complete - result is {'valid' if fuse.Shape.isValid() else 'INVALID'}")

    def get_segment_rotation(self):
        return self.rotate_segment

    def move_content(self, transform):
        """Move all segment content by the given transform."""
        logger.debug(f"\n  📦 MOVE_CONTENT called for {self.get_segment_name()}")
        logger.debug(f"    Input transform: {transform}")
        logger.debug(f"    Transform position: {transform.Base}")
        logger.debug(f"    Transform rotation: {transform.Rotation.toEuler()}")

        if self.segment_object:
            logger.debug(f"    BEFORE move_content:")
            logger.debug(f"      segment_object: {self.segment_object.Placement}")
            logger.debug(f"      lcs_base: {self.lcs_base.Placement}")
            logger.debug(f"      lcs_top: {self.lcs_top.Placement}")

            # Apply transforms
            self.segment_object.Placement = transform.multiply(self.segment_object.Placement)
            self.lcs_base.Placement = transform.multiply(self.lcs_base.Placement)
            self.lcs_top.Placement = transform.multiply(self.lcs_top.Placement)

            logger.debug(f"    AFTER move_content:")
            logger.debug(f"      segment_object: {self.segment_object.Placement}")
            logger.debug(f"      lcs_base: {self.lcs_base.Placement}")
            logger.debug(f"      lcs_top: {self.lcs_top.Placement}")

            # Also move individual wafers if they exist
            for wafer in self.wafer_list:
                wafer_obj = wafer.get_wafer()
                if wafer_obj:
                    wafer_obj.Placement = transform.multiply(wafer_obj.Placement)

            # Move the groups too
            if hasattr(self, 'main_group'):
                self.main_group.touch()
        else:
            logger.error(f"NO CONTENT: {self.get_segment_name()} has no content to move.")

        # Force visual update
        if self.segment_object and hasattr(self.segment_object, 'ViewObject'):
            self.segment_object.ViewObject.Visibility = False
            self.segment_object.ViewObject.Visibility = True

        # Force document recompute
        self.doc.recompute()

        # Force GUI update if available
        try:
            import FreeCADGui
            FreeCADGui.updateGui()
        except:
            pass
    def remove_prior_version(self):
        # TODO: not do so if making cut list???
        name = self.prefix + ".+"
        doc = FreeCAD.activeDocument()
        doc_list = doc.findObjects(Name=name)  # remove prior occurrence of set being built
        for item in doc_list:
            if item.Label != 'Parms_Master':
                doc.removeObject(item.Name)

    def make_cut_list(self, segment_no, cuts_file):
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
        """Print list giving local and global position and orientation of each wafer."""
        parm_str = f"\nConstruction list for segment: {self.get_segment_name()}\n"
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
        # logger.debug(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        return tr

    def _setup_transform_properties(self):
        """Add custom properties to store transformation data."""
        if not hasattr(self.lcs_base, 'AppliedTransformMatrix'):
            # Store the 4x4 transformation matrix as a string (FreeCAD can't store Matrix directly)
            self.lcs_base.addProperty("App::PropertyString", "AppliedTransformMatrix",
                                      "Transform", "Applied transformation matrix as string")
            self.lcs_base.addProperty("App::PropertyStringList", "CurveVerticesGroups",
                                      "Transform", "Names of associated curve vertex groups")
            self.lcs_base.addProperty("App::PropertyBool", "HasStoredTransform",
                                      "Transform", "Whether this segment has a stored transform")

        # Initialize
        self.lcs_base.HasStoredTransform = False
        self.lcs_base.CurveVerticesGroups = []

    def store_applied_transform(self, transform):
        """Store transform in FreeCAD object properties for persistence."""
        if transform:
            # Convert FreeCAD.Placement to matrix and store as string
            matrix = transform.toMatrix()
            matrix_str = ",".join([str(matrix.A[i]) for i in range(16)])

            self.lcs_base.AppliedTransformMatrix = matrix_str
            self.lcs_base.HasStoredTransform = True

            logger.debug(f"Stored transform matrix in FreeCAD object: {self.lcs_base.Label}")
        else:
            self.lcs_base.HasStoredTransform = False

    def register_curve_vertices_group(self, group_name):
        """Register a curve vertices group with this segment."""
        # Store the group name
        current_groups = list(self.lcs_base.CurveVerticesGroups)
        if group_name not in current_groups:
            current_groups.append(group_name)
            self.lcs_base.CurveVerticesGroups = current_groups

        # Force document recompute before trying to find objects
        self.doc.recompute()

        # Find and move the vertex group into this segment's visualization group
        vertex_groups = self.doc.getObjectsByLabel(group_name)
        if vertex_groups:
            vertex_group = vertex_groups[0]
            try:
                # Remove from document root if it's there
                if hasattr(vertex_group, 'removeFromDocument'):
                    pass  # It's already in the document

                # Add to visualization group
                self.visualization_group.addObject(vertex_group)
                logger.info(f"Successfully moved vertex group '{group_name}' to visualization group")
            except Exception as e:
                logger.error(f"Failed to move vertex group to visualization group: {e}")
        else:
            logger.info(f"Warning: Could not find vertex group '{group_name}' to move to visualization group")

    def register_arrow(self, arrow_obj):
        """Register an arrow with this segment's visualization group."""
        if arrow_obj:
            self.visualization_group.addObject(arrow_obj)
            logger.debug(f"Added arrow '{arrow_obj.Label}' to segment visualization group")

    def register_transform_callback(self, callback_func):
        """Register a function to be called when segment is transformed."""
        self.transform_callbacks.append(callback_func)

    def _transform_registered_vertices(self, transform):
        """Transform all registered curve vertex groups."""
        for group_name in self.lcs_base.CurveVerticesGroups:
            curve_groups = self.doc.getObjectsByLabel(group_name)
            if curve_groups:
                vertex_group_obj = curve_groups[0]

                # Transform each individual vertex in the group
                for vertex_obj in vertex_group_obj.OutList:
                    if hasattr(vertex_obj, 'Placement'):
                        current_placement = vertex_obj.Placement
                        new_placement = transform.multiply(current_placement)
                        vertex_obj.Placement = new_placement

                        # Force refresh this specific object
                        vertex_obj.touch()
                        if hasattr(vertex_obj, 'ViewObject'):
                            vertex_obj.ViewObject.Visibility = False
                            vertex_obj.ViewObject.Visibility = True

                logger.debug(f"Transformed vertex group '{group_name}' with {len(vertex_group_obj.OutList)} vertices")
            else:
                logger.error(f"Warning: Could not find vertex group '{group_name}' for transformation")

        # Force document recompute after all vertex transforms
        self.doc.recompute()

    @staticmethod
    def find_segments_with_transforms(doc=None):
        """Find all segments in document that have stored transforms."""
        if doc is None:
            doc = FreeCAD.ActiveDocument

        segments_with_transforms = []

        for obj in doc.Objects:
            if hasattr(obj, 'HasStoredTransform') and obj.HasStoredTransform:
                segments_with_transforms.append(obj)

        return segments_with_transforms

    @staticmethod
    def apply_transform_to_segment_and_vertices(segment_lcs_obj, new_transform):
        """Apply a new transform to a segment and its associated vertices (for external macros)."""
        try:
            # Get the segment object (find by LCS reference)
            segment_name = segment_lcs_obj.Label.replace('_lcs_base', '')
            segment_objects = [obj for obj in segment_lcs_obj.Document.Objects
                               if obj.Label.startswith(segment_name) and hasattr(obj, 'Shape')]

            # Apply transform to segment geometry
            for seg_obj in segment_objects:
                current_placement = seg_obj.Placement
                new_placement = new_transform.multiply(current_placement)
                seg_obj.Placement = new_placement

            # Apply to registered curve vertices
            for group_name in segment_lcs_obj.CurveVerticesGroups:
                curve_groups = segment_lcs_obj.Document.getObjectsByLabel(group_name)
                if curve_groups:
                    vertex_group_obj = curve_groups[0]
                    for vertex_obj in vertex_group_obj.OutList:
                        current_placement = vertex_obj.Placement
                        new_placement = new_transform.multiply(current_placement)
                        vertex_obj.Placement = new_placement

            # Update stored transform
            if hasattr(segment_lcs_obj, 'AppliedTransformMatrix'):
                old_transform = FlexSegment._parse_stored_transform(segment_lcs_obj)
                combined_transform = new_transform.multiply(old_transform) if old_transform else new_transform
                FlexSegment._store_transform_in_object(segment_lcs_obj, combined_transform)

            logger.debug(f"Applied external transform to segment {segment_name}")

        except Exception as e:
            logger.error(f"Error applying external transform: {e}")

    @staticmethod
    def _parse_stored_transform(lcs_obj):
        """Helper to parse stored transform from FreeCAD object."""
        if hasattr(lcs_obj, 'AppliedTransformMatrix') and lcs_obj.AppliedTransformMatrix:
            try:
                matrix_values = [float(x) for x in lcs_obj.AppliedTransformMatrix.split(',')]
                matrix = FreeCAD.Matrix()
                for i in range(16):
                    matrix.A[i] = matrix_values[i]
                return FreeCAD.Placement(matrix)
            except:
                return None
        return None

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        self.move_content(transform)

        # Store transform in FreeCAD properties for persistence
        self.store_applied_transform(transform)

        # NOW create vertices in their final transformed positions
        self._create_transformed_vertices(transform)

        # Call Python callbacks
        for callback in self.transform_callbacks:
            callback(transform)

    def validate_segment_geometry(self, tagline="SEGMENT VALIDATION: "):
        """Validate that segment geometry matches LCS positions and print detailed info.

        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.segment_object:
            return False, "No segment object exists"

        if not self.wafer_list:
            return False, "No wafers in segment"

        logger.debug(f"\n🔍 {tagline} {self.get_segment_name()}:")

        # Print base LCS information
        base_pos = self.lcs_base.Placement.Base
        base_rot = self.lcs_base.Placement.Rotation.toEuler()
        logger.debug(f"   Base LCS:")
        logger.debug(f"     Position: [{base_pos.x:.3f}, {base_pos.y:.3f}, {base_pos.z:.3f}]")
        logger.debug(f"     Rotation: [Yaw={base_rot[0]:.1f}°, Pitch={base_rot[1]:.1f}°, Roll={base_rot[2]:.1f}°]")

        # Get wafer positions and rotations
        first_wafer = self.wafer_list[0]
        last_wafer = self.wafer_list[-1]

        try:
            # Try to get LCS positions and rotations from wafer objects
            if hasattr(first_wafer, 'get_lcs1') and hasattr(first_wafer, 'get_lcs2'):
                first_start_pos = first_wafer.get_lcs1().Placement.Base
                first_start_rot = first_wafer.get_lcs1().Placement.Rotation.toEuler()
                last_end_pos = last_wafer.get_lcs2().Placement.Base
                last_end_rot = last_wafer.get_lcs2().Placement.Rotation.toEuler()
            elif hasattr(first_wafer, 'lcs1') and hasattr(last_wafer, 'lcs2'):
                first_start_pos = first_wafer.lcs1.Placement.Base
                first_start_rot = first_wafer.lcs1.Placement.Rotation.toEuler()
                last_end_pos = last_wafer.lcs2.Placement.Base
                last_end_rot = last_wafer.lcs2.Placement.Rotation.toEuler()
            else:
                # Try to get from FreeCAD objects directly
                lcs1_name = f"{self.get_segment_name()}_1_1lcs"
                lcs2_name = f"{self.get_segment_name()}_{len(self.wafer_list)}_2lcs"

                lcs1_obj = self.doc.getObject(lcs1_name)
                lcs2_obj = self.doc.getObject(lcs2_name)

                if lcs1_obj and lcs2_obj:
                    first_start_pos = lcs1_obj.Placement.Base
                    first_start_rot = lcs1_obj.Placement.Rotation.toEuler()
                    last_end_pos = lcs2_obj.Placement.Base
                    last_end_rot = lcs2_obj.Placement.Rotation.toEuler()
                else:
                    return False, "Cannot access wafer LCS positions"

        except Exception as e:
            return False, f"Error accessing wafer positions: {e}"

        logger.debug(f"   First wafer start (LCS1):")
        logger.debug(f"     Position: [{first_start_pos.x:.3f}, {first_start_pos.y:.3f}, {first_start_pos.z:.3f}]")
        logger.debug(
            f"     Rotation: [Yaw={first_start_rot[0]:.1f}°, Pitch={first_start_rot[1]:.1f}°, Roll={first_start_rot[2]:.1f}°]")

        top_pos = self.lcs_top.Placement.Base
        top_rot = self.lcs_top.Placement.Rotation.toEuler()
        logger.debug(f"   Top LCS:")
        logger.debug(f"     Position: [{top_pos.x:.3f}, {top_pos.y:.3f}, {top_pos.z:.3f}]")
        logger.debug(f"     Rotation: [Yaw={top_rot[0]:.1f}°, Pitch={top_rot[1]:.1f}°, Roll={top_rot[2]:.1f}°]")

        logger.debug(f"   Last wafer end (LCS2):")
        logger.debug(f"     Position: [{last_end_pos.x:.3f}, {last_end_pos.y:.3f}, {last_end_pos.z:.3f}]")
        logger.debug(f"     Rotation: [Yaw={last_end_rot[0]:.1f}°, Pitch={last_end_rot[1]:.1f}°, Roll={last_end_rot[2]:.1f}°]")

        # Print segment object info
        seg_pos = self.segment_object.Placement.Base
        seg_rot = self.segment_object.Placement.Rotation.toEuler()
        logger.debug(f"   Segment object (fused):")
        logger.debug(f"     Position: [{seg_pos.x:.3f}, {seg_pos.y:.3f}, {seg_pos.z:.3f}]")
        logger.debug(f"     Rotation: [Yaw={seg_rot[0]:.1f}°, Pitch={seg_rot[1]:.1f}°, Roll={seg_rot[2]:.1f}°]")

        # Check segment object bounds
        bounds = self.segment_object.Shape.BoundBox
        if bounds.isValid():
            logger.debug(f"     Bounds: Min=[{bounds.XMin:.3f}, {bounds.YMin:.3f}, {bounds.ZMin:.3f}] "
                  f"Max=[{bounds.XMax:.3f}, {bounds.YMax:.3f}, {bounds.ZMax:.3f}]")

        # Position validation
        base_error = (base_pos - first_start_pos).Length
        top_error = (top_pos - last_end_pos).Length

        logger.debug(f"\n   Position errors:")
        logger.debug(f"     Base LCS vs First wafer: {base_error:.6f}")
        logger.debug(f"     Top LCS vs Last wafer: {top_error:.6f}")

        # Rotation comparison
        logger.debug(f"\n   Rotation comparisons:")
        logger.debug(f"     Base LCS vs First wafer: ΔYaw={base_rot[0] - first_start_rot[0]:.1f}°, "
              f"ΔPitch={base_rot[1] - first_start_rot[1]:.1f}°, ΔRoll={base_rot[2] - first_start_rot[2]:.1f}°")
        logger.debug(f"     Top LCS vs Last wafer: ΔYaw={top_rot[0] - last_end_rot[0]:.1f}°, "
              f"ΔPitch={top_rot[1] - last_end_rot[1]:.1f}°, ΔRoll={top_rot[2] - last_end_rot[2]:.1f}°")

        # Validation result
        if base_error > 0.001:
            return False, f"Base LCS mismatch: {base_error:.6f} units from first wafer start"

        if top_error > 0.001:
            return False, f"Top LCS mismatch: {top_error:.6f} units from last wafer end"

        return True, "Segment geometry valid"

    def fix_segment_lcs_alignment(self):
        """Fix LCS positions to match actual wafer geometry."""
        if not self.wafer_list:
            logger.info(f"Cannot fix alignment - no wafers")
            return False

        try:
            # Get actual wafer positions
            first_wafer = self.wafer_list[0]
            last_wafer = self.wafer_list[-1]

            # Update base LCS to match first wafer
            if hasattr(first_wafer, 'lcs1'):
                self.lcs_base.Placement = FreeCAD.Placement(first_wafer.lcs1.Placement)
                logger.debug(f"Updated base LCS to match first wafer: {self.lcs_base.Placement}")

            # Update top LCS to match last wafer
            if hasattr(last_wafer, 'lcs2'):
                self.lcs_top.Placement = FreeCAD.Placement(last_wafer.lcs2.Placement)
                logger.debug(f"Updated top LCS to match last wafer: {self.lcs_top.Placement}")

            return True

        except Exception as e:
            logger.error(f"Error fixing alignment: {e}")
            return False

    def _create_transformed_vertices(self, transform):
        """Create vertices directly in their final transformed positions."""
        if not hasattr(self, 'pending_vertices'):
            return

        for group_name, curve_points in self.pending_vertices.items():
            # Remove any existing group
            existing_groups = self.doc.getObjectsByLabel(group_name)
            for group in existing_groups:
                self.doc.removeObject(group.Name)

            # Create new group
            point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)
            point_group.Label = group_name

            # Create vertices directly in final positions
            vertices = []
            for i, point in enumerate(curve_points):
                vertex_name = f"{group_name}_point_{i}"
                vertex_obj = self.doc.addObject('Part::Vertex', vertex_name)

                # Apply transform to original point to get final position
                original_placement = FreeCAD.Placement(FreeCAD.Vector(*point), FreeCAD.Rotation())
                final_placement = transform.multiply(original_placement)

                vertex_obj.Placement = final_placement
                vertices.append(vertex_obj)

            # Add to group and visualization
            point_group.addObjects(vertices)
            self.visualization_group.addObject(point_group)

            logger.debug(f"Created {len(vertices)} vertices in final positions for group '{group_name}'")

        # Clear pending vertices
        self.pending_vertices = {}

    @staticmethod
    def _store_transform_in_object(lcs_obj, transform):
        """Helper to store transform in FreeCAD object."""
        matrix = transform.toMatrix()
        matrix_str = ",".join([str(matrix.A[i]) for i in range(16)])
        lcs_obj.AppliedTransformMatrix = matrix_str
        lcs_obj.HasStoredTransform = True


