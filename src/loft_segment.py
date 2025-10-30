from core.logging_setup import get_logger, log_coord, apply_display_levels
apply_display_levels(["ERROR", "WARNING", "INFO", "DEBUG"])
# apply_display_levels(["ERROR", "WARNING", "INFO"])
logger = get_logger(__name__)
import sys
import numpy as np
import math
import csv
from core.core_utils import add_to_group, ensure_group
from wafer import Wafer, log_lcs_info, log_lcs_debug, log_lcs_info_level
import FreeCAD
import FreeCADGui
import Part
from utilities import position_to_str


class LoftSegment(object):
    def __init__(self, prefix,  show_lcs, temp_file, to_build, rotate_segment):
        self.doc = FreeCAD.ActiveDocument
        self.gui = FreeCADGui
        self.prefix = prefix + "_"
        self.rotate_segment = rotate_segment
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
        self.segment_object = None  # holder for fused segment object

        # Add bounds (set in driver:relocate_segment after final placement
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None


        try:
            self.wafer_group.addObjects([self.lcs_base, self.lcs_top])
        except Exception:
            try:
                self.wafer_group.addObject(self.lcs_base)
                self.wafer_group.addObject(self.lcs_top)
            except Exception:
                # very old FreeCAD: fall back to raw Group property
                current = list(getattr(self.wafer_group, "Group", []))
                for obj in (self.lcs_base, self.lcs_top):
                    if obj not in current:
                        current.append(obj)
                self.wafer_group.Group = current

        self.already_relocated = False  # Track relocation status
        self.stopper = False

    def remove_prior_version(self):
        name = self.prefix + ".+"
        doc = FreeCAD.activeDocument()
        doc_list = doc.findObjects(Name=name)  # remove prior occurrence of set being built
        for item in doc_list:
            doc.removeObject(item.Name)

    def get_segment_name(self):
        return self.prefix[:-1]    # strip trailing underscore

    def get_segment_object(self):
        return self.segment_object

    def get_lcs_top(self):
        # logger.debug(f"LCS TOP: {self.lcs_top.Label}")
        return self.lcs_top

    def get_lcs_base(self):
        if not self.lcs_base:
            raise ValueError(f"lcs_base not set")
        return self.lcs_base

    def get_segment_object(self):
        return self.segment_object

    def get_wafer_list(self):
        return self.wafer_list

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
        raise ValueError("Forced STOP")

        if len(self.wafer_list) == 0:
            raise ValueError("Zero Length Wafer List when building helix")
        elif len(self.wafer_list) == 1:
            # Single wafer: create a separate fused feature from the wafer's shape
            source = self.wafer_list[0].wafer
            fuse = self.doc.addObject("Part::Feature", name + "FusedResult")
            fuse.Shape = source.Shape.copy()
            # Keep the original wafer in wafer_group; segment_object will point to 'fuse'
        else:
            # Start with first wafer
            result = self.wafer_list[0].wafer.Shape.copy()

            # Fuse each subsequent wafer
            for i in range(1, len(self.wafer_list)):
                if i > 100:  # Assume a long wafer list is an error
                    raise ValueError(f"Too many wafers: {len(self.wafer_list)}")
                else:
                    waf = self.wafer_list[i]
                    log_lcs_debug(waf.lcs1, f"Wafer_{i}_Start")
                    log_lcs_debug(waf.lcs2, f"Wafer_{i}_End")
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
            self.segment_object = fuse
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

        # Add the fused result to the main group
        if self.segment_object:
            self.main_group.addObject(self.segment_object)

        self.doc.recompute()
        logger.debug(f"Fusion complete - result is {'valid' if fuse.Shape.isValid() else 'INVALID'}")

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
                # Avoid double-transform when single-wafer: skip if wafer_obj *is* segment_object
                if wafer_obj is self.segment_object:
                    wafer_obj = None
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

    def make_cut_list(self, cuts_file_obj,
                      min_lift=0.0, max_lift=45.0,
                      min_rotate=0.0, max_rotate=45.0):
        """
        Build a human-readable cut list for this segment and append to cuts_list.
        - Works with several possible wafer containers on the segment.
        - Rotation is Wafer.get_rotation_angle(expected_deg=None); CE/EC forced to 0°.
        - Accumulates a running saw position in degrees.

        Args:
            cuts_file_obj (file_obj): Output stream for cuts
            min_lift (float): Minimum lift angle in degrees.
            max_lift (float): Maximum lift angle in degrees.
            min_rotate (float): Minimum rotation angle in degrees.
            max_rotate (float): Maximum rotation angle in degrees.
        """

        seg_name = self.get_segment_name()
        wafers = self.get_wafer_list()
        if not wafers:
            none_str = f"make_cut_list: no wafers on segment '{seg_name}_'; nothing to write."
            cuts_file_obj.write(none_str)
            logger.warning(none_str)
            return

        cuts_file_obj.write(f"\tSegment '{seg_name}' cuts\n\n")
        cuts_file_obj.write("\tWafer\tType\tLift(deg)\tRotate(deg)\tOutside\t\tSawPos(deg)\n")

        rows_written = 0
        total_cylinder_length = 0
        saw_pos = 0.0
        plus_180 = True

        # Process all wafers
        for i, w in enumerate(wafers):
            next_w = wafers[i + 1] if i + 1 < len(wafers) else None
            wt = w.get_wafer_type()
            outside = w.get_outside_height()
            outside_in = int(outside)
            outside_fraction = int((outside - outside_in) * 16)
            total_cylinder_length += w.get_chord_length()
            # Calculate saw position
            if plus_180:
                add_180 = 0
                plus_180 = False
            else:
                add_180 = 180
                plus_180 = True
            if i == 0:
                saw_adj = 0.0
            else:
                saw_adj = wafers[i - 1].get_rotation_angle()
            saw_pos = (saw_pos - saw_adj) % 360.0
            corrected_saw_pos = (saw_pos + add_180) % 360.0

            # Last wafer has no cut after it
            if i == len(wafers) - 1:
                # Last wafer - no cut, just dimensions
                cuts_file_obj.write(
                    f"\t{i + 1}\t{wt}\t---\t\t---\t\t{outside_in} {outside_fraction}/16\t\t{corrected_saw_pos:.2f}\n")
            else:
                # Regular wafer with cut after it
                lift_deg = w.get_lift_angle()
                rot_deg = w.get_rotation_angle()

                # CE/EC define rotation as 0°
                if wt.endswith("C") or next_w.get_wafer_type().startswith("C"):
                    if rot_deg > 1e-6:
                        raise ValueError(f"Seg: {seg_name}, wafer: {i} has unexpected rotation: {rot_deg}")
                cuts_file_obj.write(
                    f"\t{i + 1}\t{wt}\t{lift_deg:.2f}\t\t{rot_deg:.2f}\t\t{outside_in} {outside_fraction}/16\t\t{corrected_saw_pos:.2f}\n")

                # Guardrails
                if abs(lift_deg) < min_lift or abs(lift_deg) > max_lift:
                    cuts_file_obj.write(
                        f"\n\tLift {lift_deg:.2f}° for wafer {i + 1} outside [{min_lift}, {max_lift}]°.")
                if abs(rot_deg) < min_rotate or abs(rot_deg) > max_rotate:
                    cuts_file_obj.write(
                        f"\n\yRotation {rot_deg:.2f}° for wafer {i + 1} outside [{min_rotate}, {max_rotate}]°.")

            rows_written += 1

        cuts_file_obj.write(f"\n\t\tTotal Cylinder Length: {total_cylinder_length:.2f}\n\n\n")

        cuts_file_obj.flush()
        logger.info(f"✂️  Wrote {rows_written} wafer rows ({rows_written - 1} cuts) for segment '{seg_name}'")

    def print_construction_list(self, segment_no, cons_file, global_placement, find_min_max):
        """Print list giving local and global position and orientation of each wafer."""
        parm_str = f"\nConstruction list for segment: {self.get_segment_name()}\n"
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
            num_str = str(wafer_num + 1)  # Make one-based for conformance with  in shop
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
                logger.debug(f"Successfully moved vertex group '{group_name}' to visualization group")
            except Exception as e:
                logger.error(f"Failed to move vertex group to visualization group: {e}")
        else:
            logger.warning(f"Warning: Could not find vertex group '{group_name}' to move to visualization group")

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
            if hasattr(first_wafer, 'lcs1') and hasattr(last_wafer, 'lcs2'):
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
        logger.debug(
            f"     Rotation: [Yaw={last_end_rot[0]:.1f}°, Pitch={last_end_rot[1]:.1f}°, Roll={last_end_rot[2]:.1f}°]")

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

        # Position validation with tolerance
        tol_pos = 1e-6  # meters

        def offset(a, b):
            d = (a - b).Length
            flag = "ERROR" if d > tol_pos else "OK"
            return d, flag

        d_base, f_base = offset(self.lcs_base.Placement.Base, first_start_pos)
        d_top, f_top = offset(self.lcs_top.Placement.Base, last_end_pos)

        logger.debug(f"\n   Position mismatch (m; tol={tol_pos:.1e}):")
        logger.debug(f"     Base LCS vs First wafer: {d_base:.6f}  [{f_base}]")
        logger.debug(f"     Top  LCS vs Last  wafer: {d_top:.6f}  [{f_top}]")

        # Optional: hard check
        if d_base > tol_pos or d_top > tol_pos:
            logger.warning(
                f"Segment endpoint(s) misaligned: base={d_base:.6g} m, top={d_top:.6g} m (tol={tol_pos:.1e}).")

        # Rotation comparison
        logger.debug(f"\n   Rotation comparisons:")
        logger.debug(f"     Base LCS vs First wafer: ΔYaw={base_rot[0] - first_start_rot[0]:.1f}°, "
                     f"ΔPitch={base_rot[1] - first_start_rot[1]:.1f}°, ΔRoll={base_rot[2] - first_start_rot[2]:.1f}°")
        logger.debug(f"     Top LCS vs Last wafer: ΔYaw={top_rot[0] - last_end_rot[0]:.1f}°, "
                     f"ΔPitch={top_rot[1] - last_end_rot[1]:.1f}°, ΔRoll={top_rot[2] - last_end_rot[2]:.1f}°")

        # Validation result
        if d_base > 0.001:
            return False, f"Base LCS mismatch: {d_base:.6f} units from first wafer start"

        if d_top > 0.001:
            return False, f"Top LCS mismatch: {d_top:.6f} units from last wafer end"

        return True, "Segment geometry valid"

    def fix_segment_lcs_alignment(self):
        """Fix LCS positions to match actual wafer geometry."""
        if not self.wafer_list:
            logger.warning(f"Cannot fix alignment - no wafers")
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
    def validate_ellipse_lcs_alignment(wafer_object, lcs_object, end='start'):
        """Check if the ellipse created by cutting a cylinder aligns with its LCS."""

        # Get the cylinder shape
        cylinder_shape = wafer_object.wafer.Shape

        # Get LCS position and orientation
        lcs_pos = lcs_object.Placement.Base
        lcs_x_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(1, 0, 0))
        lcs_y_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(0, 1, 0))
        lcs_z_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(0, 0, 1))

        try:
            # Create a thin disk at the LCS position
            disk_thickness = 0.01
            disk_radius = 5.0

            disk = Part.makeCylinder(disk_radius, disk_thickness,
                                     lcs_pos - lcs_z_axis * (disk_thickness / 2),
                                     lcs_z_axis)

            # Get the intersection
            common = cylinder_shape.common(disk)

            if not common or not common.Faces:
                return {'status': 'error', 'message': 'No intersection found', 'aligned': False}

            # Get the largest face
            ellipse_face = max(common.Faces, key=lambda f: f.Area)

            # Sample points on the face boundary using edges
            points = []
            n_samples = 72

            if ellipse_face.OuterWire and ellipse_face.OuterWire.Edges:
                for edge in ellipse_face.OuterWire.Edges:
                    # Sample along each edge
                    samples_per_edge = max(10, n_samples // len(ellipse_face.OuterWire.Edges))
                    try:
                        param_range = edge.ParameterRange
                        for i in range(samples_per_edge):
                            t = param_range[0] + (i / (samples_per_edge - 1)) * (param_range[1] - param_range[0])
                            point = edge.valueAt(t)
                            points.append([point.x, point.y, point.z])
                    except Exception as e:
                        logger.debug(f"Error sampling edge: {e}")
                        continue

            if len(points) < 10:
                return {'status': 'error', 'message': f'Insufficient points sampled: {len(points)}', 'aligned': False}

            points = np.array(points)

            # Calculate centroid
            centroid = points.mean(axis=0)
            centroid_vec = FreeCAD.Vector(centroid[0], centroid[1], centroid[2])

            # Project points onto LCS plane
            points_2d = []
            for p in points:
                vec = FreeCAD.Vector(p[0], p[1], p[2]) - lcs_pos
                x_comp = vec.dot(lcs_x_axis)
                y_comp = vec.dot(lcs_y_axis)
                points_2d.append([x_comp, y_comp])

            points_2d = np.array(points_2d)

            # PCA to find principal axes
            centered = points_2d - points_2d.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Sort by eigenvalue
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Major axis in 2D
            major_2d = eigenvectors[:, 0]

            # Convert to 3D
            major_axis_3d = lcs_x_axis * major_2d[0] + lcs_y_axis * major_2d[1]
            major_axis_3d.normalize()

            # Calculate alignment
            major_dot_x = abs(major_axis_3d.dot(lcs_x_axis))
            major_dot_y = abs(major_axis_3d.dot(lcs_y_axis))

            if major_dot_x > major_dot_y:
                aligned_with = 'X'
                alignment_angle = math.degrees(math.acos(min(1.0, major_dot_x)))
            else:
                aligned_with = 'Y'
                alignment_angle = math.degrees(math.acos(min(1.0, major_dot_y)))

            major_radius = math.sqrt(eigenvalues[0]) * 2
            minor_radius = math.sqrt(eigenvalues[1]) * 2
            center_error = (centroid_vec - lcs_pos).Length

            result = {
                'status': 'success',
                'ellipse_found': True,
                'major_radius': major_radius,
                'minor_radius': minor_radius,
                'center_error': center_error,
                'major_axis_aligned_with': aligned_with,
                'alignment_angle_deg': alignment_angle,
                'aligned': alignment_angle < 1.0,
                'num_points': len(points)
            }

        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'aligned': False,
                'traceback': traceback.format_exc()
            }

        # Log results
        if result['status'] == 'success':
            logger.debug(f"Ellipse-LCS Alignment ({end}): {result['major_axis_aligned_with']}-axis, "
                         f"{result['alignment_angle_deg']:.2f}°, {'ALIGNED' if result['aligned'] else 'MISALIGNED'}")
            if not result['aligned']:
                logger.warning(f"Ellipse misaligned by {result['alignment_angle_deg']:.2f}°")
        else:
            logger.error(f"Alignment check failed ({end}): {result['message']}")

        return result

    @staticmethod
    def _store_transform_in_object(lcs_obj, transform):
        """Helper to store transform in FreeCAD object."""
        matrix = transform.toMatrix()
        matrix_str = ",".join([str(matrix.A[i]) for i in range(16)])
        lcs_obj.AppliedTransformMatrix = matrix_str
        lcs_obj.HasStoredTransform = True

