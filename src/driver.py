"""
driver.py - Main workflow driver for PlantBuilder

Orchestrates the build workflow by:
- Loading configuration from YAML
- Managing FreeCAD document
- Executing operations in sequence
- Generating output files (cutting lists, placement lists)
"""

import sys
import os
import yaml
import math
from pathlib import Path
import FreeCAD as App
import Part

from core.logging_setup import get_logger
from test_transform import run_transform_test

# Get logger for this module
logger = get_logger(__name__)


class Driver:
    """Main driver class for PlantBuilder workflow"""

    def __init__(self, app, gui, doc_name):
        """
        Initialize the driver

        Args:
            app: FreeCAD App module
            gui: FreeCAD Gui module
            doc_name: Name of FreeCAD document to use
        """
        self.app = app
        self.gui = gui
        self.doc_name = doc_name
        self.config_file = None
        self.config = None
        self.doc = None
        self.segment_list = []
        self.global_settings = {}
        self.current_placement = app.Placement()
        self.output_files = {}
        self.metadata = {}


    def load_configuration(self, config_file):
        """
        Load configuration from YAML file

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        logger.info(f"Loading config: {config_file}")

        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)

            # Extract global settings
            self.global_settings = self.config.get('global_settings', {})
            self.output_files = self.config.get('output_files', {})
            self.metadata = self.config.get('metadata', {})

            logger.info("✅ Config loaded")

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def setup_document(self):
        """Setup or get FreeCAD document"""
        try:
            # Try to get existing document
            self.doc = self.app.getDocument(self.doc_name)
            logger.debug(f"Using existing '{self.doc_name}' document")
        except:
            # Create new document if it doesn't exist
            self.doc = self.app.newDocument(self.doc_name)
            logger.debug(f"Created new '{self.doc_name}' document")

        logger.debug(f"Document ready: {self.doc.Name}")

    def workflow(self):
        """Execute the complete workflow"""
        logger.info("Running: workflow()")
        #
        # # RUN TEST AND EXIT
        #
        # run_transform_test()
        # logger.info("Test complete - exiting")
        # return

        # Setup document
        self.setup_document()

        # Get workflow operations
        workflow_ops = self.config.get('workflow', [])

        if not workflow_ops:
            logger.warning("No workflow operations defined")
            return

        # Execute each operation
        for operation in workflow_ops:
            try:
                self._execute_operation(operation)
            except Exception as e:
                logger.error(f"Operation failed: {e}", exc_info=True)
                raise

        # Generate cutting list for ALL segments at the end
        if self.global_settings.get('print_cuts', False):
            self._generate_cutting_list()

        logger.info("✅ Workflow complete")

    def _execute_operation(self, operation):
        """Execute a single operation"""
        operation_type = operation.get('operation')

        if not operation_type:
            logger.warning("Operation missing 'operation' field")
            return

        # Log operation start
        description = operation.get('description', operation_type)
        logger.info(f"Executing: {description}")
        logger.debug(f"Operation details: {operation_type}")

        # Route to appropriate handler
        if operation_type == 'remove_objects':
            self._remove_objects(operation)
        elif operation_type == 'build_segment':
            self._execute_build_segment(operation)
        elif operation_type == 'set_position':
            self._execute_set_position(operation)
        else:
            logger.warning(f"Unknown operation type: {operation_type}")

    def _remove_objects(self, operation):
        """Remove objects from the FreeCAD document except those specified to keep"""
        # Check if keep_patterns key exists (even if empty)
        if 'keep_patterns' not in operation:
            logger.info("No keep_patterns specified - will remove all objects")
            keep_patterns = []
        else:
            keep_patterns = operation.get('keep_patterns', [])
            if not keep_patterns:
                logger.info("Empty keep_patterns list - will remove all objects")
            else:
                logger.info(f"Keeping objects matching patterns: {keep_patterns}")

        # Get all objects in the document
        all_objects = App.ActiveDocument.Objects
        logger.info(f"Total objects in document: {len(all_objects)}")

        # Build set of objects to keep (including their children)
        objects_to_keep = set()

        # Only process keep patterns if they exist
        if keep_patterns:
            for obj in all_objects:
                # Check if this object matches any keep pattern
                should_keep = False
                matched_pattern = None

                for pattern in keep_patterns:
                    # Exact match
                    if obj.Label == pattern or obj.Name == pattern:
                        should_keep = True
                        matched_pattern = pattern
                        break
                    # Wildcard at end: "Right*" matches "Right_Part", "Right_Loft"
                    elif pattern.endswith('*') and (
                            obj.Label.startswith(pattern[:-1]) or obj.Name.startswith(pattern[:-1])):
                        should_keep = True
                        matched_pattern = pattern
                        break
                    # Wildcard at start: "*Loft" matches "base_Loft", "Curve_Left_Loft"
                    elif pattern.startswith('*') and (
                            obj.Label.endswith(pattern[1:]) or obj.Name.endswith(pattern[1:])):
                        should_keep = True
                        matched_pattern = pattern
                        break
                    # Substring match: "Right" matches "Curve_Right_Part"
                    elif pattern in obj.Label or pattern in obj.Name:
                        should_keep = True
                        matched_pattern = pattern
                        break

                if should_keep:
                    logger.info(f"✓ MATCHED: {obj.Label} (pattern: '{matched_pattern}')")
                    objects_to_keep.add(obj)

                    # Add all children recursively
                    self._add_children_to_set(obj, objects_to_keep)

        logger.info(f"Objects to keep: {len(objects_to_keep)}")
        if objects_to_keep:
            logger.debug("Complete list of objects to keep:")
            for obj in objects_to_keep:
                logger.debug(f"  KEEP: {obj.Label}")

        # CRITICAL: Break parent-child links for kept objects whose parents will be removed
        for obj in objects_to_keep:
            # Check if this object has a parent that will be removed
            parent = self._get_parent(obj)
            if parent and parent not in objects_to_keep:
                logger.info(f"Breaking link: removing {obj.Label} from parent {parent.Label}")
                self._remove_from_parent(obj, parent)

        # Build removal list with names captured before deletion
        objects_to_remove = []
        for obj in all_objects:
            if obj not in objects_to_keep:
                objects_to_remove.append((obj.Name, obj.Label))

        logger.info(f"Objects marked for removal: {len(objects_to_remove)}")

        if len(objects_to_remove) == 0:
            logger.info("No objects to remove")
            return

        # Now remove them
        removed_count = 0
        for obj_name, obj_label in objects_to_remove:
            try:
                # Check if object still exists
                obj = App.ActiveDocument.getObject(obj_name)
                if obj:
                    logger.debug(f"Removing: {obj_label}")
                    App.ActiveDocument.removeObject(obj_name)
                    removed_count += 1
                else:
                    logger.debug(f"Already removed: {obj_label}")
            except Exception as e:
                logger.warning(f"Could not remove {obj_label}: {e}")

        logger.info(f"✓ Removed {removed_count} objects, kept {len(objects_to_keep)} objects")
        App.ActiveDocument.recompute()

    def _get_parent(self, obj):
        """Find the parent object that contains this object in its Group"""
        for potential_parent in App.ActiveDocument.Objects:
            if hasattr(potential_parent, 'Group') and obj in potential_parent.Group:
                return potential_parent
        return None

    def _remove_from_parent(self, obj, parent):
        """Remove an object from its parent's Group without deleting it"""
        if hasattr(parent, 'Group'):
            new_group = [child for child in parent.Group if child != obj]
            parent.Group = new_group
            logger.debug(f"  Removed {obj.Label} from {parent.Label}'s Group")

    def _add_children_to_set(self, obj, object_set):
        """Recursively add all children of an object to the set"""
        if hasattr(obj, 'Group'):
            for child in obj.Group:
                if child not in object_set:
                    logger.debug(f"  Keeping child: {child.Label}")
                    object_set.add(child)
                    self._add_children_to_set(child, object_set)

        if hasattr(obj, 'OutList'):
            for child in obj.OutList:
                if child not in object_set:
                    logger.debug(f"  Keeping dependency: {child.Label}")
                    object_set.add(child)
                    self._add_children_to_set(child, object_set)

    def _matches_pattern(self, name, pattern):
        """Check if name matches pattern (supports wildcards)"""
        import re
        # Convert simple wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return re.match(f"^{regex_pattern}$", name) is not None

    def _execute_build_segment(self, operation):
        """Build a segment"""
        segment_type = operation.get('segment_type', 'curve_follower')

        if segment_type == 'curve_follower':
            self._build_curve_follower_segment(operation)
        else:
            logger.warning(f"Unknown segment type: {segment_type}")

    def _build_curve_follower_segment(self, operation):
        """Build a loft-based curve follower segment"""
        segment_name = operation.get('name', 'segment')
        description = operation.get('description', '')
        curve_spec = operation.get('curve_spec', {})
        wafer_settings = operation.get('wafer_settings', {})
        segment_settings = operation.get('segment_settings', {})
        connection_spec = operation.get('connection', {})

        # Determine base_placement for this segment
        if len(self.segment_list) == 0:
            # First segment - use current_placement
            base_placement = self.current_placement
        else:
            # Subsequent segments - start at origin, will adjust after generation
            base_placement = App.Placement()

        logger.debug(f"Building segment '{segment_name}' with initial placement {base_placement}")

        # Create loft segment
        from loft_segment import LoftSegment
        segment = LoftSegment(
            doc=self.doc,
            name=segment_name,
            curve_spec=curve_spec,
            wafer_settings=wafer_settings,
            segment_settings=segment_settings,
            base_placement=base_placement,
            connection_spec=connection_spec
        )

        logger.debug(f"Created LoftSegment: {segment_name}")

        # Generate wafers (at origin for non-first segments)
        segment.generate_wafers()

        # Visualize (creates Part and adds objects)
        segment.visualize(self.doc)

        # Align to previous segment if not the first
        if len(self.segment_list) > 0:
            # Calculate adjusted placement
            adjusted_placement = self._align_segment_to_previous(segment, self.segment_list[-1])

            # Find and update the Part object
            part_name_variations = [
                f"{segment_name.replace(' ', '_')}_Part",
                f"{segment_name}_Part",
            ]

            part_obj = None
            for name_variant in part_name_variations:
                part_obj = self.doc.getObject(name_variant)
                if part_obj:
                    logger.debug(f"Found Part object: {name_variant}")
                    break

            if part_obj:
                logger.debug(f"Part placement BEFORE: {part_obj.Placement}")
                part_obj.Placement = adjusted_placement
                segment.base_placement = adjusted_placement
                logger.debug(f"Part placement AFTER: {part_obj.Placement}")
            else:
                logger.warning(f"Could not find Part object. Tried: {part_name_variations}")

        # Store segment
        self.segment_list.append(segment)

        # Update current_placement for next segment
        self.current_placement = segment.get_end_placement()
        logger.debug(f"Updated current_placement to {self.current_placement}")

        logger.info(f"✓ Created segment '{segment_name}' with {len(segment.wafer_list)} wafers")

    def _align_segment_to_previous(self, segment, prev_segment):
        """
        Align a segment to connect with the previous segment.

        Uses LCS-based alignment: aligns the current segment's entry LCS (lcs1)
        with the previous segment's exit LCS (lcs2).

        Args:
            segment: LoftSegment to be aligned
            prev_segment: LoftSegment to align to

        Returns:
            App.Placement: The adjusted placement for segment's Part
        """
        logger.debug(f"Aligning segment '{segment.name}' to '{prev_segment.name}'")

        # DEBUG: Check what's in connection_spec
        logger.debug(f"Type of connection_spec: {type(segment.connection_spec)}")
        logger.debug(f"Full connection_spec contents: {segment.connection_spec}")
        logger.debug(f"connection_spec repr: {repr(segment.connection_spec)}")

        # 1. Get previous segment's EXIT LCS in world coordinates
        if not prev_segment.wafer_list or not prev_segment.wafer_list[-1].lcs2:
            logger.warning("Previous segment has no exit LCS")
            return segment.base_placement

        prev_local_exit = prev_segment.wafer_list[-1].lcs2
        prev_world_exit = prev_segment.base_placement.multiply(prev_local_exit)
        logger.debug(f"Previous segment EXIT (world): {prev_world_exit}")

        # 2. Get current segment's ENTRY LCS in local coordinates
        if not segment.wafer_list or not segment.wafer_list[0].lcs1:
            logger.warning("Current segment has no entry LCS")
            return segment.base_placement

        curr_local_entry = segment.wafer_list[0].lcs1
        logger.debug(f"Current segment ENTRY (local): {curr_local_entry}")

        # 3. Calculate adjusted placement: prev_exit * curr_entry.inverse()
        adjusted_placement = prev_world_exit.multiply(curr_local_entry.inverse())
        logger.debug(f"Calculated adjusted placement: {adjusted_placement}")

        # Apply additional Z-axis rotation if specified in connection_spec
        rotation_angle = segment.connection_spec.get('rotation_angle', 0)
        logger.debug(f"Connection spec: {segment.connection_spec}")
        logger.debug(f"Rotation angle from connection_spec: {rotation_angle}")

        if rotation_angle != 0:
            logger.info(f"Applying Z-axis rotation: {rotation_angle}°")

            # Get the Z-axis from the PREVIOUS segment's exit LCS (the connection axis)
            connection_z_axis = prev_world_exit.Rotation.multVec(App.Vector(0, 0, 1))
            connection_point = prev_world_exit.Base

            # Create rotation around this Z-axis
            additional_rotation = App.Rotation(connection_z_axis, rotation_angle)

            # Apply rotation to the adjusted placement
            # First, rotate the position around the connection point
            offset = adjusted_placement.Base - connection_point
            rotated_offset = additional_rotation.multVec(offset)
            adjusted_placement.Base = connection_point + rotated_offset

            # Then apply rotation to the orientation
            adjusted_placement.Rotation = additional_rotation.multiply(adjusted_placement.Rotation)

            logger.info(f"Applied additional Z-axis rotation: {rotation_angle}°")
            logger.debug(f"Adjusted placement after rotation: {adjusted_placement}")
        else:
            logger.debug("No additional rotation specified (rotation_angle is 0 or not specified)")
        # 4. Verify alignment (for debugging)

        curr_world_entry = adjusted_placement.multiply(curr_local_entry)
        pos_match = (curr_world_entry.Base - prev_world_exit.Base).Length
        rot_match = curr_world_entry.Rotation.isSame(prev_world_exit.Rotation, 1e-6)

        logger.debug(f"Position difference: {pos_match}")
        logger.debug(f"Rotation match: {rot_match}")

        # Check individual axes
        prev_x = prev_world_exit.Rotation.multVec(App.Vector(1, 0, 0))
        prev_y = prev_world_exit.Rotation.multVec(App.Vector(0, 1, 0))
        prev_z = prev_world_exit.Rotation.multVec(App.Vector(0, 0, 1))

        curr_x = curr_world_entry.Rotation.multVec(App.Vector(1, 0, 0))
        curr_y = curr_world_entry.Rotation.multVec(App.Vector(0, 1, 0))
        curr_z = curr_world_entry.Rotation.multVec(App.Vector(0, 0, 1))

        logger.debug(f"Prev X={prev_x}, Curr X={curr_x}, dot={prev_x.dot(curr_x):.6f}")
        logger.debug(f"Prev Y={prev_y}, Curr Y={curr_y}, dot={prev_y.dot(curr_y):.6f}")
        logger.debug(f"Prev Z={prev_z}, Curr Z={curr_z}, dot={prev_z.dot(curr_z):.6f}")

        if pos_match < 1e-6 and rot_match:
            logger.debug("✓ LCS alignment verified")
        else:
            logger.warning(f"✗ LCS alignment verification failed: pos={pos_match}, rot={rot_match}")

        return adjusted_placement


    def _execute_set_position(self, operation):
        """Set current position and orientation"""
        position = operation.get('position', [0, 0, 0])
        rotation = operation.get('rotation', [0, 0, 0])

        # Create placement from position and rotation
        pos = App.Vector(position[0], position[1], position[2])
        rot = App.Rotation(rotation[0], rotation[1], rotation[2])
        self.current_placement = App.Placement(pos, rot)

        logger.debug(f"Position set to: {self.current_placement}")

    def _generate_cutting_list(self):
        """Generate cutting list for all segments"""
        output_file = self.output_files.get('cuts_file')
        if not os.path.isabs(output_file):
            base_dir = self.output_files.get('working_directory', '')
            output_file = os.path.join(base_dir, output_file)
            logger.debug(f"Generating cutting list with file: {output_file}")

        if not output_file:
            logger.warning("No cutting list file specified")
            return

        logger.debug(f"Generating cutting list: {output_file}")

        with open(output_file, 'w') as f:
            # Header
            f.write("=" * 90 + "\n")
            f.write("CUTTING LIST - LOFTED METHOD\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"PROJECT: {self.metadata.get('project_name', 'Unnamed Project')}\n")
            f.write(f"Total Segments: {len(self.segment_list)}\n")
            f.write("=" * 90 + "\n\n")

            for seg in self.segment_list:
                wafer_count = len(seg.wafer_list)

                # Calculate total volume for segment
                total_volume = sum(w.wafer.Volume for w in seg.wafer_list if w.wafer and hasattr(w.wafer, 'Volume'))

                # Calculate bounding box
                all_x, all_y, all_z = [], [], []
                for w in seg.wafer_list:
                    if w.wafer and hasattr(w.wafer, 'BoundBox'):
                        bb = w.wafer.BoundBox
                        all_x.extend([bb.XMin, bb.XMax])
                        all_y.extend([bb.YMin, bb.YMax])
                        all_z.extend([bb.ZMin, bb.ZMax])

                f.write(f"SEGMENT: {seg.name}\n")
                f.write("=" * 90 + "\n\n")
                f.write(f"Wafer count: {wafer_count}\n")
                f.write(f"Total volume: {total_volume:.4f}\n")
                if all_x:
                    f.write(f"X-range: {min(all_x):.2f} to {max(all_x):.2f}\n")
                    f.write(f"Y-range: {min(all_y):.2f} to {max(all_y):.2f}\n")
                    f.write(f"Z-range: {min(all_z):.2f} to {max(all_z):.2f}\n")
                f.write("\n" + "-" * 90 + "\n\n")

                # Cutting instructions header
                f.write("CUTTING INSTRUCTIONS:\n")
                f.write("  Blade° = Tilt saw blade from vertical\n")
                f.write("  Cylinder° = Rotate cylinder to this angle before cutting\n")
                f.write("  Cumulative = Total cylinder length used (mark and cut at this point)\n\n")

                # Table header
                f.write(f"{'Cut':<5} {'Length':<12} {'Blade°':<8} {'Cylinder°':<12} {'Cumulative':<13} {'Done':<6}\n")
                f.write("-" * 90 + "\n")

                # Calculate cumulative length
                cumulative = 0

                for i, wafer in enumerate(seg.wafer_list, 1):
                    if wafer.wafer:
                        geometry = wafer.geometry
                        if geometry:
                            # Get chord length (thickness of wafer)
                            chord_length = geometry.get('chord_length', 0)
                            cumulative += chord_length

                            # Calculate blade angle from normal
                            # Use the first face's normal to determine blade tilt
                            if 'ellipse1' in geometry:
                                normal1 = geometry['ellipse1'].get('normal', App.Vector(0, 0, 1))
                                blade_angle = math.degrees(math.acos(abs(normal1.z)))
                            else:
                                blade_angle = 0

                            # Get cylinder rotation angle from LCS
                            # This is the rotation around the spine axis
                            lcs1 = wafer.lcs1
                            if lcs1:
                                # Get the yaw angle (rotation around Z in the placement)
                                ypr = lcs1.Rotation.toEuler()  # Returns (yaw, pitch, roll)
                                cylinder_angle = ypr[0]  # Yaw
                            else:
                                cylinder_angle = 0

                            # Format length as fractional inches
                            length_str = self._format_fractional_inches(chord_length)
                            cumulative_str = self._format_fractional_inches(cumulative)

                            # Write row
                            f.write(
                                f"{i:<5} {length_str:<12} {blade_angle:<8.1f} {cylinder_angle:<12.1f} {cumulative_str:<13} {'[ ]':<6}\n")

                f.write("\n" + "-" * 90 + "\n")
                f.write(f"Total cylinder length required: {self._format_fractional_inches(cumulative)}\n")
                f.write(f"  ({cumulative:.3f} inches = {cumulative * 25.4:.1f} mm)\n\n")

            # Column definitions
            f.write("=" * 90 + "\n")
            f.write("COLUMN DEFINITIONS\n")
            f.write("=" * 90 + "\n\n")
            f.write("Cut:        Wafer number in sequence\n\n")
            f.write("Length:     Length of wafer measured along the chord (longest outside edge)\n")
            f.write("            This is the distance to mark on the cylinder for cutting\n\n")
            f.write("Blade°:     Blade tilt angle from vertical (half of lift angle)\n")
            f.write("            Set your saw blade to this angle for the cut\n\n")
            f.write("Cylinder°:  Rotational position of cylinder for this cut\n")
            f.write("            Rotate cylinder to this angle before making the cut\n\n")
            f.write("Cumulative: Running total of cylinder length used\n")
            f.write("            Mark this distance from the start and cut at this point\n\n")
            f.write("Done:       Checkbox to mark completion of each cut\n\n")

            f.write("=" * 90 + "\n")
            f.write("NOTES\n")
            f.write("=" * 90 + "\n\n")
            f.write("Physical Construction Process:\n")
            f.write("1. Mark cumulative length on cylinder\n")
            f.write("2. Set blade angle (Blade°)\n")
            f.write("3. Rotate cylinder to specified angle (Cylinder°)\n")
            f.write("4. Make cut\n")
            f.write("5. Wafer is removed with one angled face\n")
            f.write("6. Flip wafer 180° and repeat for next cut\n\n")

        logger.info(f"✓ Cutting list written: {output_file}")

    def _format_fractional_inches(self, decimal_inches):
        """Convert decimal inches to fractional format like '1 3/16\"'"""
        whole = int(decimal_inches)
        frac = decimal_inches - whole

        # Find closest 1/16th
        sixteenths = round(frac * 16)

        if sixteenths == 0:
            return f"{whole}\"" if whole > 0 else "0\""
        elif sixteenths == 16:
            return f"{whole + 1}\""
        else:
            # Simplify fraction
            num = sixteenths
            den = 16
            while num % 2 == 0 and den % 2 == 0:
                num //= 2
                den //= 2

            if whole > 0:
                return f"{whole} {num}/{den}\""
            else:
                return f"{num}/{den}\""

    def build_lofted_cut_list(self, segment, output_file):
        """Build cutting list for lofted segments"""

        with open(output_file, 'w') as cuts_file:
            # Header
            cuts_file.write("=" * 90 + "\n")
            cuts_file.write("CUTTING LIST - LOFTED METHOD\n")
            cuts_file.write("=" * 90 + "\n\n")

            cuts_file.write(f"SEGMENT: {segment.name}\n")
            cuts_file.write("=" * 90 + "\n\n")

            # Segment stats
            total_volume = sum(w.wafer.Volume for w in segment.wafer_list if w.wafer is not None)
            cuts_file.write(f"Wafer count: {len(segment.wafer_list)}\n")
            cuts_file.write(f"Total volume: {total_volume:.4f}\n")

            # Get bounds
            if segment.wafer_list:
                first_wafer = segment.wafer_list[0]
                last_wafer = segment.wafer_list[-1]

                if hasattr(first_wafer, 'geometry') and hasattr(last_wafer, 'geometry'):
                    f_geom = first_wafer.geometry
                    l_geom = last_wafer.geometry

                    x_vals = [f_geom.get('center1', App.Vector()).x, f_geom.get('center2', App.Vector()).x,
                              l_geom.get('center1', App.Vector()).x, l_geom.get('center2', App.Vector()).x]
                    y_vals = [f_geom.get('center1', App.Vector()).y, f_geom.get('center2', App.Vector()).y,
                              l_geom.get('center1', App.Vector()).y, l_geom.get('center2', App.Vector()).y]
                    z_vals = [f_geom.get('center1', App.Vector()).z, f_geom.get('center2', App.Vector()).z,
                              l_geom.get('center1', App.Vector()).z, l_geom.get('center2', App.Vector()).z]

                    cuts_file.write(f"X-range: {min(x_vals):.2f} to {max(x_vals):.2f}\n")
                    cuts_file.write(f"Y-range: {min(y_vals):.2f} to {max(y_vals):.2f}\n")
                    cuts_file.write(f"Z-range: {min(z_vals):.2f} to {max(z_vals):.2f}\n")

            cuts_file.write("\n" + "-" * 90 + "\n\n")

            # Instructions
            cuts_file.write("CUTTING INSTRUCTIONS:\n")
            cuts_file.write("  Blade° = Tilt saw blade from vertical\n")
            cuts_file.write("  Cylinder° = Rotate cylinder to this angle before cutting\n")
            cuts_file.write("  Cumulative = Total cylinder length used (mark and cut at this point)\n\n")

            # Column headers
            cuts_file.write(f"{'Cut':<4} {'Length':<10} {'Blade°':<7} {'Rot°':<6} {'Cylinder°':<10} ")
            cuts_file.write(f"{'Collin°':<10} {'Azimuth°':<10} {'Cumulative':<12} {'Done':<6}\n")
            cuts_file.write("-" * 90 + "\n")

            # Wafer data
            cumulative_length = 0.0
            cumulative_rotation = 0.0

            for i, wafer in enumerate(segment.wafer_list):
                if wafer.wafer is None:
                    continue

                geom = wafer.geometry

                # Extract data
                chord_length = geom.get('chord_length', 0)
                lift_angle = geom.get('lift_angle_deg', 0)
                blade_angle = lift_angle / 2.0
                rotation = geom.get('rotation_angle_deg', 0)
                collinearity = geom.get('collinearity_angle_deg', 0)
                azimuth = geom.get('chord_azimuth_deg', 0)

                # Calculate cylinder angle
                if i == 0:
                    cylinder_angle = 0.0
                else:
                    cumulative_rotation += rotation
                    if i % 2 == 1:
                        cylinder_angle = (cumulative_rotation + 180.0) % 360.0
                    else:
                        cylinder_angle = cumulative_rotation % 360.0

                cumulative_length += chord_length

                # Format output
                cut_num = f"{i + 1:<4}"
                length_str = self._format_inches(chord_length)
                blade = f"{blade_angle:<7.1f}"
                rotation_str = f"{rotation:<6.0f}"
                cylinder = f"{cylinder_angle:<10.0f}"
                collin_str = f"{collinearity:<10.4f}"
                azimuth_str = f"{azimuth:<10.2f}"
                cumul_str = self._format_inches(cumulative_length)
                done_mark = "[ ]"

                cuts_file.write(f"{cut_num} {length_str:<10} {blade} {rotation_str} {cylinder} ")
                cuts_file.write(f"{collin_str} {azimuth_str} {cumul_str:<12} {done_mark}\n")

            # Summary
            cuts_file.write("\n" + "-" * 90 + "\n")
            cuts_file.write(f"Total cylinder length required: {self._format_inches(cumulative_length)}\n")
            cuts_file.write(f"  ({cumulative_length:.3f} inches = {cumulative_length * 25.4:.1f} mm)\n\n")

            # Column definitions (if enabled)
            include_definitions = self.global_settings.get('include_cut_list_definitions', True)

            if include_definitions:
                self._write_cut_list_definitions(cuts_file)

    def _write_cut_list_definitions(self, cuts_file):
        """Write column definitions to cutting list"""
        cuts_file.write("=" * 90 + "\n")
        cuts_file.write("COLUMN DEFINITIONS\n")
        cuts_file.write("=" * 90 + "\n\n")

        cuts_file.write("Cut:        Wafer number in sequence\n\n")

        cuts_file.write("Length:     Length of wafer measured along the chord (longest outside edge)\n")
        cuts_file.write("            This is the distance to mark on the cylinder for cutting\n\n")

        cuts_file.write("Blade°:     Blade tilt angle from vertical (half of lift angle)\n")
        cuts_file.write("            Set your saw blade to this angle for the cut\n\n")

        cuts_file.write("Rot°:       Rotation angle - the incremental twist of the curve at this wafer\n")
        cuts_file.write("            This controls the Z-rise and torsion of the structure\n\n")

        cuts_file.write("Cylinder°:  Absolute rotational position of cylinder for this cut\n")
        cuts_file.write("            Includes 180° flip between wafers plus accumulated rotation\n")
        cuts_file.write("            Rotate cylinder to this angle before making the cut\n\n")

        cuts_file.write("Collin°:    Collinearity angle - exterior angle between consecutive chord vectors\n")
        cuts_file.write("            Measures how much the chord direction changes from previous wafer\n")
        cuts_file.write("            Interior angle = 180° - Collin°\n")
        cuts_file.write("            Larger values indicate sharper curves\n\n")

        cuts_file.write("Azimuth°:   Compass direction of chord in XY plane, measured from +Y axis\n")
        cuts_file.write("            Negative = clockwise from North, Positive = counterclockwise\n")
        cuts_file.write("            Shows how the structure curves in the horizontal plane\n\n")

        cuts_file.write("Cumulative: Running total of cylinder length used\n")
        cuts_file.write("            Mark this distance from the start and cut at this point\n\n")

        cuts_file.write("Done:       Checkbox to mark completion of each cut\n\n")

        cuts_file.write("=" * 90 + "\n")
        cuts_file.write("NOTES\n")
        cuts_file.write("=" * 90 + "\n\n")

        cuts_file.write("Physical Construction Process:\n")
        cuts_file.write("1. Mark cumulative length on cylinder\n")
        cuts_file.write("2. Set blade angle (Blade°)\n")
        cuts_file.write("3. Rotate cylinder to specified angle (Cylinder°)\n")
        cuts_file.write("4. Make cut\n")
        cuts_file.write("5. Wafer is removed with one angled face\n")
        cuts_file.write("6. Flip wafer 180° and repeat for next cut\n\n")

        cuts_file.write("Geometry:\n")
        cuts_file.write("- Length, Blade°, and Rot° are the primary geometric parameters\n")
        cuts_file.write("- Cylinder° is derived from Rot° for cutting convenience\n")
        cuts_file.write("- Collin° and Azimuth° are calculated values for reference\n\n")

    def _format_inches(self, value):
        """Format a value as fractional inches"""
        inches = int(value)
        fraction = value - inches

        # Convert to nearest 1/16th
        sixteenths = round(fraction * 16)

        if sixteenths == 0:
            return f"{inches}\""
        elif sixteenths == 16:
            return f"{inches + 1}\""
        else:
            # Simplify fraction
            if sixteenths % 8 == 0:
                return f"{inches} {sixteenths // 8}/2\""
            elif sixteenths % 4 == 0:
                return f"{inches} {sixteenths // 4}/4\""
            elif sixteenths % 2 == 0:
                return f"{inches} {sixteenths // 2}/8\""
            else:
                return f"{inches} {sixteenths}/16\""

    def _format_placement(self, placement):
        """Format a placement for display"""
        pos = placement.Base
        rot = placement.Rotation
        angles = rot.toEuler()
        return f"Pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}), Rot=({angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f})"

