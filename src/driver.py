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

        # RUN TEST AND EXIT

        run_transform_test()
        logger.info("Test complete - exiting")
        return

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
            self._execute_remove_objects(operation)
        elif operation_type == 'build_segment':
            self._execute_build_segment(operation)
        elif operation_type == 'set_position':
            self._execute_set_position(operation)
        else:
            logger.warning(f"Unknown operation type: {operation_type}")

    def _execute_remove_objects(self, operation):
        """Remove objects matching patterns"""
        patterns = operation.get('patterns', [])

        if not patterns:
            logger.warning("No patterns specified for remove_objects operation")
            return

        logger.debug(f"Operation details: remove_objects")

        # Build list of objects to remove FIRST (before deleting anything)
        objects_to_remove = []
        for obj in self.doc.Objects:
            try:
                obj_name = obj.Name  # Access Name before any deletions
                for pattern in patterns:
                    if self._matches_pattern(obj_name, pattern):
                        objects_to_remove.append(obj)
                        break  # Don't check other patterns for this object
            except ReferenceError:
                # Object already deleted (e.g., child of a Part)
                continue

        # Now remove all objects
        removed_count = 0
        for obj in objects_to_remove:
            try:
                logger.debug(f"Removing: {obj.Name}")
                self.doc.removeObject(obj.Name)
                removed_count += 1
            except ReferenceError:
                # Object already deleted as part of removing a parent
                logger.debug(f"Object already removed (was child of parent)")
                pass
            except Exception as e:
                logger.warning(f"Could not remove object: {e}")

        logger.info(f"Removed {removed_count} objects")

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

        # Determine base_placement for this segment
        # For now, just use identity - we'll adjust after generation
        if len(self.segment_list) == 0:
            # First segment - use current_placement
            base_placement = self.current_placement
        else:
            # Subsequent segments - start at identity, will adjust after
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
            base_placement=base_placement
        )

        logger.debug(f"Created LoftSegment: {segment_name}")

        # Generate wafers (at origin for non-first segments)
        segment.generate_wafers()

        # Visualize (creates Part and adds objects)
        segment.visualize(self.doc)

        # Adjust Part placement for segment alignment (segments 2+)
        if len(self.segment_list) > 0:
            # Get the world coordinates of the two LCS we need to align

            # 1. Previous segment's EXIT (lcs2 of last wafer) in WORLD coordinates
            prev_segment = self.segment_list[-1]
            if prev_segment.wafer_list and prev_segment.wafer_list[-1].lcs2:
                # Local lcs2 from previous segment
                prev_local_exit = prev_segment.wafer_list[-1].lcs2
                # Transform to world coordinates
                prev_world_exit = prev_segment.base_placement.multiply(prev_local_exit)
                logger.debug(f"Previous segment EXIT (world): {prev_world_exit}")
            else:
                logger.warning("Previous segment has no exit LCS")
                prev_world_exit = prev_segment.base_placement

            # 2. Current segment's ENTRY (lcs1 of first wafer) in WORLD coordinates (before adjustment)
            if segment.wafer_list and segment.wafer_list[0].lcs1:
                # Local lcs1 from current segment
                curr_local_entry = segment.wafer_list[0].lcs1
                # Current segment Part is at identity, so world = local
                curr_world_entry = segment.base_placement.multiply(curr_local_entry)
                logger.debug(f"Current segment ENTRY (world, before): {curr_world_entry}")
                logger.debug(f"Current segment ENTRY (local): {curr_local_entry}")
            else:
                logger.warning("Current segment has no entry LCS")
                curr_local_entry = App.Placement()

            # 3. Calculate Part placement adjustment
            # We want: prev_world_exit = adjusted_part_placement * curr_local_entry
            # So: adjusted_part_placement = prev_world_exit * curr_local_entry.inverse()

            adjusted_placement = prev_world_exit.multiply(curr_local_entry.inverse())
            logger.debug(f"Adjusted Part placement: {adjusted_placement}")

            logger.debug(f"Verification - multiplying back:")
            test_result = adjusted_placement.multiply(curr_local_entry)
            logger.debug(f"adjusted_placement * curr_local_entry = {test_result}")
            logger.debug(f"Should equal prev_world_exit = {prev_world_exit}")

            # 4. Find and update the Part object
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

                # Verify: compute where entry is NOW in world coordinates
                new_world_entry = adjusted_placement.multiply(curr_local_entry)
                logger.debug(f"Current segment ENTRY (world, after): {new_world_entry}")
                logger.debug(f"Should match previous EXIT: {prev_world_exit}")

                prev_local_exit = prev_segment.wafer_list[-1].lcs2
                logger.debug(f"Previous segment EXIT (local): {prev_local_exit}")
            else:
                logger.warning(f"Could not find Part object. Tried: {part_name_variations}")

        # Store segment
        self.segment_list.append(segment)

        # Update current_placement for next segment
        self.current_placement = segment.get_end_placement()
        logger.debug(f"Updated current_placement to {self.current_placement}")

        logger.info(f"✓ Created segment '{segment_name}' with {len(segment.wafer_list)} wafers")

        # Generate cutting list if requested
        if self.global_settings.get('print_cuts', False):
            self._generate_cutting_list(segment)


    def _execute_set_position(self, operation):
        """Set current position and orientation"""
        position = operation.get('position', [0, 0, 0])
        rotation = operation.get('rotation', [0, 0, 0])

        # Create placement from position and rotation
        pos = App.Vector(position[0], position[1], position[2])
        rot = App.Rotation(rotation[0], rotation[1], rotation[2])
        self.current_placement = App.Placement(pos, rot)

        logger.debug(f"Position set to: {self.current_placement}")

    def _generate_cutting_list(self, segment):
        """Generate cutting list for a segment"""
        output_files = self.config.get('output_files', {})
        working_dir = output_files.get('working_directory', '/tmp')
        cuts_filename = output_files.get('cuts_file', 'cutting_list.txt')

        output_file = os.path.join(working_dir, cuts_filename)

        logger.debug(f"Generating cutting list: {output_file}")

        try:
            self.build_lofted_cut_list(segment, output_file)
            logger.info(f"✓ Cutting list written: {output_file}")
        except Exception as e:
            logger.error(f"Failed to generate cutting list: {e}")

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

