try:
    from core.logging_setup import get_logger, log_coord, apply_display_levels

    apply_display_levels(["ERROR", "WARNING", "INFO", "COORD", "DEBUG"])
    # apply_display_levels(["ERROR", "WARNING", "INFO"])
except Exception:
    import logging

    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import Part
import FreeCAD
import FreeCADGui
import numpy as np
import re
import yaml
from typing import Dict, Any, Optional
from curve_follower import CurveFollower
from flex_segment import FlexSegment
import curves
from loft_segment import LoftSegment
from curve_follower_loft import CurveFollowerLoft
from curve_follower_loft import get_curve_points_from_curves_module as get_curve_points
from curve_follower_loft import create_sampler_function
from wafer_loft import LoftWaferGenerator


class Driver(object):
    """Plant Builder Driver supporting YA    level = "DEBUG"ML-based project configuration."""

    def __init__(self, App, Gui, assembly_name):
        """Initialize the Driver with FreeCAD integration.

        Args:
            App: FreeCAD Application object
            Gui: FreeCAD GUI object
            assembly_name: Name of the parent assembly document
            master_spreadsheet: Name of the master parameter spreadsheet
        """
        self.App = App
        self.Gui = Gui
        self.doc = App.activeDocument()
        print(f"DOCS: {App.listDocuments()}", flush=True)
        self.parent_assembly = App.listDocuments()[assembly_name]
        if not self.parent_assembly:
            raise ValueError(f"Assembly {assembly_name} not found.")

        # Project configuration (loaded from YAML)
        self.project_config = None
        self.global_settings = None
        self.curve_templates = {}
        self.workflows = {}

        # Build state
        self.output_settings = None
        self.segment_list = []
        self.path_place_list = None

        # result bounds
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        self.relocate_segments_tf = None
        self.stopper = False
        self.use_loft = False

        # Utility functions
        self.get_object_by_label = self._gobj()
        FreeCAD.gobj = self.get_object_by_label

    def load_configuration(self, config_file):
        """Load and validate YAML configuration"""
        print(f"Loading config: {config_file}")

        with open(config_file, 'r') as f:
            self.project_config = yaml.safe_load(f)

        self.global_settings = self.project_config.get('global_settings', {})

        # ← ADD THIS LINE
        self.output_settings = self.project_config.get('output_files', {})

        # Validate curve availability
        available_curves = self._validate_curve_availability()
        print(f"Available curve types: {sorted(available_curves)}")

        # ... rest of validation ...

    def _apply_global_settings(self) -> None:
        """Apply global settings from the YAML configuration."""
        global_settings = self.project_config.get('global_settings', {})

        # Set relocate segments flag
        self.relocate_segments_tf = global_settings.get('relocate_segments', True)

        # Remove existing objects if specified
        if global_settings.get('remove_existing', False):
            self._remove_existing_objects()

    # In driver.py, add this method to the Driver class

    def _validate_curve_availability(self):
        """
        Check what curve generation functions are available

        Returns:
            set of available curve types
        """

        # Let the curves module tell us what it provides
        return curves.get_available_curves()

    def _remove_existing_objects(self) -> None:
        """Remove existing objects based on global settings."""
        global_settings = self.project_config.get('global_settings', {})
        do_cuts = global_settings.get('print_cuts', False)

        remove_string = "K.+|L+.|N+.|base_lcs.*"
        if not do_cuts:
            remove_string += "|.+|e.+|E.+|f.+|A.+"

        self.remove_objects_re(remove_string)

    def workflow(self, workflow_name: Optional[str] = None) -> None:
        """Execute the specified workflow.

        Args:
            workflow_name: Name of workflow to execute. If None, uses main workflow.
        """
        # Determine which workflow to execute
        if workflow_name and workflow_name in self.workflows:
            operations = self.workflows[workflow_name]
            logger.info(f"Executing alternative workflow: {workflow_name}")
        else:
            operations = self.project_config['workflow']

        # Execute operations in sequence
        for operation in operations:
            self._execute_operation(operation)

        # Post-processing
        self.set_composite_bounds()
        print(f"Composite bounds: {self.get_composite_bounds()}")
        self._generate_output_files()

        FreeCAD.ActiveDocument.recompute()

    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """Execute a single workflow operation.

        Args:
            operation: Dictionary containing operation definition
        """
        logger.info(f"operation:{operation}")
        op_type = operation.get('operation')
        description = operation.get('description', '')

        if description:
            logger.info(f"Executing: {description}")

        if op_type == 'remove_objects':
            self._execute_remove_objects(operation)
        elif op_type == 'set_position':
            self._execute_set_position(operation)
        elif op_type == 'build_segment':
            self._execute_build_segment(operation)
        elif op_type == 'validate_reconstruction':
            self._execute_validate_reconstruction(operation)
        elif op_type == 'reconstruct_wafers':
            self._execute_reconstruct_wafers(operation)
        else:
            logger.error(f"Unknown operation type: {op_type}")

    def _execute_remove_objects(self, operation: Dict[str, Any]) -> None:
        """Execute remove_objects operation."""
        patterns = operation.get('patterns', [])
        for pattern in patterns:
            # logger.debug(f"Removing objects matching: {pattern}")
            self.remove_objects_re(pattern)

    def _execute_set_position(self, operation: Dict[str, Any]) -> None:
        """Execute set_position operation."""
        position = operation.get('position', [0, 0, 0])
        rotation = operation.get('rotation', [0, 0, 0])

        # Store initial position for first segment
        self.initial_position = FreeCAD.Placement(
            FreeCAD.Vector(position[0], position[1], position[2]),
            FreeCAD.Rotation(rotation[0], rotation[1], rotation[2])
        )

        logger.debug(f"Set initial position: {position}, rotation: {rotation}")

    def _execute_build_segment(self, operation: Dict[str, Any]) -> None:
        """Execute build_segment operation."""
        segment_type = operation.get('segment_type')
        name = operation.get('name')

        if not name:
            raise ValueError("Segment name is required")
        if self.stopper:
            raise ValueError("END HERE")
        self.stopper = False
        # Remove existing objects with this name
        self.remove_objects_re(rf"{name}.*")

        if segment_type == 'curve_follower':
            self._build_curve_follower_segment(operation)
        else:
            raise ValueError(f"Unknown segment type: {segment_type}")

    def _build_curve_follower_segment(self, operation):
        """Build curve follower segment"""

        segment_name = operation.get('name', 'unnamed_segment')

        # Extract parameters from operation
        curve_spec = operation.get('curve_spec', {})
        wafer_settings = operation.get('wafer_settings', {})
        segment_settings = operation.get('segment_settings', {})

        # Get segment base placement
        if not hasattr(self, 'segment_base_placement'):
            self.segment_base_placement = self.App.Placement()

        segment_base_placement = self.segment_base_placement

        print(f"Building segment '{segment_name}' at placement: {segment_base_placement}")

        # Determine which approach to use
        segment_loft = operation.get('lofted_segment', False)

        try:
            if segment_loft:
                # ============================================
                # NEW LOFT-BASED APPROACH
                # ============================================
                from loft_segment import LoftSegment

                # Create segment
                segment = LoftSegment(
                    doc=self.doc,
                    name=segment_name,
                    curve_spec=curve_spec,
                    wafer_settings=wafer_settings,
                    segment_settings=segment_settings,
                    base_placement=segment_base_placement
                )

                # Generate wafers
                segment.generate_wafers()

                # Visualize if requested
                if self.doc:
                    show_lcs = segment_settings.get('show_lcs', True)
                    show_cutting_planes = segment_settings.get('show_cutting_planes', True)
                    segment.visualize(show_lcs, show_cutting_planes)

                # Add to segment list
                if not hasattr(self, 'segment_list'):
                    self.segment_list = []
                self.segment_list.append(segment)

                # Update placement for next segment (if concatenating)
                self.segment_base_placement = segment.get_end_placement()

                print(f"✓ Created loft segment '{segment_name}' with {segment.get_wafer_count()} wafers")

            else:
                # ============================================
                # OLD DIRECT APPROACH (using flex_segment)
                # ============================================
                from flex_segment import FlexSegment

                # Create segment using old approach
                segment = FlexSegment(
                    doc=self.doc,
                    name=segment_name,
                    curve_spec=curve_spec,
                    wafer_settings=wafer_settings,
                    segment_settings=segment_settings,
                    base_placement=segment_base_placement
                )

                # Generate wafers (uses old CurveFollower internally)
                segment.generate_wafers()

                # Add to segment list
                if not hasattr(self, 'segment_list'):
                    self.segment_list = []
                self.segment_list.append(segment)

                # Update placement for next segment
                self.segment_base_placement = segment.get_end_placement()

                print(f"✓ Created flex segment '{segment_name}' with {segment.get_wafer_count()} wafers")

        except Exception as e:
            print(f"✗ Error creating segment '{segment_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_output_files(self):
        """Generate output files"""

        # Determine which method based on segment types
        if not hasattr(self, 'segment_list') or not self.segment_list:
            print("No segments to output")
            return

        # Check if we have lofted segments
        has_lofted = any(hasattr(seg, 'follower') for seg in self.segment_list)

        # Generate cut list
        cuts_file_path = self.output_settings.get('working_directory') + self.output_settings.get('cuts_file')
        if cuts_file_path:
            print(f"CUTS: {cuts_file_path}")
            with open(cuts_file_path, 'w') as cuts_file:
                if has_lofted:
                    self.build_lofted_cut_list(cuts_file)
                else:
                    self.build_cut_list(cuts_file)

    # Utility methods (mostly unchanged)
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

    def relocate_segment(self):
        """Relocate segments end to end as set in the parameters"""
        # return
        if not self.relocate_segments_tf:
            return

        if not self.segment_list:
            return

        segment = self.segment_list[-1]
        segment_name = segment.get_segment_name()

        # Check if already relocated
        if hasattr(segment, 'already_relocated') and segment.already_relocated:
            return

        logger.debug(f"\n🔧 RELOCATING SEGMENT: {segment_name}")

        # Validate segment before relocation
        is_valid, error_msg = segment.validate_segment_geometry()
        if not is_valid:
            if segment.fix_segment_lcs_alignment():
                is_valid, error_msg = segment.validate_segment_geometry()
                if not is_valid:
                    raise ValueError(f"Cannot relocate invalid segment: {error_msg}")
            else:
                raise ValueError(f"Cannot fix segment alignment - segment not valid")

        if len(self.segment_list) == 1:
            # First segment
            if hasattr(self, 'initial_position') and self.initial_position:
                logger.debug(f"  First segment - applying initial position: {self.initial_position}")
                segment.move_to_top(self.initial_position)
        else:
            # Align to previous segment
            prev_segment = self.segment_list[-2]
            prev_end_lcs = prev_segment.get_lcs_top()
            current_start_lcs = segment.get_lcs_base()

            logger.debug(f"\n  🔍 BEFORE ALIGNMENT:")
            logger.debug(f"    Previous segment '{prev_segment.get_segment_name()}' ends at:")
            logger.debug(f"      Position: {prev_end_lcs.Placement.Base}")
            logger.debug(f"      Rotation: {prev_end_lcs.Placement.Rotation.toEuler()}")
            logger.debug(f"    Current segment '{segment_name}' starts at:")
            logger.debug(f"      Position: {current_start_lcs.Placement.Base}")
            logger.debug(f"      Rotation: {current_start_lcs.Placement.Rotation.toEuler()}")
            logger.debug(f"    Segment object at:")
            logger.debug(f"      Position: {segment.segment_object.Placement.Base}")
            logger.debug(f"      Rotation: {segment.segment_object.Placement.Rotation.toEuler()}")

            # Calculate alignment transform
            target_placement = prev_end_lcs.Placement
            current_placement = current_start_lcs.Placement
            align_transform = target_placement.multiply(current_placement.inverse())

            logger.debug(f"\n  📐 CALCULATED TRANSFORM:")
            logger.debug(f"    Position: {align_transform.Base}")
            logger.debug(f"    Rotation: {align_transform.Rotation.toEuler()}")
            # Apply the transform
            logger.debug(f"\n  🔨 CALLING move_to_top with transform...")
            segment.move_to_top(align_transform)

            logger.debug(f"\n  🔍 AFTER ALIGNMENT:")
            logger.debug(f"    Current segment '{segment_name}' base now at:")
            logger.debug(f"      Position: {segment.get_lcs_base().Placement.Base}")
            logger.debug(f"      Rotation: {segment.get_lcs_base().Placement.Rotation.toEuler()}")
            logger.debug(f"    Segment object now at:")
            logger.debug(f"      Position: {segment.segment_object.Placement.Base}")
            logger.debug(f"      Rotation: {segment.segment_object.Placement.Rotation.toEuler()}")
            logger.debug(f"    Segment bounds:")
            bounds = segment.segment_object.Shape.BoundBox
            logger.debug(f"      Min: [{bounds.XMin:.3f}, {bounds.YMin:.3f}, {bounds.ZMin:.3f}]")
            logger.debug(f"      Max: [{bounds.XMax:.3f}, {bounds.YMax:.3f}, {bounds.ZMax:.3f}]")

        # Apply segment rotation if specified
        angle = segment.get_segment_rotation()
        if angle != 0:
            base_pos = segment.get_lcs_base().Placement
            rotation = FreeCAD.Placement(
                FreeCAD.Vector(0, 0, 0),
                FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), angle)
            )
            transform = base_pos.multiply(rotation).multiply(base_pos.inverse())
            segment.move_content(transform)

        # Store transform for debugging
        if hasattr(segment, 'store_applied_transform'):
            segment.store_applied_transform(
                align_transform if len(self.segment_list) > 1 else None)  # self.initial_position)

        segment.already_relocated = True
        logger.debug(f"\n✅ Completed relocation for segment '{segment_name}'")

        self.doc.recompute()
        segment.set_bounds()

    def build_cut_list(self, cuts_file):
        """
        Build cut list for all segments (old method)
        """
        cuts_file.write("=" * 80 + "\n")
        cuts_file.write("CUTTING LIST - DIRECT METHOD\n")
        cuts_file.write("=" * 80 + "\n\n")

        if not hasattr(self, 'segment_list') or not self.segment_list:
            cuts_file.write("No segments generated\n")
            return

        # Iterate through segments
        for seg_idx, segment in enumerate(self.segment_list):
            cuts_file.write(f"\n{'=' * 80}\n")
            cuts_file.write(f"SEGMENT {seg_idx}: {segment.name}\n")
            cuts_file.write(f"{'=' * 80}\n\n")

            # Segment info
            cuts_file.write(f"Wafer count: {segment.get_wafer_count()}\n")

            bounds = segment.get_bounds()
            if bounds['x_min'] is not None:
                cuts_file.write(f"X-range: {bounds['x_min']:.2f} to {bounds['x_max']:.2f}\n")
                cuts_file.write(f"Y-range: {bounds['y_min']:.2f} to {bounds['y_max']:.2f}\n")
                cuts_file.write(f"Z-range: {bounds['z_min']:.2f} to {bounds['z_max']:.2f}\n")

            cuts_file.write("\n" + "-" * 80 + "\n\n")

            # Column headers
            cuts_file.write(f"{'Index':<6} {'Volume':<10} {'Details'}\n")
            cuts_file.write("-" * 80 + "\n")

            # Wafer details
            for wafer in segment.wafer_list:
                if wafer.wafer is None:
                    cuts_file.write(f"{wafer.index:<6} FAILED\n")
                    continue

                cuts_file.write(f"{wafer.index:<6} {wafer.volume:<10.4f}\n")

    def build_lofted_cut_list(self, cuts_file):
        """Build cut list for all segments (new loft method)"""

        cuts_file.write("=" * 90 + "\n")
        cuts_file.write("CUTTING LIST - LOFTED METHOD\n")
        cuts_file.write("=" * 90 + "\n\n")

        if not hasattr(self, 'segment_list') or not self.segment_list:
            cuts_file.write("No segments generated\n")
            return

        # Get settings for what to include
        show_lcs_in_cutlist = self.output_settings.get('show_lcs_in_cutlist', False)  # Default: don't show

        # Constants
        KERF_PER_CUT = 0.125  # 1/8 inch per cut

        # Iterate through segments
        for seg_idx, segment in enumerate(self.segment_list):
            cuts_file.write(f"\n{'=' * 90}\n")
            cuts_file.write(f"SEGMENT {seg_idx}: {segment.name}\n")
            cuts_file.write(f"{'=' * 90}\n\n")

            # Segment info
            cuts_file.write(f"Wafer count: {segment.get_wafer_count()}\n")
            cuts_file.write(f"Total volume: {segment.get_total_volume():.4f}\n")

            bounds = segment.get_bounds()
            if bounds['x_min'] is not None:
                cuts_file.write(f"X-range: {bounds['x_min']:.2f} to {bounds['x_max']:.2f}\n")
                cuts_file.write(f"Y-range: {bounds['y_min']:.2f} to {bounds['y_max']:.2f}\n")
                cuts_file.write(f"Z-range: {bounds['z_min']:.2f} to {bounds['z_max']:.2f}\n")

            cuts_file.write("\n" + "-" * 90 + "\n\n")

            # Instructions
            cuts_file.write("CUTTING INSTRUCTIONS:\n")
            cuts_file.write("  Blade° = Tilt saw blade from vertical\n")
            cuts_file.write("  Cylinder° = Rotate cylinder to this angle before cutting\n")
            cuts_file.write("  Cumulative = Total cylinder length used (mark and cut at this point)\n")
            cuts_file.write("\n")

            # Column headers
            cuts_file.write(f"{'Cut':<4} {'Length':<10} {'Blade°':<7} {'Rot°':<6} {'Cylinder°':<10} ")
            cuts_file.write(f"{'Collin°':<10} {'Azimuth°':<10} {'Cumulative':<12} {'Done':<6}\n")

            cuts_file.write("-" * 90 + "\n")

            # Track cumulative values
            cumulative_rotation = 0.0  # Cumulative rotation in degrees
            total_cylinder_length = 0.0  # Total length in inches

            # Wafer details
            for wafer_idx, wafer in enumerate(segment.wafer_list):
                if wafer.wafer is None:
                    cuts_file.write(f"{wafer_idx + 1:<4} FAILED\n")
                    continue

                geom = wafer.geometry

                # Get values
                lift_deg = geom.get('lift_angle_deg', 0)
                rotation_deg = geom.get('rotation_angle_deg', 0)
                chord_length = geom.get('chord_length', 0)

                # Blade tilt is always half the lift angle (cutting at bisector)
                blade_tilt = lift_deg / 2.0

                # Calculate cylinder rotation angle (cumulative rotation with alternating 180° flip)
                # First cut starts at 0°
                if wafer_idx == 0:
                    cylinder_angle = 0.0
                else:
                    # Accumulate rotation and add 180° for the flip
                    cumulative_rotation += rotation_deg
                    if wafer_idx % 2 == 1:
                        # Odd cuts: add 180°
                        cylinder_angle = cumulative_rotation + 180.0
                    else:
                        # Even cuts: just cumulative
                        cylinder_angle = cumulative_rotation
                collinearity = geom.get('collinearity_angle_deg', 0)
                azimuth = geom.get('chord_azimuth_deg', 0)

                # Normalize cylinder angle to 0-360 range
                cylinder_angle = cylinder_angle % 360.0

                # Add to total cylinder length (chord + kerf)
                total_cylinder_length += chord_length + KERF_PER_CUT

                # Format values
                cut_num = f"{wafer_idx + 1:<4}"  # 1-indexed for user
                length_str = self._format_length_inches(chord_length)
                blade = f"{blade_tilt:<7.1f}"  # To tenths of a degree
                rotation = f"{rotation_deg:<6.0f}"  # Nearest whole degree (for reference)
                cylinder = f"{cylinder_angle:<10.0f}"  # Nearest whole degree
                cumul_str = self._format_length_inches(total_cylinder_length)
                collin_str = f"{collinearity:<10.4f}"
                azimuth_str = f"{azimuth:<10.2f}"
                done_mark = "[ ]"

                cuts_file.write(
                    f"{cut_num} {length_str:<10} {blade} {rotation} {cylinder} {collin_str} {azimuth_str} {cumul_str:<12} {done_mark}\n")

                # LCS information (optional)
                if show_lcs_in_cutlist and wafer.lcs1 and wafer.lcs2:
                    cuts_file.write(f"     LCS1: {self._format_placement(wafer.lcs1)}\n")
                    cuts_file.write(f"     LCS2: {self._format_placement(wafer.lcs2)}\n")

                cuts_file.write("\n")

            # Summary
            cuts_file.write("-" * 90 + "\n")
            summary_length = self._format_length_inches(total_cylinder_length)
            cuts_file.write(f"Total cylinder length required: {summary_length}\n")
            cuts_file.write(f"  ({total_cylinder_length:.3f} inches = ")
            cuts_file.write(f"{total_cylinder_length * 25.4:.1f} mm)\n\n")

    def _format_length_inches(self, length_inches):
        """
        Format length as inches + nearest 32nd

        Args:
            length_inches: Length in inches (float)

        Returns:
            String like "5-11/32\"" or "12-1/2\""
        """
        # Get whole inches
        whole = int(length_inches)

        # Get fractional part
        frac = length_inches - whole

        # Convert to 32nds
        thirty_seconds = round(frac * 32)

        # Handle rounding to next whole inch
        if thirty_seconds >= 32:
            whole += 1
            thirty_seconds = 0

        # Format
        if thirty_seconds == 0:
            return f'{whole}"'
        else:
            # Simplify fraction if possible
            numerator = thirty_seconds
            denominator = 32

            # Reduce fraction
            from math import gcd
            divisor = gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor

            return f'{whole}-{numerator}/{denominator}"'

    def build_lofted_placement_list(self, place_file):
        """
        Build placement list for lofted segments

        Args:
            place_file: Open file handle for writing
        """
        place_file.write("=" * 80 + "\n")
        place_file.write("PLACEMENT LIST - LOFTED SEGMENT METHOD\n")
        place_file.write("=" * 80 + "\n\n")

        if not hasattr(self, 'wafer_list') or not self.wafer_list:
            place_file.write("No wafers generated\n")
            return

        place_file.write(f"Segment: {self.current_segment_name}\n")
        place_file.write(f"Total wafers: {len(self.wafer_list)}\n\n")

        place_file.write(f"{'Index':<6} {'LCS1 Placement':<60} {'LCS2 Placement':<60}\n")
        place_file.write("-" * 130 + "\n")

        for wafer in self.wafer_list:
            if wafer.wafer is None:
                place_file.write(f"{wafer.index:<6} FAILED\n")
                continue

            lcs1_str = self._format_placement(wafer.lcs1) if wafer.lcs1 else "N/A"
            lcs2_str = self._format_placement(wafer.lcs2) if wafer.lcs2 else "N/A"

            place_file.write(f"{wafer.index:<6} {lcs1_str:<60} {lcs2_str:<60}\n")

        place_file.write("\n" + "=" * 80 + "\n")

    def _format_placement(self, placement):
        """
        Format a FreeCAD Placement for output

        Args:
            placement: App.Placement object

        Returns:
            Formatted string
        """
        if placement is None:
            return "None"

        pos = placement.Base
        rot = placement.Rotation
        angles = rot.toEuler()  # Yaw, Pitch, Roll in degrees

        return f"Pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}) Rot=({angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f})"

    def build_place_list(self, filename: Optional[str] = "default_places_file"):
        """Build placement list file."""
        logger.info(f"Building place list: {filename}")
        min_max = [[0, 0], [0, 0], [0, 0]]

        def find_min_max(base):
            for i in range(3):
                if base[i] < min_max[i][0]:
                    min_max[i][0] = np.round(base[i], 3)
                if base[i] > min_max[i][1]:
                    min_max[i][1] = np.round(base[i], 3)

        with open(filename, "w+") as place_file:
            place_file.write("Wafer Placement:\n\n\n")
            global_placement = FreeCAD.Placement(FreeCAD.Vector(0, 0, 0), FreeCAD.Rotation(0, 0, 0))

            for nbr, segment in enumerate(self.segment_list):
                logger.debug(f"Segment: {segment.get_segment_name()}")
                global_placement = segment.print_construction_list(nbr, place_file, global_placement, find_min_max)

            min_max_str = f"\nGlobal Min Max:\n\tX: {min_max[0][0]} - {min_max[0][1]}, "
            min_max_str += f"Y: {min_max[1][0]} - {min_max[1][1]}, Z: {min_max[2][0]} - {min_max[2][1]}"
            place_file.write(f"{min_max_str}")

        # After writing the placement list, clean up construction LCSs
        for segment in self.segment_list:
            try:
                # Force deletion of per-wafer LCSs; base/top are preserved.
                segment.cleanup_wafer_lcs(keep_debug=False)
            except Exception as e:
                logger.warning(f"Cleanup failed for segment {segment.get_segment_name()}: {e}")

    def remove_objects_re(self, remove_string: str) -> None:
        """Remove objects containing 'name' as a part of a label.

        Args:
            remove_string: Raw string containing a regular expression
        """
        pattern = re.compile(remove_string)
        doc_list = [obj for obj in self.doc.Objects if pattern.match(obj.Label)]

        for item in doc_list:
            try:
                self.doc.removeObject(item.Label)
            except Exception as e:
                logger.debug(f"Remove object exception: {e}")

    def set_composite_bounds(self):
        """Calculate composite bounding box across all segments"""

        if not hasattr(self, 'segment_list') or not self.segment_list:
            print("Composite bounds: No segments")
            return

        # Initialize with None
        comp_x_min = comp_x_max = None
        comp_y_min = comp_y_max = None
        comp_z_min = comp_z_max = None

        # Iterate through all segments
        for segment in self.segment_list:
            seg_bounds = segment.get_bounds()

            # Update x bounds
            if seg_bounds['x_min'] is not None:
                comp_x_min = seg_bounds['x_min'] if comp_x_min is None else min(comp_x_min, seg_bounds['x_min'])
                comp_x_max = seg_bounds['x_max'] if comp_x_max is None else max(comp_x_max, seg_bounds['x_max'])

            # Update y bounds
            if seg_bounds['y_min'] is not None:
                comp_y_min = seg_bounds['y_min'] if comp_y_min is None else min(comp_y_min, seg_bounds['y_min'])
                comp_y_max = seg_bounds['y_max'] if comp_y_max is None else max(comp_y_max, seg_bounds['y_max'])

            # Update z bounds
            if seg_bounds['z_min'] is not None:
                comp_z_min = seg_bounds['z_min'] if comp_z_min is None else min(comp_z_min, seg_bounds['z_min'])
                comp_z_max = seg_bounds['z_max'] if comp_z_max is None else max(comp_z_max, seg_bounds['z_max'])

        # Store composite bounds
        self.composite_x_min = comp_x_min
        self.composite_x_max = comp_x_max
        self.composite_y_min = comp_y_min
        self.composite_y_max = comp_y_max
        self.composite_z_min = comp_z_min
        self.composite_z_max = comp_z_max

        # Print with None handling
        if comp_x_min is not None:
            print(
                f"Composite bounds: X({comp_x_min:.2f},{comp_x_max:.2f}) Y({comp_y_min:.2f},{comp_y_max:.2f}) Z({comp_z_min:.2f},{comp_z_max:.2f})")
        else:
            print("Composite bounds: None (no valid wafers)")

    def get_composite_bounds(self):
        """Return wafer extents in each dimension"""
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    def _execute_validate_reconstruction(self, operation: Dict[str, Any]):
        """Execute reconstruction validation operation."""
        from reconstruction.reconstruction_workflow import ReconstructionWorkflow

        segment_name = operation.get('segment_name')
        cutlist_file = operation.get('cutlist_file')
        create_vertices = operation.get('create_vertices', False)
        match_original = operation.get('match_original_position', False)

        # Get the original segment
        original_segment = None
        for seg in self.segment_list:
            if seg.get_segment_name() == segment_name:
                original_segment = seg
                break

        if not original_segment:
            logger.error(f"Cannot find original segment '{segment_name}' for validation")
            return

        # Create workflow with correct units (inches, not mm)
        workflow = ReconstructionWorkflow(self.doc, units_per_inch=1.0)  # ← ADD THIS

        # Get original starting LCS if requested
        starting_lcs = None
        if match_original:
            starting_lcs = original_segment.get_lcs_base()
            logger.info(f"Original segment starts at: {starting_lcs.Placement.Base}")
            logger.info(f"Original segment rotation: {starting_lcs.Placement.Rotation.toEuler()}")

        # Reconstruct
        result = workflow.reconstruct_from_cutlist_with_start(
            cutlist_file=cutlist_file,
            cylinder_diameter=1.875,  # inches
            create_vertices_only=create_vertices,
            starting_lcs=starting_lcs
        )

        logger.info(f"Reconstruction complete: {len(result.segments)} segment(s)")

    def _execute_reconstruct_wafers(self, operation):
        """Reconstruct wafer structure from cut list parameters"""
        from wafer_reconstructor import reconstruct_from_segment

        segment_name = operation.get('segment_name', '')
        rotation_multiplier = operation.get('rotation_multiplier', 1.0)
        name_prefix = operation.get('name_prefix', 'Recon')

        segment = None
        for seg in self.segment_list:
            if seg.name == segment_name:
                segment = seg
                break

        if segment is None:
            print(f"✗ Segment '{segment_name}' not found")
            return

        # Get initial azimuth for alignment
        if segment.wafer_list and segment.wafer_list[0].geometry:
            initial_azimuth = segment.wafer_list[0].geometry.get('chord_azimuth_deg', 0)
        else:
            initial_azimuth = 0

        # Extract cut data
        from wafer_reconstructor import extract_cut_data_from_segment, WaferReconstructor

        radius = segment.wafer_list[0].geometry.get('ellipse1', {}).get('minor_axis', 0.9375)
        reconstructor = WaferReconstructor(cylinder_radius=radius)

        cut_data = extract_cut_data_from_segment(segment)
        reconstructor.build_from_cut_list(
            cut_data,
            rotation_multiplier=rotation_multiplier,
            initial_orientation={'azimuth': initial_azimuth}
        )
        reconstructor.visualize_in_freecad(self.doc, name_prefix=name_prefix)

        print(f"✓ Reconstructed segment '{segment_name}' with rotation_multiplier={rotation_multiplier}")
