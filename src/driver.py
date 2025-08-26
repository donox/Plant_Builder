try:
    from core.logging_setup import get_logger
except Exception:
    import logging
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import sys
import os
import Part
import FreeCAD
import FreeCADGui
import numpy as np
import re
import yaml
from typing import Dict, Any, List, Optional
from wafer import Wafer
from curve_follower import CurveFollower
from flex_segment import FlexSegment
from curves import Curves
from make_helix import MakeHelix
from test_get_rotation_angle_freecad_wrapper import run_freecad_wrapper
# import pydevd_pycharm


# pip install pydevd-pycharm~=241.15989.155
# pip install pyyaml

class Driver(object):
    """Plant Builder Driver supporting YA    level = "DEBUG"ML-based project configuration."""

    def __init__(self, App, Gui, assembly_name, master_spreadsheet):
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
        self.parent_assembly = App.listDocuments()[assembly_name]
        if not self.parent_assembly:
            raise ValueError(f"Assembly {assembly_name} not found.")

        # Project configuration (loaded from YAML)
        self.project_config = None
        self.curve_templates = {}
        self.workflows = {}

        # Build state
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

        # Utility functions
        self.get_object_by_label = self._gobj()
        FreeCAD.gobj = self.get_object_by_label

        # Support for remote debugging to FreeCAD
        # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

    def load_yaml_config(self, yaml_file_path: str) -> None:
        """Load project configuration from YAML file.

        Args:
            yaml_file_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file has syntax errors
            ValueError: If required sections are missing
        """
        try:
            logger.info(f"YAML FILE PATH: {yaml_file_path}")
            with open(yaml_file_path, 'r') as file:
                self.project_config = yaml.safe_load(file)

            # Validate required sections
            required_sections = ['metadata', 'global_settings', 'workflow']
            for section in required_sections:
                if section not in self.project_config:
                    raise ValueError(f"Required section '{section}' missing from YAML config")

            # Extract templates and alternative workflows
            self.curve_templates = self.project_config.get('curve_templates', {})
            self.workflows = self.project_config.get('workflows', {})

            # Apply global settings
            self._apply_global_settings()

        except FileNotFoundError:
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_file_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML syntax error: {e}")

    def _apply_global_settings(self) -> None:
        """Apply global settings from the YAML configuration."""
        global_settings = self.project_config.get('global_settings', {})

        # Set relocate segments flag
        self.relocate_segments_tf = global_settings.get('relocate_segments', True)

        # Remove existing objects if specified
        if global_settings.get('remove_existing', False):
            self._remove_existing_objects()

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
        elif op_type == 'run_tests':
            self._execute_run_tests(operation)
        else:
            logger.error(f"Unknown operation type: {op_type}")

    def _execute_run_tests(self, operation ):
        run_freecad_wrapper()
        foo = 3/0

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

    def _build_curve_follower_segment(self, operation: Dict[str, Any]) -> None:
        """Build a curve follower segment."""
        name = operation['name']
        curve_spec = operation.get('curve_spec', {})
        wafer_settings = operation.get('wafer_settings', {})
        segment_settings = operation.get('segment_settings', {})

        # Handle curve template references
        if isinstance(curve_spec, str):
            if curve_spec not in self.curve_templates:
                raise ValueError(f"Curve template '{curve_spec}' not found")
            curve_spec = self.curve_templates[curve_spec].copy()

        # Extract settings with defaults
        cylinder_diameter = wafer_settings.get('cylinder_diameter', 2.0)
        min_height = wafer_settings.get('min_height', 1.0)
        max_chord = wafer_settings.get('max_chord', 0.5)

        show_lcs = segment_settings.get('show_lcs', True)
        build_segment = segment_settings.get('build_segment', True)
        rotate_segment = segment_settings.get('rotate_segment', 0.0)
        add_curve_vertices = segment_settings.get('add_curve_vertices', False)

        # Get temp file setting
        temp_file = self.project_config.get('global_settings', {}).get('temp_file', 'temp.dat')

        # Create segment
        segment = FlexSegment(name, show_lcs, temp_file, build_segment, rotate_segment)
        self.segment_list.append(segment)

        try:
            # Create curve follower
            follower = CurveFollower(
                doc=self.doc,
                segment=segment,
                cylinder_diameter=cylinder_diameter,
                curve_spec=curve_spec,
                min_height=min_height,
                max_chord=max_chord
            )

            # Get the actual segment base position
            segment_base = segment.get_lcs_base()
            logger.debug(f"Segment base placement: {segment_base.Placement}")

            # Process wafers
            follower.process_wafers(add_curve_vertices=False)

            # Fuse wafers if any were created
            if segment.get_wafer_count() > 0:
                segment.fuse_wafers()

                segment_obj = segment.get_segment_object()

                if segment_obj:
                    logger.info(f"Successfully created segment '{name}' with {segment.get_wafer_count()} wafers")

                    # Add curve vertices BEFORE relocation
                    if add_curve_vertices:
                        logger.debug("Adding curve vertices...")
                        follower.add_curve_visualization()

                    # Relocate segment - ONLY CALL THIS ONCE!
                    self.relocate_segment()
                    logger.debug(f"Completed relocation for segment '{name}'")

                else:
                    logger.warning(f"Warning: Segment '{name}' created wafers but fusing failed")
            else:
                logger.warning(f"Warning: No wafers created for segment '{name}'")

            # Force recompute
            FreeCAD.ActiveDocument.recompute()
            FreeCADGui.updateGui()

        except Exception as e:
            logger.error(f"Error creating curve follower segment '{name}': {e}")
            raise

    def _generate_output_files(self) -> None:
        """Generate output files based on global settings."""
        global_settings = self.project_config.get('global_settings', {})
        output_files = self.project_config.get('output_files', {})

        direct = ""
        if output_files.get('working_directory', False):
            direct = output_files.get('working_directory', "")

        if global_settings.get('print_cuts', False):
            cuts_file = output_files.get('cuts_file', 'cutting_list.txt')
            cuts_file = direct + cuts_file
            logger.info(f"CUTS: {cuts_file}")
            self.build_cut_list(cuts_file)

        if global_settings.get('print_place', False):
            place_file = output_files.get('place_file', 'placement_list.txt')
            place_file = direct + place_file
            self.build_place_list(place_file)

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

    def _get_workflow(self):
        return self.get_parm("workflow")

    def validate_segment_connection(self, prev_segment, curr_segment):
        """Validate that two segments can be connected properly.

        Args:
            prev_segment: Previous segment (must end with circular cut)
            curr_segment: Current segment (must start with circular cut)

        Raises:
            ValueError: If segments cannot be connected
        """
        if not prev_segment.wafer_list:
            raise ValueError(f"Previous segment {prev_segment.get_segment_name()} has no wafers")

        if not curr_segment.wafer_list:
            raise ValueError(f"Current segment {curr_segment.get_segment_name()} has no wafers")

        # Check last wafer of previous segment
        last_wafer = prev_segment.wafer_list[-1]
        if hasattr(last_wafer, 'wafer_type'):
            if last_wafer.wafer_type[1] != 'C':
                raise ValueError(
                    f"Previous segment {prev_segment.get_segment_name()} must end with circular cut, found {last_wafer.wafer_type}")

        # Check first wafer of current segment
        first_wafer = curr_segment.wafer_list[0]
        if hasattr(first_wafer, 'wafer_type'):
            if first_wafer.wafer_type[0] != 'C':
                raise ValueError(
                    f"Current segment {curr_segment.get_segment_name()} must start with circular cut, found {first_wafer.wafer_type}")

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
                raise ValueError(f"Cannot fix segment alignment: {error_msg}")

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
            # target_placement.Rotation = rot    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            segment.store_applied_transform(align_transform if len(self.segment_list) > 1 else None) # self.initial_position)

        segment.already_relocated = True
        logger.debug(f"\n✅ Completed relocation for segment '{segment_name}'")

        self.doc.recompute()
        segment.set_bounds()

    def build_cut_list(self, filename: Optional[str] = None):
        """Build cutting list file."""
        if filename is None:
            filename = self.get_parm("cuts_file")

        logger.info(f"Building cut list: {filename}")
        with open(filename, "w+") as cuts_file:
            cuts_file.write(f"Cut List for: {self.project_config.get('metadata', {})['project_name']}\n\n")
            cuts_file.write(f'Project Bounds\n')
            cuts_file.write(f'\tX-min: {self.x_min:.2f}\tX_max: {self.x_max:.2f}\n')
            cuts_file.write(f'\tY-min: {self.y_min:.2f}\tY_max: {self.y_max:.2f}\n')
            cuts_file.write(f'\tZ-min: {self.z_min:.2f}\tZ_max: {self.z_max:.2f}\n\n')
            cuts_file.write("Cutting order:\n")
            for nbr, segment in enumerate(self.segment_list):
                segment.make_cut_list(cuts_file)

    def build_place_list(self, filename: Optional[str] = None):
        """Build placement list file."""
        if filename is None:
            filename = self.get_parm("place_file")

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
            if item.Label != 'Parms_Master':
                try:
                    self.doc.removeObject(item.Label)
                except Exception as e:
                    logger.debug(f"Remove object exception: {e}")

    def set_composite_bounds(self):
        """Set bounds of result object based on all segments"""
        if not self.segment_list:
            return
        results = [0, 0, 0, 0, 0, 0]
        for segment in self.segment_list:
            seg_bounds = segment.get_bounds()
            for nbr,  bnd in enumerate(seg_bounds):
                if not bnd:
                    break
                res = results[nbr]
                if bnd > 0:
                    if res < bnd:
                        res = bnd
                elif bnd < 0:
                    if res > bnd:
                        res = bnd
                results[nbr] = res
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = results

    def get_composite_bounds(self):
        """Return wafer extents in each dimension"""
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    @staticmethod
    def make_tf(variable_name, parent_parms):
        """Convert string boolean to actual boolean."""
        logger.debug(f"make_tf: Variable: {variable_name}, parent: {parent_parms}")
        try:
            if parent_parms.get(variable_name) == "True":
                logger.debug(f"{variable_name} = True")
                return True
            else:
                logger.debug(f"{variable_name} = False")
                return False
        except Exception as e:
            logger.error(f"Exception: {e} on reference to {variable_name}")
            raise e

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move object_1 to object_2's placement"""
        # Since object_1 (current segment base) is at origin,
        # the transform is simply object_2's placement
        return FreeCAD.Placement(object_2.Placement)

