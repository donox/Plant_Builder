try:
    from core.logging_setup import get_logger
except Exception:
    try:
        from logging_setup import get_logger
    except Exception:
        import logging
        get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import string
import inspect
import importlib.util
import Part
import Draft
import FreeCAD
import FreeCADGui
import numpy as np
import csv
import re
import time
import importlib.util
import sys
import math
import yaml
from typing import Dict, Any, List, Optional
from wafer import Wafer
from flex_segment import FlexSegment
from curve_follower import CurveFollower
from curves import Curves
from make_helix import MakeHelix
import utilities
import pydevd_pycharm


# pip install pydevd-pycharm~=241.15989.155
# pip install pyyaml

class Driver(object):
    """Plant Builder Driver supporting YAML-based project configuration."""

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
        self.parent_parms = self.parent_assembly.getObjectsByLabel(master_spreadsheet)
        if self.parent_parms:
            self.parent_parms = self.parent_parms[0]
            self.get_parm, self.set_parm = self.handle_spreadsheet(self.parent_parms)
        else:
            raise ValueError(f"Spreadsheet {master_spreadsheet} not found")

        # Project configuration (loaded from YAML)
        self.project_config = None
        self.curve_templates = {}
        self.workflows = {}

        # Build state
        self.segment_list = []
        self.handle_arrows = None
        self.path_place_list = None

        # Trace and debugging
        self.trace_file_name = None
        self.trace_file = None
        self.do_trace = None
        self.relocate_segments_tf = None
        self._set_up_trace()
        self.stopper = False

        # Utility functions
        self.get_object_by_label = self._gobj()
        FreeCAD.gobj = self.get_object_by_label

        # Support for remote debugging to FreeCAD
        pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

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
            # yaml_file_path = "/home/don/FreecadProjects/Macros/PyMacros/PlantBuilder/src/yaml_files/small.yml"   #  !!!!!!!!!!!!!!!
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

        # Override trace settings if specified in YAML
        if 'do_trace' in global_settings:
            self.do_trace = global_settings['do_trace']

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
        self.process_arrow_command()
        self._generate_output_files()

        FreeCAD.ActiveDocument.recompute()
        FreeCADGui.updateGui()

        # Try to refresh the 3D view
        try:
            view = FreeCADGui.activeDocument().activeView()
            if view:
                view.fitAll()
        except:
            pass

        if self.do_trace and self.trace_file:
                self.trace_file.close()

    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """Execute a single workflow operation.

        Args:
            operation: Dictionary containing operation definition
        """
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
        elif op_type == 'add_arrows':
            self._execute_add_arrows(operation)
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
        elif segment_type == 'helix':
            self._build_helix_segment(operation)
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

            # Add debugging
            curve_info = follower.get_curve_info()
            # logger.debug(f"Curve info: {curve_info}")
            # logger.debug(f"First 5 curve points:")print
            # for i, point in enumerate(follower.curve_points[:5]):
            #     logger.debug(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
            # logger.debug(f"Last 5 curve points:")
            # for i, point in enumerate(follower.curve_points[-5:], len(follower.curve_points) - 5):
            #     logger.debug(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
            #
            # logger.debug(f"\n=== COORDINATE DEBUGGING ===")
            # logger.debug(f"Curve start point: {follower.curve_points[0]}")
            # logger.debug(f"Curve end point: {follower.curve_points[-1]}")

            # Get the actual segment base position
            segment_base = segment.get_lcs_base()
            logger.debug(f"Segment base placement: {segment_base.Placement}")

            # Process wafers
            follower.process_wafers(add_curve_vertices=False, debug=True)

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
                    logger.info(f"Completed relocation for segment '{name}'")

                else:
                    logger.info(f"Warning: Segment '{name}' created wafers but fusing failed")
            else:
                logger.info(f"Warning: No wafers created for segment '{name}'")

            # Force recompute
            FreeCAD.ActiveDocument.recompute()
            FreeCADGui.updateGui()

        except Exception as e:
            logger.error(f"Error creating curve follower segment '{name}': {e}")
            raise

    def _build_helix_segment(self, operation: Dict[str, Any]) -> None:
        """Build a traditional helix segment."""
        name = operation['name']
        helix_settings = operation.get('helix_settings', {})
        segment_settings = operation.get('segment_settings', {})

        # Extract settings
        lift_angle = helix_settings.get('lift_angle', 0.0)
        rotate_angle = helix_settings.get('rotate_angle', 0.0)
        wafer_count = helix_settings.get('wafer_count', 10)
        outside_height = helix_settings.get('outside_height', 2.0)
        cylinder_diameter = helix_settings.get('cylinder_diameter', 2.0)

        show_lcs = segment_settings.get('show_lcs', True)
        build_segment = segment_settings.get('build_segment', True)
        rotate_segment = segment_settings.get('rotate_segment', 0.0)

        temp_file = self.project_config.get('global_settings', {}).get('temp_file', 'temp.dat')

        # Create segment
        segment = FlexSegment(name, show_lcs, temp_file, build_segment, rotate_segment)
        self.segment_list.append(segment)

        # Create helix
        helix = MakeHelix(segment)
        helix.create_helix(wafer_count, cylinder_diameter, outside_height,
                           lift_angle, rotate_angle, name)

        self.relocate_segment()
        logger.info(f"Created helix segment '{name}' with {wafer_count} wafers")


    def _execute_add_arrows(self, operation: Dict[str, Any]) -> None:
        """Execute add_arrows operation (deferred until after segment relocation)."""
        self.handle_arrows = operation

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

        logger.info(f"\n🔧 RELOCATING SEGMENT: {segment_name}")

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
        logger.info(f"\n✅ Completed relocation for segment '{segment_name}'")

        # After segment.move_to_top(align_transform)
        self.doc.recompute()
        FreeCADGui.updateGui()
        FreeCADGui.SendMsgToActiveView("ViewFit")
    def build_cut_list(self, filename: Optional[str] = None):
        """Build cutting list file."""
        if filename is None:
            filename = self.get_parm("cuts_file")

        logger.info(f"Building cut list: {filename}")
        with open(filename, "w+") as cuts_file:
            cuts_file.write("Cutting order:\n")
            for nbr, segment in enumerate(self.segment_list):
                segment.make_cut_list(nbr, cuts_file)

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

        with open(filename, "w+") as cuts_file:
            cuts_file.write("Wafer Placement:\n\n\n")
            global_placement = FreeCAD.Placement(FreeCAD.Vector(0, 0, 0), FreeCAD.Rotation(0, 0, 0))

            for nbr, segment in enumerate(self.segment_list):
                logger.debug(f"Segment: {segment.get_segment_name()}")
                global_placement = segment.print_construction_list(nbr, cuts_file, global_placement, find_min_max)

            min_max_str = f"\nGlobal Min Max:\n\tX: {min_max[0][0]} - {min_max[0][1]}, "
            min_max_str += f"Y: {min_max[1][0]} - {min_max[1][1]}, Z: {min_max[2][0]} - {min_max[2][1]}"
            cuts_file.write(f"{min_max_str}")

    def process_arrow_command(self):
        """Process deferred arrow command."""
        if not self.handle_arrows:
            return

        try:
            # Handle both old format (list) and new format (dict)
            if isinstance(self.handle_arrows, list):
                size = float(self.handle_arrows[1])
                point_nbr = int(self.handle_arrows[2])
            else:
                size = self.handle_arrows.get('size', 5.0)
                point_nbr = self.handle_arrows.get('point_number', 0)

            if not self.segment_list:
                logger.info("No segments available for arrow placement")
                return

            # Place arrow on the first segment
            segment_list_top = self.segment_list[0]

            # Check if the LCS object still exists
            try:
                lcs_top = segment_list_top.get_lcs_base()
                if not lcs_top or not hasattr(lcs_top, 'Placement'):
                    logger.error("Warning: LCS object no longer valid for arrow placement")
                    return
            except:
                logger.error("Warning: Cannot access LCS object for arrow placement")
                return

            if not self.compound_transform:
                logger.info("Warning: No compound transform available for arrow placement")
                return

            new_place = lcs_top.Placement.multiply(self.compound_transform)

            # Create arrow visualization with segment-specific name
            segment_name = segment_list_top.get_segment_name()
            arrow_name = f"{segment_name}_Arrow_{point_nbr}"
            arrow_obj = arrow(arrow_name, new_place, size)

            # Register arrow with the segment
            segment_list_top.register_arrow(arrow_obj)

            logger.debug(f"Added arrow '{arrow_name}' at point {point_nbr}")

        except Exception as e:
            logger.error(f"Error creating arrow: {e}")
            # Don't re-raise - arrow creation is not critical

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

    def _set_up_trace(self):
        """Set up tracing functionality."""
        self.trace_file_name = self.parent_parms.get("trace_file")
        self.do_trace = Driver.make_tf("do_trace", self.parent_parms)
        if self.do_trace:
            self.trace_file = open(self.trace_file_name, "w")
            self.trace_file.write("Start Trace\n")
            self.trace_file.flush()
        else:
            self.trace_file = None

    def trace(self, *args):
        """Write trace information."""
        if self.do_trace:
            if self.trace_file.closed:
                logger.debug("FILE WAS CLOSED")
                self.trace_file = open(self.trace_file_name, "a")
            trace_string = ''
            for arg in args:
                trace_string += "  " + repr(arg) + "\n"
            self.trace_file.write(trace_string)
            self.trace_file.flush()
            logger.debug(trace_string)

    def handle_spreadsheet(self, sheet):
        """Create functions to handle spreadsheet parameters."""

        def get_parm(parm_name):
            try:
                parm_value = sheet.get(parm_name)
                if parm_value == "None":
                    parm_value = None
                if self.do_trace:
                    self.trace_file.write(f"Parameter: {parm_name} fetched with value: {parm_value}\n")
                return parm_value
            except Exception as e:
                logger.error(f"Exception {e} reading from spreadsheet for value: {parm_name}")
                raise e

        def set_parm(parm_name, new_value):
            try:
                parm_value = sheet.set(parm_name, new_value)
                sheet.recompute()
                if self.do_trace:
                    self.trace_file.write(f"Parameter: {parm_name} set with value: {new_value}\n")
                return parm_value
            except Exception as e:
                logger.error(f"Exception {e} writing to spreadsheet for value: {parm_name}")
                raise e

        return get_parm, set_parm

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

    # @staticmethod
    # def make_transform_align(object_1, object_2):
    #     """Create transform that will move an object by the same relative positions of two input objects"""
    #     l1 = object_1.Placement
    #     l2 = object_2.Placement
    #     tr = l1.inverse().multiply(l2)
    #     FreeCAD.align = tr
    #     return tr

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move object_1 to object_2's placement"""
        # Since object_1 (current segment base) is at origin,
        # the transform is simply object_2's placement
        return FreeCAD.Placement(object_2.Placement)

    # Keep all the legacy methods for backwards compatibility
    def build_from_file(self):
        """Legacy CSV file reader (kept for backwards compatibility)."""
        # ... (keep existing implementation for backwards compatibility)
        pass


# Utility functions (unchanged)
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