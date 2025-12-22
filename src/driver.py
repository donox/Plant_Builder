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
import FreeCAD as App

from core.logging_setup import get_logger
# from test_transform import run_transform_test

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
            # logger.debug(f"Using existing '{self.doc_name}' document")
        except:
            # Create new document if it doesn't exist
            self.doc = self.app.newDocument(self.doc_name)
            logger.debug(f"Created new '{self.doc_name}' document")

        # logger.debug(f"Document ready: {self.doc.Name}")

    def workflow(self):
        """Execute the workflow defined in the config"""
        logger.info("Running: workflow()")

        # Setup document
        self.setup_document()

        # Get workflow mode setting
        workflow_mode = self.config.get('global_settings', {}).get('workflow_mode', None)

        if workflow_mode == "first_pass":
            logger.info("🎬 FIRST PASS MODE: Generate all segments + create editable closing curve")
        elif workflow_mode == "second_pass":
            logger.info("🔧 SECOND PASS MODE: Skip all operations except close_loop")
        else:
            logger.info("▶️  SINGLE RUN MODE: Normal execution")

        workflow_operations = self.config.get('workflow', [])

        for operation in workflow_operations:
            operation_type = operation.get('operation')
            description = operation.get('description', operation_type)

            # In second pass mode, skip everything except close_loop
            if workflow_mode == "second_pass" and operation_type != 'close_loop':
                logger.debug(f"⏭️  Skipping '{description}' (second pass mode)")
                continue

            logger.info(f"Executing: {description}")

            try:
                self._execute_operation(operation)
            except Exception as e:
                logger.error(f"Operation failed: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
        # logger.debug(f"Operation details: {operation_type}")

        # Route to appropriate handler
        if operation_type == 'remove_objects':
            self._remove_objects(operation)
        elif operation_type == 'build_segment':
            self._execute_build_segment(operation)
        elif operation_type == 'set_position':
            self._execute_set_position(operation)
        elif operation_type == 'close_loop':
            self._close_loop(operation)
        elif operation_type == 'test_bezier_closing':
            self._test_bezier_closing(operation)
        else:
            logger.warning(f"Unknown operation type: {operation_type}")

    def _test_bezier_closing(self, operation):
        """
        Run Bezier closing curve test

        Args:
            operation: Dictionary with test parameters
        """
        from test_bezier_closing import BezierClosingTest

        logger.info("Running Bezier closing test")

        tester = BezierClosingTest(self.doc)

        # Check if using explicit points
        use_explicit = operation.get('use_explicit_points', False)
        end_points_list = operation.get('end_points', None)
        start_points_list = operation.get('start_points', None)

        result = tester.run_test(
            num_end_points=operation.get('num_end_points', 3),
            num_start_points=operation.get('num_start_points', 3),
            separation=operation.get('separation', 10.0),
            tension=operation.get('tension', 0.5),
            curvature_weight=operation.get('curvature_weight', 0.2),
            cylinder_radius=operation.get('cylinder_radius', 1.0),
            seed=operation.get('seed', None),
            use_explicit_points=use_explicit,
            end_points_list=end_points_list,
            start_points_list=start_points_list
        )

        logger.info("✓ Bezier test complete")

    def _close_loop(self, operation):
        """
        Create a closing segment that connects the last segment back to the first
        Uses multiple LCS from each segment for smooth curvature continuity.
        """
        workflow_mode = self.config.get('global_settings', {}).get('workflow_mode', None)

        # Handle second pass mode - reconstruct segments from document
        # In second_pass mode, remove any existing closing segments (but NOT editable curves!)
        if workflow_mode == "second_pass":
            objects_to_remove = []
            for obj in self.doc.Objects:
                if 'closing_segment' in obj.Label:
                    # Don't delete editable curves (they start with "EDIT_")
                    if obj.Label.startswith('EDIT_'):
                        logger.debug(f"Preserving editable curve: {obj.Label}")
                        continue
                    objects_to_remove.append(obj.Name)

            if objects_to_remove:
                logger.info(f"Removing {len(objects_to_remove)} old closing segment objects")
                for obj_name in objects_to_remove:
                    try:
                        self.doc.removeObject(obj_name)
                    except:
                        pass
                self.doc.recompute()
            logger.info("Second pass mode: Looking for existing segments in document")
            # DEBUG: Show what objects exist
            logger.debug(f"All objects in document: {[obj.Label for obj in self.doc.Objects]}")
            logger.debug(f"Object types: {[(obj.Label, obj.TypeId) for obj in self.doc.Objects]}")

            # Find all segment Part objects - be more lenient
            segment_parts = []
            for obj in self.doc.Objects:
                if '_Part' in obj.Label and 'closing_segment' not in obj.Label:
                    segment_parts.append(obj)
                    logger.debug(f"Found segment part: {obj.Label} ({obj.TypeId})")

            if len(segment_parts) < 2:
                raise ValueError(f"Second pass mode requires at least 2 existing segments. Found: {len(segment_parts)}")

            logger.info(f"Found {len(segment_parts)} existing segments")

            # Reconstruct first and last segments
            first_segment = self._reconstruct_segment_from_part(segment_parts[0])
            last_segment = self._reconstruct_segment_from_part(segment_parts[-1])

            # Store in segment_list for the close operation
            self.segment_list = [first_segment, last_segment]

            logger.info(f"Reconstructed segments: '{first_segment.name}' (first) and '{last_segment.name}' (last)")
        else:
            # Normal mode or first pass mode: use built segments
            if len(self.segment_list) < 2:
                logger.warning("Need at least 2 segments to close a loop")
                return

        first_segment = self.segment_list[0]
        last_segment = self.segment_list[-1]

        logger.info("Creating closing segment with smooth curvature matching")

        # Get wafer settings and segment settings
        wafer_settings = operation.get('wafer_settings', {})
        segment_settings = operation.get('segment_settings', {})

        # Create curve_spec with closing curve
        curve_spec = {
            'type': 'closing_curve',
            'parameters': {
                'start_segment': first_segment,
                'end_segment': last_segment,
                'num_lcs_per_end': operation.get('num_lcs_per_end', 3),
                'tension': operation.get('tension', 0.5),
                'points': operation.get('points', 50),
                'cylinder_radius': wafer_settings.get('cylinder_diameter', 2.0) / 2.0,
                # Pass through edit-related parameters
                'use_edited_curve': operation.get('use_edited_curve'),
                'create_editable_curve': operation.get('create_editable_curve', workflow_mode == "first_pass")
            }
        }

        # Determine base placement for closing segment
        use_edited_curve = operation.get('use_edited_curve')

        if use_edited_curve:
            # For edited curves in world coordinates, place segment at the exit point
            # so the local coordinate transformation aligns correctly
            last_wafer = last_segment.wafer_list[-1]
            base_placement_for_closing = last_segment.base_placement.multiply(last_wafer.lcs2)
            logger.info(f"Edited curve mode - placing segment at exit LCS: {base_placement_for_closing.Base}")
        else:
            # For auto-generated curves, create at origin (will be aligned later)
            base_placement_for_closing = App.Placement()
            logger.debug("Auto-generated curve mode - placing segment at origin")

        # Create the closing segment
        from loft_segment import LoftSegment
        closing_segment = LoftSegment(
            doc=self.doc,
            name=operation.get('name', 'closing_segment'),
            curve_spec=curve_spec,
            wafer_settings=wafer_settings,
            segment_settings=segment_settings,
            base_placement=base_placement_for_closing,
            connection_spec={}
        )

        logger.debug(f"Created LoftSegment: {closing_segment.name}")

        # Generate wafers (at origin)
        closing_segment.generate_wafers()

        # In first pass mode, stop after creating editable curve
        if workflow_mode == "first_pass":
            logger.info("⏸️  FIRST PASS COMPLETE")
            logger.info("    1. Edit the created curve using FreeCAD tools")
            logger.info("    2. Change workflow_mode to 'second_pass' in YAML")
            logger.info("    3. Add use_edited_curve parameter to close_loop operation")
            logger.info("    4. Re-run the workflow")
            return

        # Continue with normal processing (visualize, align, etc.)
        # Visualize (creates Part and adds objects)
        closing_segment.visualize(self.doc)

        # Continue with normal processing (visualize, align, etc.)
        # Visualize (creates Part and adds objects)
        closing_segment.visualize(self.doc)

        # Check if we used an edited curve
        used_edited_curve = operation.get('use_edited_curve') is not None

        if not used_edited_curve:
            # Only apply alignment if we auto-generated the curve
            # Special alignment for closing segment
            adjusted_placement = self._align_segment_to_previous(closing_segment, last_segment)

            # CRITICAL: Apply 180° rotation in LOCAL coordinates
            local_rotation = App.Rotation(App.Vector(0, 1, 0), 180)
            adjusted_placement = App.Placement(
                adjusted_placement.Base,
                adjusted_placement.Rotation.multiply(local_rotation)
            )
            logger.debug("Applied 180° rotation around local Y-axis to closing segment")

            # Find and update the Part object
            part_name = f"{closing_segment.name}_Part"
            part_obj = self.doc.getObject(part_name)

            if part_obj:
                part_obj.Placement = adjusted_placement
                closing_segment.base_placement = adjusted_placement
                logger.debug(f"Aligned closing segment to last segment")
            else:
                logger.warning(f"Could not find Part object: {part_name}")
        else:
            logger.info("Using edited curve - skipping alignment (wafers already in correct position)")

        # Store segment
        self.segment_list.append(closing_segment)

        # Verify the closure
        end_placement = closing_segment.get_end_placement()
        start_placement = first_segment.get_start_placement()

        gap = end_placement.Base.distanceToPoint(start_placement.Base)
        logger.info(f"Loop closure gap: {gap:.3f}")

        # Debug: show actual segment length
        if len(closing_segment.wafer_list) > 0:
            first_wafer_pos = closing_segment.base_placement.multVec(
                closing_segment.wafer_list[0].lcs1.Base
            )
            last_wafer_pos = closing_segment.base_placement.multVec(
                closing_segment.wafer_list[-1].lcs2.Base
            )
            actual_length = first_wafer_pos.distanceToPoint(last_wafer_pos)
            logger.warning(f"Closing segment actual length: {actual_length:.3f} (expected ~{gap:.3f})")

        if gap > 1.0:
            logger.warning(f"Large closure gap detected: {gap:.3f} - loop may not be properly closed")

        logger.info("✓ Closing segment created with curvature continuity")

    def _reconstruct_segment_from_part(self, part_obj):
        """
        Reconstruct a LoftSegment from an existing Part object in the document.
        Used in second_pass mode to work with pre-existing segments.
        """
        from loft_segment import LoftSegment
        from wafer_loft import Wafer
        import FreeCAD as App

        # Extract segment name from Part label (remove '_Part' suffix)
        segment_name = part_obj.Label.replace('_Part', '')

        logger.debug(f"Reconstructing segment '{segment_name}' from Part object")

        # Create minimal LoftSegment with placeholder values
        segment = LoftSegment(
            doc=self.doc,
            name=segment_name,
            curve_spec={'type': 'reconstructed'},
            wafer_settings={'cylinder_diameter': 2.0},
            segment_settings={},
            base_placement=part_obj.Placement,
            connection_spec={}
        )

        # Find associated wafer objects
        wafer_prefix = f"Wafer_{segment_name}_"
        wafer_objects = []
        for obj in self.doc.Objects:
            if obj.Label.startswith(wafer_prefix):
                wafer_objects.append(obj)

        # Sort by number in name
        wafer_objects.sort(key=lambda obj: int(obj.Label.split('_')[-1]) if obj.Label.split('_')[-1].isdigit() else 0)

        if wafer_objects:
            logger.debug(f"Found {len(wafer_objects)} wafer objects for '{segment_name}'")

            # Find LCS objects for each wafer
            for i, wafer_obj in enumerate(wafer_objects):
                # Look for associated LCS objects
                lcs1_name = f"LCS_{segment_name}_{i}_1"
                lcs2_name = f"LCS_{segment_name}_{i}_2"

                lcs1_obj = self.doc.getObject(lcs1_name)
                lcs2_obj = self.doc.getObject(lcs2_name)

                if lcs1_obj and lcs2_obj:
                    # Create minimal Wafer with all required parameters
                    wafer = Wafer(
                        solid=wafer_obj,  # Use the wafer object itself
                        index=i,
                        plane1=None,  # Not needed for reconstruction
                        plane2=None,  # Not needed for reconstruction
                        geometry=None,  # Not needed for reconstruction
                        lcs1=lcs1_obj.Placement,
                        lcs2=lcs2_obj.Placement
                    )
                else:
                    # Approximate LCS from wafer placement
                    wafer = Wafer(
                        solid=wafer_obj,
                        index=i,
                        plane1=None,
                        plane2=None,
                        geometry=None,
                        lcs1=wafer_obj.Placement,
                        lcs2=App.Placement(
                            wafer_obj.Placement.Base + wafer_obj.Placement.Rotation.multVec(App.Vector(0, 0, 1)),
                            wafer_obj.Placement.Rotation
                        )
                    )
                segment.wafer_list.append(wafer)
        else:
            logger.warning(f"No wafer objects found for '{segment_name}', creating dummy wafer")

            # Create single dummy wafer at Part placement
            dummy_wafer = Wafer(
                solid=None,
                index=0,
                plane1=None,
                plane2=None,
                geometry=None,
                lcs1=part_obj.Placement,
                lcs2=App.Placement(
                    part_obj.Placement.Base + part_obj.Placement.Rotation.multVec(App.Vector(0, 0, 1)),
                    part_obj.Placement.Rotation
                )
            )
            segment.wafer_list = [dummy_wafer]

        logger.debug(f"Reconstructed segment '{segment_name}' with {len(segment.wafer_list)} wafers")
        return segment

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
            # logger.debug("Complete list of objects to keep:")
            for obj in objects_to_keep:
                # logger.debug(f"  KEEP: {obj.Label}")
                pass

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
                    # logger.debug(f"Removing: {obj_label}")
                    App.ActiveDocument.removeObject(obj_name)
                    removed_count += 1
                # else:
                #     logger.debug(f"Already removed: {obj_label}")
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
            # logger.debug(f"  Removed {obj.Label} from {parent.Label}'s Group")

    def _add_children_to_set(self, obj, object_set):
        """Recursively add all children of an object to the set"""
        if hasattr(obj, 'Group'):
            for child in obj.Group:
                if child not in object_set:
                    # logger.debug(f"  Keeping child: {child.Label}")
                    object_set.add(child)
                    self._add_children_to_set(child, object_set)

        if hasattr(obj, 'OutList'):
            for child in obj.OutList:
                if child not in object_set:
                    # logger.debug(f"  Keeping dependency: {child.Label}")
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
        # Now only curve_follower segments are supported.
        self._build_curve_follower_segment(operation)

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
                part_obj.Placement = adjusted_placement
                segment.base_placement = adjusted_placement
            else:
                logger.warning(f"Could not find Part object. Tried: {part_name_variations}")

        # Apply rotate_segment if specified (rotates around segment's base Z-axis)
        rotate_segment = segment_settings.get('rotate_segment', 0)
        if rotate_segment != 0 and part_obj:
            logger.info(f"Applying segment rotation: {rotate_segment}°")

            # Get the entry LCS (first wafer's first LCS) as rotation axis
            if len(segment.wafer_list) > 0 and segment.wafer_list[0].lcs1:
                entry_lcs = segment.wafer_list[0].lcs1

                # Transform entry LCS to world coordinates
                if len(self.segment_list) > 0:
                    # Non-first segment - entry LCS is in adjusted_placement coordinates
                    entry_lcs_world = adjusted_placement.multiply(entry_lcs)
                else:
                    # First segment
                    entry_lcs_world = entry_lcs

                # Get Z-axis and rotation point from entry LCS
                rotation_axis = entry_lcs_world.Rotation.multVec(App.Vector(0, 0, 1))
                rotation_point = entry_lcs_world.Base

                # Create rotation
                rotation = App.Rotation(rotation_axis, rotate_segment)

                # Apply to Part object
                current_placement = part_obj.Placement

                # Rotate position around rotation point
                offset = current_placement.Base - rotation_point
                rotated_offset = rotation.multVec(offset)
                current_placement.Base = rotation_point + rotated_offset

                # Apply rotation to orientation
                current_placement.Rotation = rotation.multiply(current_placement.Rotation)

                part_obj.Placement = current_placement
                segment.base_placement = current_placement
                logger.debug(f"Rotated segment {rotate_segment}° around entry Z-axis")

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
        # logger.debug(f"Type of connection_spec: {type(segment.connection_spec)}")
        # logger.debug(f"Full connection_spec contents: {segment.connection_spec}")
        # logger.debug(f"connection_spec repr: {repr(segment.connection_spec)}")

        # 1. Get previous segment's EXIT LCS in world coordinates
        if not prev_segment.wafer_list or not prev_segment.wafer_list[-1].lcs2:
            logger.warning("Previous segment has no exit LCS")
            return segment.base_placement

        prev_local_exit = prev_segment.wafer_list[-1].lcs2
        prev_world_exit = prev_segment.base_placement.multiply(prev_local_exit)
        # logger.debug(f"Previous segment EXIT (world): {prev_world_exit}")

        # 2. Get current segment's ENTRY LCS in local coordinates
        if not segment.wafer_list or not segment.wafer_list[0].lcs1:
            logger.warning("Current segment has no entry LCS")
            return segment.base_placement

        curr_local_entry = segment.wafer_list[0].lcs1
        # logger.debug(f"Current segment ENTRY (local): {curr_local_entry}")

        # 3. Calculate adjusted placement: prev_exit * curr_entry.inverse()
        adjusted_placement = prev_world_exit.multiply(curr_local_entry.inverse())
        # logger.debug(f"Calculated adjusted placement: {adjusted_placement}")

        # Apply additional Z-axis rotation if specified in connection_spec
        rotation_angle = segment.connection_spec.get('rotation_angle', 0)
        # logger.debug(f"Connection spec: {segment.connection_spec}")
        # logger.debug(f"Rotation angle from connection_spec: {rotation_angle}")

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
        #     logger.debug(f"Adjusted placement after rotation: {adjusted_placement}")
        # else:
        #     logger.debug("No additional rotation specified (rotation_angle is 0 or not specified)")

        # 4. Verify alignment (for debugging)

        curr_world_entry = adjusted_placement.multiply(curr_local_entry)
        pos_match = (curr_world_entry.Base - prev_world_exit.Base).Length
        rot_match = curr_world_entry.Rotation.isSame(prev_world_exit.Rotation, 1e-6)

        # logger.debug(f"Position difference: {pos_match}")
        # logger.debug(f"Rotation match: {rot_match}")

        # Check individual axes
        # prev_x = prev_world_exit.Rotation.multVec(App.Vector(1, 0, 0))
        # prev_y = prev_world_exit.Rotation.multVec(App.Vector(0, 1, 0))
        # prev_z = prev_world_exit.Rotation.multVec(App.Vector(0, 0, 1))
        #
        # curr_x = curr_world_entry.Rotation.multVec(App.Vector(1, 0, 0))
        # curr_y = curr_world_entry.Rotation.multVec(App.Vector(0, 1, 0))
        # curr_z = curr_world_entry.Rotation.multVec(App.Vector(0, 0, 1))

        # logger.debug(f"Prev X={prev_x}, Curr X={curr_x}, dot={prev_x.dot(curr_x):.6f}")
        # logger.debug(f"Prev Y={prev_y}, Curr Y={curr_y}, dot={prev_y.dot(curr_y):.6f}")
        # logger.debug(f"Prev Z={prev_z}, Curr Z={curr_z}, dot={prev_z.dot(curr_z):.6f}")

        if pos_match < 1e-6 and rot_match:
            # logger.debug("✓ LCS alignment verified")
            pass
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

        # logger.debug(f"Position set to: {self.current_placement}")

    def _generate_cutting_list(self):
        """Generate cutting list for all segments"""
        output_file = self.output_files.get('cuts_file')
        if not os.path.isabs(output_file):
            base_dir = self.output_files.get('working_directory', '')
            output_file = os.path.join(base_dir, output_file)
            logger.info(f"Generating cutting list with file: {output_file}")

        if not output_file:
            logger.warning("No cutting list file specified")
            return

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
        """Convert decimal inches to fractional format like '1 3/16'"""
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
            print(f"Include definitions XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: {include_definitions}")
            # Summary
            cuts_file.write("\n" + "-" * 90 + "\n")
            cuts_file.write(f"Total cylinder length required: {self._format_inches(cumulative_length)}\n")
            cuts_file.write(f"  ({cumulative_length:.3f} inches = {cumulative_length * 25.4:.1f} mm)\n\n")

            # Column definitions (if enabled)
            include_definitions = self.global_settings.get('include_cut_list_definitions', False)
            print(f"Include definitions: {include_definitions}")

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