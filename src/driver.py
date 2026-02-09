"""
driver.py - Main workflow driver for PlantBuilder

Orchestrates the build workflow by:
- Loading configuration from YAML
- Managing FreeCAD document
- Executing operations in sequence
- Generating output files (cutting lists, placement lists)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import FreeCAD as App

from config.loader import load_config
from core.logging_setup import get_logger
from curves import Curves
from core.metadata import apply_metadata


logger = get_logger(__name__)


class Driver:
    """Main driver class for PlantBuilder workflow"""

    def __init__(self, app, gui, doc_name: str):
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

        self.config_file: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None

        self.doc = self._get_or_create_document(doc_name)

        self.segment_list = []
        self.global_settings: Dict[str, Any] = {}
        self.current_placement = app.Placement()
        self.output_files: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Document management
    # -------------------------------------------------------------------------

    def _get_or_create_document(self, doc_name: str):
        """
        Return an existing FreeCAD document by name, or create it if missing.
        Also makes it the active document.
        """
        if self.app is None:
            return None

        # Prefer active doc if it exists and matches
        doc = self.app.ActiveDocument
        if doc is not None and (not doc_name or doc.Name == doc_name):
            return doc

        # Try to find an open doc by name
        try:
            if doc_name and doc_name in self.app.listDocuments():
                doc = self.app.getDocument(doc_name)
                self.app.setActiveDocument(doc.Name)
                if self.gui:
                    self.gui.ActiveDocument = self.gui.getDocument(doc.Name)
                return doc
        except Exception:
            pass

        # Create if missing
        if not doc_name:
            doc_name = "PlantBuilder"

        doc = self.app.newDocument(doc_name)
        self.app.setActiveDocument(doc.Name)
        if self.gui:
            self.gui.ActiveDocument = self.gui.getDocument(doc.Name)

        logger.debug(f"Created new '{doc.Name}' document")
        return doc

    def setup_document(self):
        """Ensure the document is ready and active."""
        self.doc = self._get_or_create_document(self.doc_name)

    # -------------------------------------------------------------------------
    # Config & metadata
    # -------------------------------------------------------------------------

    def load_configuration(self, config_file: str):
        self.config_file = config_file

        loaded = load_config(
            config_file,
            yaml_base_dir=os.path.join(os.path.dirname(__file__), "yaml_files", "base"),
        )

        self.config = loaded.data
        self.global_settings = self.config.get("global_settings", {}) or {}
        self.output_files = self.config.get("output_files", {}) or {}
        self.metadata = self.config.get("metadata", {}) or {}

        # Apply metadata immediately (but make sure remove_objects keeps it)
        if self.doc and self.metadata:
            apply_metadata(self.doc, self.metadata, source_config=config_file)

    # -------------------------------------------------------------------------
    # Workflow execution
    # -------------------------------------------------------------------------

    def workflow(self):
        """Execute the workflow defined in the config"""
        logger.info("Running: workflow()")

        # Setup document
        self.setup_document()

        if not self.config:
            raise RuntimeError("No configuration loaded. Call load_configuration() first.")

        workflow_mode = self.global_settings.get("workflow_mode", None)

        if workflow_mode == "first_pass":
            logger.info("🎬 FIRST PASS MODE: Generate all segments + create editable closing curve")
        elif workflow_mode == "second_pass":
            logger.info("🔧 SECOND PASS MODE: Skip all operations except close_loop")
        else:
            logger.info("▶️  SINGLE RUN MODE: Normal execution")

        workflow_operations = self.config.get("workflow", []) or []

        for operation in workflow_operations:
            operation_type = operation.get("operation")
            description = operation.get("description", operation_type)

            # Second pass: skip everything but close_loop
            if workflow_mode == "second_pass" and operation_type != "close_loop":
                logger.debug(f"⏭️  Skipping '{description}' (second pass mode)")
                continue

            logger.info(f"Executing: {description}")

            try:
                self._execute_operation(operation)
            except RuntimeError as e:
                logger.error(f"Operation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Operation failed: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise

        # Generate ONE cutting list for all segments at end (lofted method only)
        if self.global_settings.get("print_cuts", False):
            self._write_lofted_cut_list()

        logger.info("✅ Workflow complete")

    def _execute_operation(self, operation: Dict[str, Any]):
        operation_type = operation.get("operation")
        if not operation_type:
            logger.warning("Operation missing 'operation' field")
            return

        if operation_type == "remove_objects":
            self._remove_objects(operation)
        elif operation_type == "build_segment":
            self._execute_build_segment(operation)
        elif operation_type == "set_position":
            self._execute_set_position(operation)
        elif operation_type == "close_loop":
            self._close_loop(operation)
        elif operation_type == "export_curve":
            self._execute_export_curve(operation)
        else:
            logger.warning(f"Unknown operation type: {operation_type}")

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def _execute_export_curve(self, operation: Dict[str, Any]) -> None:
        """
        Export a composite curve from existing segments as a single FreeCAD object.

        YAML example:

        - operation export_curve
          name FinalCurve
          description Composite of base + Curve Left + Curve Right + close_seg
          segments
            - base
            - Curve Left
            - Curve Right
            - close_seg
          sampling
            per_segment 100
        """
        curve_name = operation.get("name", "ExportedCurve")
        segment_names = operation.get("segments", [])
        sampling_cfg = operation.get("sampling", {})
        points_per_segment = int(sampling_cfg.get("per_segment", 100))

        if not segment_names:
            raise ValueError("export_curve operation requires a non-empty 'segments' list.")

        if points_per_segment < 2:
            raise ValueError(
                f"export_curve 'sampling.per_segment' must be >= 2, got {points_per_segment}."
            )

        if not self.segment_list:
            raise ValueError(
                "No segments available in driver.segment_list; "
                "export_curve must run after build_segment operations."
            )

        logger.info(
            f"Exporting composite curve '{curve_name}' from segments {segment_names} "
            f"with {points_per_segment} points per segment."
        )

        # 1) Resolve LoftSegment instances in the order given.
        ordered_segments = []
        for seg_name in segment_names:
            match = None
            for seg in self.segment_list:
                logger.debug(f"NAMES OF SEGMENTS: {seg_name} -> {seg.name}")
                if seg.name == seg_name:
                    match = seg
                    break
            if match is None:
                raise ValueError(
                    f"export_curve: could not find segment named '{seg_name}' "
                    "in driver.segment_list."
                )
            ordered_segments.append(match)

        all_points_world: list[App.Vector] = []

        # 2) For each segment, regenerate its curve points in local coordinates
        #    via Curves + curve_spec, then map to world using segment.base_placement. [file:2][file:7]
        for idx, segment in enumerate(ordered_segments):
            if not segment.curve_spec:
                raise ValueError(
                    f"export_curve: segment '{segment.name}' has no curve_spec."
                )

            curve_spec = segment.curve_spec
            curve_type = curve_spec.get("type")

            if curve_type == "existing_curve":
                logger.info(
                    f"export_curve: skipping segment '{segment.name}' "
                    f"(type {curve_type}) for parametric sampling."
                )
                continue

            # Use the Curves class to generate the transformed curve points.
            curves_instance = Curves(self.doc, curve_spec)
            points_array = curves_instance.get_curve_points()  # numpy array (N, 3). [file:2]

            if points_array is None or len(points_array) == 0:
                raise ValueError(
                    f"export_curve: Curves returned no points for segment '{segment.name}'."
                )

            # Resample / thin to points_per_segment, using simple index-based sampling
            # for v1. This keeps behavior predictable and easy to adjust later.
            total_pts = len(points_array)
            if total_pts <= points_per_segment:
                indices = range(total_pts)
            else:
                # Evenly distributed indices from 0 .. total_pts-1
                indices = [
                    int(round(i * (total_pts - 1) / (points_per_segment - 1)))
                    for i in range(points_per_segment)
                ]

            segment_points_local: list[App.Vector] = []
            for i in indices:
                p = points_array[i]
                segment_points_local.append(App.Vector(float(p[0]), float(p[1]), float(p[2])))

            # Transform to world coordinates using the segment's base_placement. [file:7]
            segment_points_world = segment.transform_curve_points(segment_points_local)

            # Avoid duplicating the join point between consecutive segments
            if all_points_world and segment_points_world:
                last_global = all_points_world[-1]
                first_new = segment_points_world[0]
                if (last_global.sub(first_new)).Length <= 1e-6:
                    segment_points_world = segment_points_world[1:]

            all_points_world.extend(segment_points_world)

            logger.info(
                f"export_curve: segment '{segment.name}' contributed "
                f"{len(segment_points_world)} world points "
                f"(source points: {total_pts})."
            )

        if len(all_points_world) < 2:
            raise ValueError(
                f"export_curve: composite curve '{curve_name}' has fewer than 2 points."
            )

        # 3) Build a BSpline curve through all world points and create a Part::Feature.
        import Part  # local import to avoid circular issues

        bspline = Part.BSplineCurve()
        bspline.interpolate(all_points_world)

        doc = self.doc
        if doc is None:
            raise ValueError("export_curve: driver.doc is None; cannot create curve object.")

        # Remove any existing object with this label, to keep things clean.
        existing = [obj for obj in doc.Objects if obj.Label == curve_name]
        for obj in existing:
            try:
                doc.removeObject(obj.Name)
            except Exception:
                pass

        curve_obj = doc.addObject("Part::Feature", "ExportedCurve")
        curve_obj.Label = curve_name
        curve_obj.Shape = bspline.toShape()

        # Simple styling so it is visible.
        if hasattr(curve_obj, "ViewObject"):
            curve_obj.ViewObject.LineColor = (0.0, 1.0, 0.0)
            curve_obj.ViewObject.LineWidth = 3.0

        doc.recompute()

        # 4) Basic diagnostics: length and point count.
        total_length = 0.0
        for i in range(len(all_points_world) - 1):
            total_length += all_points_world[i].distanceToPoint(all_points_world[i + 1])

        logger.info(
            f"export_curve: created curve '{curve_name}' with "
            f"{len(all_points_world)} points, approximate length {total_length:.3f}."
        )


    def _close_loop(self, operation):
        """
        Create a closing segment that connects the last segment back to the first.
        Uses multiple LCS from each segment for smooth curvature continuity.
        """
        workflow_mode = self.global_settings.get("workflow_mode", None)

        if workflow_mode == "second_pass":
            objects_to_remove = []
            for obj in self.doc.Objects:
                if "closing_segment" in obj.Label:
                    if obj.Label.startswith("EDIT_"):
                        logger.debug(f"Preserving editable curve: {obj.Label}")
                        continue
                    objects_to_remove.append(obj.Name)

            if objects_to_remove:
                logger.info(f"Removing {len(objects_to_remove)} old closing segment objects")
                for obj_name in objects_to_remove:
                    try:
                        self.doc.removeObject(obj_name)
                    except Exception:
                        pass
                self.doc.recompute()

            logger.info("Second pass mode: Looking for existing segments in document")

            segment_parts = []
            for obj in self.doc.Objects:
                if "_Part" in obj.Label and "closing_segment" not in obj.Label:
                    segment_parts.append(obj)

            if len(segment_parts) < 2:
                raise ValueError(
                    f"Second pass mode requires at least 2 existing segments. Found: {len(segment_parts)}"
                )

            first_segment = self._reconstruct_segment_from_part(segment_parts[0])
            last_segment = self._reconstruct_segment_from_part(segment_parts[-1])
            self.segment_list = [first_segment, last_segment]
        else:
            if len(self.segment_list) < 2:
                logger.warning("Need at least 2 segments to close a loop")
                return

        first_segment = self.segment_list[0]
        last_segment = self.segment_list[-1]

        wafer_settings = operation.get("wafer_settings", {}) or {}
        segment_settings = operation.get("segment_settings", {}) or {}

        curve_spec = {
            "type": "closing_curve",
            "parameters": {
                "start_segment": first_segment,
                "end_segment": last_segment,
                "num_lcs_per_end": operation.get("num_lcs_per_end", 3),
                "tension": operation.get("tension", 0.5),
                "points": operation.get("points", 50),
                "cylinder_radius": float(wafer_settings.get("cylinder_diameter", 2.0)) / 2.0,
                "use_edited_curve": operation.get("use_edited_curve"),
                "create_editable_curve": operation.get("create_editable_curve", workflow_mode == "first_pass"),
                "max_closing_angle": operation.get("max_closing_angle", 90.0),
                "entry_helper_length": operation.get("entry_helper_length"),
                "exit_helper_length": operation.get("exit_helper_length"),
            },
        }

        use_edited_curve = operation.get("use_edited_curve")

        if use_edited_curve:
            last_wafer = last_segment.wafer_list[-1]
            base_placement_for_closing = last_segment.base_placement.multiply(last_wafer.lcs2)
            logger.info(f"Edited curve mode - placing segment at exit LCS: {base_placement_for_closing.Base}")
        else:
            base_placement_for_closing = App.Placement()

        from loft_segment import LoftSegment

        closing_segment = LoftSegment(
            doc=self.doc,
            name=operation.get("name", "closing_segment"),
            curve_spec=curve_spec,
            wafer_settings=wafer_settings,
            segment_settings=segment_settings,
            base_placement=base_placement_for_closing,
            connection_spec={},
        )

        closing_segment.generate_wafers()

        if workflow_mode == "first_pass":
            logger.info("⏸️  FIRST PASS COMPLETE")
            logger.info("    1. Edit the created curve using FreeCAD tools")
            logger.info("    2. Change workflow_mode to 'second_pass' in YAML")
            logger.info("    3. Add use_edited_curve parameter to close_loop operation")
            logger.info("    4. Re-run the workflow")
            return

        closing_segment.visualize(self.doc)

        used_edited_curve = operation.get("use_edited_curve") is not None

        if not used_edited_curve:
            adjusted_placement = self._align_segment_to_previous(closing_segment, last_segment)

            local_rotation = App.Rotation(App.Vector(0, 1, 0), 180)
            adjusted_placement = App.Placement(
                adjusted_placement.Base,
                adjusted_placement.Rotation.multiply(local_rotation),
            )

            part_name = f"{closing_segment.name}_Part"
            part_obj = self.doc.getObject(part_name)

            if part_obj:
                part_obj.Placement = adjusted_placement
                closing_segment.base_placement = adjusted_placement
            else:
                logger.warning(f"Could not find Part object: {part_name}")
        else:
            logger.info("Using edited curve - skipping alignment (wafers already in correct position)")

        self.segment_list.append(closing_segment)

        end_placement = closing_segment.get_end_placement()
        start_placement = first_segment.get_start_placement()

        gap = end_placement.Base.distanceToPoint(start_placement.Base)
        logger.info(f"Loop closure gap: {gap:.3f}")

        if gap > 1.0:
            logger.warning(f"Large closure gap detected: {gap:.3f} - loop may not be properly closed")

        logger.info("✓ Closing segment created with curvature continuity")

    def _reconstruct_segment_from_part(self, part_obj):
        from loft_segment import LoftSegment
        from wafer_loft import Wafer

        segment_name = part_obj.Label.replace("_Part", "")

        segment = LoftSegment(
            doc=self.doc,
            name=segment_name,
            curve_spec={"type": "reconstructed"},
            wafer_settings={"cylinder_diameter": 2.0},
            segment_settings={},
            base_placement=part_obj.Placement,
            connection_spec={},
        )

        wafer_prefix = f"Wafer_{segment_name}_"
        wafer_objects = [obj for obj in self.doc.Objects if obj.Label.startswith(wafer_prefix)]

        wafer_objects.sort(
            key=lambda obj: int(obj.Label.split("_")[-1]) if obj.Label.split("_")[-1].isdigit() else 0
        )

        if wafer_objects:
            for i, wafer_obj in enumerate(wafer_objects):
                lcs1_name = f"LCS_{segment_name}_{i}_1"
                lcs2_name = f"LCS_{segment_name}_{i}_2"

                lcs1_obj = self.doc.getObject(lcs1_name)
                lcs2_obj = self.doc.getObject(lcs2_name)

                if lcs1_obj and lcs2_obj:
                    wafer = Wafer(
                        solid=wafer_obj,
                        index=i,
                        plane1=None,
                        plane2=None,
                        geometry=None,
                        lcs1=lcs1_obj.Placement,
                        lcs2=lcs2_obj.Placement,
                    )
                else:
                    wafer = Wafer(
                        solid=wafer_obj,
                        index=i,
                        plane1=None,
                        plane2=None,
                        geometry=None,
                        lcs1=wafer_obj.Placement,
                        lcs2=App.Placement(
                            wafer_obj.Placement.Base + wafer_obj.Placement.Rotation.multVec(App.Vector(0, 0, 1)),
                            wafer_obj.Placement.Rotation,
                        ),
                    )
                segment.wafer_list.append(wafer)
        else:
            logger.warning(f"No wafer objects found for '{segment_name}', creating dummy wafer")
            dummy = Wafer(
                solid=None,
                index=0,
                plane1=None,
                plane2=None,
                geometry=None,
                lcs1=part_obj.Placement,
                lcs2=App.Placement(
                    part_obj.Placement.Base + part_obj.Placement.Rotation.multVec(App.Vector(0, 0, 1)),
                    part_obj.Placement.Rotation,
                ),
            )
            segment.wafer_list = [dummy]

        return segment

    def _remove_objects(self, operation):
        """
        Remove objects from the document except those specified to keep.

        NOTE: Always preserves ModelMetadata so it remains visible even when configs
        use remove_objects at the start of a run.
        """
        keep_patterns = operation.get("keep_patterns", None)

        if keep_patterns is None:
            logger.info("No keep_patterns specified - will remove all objects")
            keep_patterns = []
        elif not keep_patterns:
            logger.info("Empty keep_patterns list - will remove all objects")
        else:
            logger.info(f"Keeping objects matching patterns: {keep_patterns}")

        # Always keep metadata object
        if "ModelMetadata" not in keep_patterns:
            keep_patterns = list(keep_patterns) + ["ModelMetadata"]

        all_objects = App.ActiveDocument.Objects
        objects_to_keep = set()

        if keep_patterns:
            for obj in all_objects:
                matched = False
                for pattern in keep_patterns:
                    if obj.Label == pattern or obj.Name == pattern:
                        matched = True
                        break
                    elif pattern.endswith("*") and (obj.Label.startswith(pattern[:-1]) or obj.Name.startswith(pattern[:-1])):
                        matched = True
                        break
                    elif pattern.startswith("*") and (obj.Label.endswith(pattern[1:]) or obj.Name.endswith(pattern[1:])):
                        matched = True
                        break
                    elif pattern in obj.Label or pattern in obj.Name:
                        matched = True
                        break

                if matched:
                    objects_to_keep.add(obj)
                    self._add_children_to_set(obj, objects_to_keep)

        # Break parent-child links for kept objects whose parents will be removed
        for obj in objects_to_keep:
            parent = self._get_parent(obj)
            if parent and parent not in objects_to_keep:
                self._remove_from_parent(obj, parent)

        objects_to_remove = []
        for obj in all_objects:
            if obj not in objects_to_keep:
                objects_to_remove.append((obj.Name, obj.Label))

        removed_count = 0
        for obj_name, obj_label in objects_to_remove:
            try:
                obj = App.ActiveDocument.getObject(obj_name)
                if obj:
                    App.ActiveDocument.removeObject(obj_name)
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove {obj_label}: {e}")

        logger.info(f"✓ Removed {removed_count} objects, kept {len(objects_to_keep)} objects")
        App.ActiveDocument.recompute()

    def _get_parent(self, obj):
        for potential_parent in App.ActiveDocument.Objects:
            if hasattr(potential_parent, "Group") and obj in potential_parent.Group:
                return potential_parent
        return None

    def _remove_from_parent(self, obj, parent):
        if hasattr(parent, "Group"):
            parent.Group = [child for child in parent.Group if child != obj]

    def _add_children_to_set(self, obj, object_set):
        if hasattr(obj, "Group"):
            for child in obj.Group:
                if child not in object_set:
                    object_set.add(child)
                    self._add_children_to_set(child, object_set)

        if hasattr(obj, "OutList"):
            for child in obj.OutList:
                if child not in object_set:
                    object_set.add(child)
                    self._add_children_to_set(child, object_set)

    def _execute_build_segment(self, operation):
        # Only curve_follower supported at this point
        self._build_curve_follower_segment(operation)

    def _build_curve_follower_segment(self, operation):
        segment_name = operation.get("name", "segment")
        curve_spec = operation.get("curve_spec", {}) or {}
        wafer_settings = operation.get("wafer_settings", {}) or {}
        segment_settings = operation.get("segment_settings", {}) or {}
        connection_spec = operation.get("connection", {}) or {}

        if len(self.segment_list) == 0:
            base_placement = self.current_placement
        else:
            base_placement = App.Placement()

        from loft_segment import LoftSegment

        segment = LoftSegment(
            doc=self.doc,
            name=segment_name,
            curve_spec=curve_spec,
            wafer_settings=wafer_settings,
            segment_settings=segment_settings,
            base_placement=base_placement,
            connection_spec=connection_spec,
        )

        segment.generate_wafers()
        segment.visualize(self.doc)

        part_obj = None
        if len(self.segment_list) > 0:
            adjusted_placement = self._align_segment_to_previous(segment, self.segment_list[-1])

            part_name_variations = [
                f"{segment_name.replace(' ', '_')}_Part",
                f"{segment_name}_Part",
            ]
            for name_variant in part_name_variations:
                part_obj = self.doc.getObject(name_variant)
                if part_obj:
                    break

            if part_obj:
                part_obj.Placement = adjusted_placement
                segment.base_placement = adjusted_placement
            else:
                logger.warning(f"Could not find Part object. Tried: {part_name_variations}")

        rotate_segment = segment_settings.get("rotate_segment", 0)
        if rotate_segment != 0 and part_obj:
            logger.info(f"Applying segment rotation: {rotate_segment}°")

            if len(segment.wafer_list) > 0 and segment.wafer_list[0].lcs1:
                entry_lcs = segment.wafer_list[0].lcs1
                entry_lcs_world = adjusted_placement.multiply(entry_lcs) if len(self.segment_list) > 0 else entry_lcs

                rotation_axis = entry_lcs_world.Rotation.multVec(App.Vector(0, 0, 1))
                rotation_point = entry_lcs_world.Base

                rotation = App.Rotation(rotation_axis, rotate_segment)

                current_placement = part_obj.Placement
                offset = current_placement.Base - rotation_point
                rotated_offset = rotation.multVec(offset)
                current_placement.Base = rotation_point + rotated_offset
                current_placement.Rotation = rotation.multiply(current_placement.Rotation)

                part_obj.Placement = current_placement
                segment.base_placement = current_placement

        self.segment_list.append(segment)
        self.current_placement = segment.get_end_placement()

        logger.info(f"✓ Created segment '{segment_name}' with {len(segment.wafer_list)} wafers")

    def _align_segment_to_previous(self, segment, prev_segment):
        if not prev_segment.wafer_list or not prev_segment.wafer_list[-1].lcs2:
            logger.warning("Previous segment has no exit LCS")
            return segment.base_placement

        prev_local_exit = prev_segment.wafer_list[-1].lcs2
        prev_world_exit = prev_segment.base_placement.multiply(prev_local_exit)

        if not segment.wafer_list or not segment.wafer_list[0].lcs1:
            logger.warning("Current segment has no entry LCS")
            return segment.base_placement

        curr_local_entry = segment.wafer_list[0].lcs1
        adjusted_placement = prev_world_exit.multiply(curr_local_entry.inverse())

        rotation_angle = segment.connection_spec.get("rotation_angle", 0)
        if rotation_angle != 0:
            logger.info(f"Applying Z-axis rotation: {rotation_angle}°")
            connection_z_axis = prev_world_exit.Rotation.multVec(App.Vector(0, 0, 1))
            connection_point = prev_world_exit.Base
            additional_rotation = App.Rotation(connection_z_axis, rotation_angle)

            offset = adjusted_placement.Base - connection_point
            rotated_offset = additional_rotation.multVec(offset)
            adjusted_placement.Base = connection_point + rotated_offset
            adjusted_placement.Rotation = additional_rotation.multiply(adjusted_placement.Rotation)

        # Verify alignment (warn only)
        curr_world_entry = adjusted_placement.multiply(curr_local_entry)
        pos_match = (curr_world_entry.Base - prev_world_exit.Base).Length
        rot_match = curr_world_entry.Rotation.isSame(prev_world_exit.Rotation, 1e-6)
        if not (pos_match < 1e-6 and rot_match):
            logger.warning(f"✗ LCS alignment verification failed: pos={pos_match}, rot={rot_match}")

        return adjusted_placement

    def _execute_set_position(self, operation):
        position = operation.get("position", [0, 0, 0])
        rotation = operation.get("rotation", [0, 0, 0])

        pos = App.Vector(position[0], position[1], position[2])
        rot = App.Rotation(rotation[0], rotation[1], rotation[2])
        self.current_placement = App.Placement(pos, rot)

    # -------------------------------------------------------------------------
    # Cutting list (SINGLE, lofted method, ALL segments)
    # -------------------------------------------------------------------------

    def _write_lofted_cut_list(self):
        """Generate ONE lofted-method cutting list file for all segments."""
        output_file = self.output_files.get("cuts_file")
        if not output_file:
            logger.warning("No cutting list file specified (output_files.cuts_file)")
            return

        if not os.path.isabs(output_file):
            base_dir = self.output_files.get("working_directory", "")
            output_file = os.path.join(base_dir, output_file)

        include_definitions = bool(self.global_settings.get("include_cut_list_definitions", False))

        with open(output_file, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("CUTTING LIST - LOFTED METHOD\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"PROJECT: {self.metadata.get('project_name', 'Unnamed Project')}\n")
            f.write(f"Total Segments: {len(self.segment_list)}\n")
            f.write("=" * 90 + "\n\n")

            for seg in self.segment_list:
                self._write_lofted_segment_block(f, seg)

            if include_definitions:
                self._write_cut_list_definitions(f)

        logger.info(f"✓ Cutting list written: {output_file}")

    def _write_lofted_segment_block(self, f, segment):
        """Write one segment section using lofted geometry definitions."""
        wafer_count = len(segment.wafer_list)

        total_volume = 0.0
        for w in segment.wafer_list:
            if getattr(w, "wafer", None) is not None and hasattr(w.wafer, "Volume"):
                total_volume += w.wafer.Volume

        # Calculate bounding box for the segment
        wafer_shapes = []
        for w in segment.wafer_list:
            if getattr(w, "wafer", None) is not None:
                wafer_shapes.append(w.wafer)

        if wafer_shapes:
            # Create compound of all wafers
            import Part
            compound = Part.makeCompound(wafer_shapes)
            bbox = compound.BoundBox

            # Get dimensions
            bbox_x = bbox.XLength
            bbox_y = bbox.YLength
            bbox_z = bbox.ZLength
            bbox_volume = bbox_x * bbox_y * bbox_z

            # Get center point
            bbox_center = (
                (bbox.XMin + bbox.XMax) / 2,
                (bbox.YMin + bbox.YMax) / 2,
                (bbox.ZMin + bbox.ZMax) / 2
            )
        else:
            bbox_x = bbox_y = bbox_z = bbox_volume = 0.0
            bbox_center = (0, 0, 0)

        # Now write to file (replace lines 896-899):
        f.write(f"SEGMENT: {segment.name}\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Wafer count: {wafer_count}\n")
        f.write(f"Total volume: {total_volume:.4f}\n")
        f.write(f"Bounding box (X × Y × Z): {bbox_x:.2f} × {bbox_y:.2f} × {bbox_z:.2f}\n")
        f.write(f"Bounding box volume: {bbox_volume:.4f}\n")
        f.write(f"Bounding box center: ({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f})\n")
        f.write(f"Packing efficiency: {(total_volume / bbox_volume * 100):.1f}%\n\n")
        f.write("-" * 90 + "\n\n")

        f.write("CUTTING INSTRUCTIONS:\n")
        f.write("  Blade° = Tilt saw blade from vertical\n")
        f.write("  Rot° = Incremental twist at this wafer\n")
        f.write("  Cylinder° = Absolute rotation to set before cutting\n")
        f.write("  Cumulative = Total cylinder length used (mark and cut)\n\n")

        f.write(f"{'Cut':<4} {'Length':<10} {'Blade°':<7} {'Rot°':<6} {'Cylinder°':<10} ")
        f.write(f"{'Collin°':<10} {'Azimuth°':<10} {'Cumulative':<12} {'Done':<6}\n")
        f.write("-" * 90 + "\n")

        cumulative_length = 0.0
        cumulative_rotation = 0.0

        for i, wafer in enumerate(segment.wafer_list):
            if getattr(wafer, "wafer", None) is None:
                continue

            geom = wafer.geometry or {}

            chord_length = float(geom.get("chord_length", 0.0))
            lift_angle = float(geom.get("lift_angle_deg", 0.0))
            blade_angle = lift_angle / 2.0
            rotation = float(geom.get("rotation_angle_deg", 0.0))
            collinearity = float(geom.get("collinearity_angle_deg", 0.0))
            azimuth = float(geom.get("chord_azimuth_deg", 0.0))

            if i == 0:
                cylinder_angle = 0.0
            else:
                cumulative_rotation += rotation
                if i % 2 == 1:
                    cylinder_angle = (cumulative_rotation + 180.0) % 360.0
                else:
                    cylinder_angle = cumulative_rotation % 360.0

            cumulative_length += chord_length

            length_str = self._format_fractional_inches(chord_length)
            cumul_str = self._format_fractional_inches(cumulative_length)

            f.write(
                f"{i + 1:<4} {length_str:<10} {blade_angle:<7.1f} {rotation:<6.0f} {cylinder_angle:<10.0f} "
                f"{collinearity:<10.4f} {azimuth:<10.2f} {cumul_str:<12} {'[ ]':<6}\n"
            )

        f.write("\n" + "-" * 90 + "\n")
        f.write(f"Total cylinder length required: {self._format_fractional_inches(cumulative_length)}\n")
        f.write(f"  ({cumulative_length:.3f} inches = {cumulative_length * 25.4:.1f} mm)\n\n")

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
        cuts_file.write("            Interior angle = 180° - Collin°\n")
        cuts_file.write("            Larger values indicate sharper curves\n\n")
        cuts_file.write("Azimuth°:   Compass direction of chord in XY plane, measured from +Y axis\n")
        cuts_file.write("            Negative = clockwise from North, Positive = counterclockwise\n\n")
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

    # -------------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------------

    def _format_fractional_inches(self, decimal_inches: float) -> str:
        """Convert decimal inches to fractional format like '1 3/16"' (nearest 1/16)."""
        whole = int(decimal_inches)
        frac = decimal_inches - whole

        sixteenths = round(frac * 16)

        if sixteenths == 0:
            return f'{whole}"' if whole > 0 else '0"'
        if sixteenths == 16:
            return f'{whole + 1}"'

        num = sixteenths
        den = 16
        while num % 2 == 0 and den % 2 == 0:
            num //= 2
            den //= 2

        if whole > 0:
            return f'{whole} {num}/{den}"'
        return f'{num}/{den}"'

    def _format_placement(self, placement):
        pos = placement.Base
        rot = placement.Rotation
        angles = rot.toEuler()
        return f"Pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}), Rot=({angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f})"
