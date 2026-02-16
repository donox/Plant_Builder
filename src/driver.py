"""
driver.py - Main workflow driver for PlantBuilder

Orchestrates the build workflow by:
- Loading configuration from YAML
- Managing FreeCAD document
- Executing operations in sequence
- Generating output files (cutting lists, placement lists)
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import FreeCAD as App

from config.loader import load_config
from core.logging_setup import get_logger
from core.wafer_settings import WaferSettings
from curves import Curves, compute_tangent_direction
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

            # Second pass: skip everything but close_curve / close_loop
            if workflow_mode == "second_pass" and operation_type not in ("close_loop", "close_curve"):
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
        elif operation_type in ("close_curve", "close_loop"):
            self._close_curve(operation)
        elif operation_type == "export_curve":
            self._execute_export_curve(operation)
        else:
            logger.warning(f"Unknown operation type: {operation_type}")

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def _combine_segment_curves(self, segments, points_per_segment):
        """Regenerate and combine curve points from multiple segments into world-coordinate list.

        Args:
            segments: List of LoftSegment instances (in order).
            points_per_segment: Number of points to sample per segment.

        Returns:
            List of App.Vector in world coordinates.
        """
        all_points_world: list[App.Vector] = []

        for idx, segment in enumerate(segments):
            if not segment.curve_spec:
                raise ValueError(
                    f"Segment '{segment.name}' has no curve_spec."
                )

            curve_spec = segment.curve_spec
            curve_type = curve_spec.get("type")

            if curve_type == "existing_curve":
                logger.info(
                    f"Skipping segment '{segment.name}' "
                    f"(type {curve_type}) for parametric sampling."
                )
                continue

            # Use the Curves class to generate the transformed curve points.
            curves_instance = Curves(self.doc, curve_spec)
            points_array = curves_instance.get_curve_points()

            if points_array is None or len(points_array) == 0:
                raise ValueError(
                    f"Curves returned no points for segment '{segment.name}'."
                )

            total_pts = len(points_array)
            if total_pts <= points_per_segment:
                indices = range(total_pts)
            else:
                indices = [
                    int(round(i * (total_pts - 1) / (points_per_segment - 1)))
                    for i in range(points_per_segment)
                ]

            segment_points_local: list[App.Vector] = []
            for i in indices:
                p = points_array[i]
                segment_points_local.append(App.Vector(float(p[0]), float(p[1]), float(p[2])))

            segment_points_world = segment.transform_curve_points(segment_points_local)

            # Avoid duplicating the join point between consecutive segments
            if all_points_world and segment_points_world:
                last_global = all_points_world[-1]
                first_new = segment_points_world[0]
                if (last_global.sub(first_new)).Length <= 1e-6:
                    segment_points_world = segment_points_world[1:]

            all_points_world.extend(segment_points_world)

            logger.info(
                f"Segment '{segment.name}' contributed "
                f"{len(segment_points_world)} world points "
                f"(source points: {total_pts})."
            )

        return all_points_world

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

        # 2) Combine all segment curves into world-coordinate points.
        all_points_world = self._combine_segment_curves(ordered_segments, points_per_segment)

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


    def _close_curve(self, operation):
        """
        Close an open multi-segment curve by generating a B-spline closing
        curve constrained by endpoint tangent directions, combining all segments
        into a single composite closed curve, and building wafers on it.

        The composite curve triggers the is_closed gap-closing logic in
        generate_all_wafers_by_slicing().
        """
        import Part

        if len(self.segment_list) < 1:
            logger.warning("Need at least 1 segment to close a curve")
            return

        first_segment = self.segment_list[0]
        last_segment = self.segment_list[-1]

        wafer_settings = WaferSettings.from_dict(operation.get("wafer_settings", {}) or {})
        segment_settings = operation.get("segment_settings", {}) or {}
        num_lcs_per_end = int(operation.get("num_lcs_per_end", 3))
        closing_points_count = int(operation.get("points", 50))
        max_closing_angle = float(operation.get("max_closing_angle", 90.0))
        sampling_cfg = operation.get("sampling", {})
        points_per_segment = int(sampling_cfg.get("per_segment", 100))
        curve_name = operation.get("name", "ClosedCurve")

        # ------------------------------------------------------------------
        # 1) Extract endpoint geometry from first and last segments
        # ------------------------------------------------------------------
        # Exit point P0 = last segment's last wafer's lcs2 (world coords)
        last_wafers = last_segment.wafer_list
        first_wafers = first_segment.wafer_list

        if not last_wafers or not first_wafers:
            raise ValueError("First and last segments must have wafers to close curve")

        P0 = last_segment.base_placement.multiply(last_wafers[-1].lcs2).Base
        P1 = first_segment.base_placement.multiply(first_wafers[0].lcs1).Base

        # Exit tangent T0: averaged direction from last few wafers' lcs2 positions
        num_exit = min(num_lcs_per_end, len(last_wafers))
        exit_lcs_list = []
        for i in range(num_exit):
            wafer_idx = -(num_exit - i)
            wafer = last_wafers[wafer_idx]
            if wafer.lcs2:
                exit_lcs_list.append(last_segment.base_placement.multiply(wafer.lcs2))

        # Entry tangent T1: averaged direction from first few wafers' lcs1 positions
        num_entry = min(num_lcs_per_end, len(first_wafers))
        entry_lcs_list = []
        for i in range(num_entry):
            wafer = first_wafers[i]
            if wafer.lcs1:
                entry_lcs_list.append(first_segment.base_placement.multiply(wafer.lcs1))

        if len(exit_lcs_list) < 1 or len(entry_lcs_list) < 1:
            raise ValueError("Need at least 1 LCS per end to compute tangent directions")

        T0 = compute_tangent_direction(exit_lcs_list, reverse=False)  # forward at exit (away from segment)
        T1 = compute_tangent_direction(entry_lcs_list, reverse=False)  # forward at entry (into first segment)

        logger.info(f"Closing curve: P0={P0}, P1={P1}")
        logger.info(f"  Exit tangent T0={T0}")
        logger.info(f"  Entry tangent T1={T1}")

        # ------------------------------------------------------------------
        # 2) Angular separation check
        # ------------------------------------------------------------------
        gap_vec = P1 - P0
        gap_distance = gap_vec.Length
        if gap_distance < 1e-6:
            logger.info("Endpoints already coincide, gap distance ~0")
            gap_dir = T0
        else:
            gap_dir = App.Vector(gap_vec)
            gap_dir.normalize()

        exit_turn_angle = math.degrees(T0.getAngle(gap_dir))
        entry_turn_angle = math.degrees(T1.getAngle(gap_dir))
        max_turn = max(exit_turn_angle, entry_turn_angle)

        logger.info(
            f"  Exit turn angle: {exit_turn_angle:.1f} deg, "
            f"Entry turn angle: {entry_turn_angle:.1f} deg"
        )

        if max_turn > max_closing_angle:
            logger.error(
                f"Closing angle too large: max turning angle = {max_turn:.1f} deg "
                f"(limit = {max_closing_angle:.1f} deg). "
                f"P0={P0}, P1={P1}, gap={gap_distance:.3f}. "
                f"Add/modify segments to bring endpoints into range."
            )
            return

        # ------------------------------------------------------------------
        # 3) Generate closing B-spline
        # ------------------------------------------------------------------
        # Scale tangent magnitudes proportional to gap distance for smooth shape
        tangent_scale = gap_distance / 3.0 if gap_distance > 1e-6 else 1.0
        T0_scaled = App.Vector(T0.x * tangent_scale, T0.y * tangent_scale, T0.z * tangent_scale)
        # T1 points forward into first segment = the direction the BSpline should arrive at P1
        T1_scaled = App.Vector(T1.x * tangent_scale, T1.y * tangent_scale, T1.z * tangent_scale)

        bspline = Part.BSplineCurve()
        bspline.interpolate(
            Points=[P0, P1],
            Tangents=[T0_scaled, T1_scaled],
            TangentFlags=[True, True],
        )

        # Sample the closing B-spline into N points
        closing_points = []
        for i in range(closing_points_count):
            u = i / max(1, closing_points_count - 1)
            param = bspline.FirstParameter + u * (bspline.LastParameter - bspline.FirstParameter)
            closing_points.append(bspline.value(param))

        closing_length = sum(
            closing_points[i].distanceToPoint(closing_points[i + 1])
            for i in range(len(closing_points) - 1)
        )
        logger.info(
            f"  Closing B-spline: {len(closing_points)} points, "
            f"length={closing_length:.3f}, gap={gap_distance:.3f}"
        )

        # ------------------------------------------------------------------
        # 4) Combine all segments + closing curve into composite closed BSpline
        # ------------------------------------------------------------------
        all_points_world = self._combine_segment_curves(self.segment_list, points_per_segment)

        if len(all_points_world) < 2:
            raise ValueError("Composite curve has fewer than 2 points from segments")

        # Append closing curve points (skip first if it duplicates last segment point)
        if closing_points:
            if (all_points_world[-1].sub(closing_points[0])).Length <= 1e-6:
                closing_points = closing_points[1:]
            all_points_world.extend(closing_points)

        # Remove duplicate endpoint if closing curve lands on the start point
        if len(all_points_world) > 2:
            if (all_points_world[-1].sub(all_points_world[0])).Length <= 1e-3:
                all_points_world = all_points_world[:-1]

        logger.info(f"  Composite curve: {len(all_points_world)} total points")

        # Build a periodic (closed) BSpline so tangent continuity is maintained
        # at the junction between the closing curve and the first segment.
        composite_bspline = Part.BSplineCurve()
        composite_bspline.interpolate(all_points_world, PeriodicFlag=True)
        composite_wire = Part.Wire([composite_bspline.toShape()])

        doc = self.doc
        if doc is None:
            raise ValueError("driver.doc is None; cannot create curve object.")

        # Remove existing object with same label
        for obj in doc.Objects:
            if obj.Label == curve_name:
                try:
                    doc.removeObject(obj.Name)
                except Exception:
                    pass

        curve_obj = doc.addObject("Part::Feature", "ClosedCurve")
        curve_obj.Label = curve_name
        curve_obj.Shape = composite_wire

        if hasattr(curve_obj, "ViewObject"):
            curve_obj.ViewObject.LineColor = (0.0, 1.0, 0.0)
            curve_obj.ViewObject.LineWidth = 3.0

        doc.recompute()

        # ------------------------------------------------------------------
        # 5) Build wafers on the composite closed curve
        # ------------------------------------------------------------------
        # Create a new LoftSegment using the existing_curve type to build
        # wafers from the composite wire.
        from loft_segment import LoftSegment

        composite_curve_spec = {
            "type": "existing_curve",
            "from_label": curve_name,
            "num_samples": len(all_points_world),
        }

        closing_segment = LoftSegment(
            doc=doc,
            name=f"{curve_name}_wafers",
            curve_spec=composite_curve_spec,
            wafer_settings=wafer_settings,
            segment_settings=segment_settings,
            base_placement=App.Placement(),
            connection_spec={},
        )

        closing_segment.generate_wafers()
        closing_segment.visualize(doc)

        self.segment_list.append(closing_segment)

        logger.info(
            f"close_curve: created '{curve_name}' with "
            f"{len(closing_segment.wafer_list)} wafers on composite closed curve"
        )

    def _reconstruct_segment_from_part(self, part_obj):
        from loft_segment import LoftSegment
        from wafer_loft import Wafer

        segment_name = part_obj.Label.replace("_Part", "")

        segment = LoftSegment(
            doc=self.doc,
            name=segment_name,
            curve_spec={"type": "reconstructed"},
            wafer_settings=WaferSettings(),
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
        wafer_settings = WaferSettings.from_dict(operation.get("wafer_settings", {}) or {})
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

        # Now write to file
        f.write(f"SEGMENT: {segment.name}\n")
        f.write("=" * 90 + "\n\n")

        # Write parameter settings
        self._write_segment_parameters(f, segment)

        f.write(f"Wafer count: {wafer_count}\n")
        f.write(f"Total volume: {total_volume:.4f}\n")
        f.write(f"Bounding box (X × Y × Z): {bbox_x:.2f} × {bbox_y:.2f} × {bbox_z:.2f}\n")
        f.write(f"Bounding box volume: {bbox_volume:.4f}\n")
        f.write(f"Bounding box center: ({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f})\n")
        f.write(f"Packing efficiency: {(total_volume / bbox_volume * 100):.1f}%\n\n")
        f.write("-" * 90 + "\n\n")

        # Compute initial_offset: angle from world +Z to first wafer's major
        # axis in the cross-section plane (saw-table coordinate system).
        initial_offset = 0.0
        for wafer in segment.wafer_list:
            if getattr(wafer, "wafer", None) is None:
                continue
            geom0 = wafer.geometry or {}
            chord_vec = geom0.get("chord_vector")
            ellipse1 = geom0.get("ellipse1")
            if chord_vec is not None and ellipse1 is not None:
                major_axis = ellipse1.get("major_axis_vector")
                if major_axis is not None:
                    chord_dir = App.Vector(chord_vec.x, chord_vec.y, chord_vec.z)
                    chord_dir.normalize()
                    # Project world +Z onto cross-section plane (perpendicular to chord)
                    z_up = App.Vector(0, 0, 1)
                    z_dot = z_up.dot(chord_dir)
                    up_proj = z_up - chord_dir * z_dot
                    if up_proj.Length < 1e-6:
                        # Chord is near-vertical, fall back to +X as "up"
                        x_right = App.Vector(1, 0, 0)
                        x_dot = x_right.dot(chord_dir)
                        up_proj = x_right - chord_dir * x_dot
                    up_proj.normalize()
                    # "right" = chord × up (clockwise from up looking along chord)
                    right_proj = chord_dir.cross(up_proj)
                    right_proj.normalize()
                    # Project major axis onto cross-section
                    ma = App.Vector(major_axis.x, major_axis.y, major_axis.z)
                    ma_dot = ma.dot(chord_dir)
                    major_proj = ma - chord_dir * ma_dot
                    if major_proj.Length > 1e-6:
                        major_proj.normalize()
                        initial_offset = math.degrees(
                            math.atan2(major_proj.dot(right_proj), major_proj.dot(up_proj))
                        )
            break  # only need the first valid wafer

        logger.info("Cylinder° initial_offset from +Z to first major axis: %.1f°", initial_offset)

        f.write("CUTTING INSTRUCTIONS:\n")
        f.write("  Blade° = Tilt saw blade from vertical\n")
        f.write("  Rot° = Incremental twist at this wafer\n")
        f.write("  Cylinder° = Absolute rotation from top (0°=12 o'clock, CW looking toward blade)\n")
        f.write(f"  Cylinder° initial offset: {initial_offset:.1f}°\n")
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
                cylinder_angle = (cylinder_angle + initial_offset) % 360.0

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

    def _write_segment_parameters(self, f, segment):
        """Write curve and wafer parameter settings for a segment."""
        # Curve spec
        curve_spec = getattr(segment, "curve_spec", None) or {}
        curve_type = curve_spec.get("type", "unknown")
        curve_params = curve_spec.get("parameters", {})

        f.write(f"Curve type: {curve_type}\n")
        if curve_params:
            f.write("Curve parameters:\n")
            for key, value in curve_params.items():
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Wafer settings
        ws = getattr(segment, "wafer_settings", None)
        if ws is not None:
            f.write("Wafer settings:\n")
            f.write(f"  cylinder_diameter: {ws.cylinder_diameter}\n")
            f.write(f"  max_chord: {ws.max_chord}\n")
            f.write(f"  min_height: {ws.min_height}\n")
            f.write(f"  min_inner_chord: {ws.min_inner_chord}\n")
            f.write(f"  profile_density: {ws.profile_density}\n")
            if ws.max_wafer_count is not None:
                f.write(f"  max_wafer_count: {ws.max_wafer_count}\n")
        f.write("\n")

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
        cuts_file.write("            Saw-table coords: 0°=top (12 o'clock), CW looking toward blade\n")
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
