"""Assemble reconstructed wafers into complete segments."""

from typing import List, Optional, Any
import FreeCAD
import numpy as np

try:
    from core.logging_setup import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .cut_list_parser import SegmentData, WaferSpec
from .wafer_reconstructor import WaferReconstructor


class AssembledSegment:
    """Container for assembled segment data."""

    def __init__(self, segment_name: str):
        self.segment_name = segment_name
        self.wafer_objects = []
        self.lcs_chain = []
        self.centerline_path = []
        self.fused_object = None

    def __repr__(self):
        return (f"AssembledSegment('{self.segment_name}', "
                f"{len(self.wafer_objects)} wafers)")


class SegmentAssembler:
    """Assemble wafers from cut list into complete segment."""

    def __init__(self, doc: Any):
        """Initialize assembler.

        Args:
            doc: FreeCAD document
        """
        self.doc = doc

    def assemble_segment(self,
                        segment_data: SegmentData,
                        cylinder_diameter: float,
                        starting_lcs: Optional[Any] = None,
                        fuse_wafers: bool = True
                       ) -> AssembledSegment:
        """Assemble all wafers in a segment.

        Args:
            segment_data: Parsed segment data from cut list
            cylinder_diameter: Cylinder diameter
            starting_lcs: Optional starting LCS (defaults to origin)
            fuse_wafers: Whether to fuse wafers into single object

        Returns:
            AssembledSegment with all geometry
        """
        logger.info(f"Assembling segment '{segment_data.segment_name}' "
                   f"with {len(segment_data.wafers)} wafers")

        result = AssembledSegment(segment_data.segment_name)
        reconstructor = WaferReconstructor(self.doc, cylinder_diameter)

        # Create starting LCS if not provided
        if starting_lcs is None:
            starting_lcs = self._create_origin_lcs(segment_data.segment_name)

        result.lcs_chain.append(starting_lcs)
        result.centerline_path.append(starting_lcs.Placement.Base)

        # Create each wafer sequentially
        current_lcs = starting_lcs
        prev_wafer_spec = None

        for i, wafer_spec in enumerate(segment_data.wafers):
            try:
                wafer_obj, end_lcs, _ = reconstructor.create_wafer_from_spec(
                    wafer_spec,
                    current_lcs,
                    prev_wafer_spec
                )

                result.wafer_objects.append(wafer_obj)
                result.lcs_chain.append(end_lcs)
                result.centerline_path.append(end_lcs.Placement.Base)

                current_lcs = end_lcs
                prev_wafer_spec = wafer_spec

            except Exception as e:
                logger.error(f"Error creating wafer {wafer_spec.wafer_num}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Fuse wafers if requested
        if fuse_wafers and len(result.wafer_objects) > 1:
            result.fused_object = self._fuse_wafers(
                result.wafer_objects,
                segment_data.segment_name
            )

        logger.info(f"Successfully assembled {len(result.wafer_objects)} wafers")

        return result

    def _create_origin_lcs(self, segment_name: str) -> Any:
        """Create LCS at origin for segment start.

        Args:
            segment_name: Name for the LCS

        Returns:
            LCS object at origin
        """
        lcs = self.doc.addObject('PartDesign::CoordinateSystem',
                                f'{segment_name}_origin_lcs')
        lcs.Placement = FreeCAD.Placement(
            FreeCAD.Vector(0, 0, 0),
            FreeCAD.Rotation(0, 0, 0)
        )
        lcs.Visibility = False

        return lcs

    def _fuse_wafers(self,
                    wafer_objects: List[Any],
                    segment_name: str) -> Any:
        """Fuse wafer objects into single shape.

        Args:
            wafer_objects: List of wafer Part objects
            segment_name: Name for fused object

        Returns:
            Fused Part object or None
        """
        logger.info(f"Fusing {len(wafer_objects)} wafers...")

        try:
            # Start with first wafer shape
            result_shape = wafer_objects[0].Shape.copy()

            # Fuse subsequent wafers
            for i, wafer_obj in enumerate(wafer_objects[1:], start=2):
                try:
                    result_shape = result_shape.fuse(wafer_obj.Shape)
                    logger.debug(f"  Fused wafer {i}/{len(wafer_objects)}")
                except Exception as e:
                    logger.warning(f"  Error fusing wafer {i}: {e}")
                    continue

            # Create result object
            fused = self.doc.addObject("Part::Feature",
                                      f"{segment_name}_reconstructed_fused")
            fused.Shape = result_shape
            fused.ViewObject.DisplayMode = "Shaded"

            logger.info("Fusion complete")
            return fused

        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return None

    def create_centerline_vertices(self,
                                   lcs_chain: List[Any],
                                   group_name: str = "reconstructed_centerline"
                                  ) -> Any:
        """Create vertex objects at each LCS position for visualization.

        Args:
            lcs_chain: List of LCS objects along centerline
            group_name: Name for vertex group

        Returns:
            Group object containing vertices
        """
        logger.info(f"Creating {len(lcs_chain)} centerline vertices")

        # Create group
        vertex_group = self.doc.addObject("App::DocumentObjectGroup", group_name)
        vertices = []

        # Create vertex at each LCS position
        for i, lcs in enumerate(lcs_chain):
            import Part
            vertex_shape = Part.Vertex(lcs.Placement.Base)
            vertex_obj = self.doc.addObject("Part::Feature", f"{group_name}_v{i}")
            vertex_obj.Shape = vertex_shape
            vertices.append(vertex_obj)

        # Add to group
        vertex_group.addObjects(vertices)

        logger.info(f"Created vertex group '{group_name}'")
        return vertex_group