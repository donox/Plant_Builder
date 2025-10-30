"""High-level workflow for cut list reconstruction and validation."""

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

try:
    from core.logging_setup import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

import FreeCAD

from reconstruction.cut_list_parser import CutListParser
from reconstruction.segment_assembler import SegmentAssembler, AssembledSegment
from reconstruction.reconstruction_validator import ReconstructionValidator, ValidationReport


class ReconstructionResult:
    """Container for reconstruction results."""

    def __init__(self):
        self.segments = []
        self.validation_report = None

    def __repr__(self):
        return f"ReconstructionResult({len(self.segments)} segments)"


class ReconstructionWorkflow:
    """Main workflow for cut list reconstruction and validation."""

    def __init__(self, doc: Any, units_per_inch: float = 25.4):
        """Initialize workflow.

        Args:
            doc: FreeCAD document
        """
        self.doc = doc
        self.parser = CutListParser()
        self.assembler = SegmentAssembler(doc)
        self.validator = ReconstructionValidator()

    def reconstruct_from_cutlist(self,
                                 cutlist_file: str,
                                 cylinder_diameter: float,
                                 create_vertices_only: bool = False,
                                 fuse_wafers: bool = True
                                 ) -> ReconstructionResult:
        """Reconstruct geometry from cutting list.

        Args:
            cutlist_file: Path to cutting list file
            cylinder_diameter: Cylinder diameter in model units
            create_vertices_only: If True, only create centerline vertices
            fuse_wafers: Whether to fuse wafers into single object

        Returns:
            ReconstructionResult with all segments
        """
        logger.info("=" * 80)
        logger.info("STARTING RECONSTRUCTION FROM CUT LIST")
        logger.info("=" * 80)
        logger.info(f"Cut list file: {cutlist_file}")
        logger.info(f"Cylinder diameter: {cylinder_diameter}")
        logger.info(f"Vertices only: {create_vertices_only}")

        # Parse cut list
        parsed_data = self.parser.parse_file(cutlist_file)

        result = ReconstructionResult()

        # Process each segment
        for segment_data in parsed_data['segments']:
            logger.info(f"\nProcessing segment: {segment_data.segment_name}")

            if create_vertices_only:
                # Create only centerline vertices
                assembled = self.assembler.assemble_segment(
                    segment_data,
                    cylinder_diameter,
                    fuse_wafers=False
                )

                # Create vertex visualization
                vertex_group = self.assembler.create_centerline_vertices(
                    assembled.lcs_chain,
                    f"{segment_data.segment_name}_reconstructed_vertices"
                )

                logger.info(f"Created {len(assembled.lcs_chain)} vertices")

            else:
                # Create full wafer geometry
                assembled = self.assembler.assemble_segment(
                    segment_data,
                    cylinder_diameter,
                    fuse_wafers=fuse_wafers
                )

            result.segments.append(assembled)

        logger.info("\n" + "=" * 80)
        logger.info("RECONSTRUCTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total segments processed: {len(result.segments)}")

        self.doc.recompute()

        return result

    def validate_against_original(self,
                                  cutlist_file: str,
                                  original_curve_spec: Dict[str, Any],
                                  cylinder_diameter: float,
                                  output_report: Optional[str] = None
                                  ) -> ValidationReport:
        """Full validation pipeline.

        Args:
            cutlist_file: Path to cutting list file
            original_curve_spec: Curve specification dictionary
            cylinder_diameter: Cylinder diameter
            output_report: Optional path for validation report

        Returns:
            ValidationReport
        """
        logger.info("=" * 80)
        logger.info("STARTING VALIDATION PIPELINE")
        logger.info("=" * 80)

        # Reconstruct from cut list
        reconstruction = self.reconstruct_from_cutlist(
            cutlist_file,
            cylinder_diameter,
            create_vertices_only=False,
            fuse_wafers=False
        )

        # Generate original curve
        from curves import Curves
        curves = Curves(self.doc, original_curve_spec)
        original_points = curves.get_curve_points()

        logger.info(f"\nOriginal curve: {len(original_points)} points")

        # Compare for first segment
        if reconstruction.segments:
            segment = reconstruction.segments[0]

            # Geometry comparison
            validation_report = self.validator.compare_paths(
                original_points,
                segment.lcs_chain
            )

            # Parse cut list for rotation comparison
            parsed_data = self.parser.parse_file(cutlist_file)
            segment_data = parsed_data['segments'][0]

            # Extract rotations from cut list
            cutlist_rotations = [
                w.rotation_angle for w in segment_data.wafers
                if w.rotation_angle is not None
            ]

            # Calculate expected rotations from curve
            # (This would need actual curve analysis - placeholder for now)
            expected_rotations = cutlist_rotations  # Placeholder

            rotation_report = self.validator.compare_rotations(
                expected_rotations,
                cutlist_rotations
            )

            # Update validation report with rotation data
            validation_report.total_rotation_error = rotation_report['total_error']

            # Generate report if requested
            if output_report:
                self.validator.generate_report(
                    validation_report,
                    rotation_report,
                    output_report
                )

            return validation_report
        else:
            logger.error("No segments reconstructed!")
            return ValidationReport(
                max_deviation=float('inf'),
                rms_deviation=float('inf'),
                total_rotation_error=float('inf'),
                per_wafer_deviations=[],
                passed=False,
                issues=["No segments reconstructed"]
            )

    def reconstruct_single_segment(self,
                                   segment_name: str,
                                   cutlist_file: str,
                                   cylinder_diameter: float
                                   ) -> Optional[AssembledSegment]:
        """Reconstruct a single named segment from cut list.

        Args:
            segment_name: Name of segment to reconstruct
            cutlist_file: Path to cutting list file
            cylinder_diameter: Cylinder diameter

        Returns:
            AssembledSegment or None if not found
        """
        parsed_data = self.parser.parse_file(cutlist_file)

        # Find the requested segment
        for segment_data in parsed_data['segments']:
            if segment_data.segment_name == segment_name:
                logger.info(f"Reconstructing segment: {segment_name}")

                assembled = self.assembler.assemble_segment(
                    segment_data,
                    cylinder_diameter,
                    fuse_wafers=True
                )

                self.doc.recompute()
                return assembled

        logger.error(f"Segment '{segment_name}' not found in cut list")
        return None

    def reconstruct_from_cutlist_with_start(self,
                                            cutlist_file: str,
                                            cylinder_diameter: float,
                                            create_vertices_only: bool = False,
                                            starting_lcs: Any = None
                                            ) -> ReconstructionResult:
        """Reconstruct with specified starting LCS."""
        logger.info("=" * 80)
        logger.info("STARTING RECONSTRUCTION FROM CUT LIST")
        logger.info("=" * 80)
        logger.info(f"Cut list file: {cutlist_file}")
        logger.info(f"Cylinder diameter: {cylinder_diameter}")
        logger.info(f"Vertices only: {create_vertices_only}")

        if starting_lcs:
            logger.info(f"Starting position: {starting_lcs.Placement.Base}")
            logger.info(f"Starting rotation: {starting_lcs.Placement.Rotation.toEuler()}")

        # Parse cut list
        parsed_data = self.parser.parse_file(cutlist_file)

        result = ReconstructionResult()

        # Process each segment
        for segment_data in parsed_data['segments']:
            logger.info(f"\nProcessing segment: {segment_data.segment_name}")

            if create_vertices_only:
                # Create only centerline vertices
                assembled = self.assembler.assemble_segment(
                    segment_data,
                    cylinder_diameter,
                    starting_lcs=starting_lcs,  # Pass the starting LCS
                    fuse_wafers=False
                )

                # Create vertex visualization
                vertex_group = self.assembler.create_centerline_vertices(
                    assembled.lcs_chain,
                    f"{segment_data.segment_name}_reconstructed_vertices"
                )

                logger.info(f"Created {len(assembled.lcs_chain)} vertices")

            else:
                # Create full wafer geometry
                assembled = self.assembler.assemble_segment(
                    segment_data,
                    cylinder_diameter,
                    starting_lcs=starting_lcs,  # Pass the starting LCS
                    fuse_wafers=True
                )

            result.segments.append(assembled)

        logger.info("\n" + "=" * 80)
        logger.info("RECONSTRUCTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total segments processed: {len(result.segments)}")

        self.doc.recompute()

        return result