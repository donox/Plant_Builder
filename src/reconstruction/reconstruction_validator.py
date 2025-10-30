"""Validation tools for comparing reconstructed geometry with original."""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

try:
    from core.logging_setup import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from geometry validation."""
    max_deviation: float
    rms_deviation: float
    total_rotation_error: float
    per_wafer_deviations: List[Dict[str, float]]
    passed: bool
    issues: List[str]

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return (f"ValidationReport({status}: "
                f"max_dev={self.max_deviation:.3f}, "
                f"rms_dev={self.rms_deviation:.3f}, "
                f"{len(self.issues)} issues)")


class ReconstructionValidator:
    """Validate reconstructed wafers against original curve."""

    def __init__(self, tolerance: float = 0.1):
        """Initialize validator.

        Args:
            tolerance: Maximum acceptable deviation in model units
        """
        self.tolerance = tolerance

    def compare_paths(self,
                     original_curve_points: np.ndarray,
                     reconstructed_lcs_chain: List[Any]
                    ) -> ValidationReport:
        """Compare original curve with reconstructed centerline.

        Args:
            original_curve_points: Original curve points (N x 3 array)
            reconstructed_lcs_chain: LCS chain from reconstruction

        Returns:
            ValidationReport with comparison results
        """
        logger.info("Comparing original curve with reconstructed path...")

        # Convert LCS chain to points
        reconstructed_points = np.array([
            [lcs.Placement.Base.x, lcs.Placement.Base.y, lcs.Placement.Base.z]
            for lcs in reconstructed_lcs_chain
        ])

        # Calculate deviations
        deviations = []
        per_wafer = []

        for i, recon_pt in enumerate(reconstructed_points):
            # Find closest point on original curve
            distances = np.sqrt(np.sum((original_curve_points - recon_pt) ** 2, axis=1))
            min_dist = np.min(distances)
            closest_idx = np.argmin(distances)

            deviations.append(min_dist)
            per_wafer.append({
                'wafer_num': i,
                'deviation': min_dist,
                'reconstructed': recon_pt.tolist(),
                'closest_original_idx': int(closest_idx),
                'closest_original': original_curve_points[closest_idx].tolist()
            })

        max_dev = np.max(deviations)
        rms_dev = np.sqrt(np.mean(np.array(deviations) ** 2))

        # Check if passed
        passed = max_dev <= self.tolerance
        issues = []

        if not passed:
            issues.append(f"Maximum deviation {max_dev:.3f} exceeds tolerance {self.tolerance:.3f}")

        # Log results
        logger.info(f"  Max deviation: {max_dev:.3f}")
        logger.info(f"  RMS deviation: {rms_dev:.3f}")
        logger.info(f"  Tolerance: {self.tolerance:.3f}")
        logger.info(f"  Status: {'PASSED' if passed else 'FAILED'}")

        return ValidationReport(
            max_deviation=max_dev,
            rms_deviation=rms_dev,
            total_rotation_error=0.0,
            per_wafer_deviations=per_wafer,
            passed=passed,
            issues=issues
        )