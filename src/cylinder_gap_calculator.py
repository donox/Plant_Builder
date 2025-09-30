"""Gap calculation logic for adjacent cylinders in curved sequences."""

import math
import numpy as np
from typing import List, Tuple, Dict, Any
from core.logging_setup import get_logger
import Part
import FreeCAD

logger = get_logger(__name__)


class CylinderGapCalculator:
    """Handles gap calculations between adjacent cylinders in curved sequences."""

    def __init__(self, cylinder_radius: float):
        self.cylinder_radius = cylinder_radius
        self.cylinder_diameter = cylinder_radius * 2

    def calculate_extensions_for_sequence(self, wafer_data_list: List[Tuple]) -> List[Tuple[float, float]]:
        """Calculate extensions for all wafers in sequence."""
        extensions = []

        for i, wafer_data in enumerate(wafer_data_list):
            start_ext = self._calculate_start_extension(wafer_data, i, wafer_data_list)
            end_ext = self._calculate_end_extension(wafer_data, i, wafer_data_list)
            extensions.append((start_ext, end_ext))

        logger.debug(f"Calculated extensions for {len(extensions)} wafers")
        return extensions

    def _calculate_start_extension(self, current_wafer: Tuple, index: int, all_wafers: List[Tuple]) -> float:
        if index == 0:
            return self._minimal_extension()

        prev_wafer = all_wafers[index - 1]
        return self._calculate_interface_extension(prev_wafer, current_wafer, 'end_to_start')

    def _calculate_end_extension(self, current_wafer: Tuple, index: int, all_wafers: List[Tuple]) -> float:
        if index == len(all_wafers) - 1:
            return self._minimal_extension()

        next_wafer = all_wafers[index + 1]
        return self._calculate_interface_extension(current_wafer, next_wafer, 'start_to_end')

    def _calculate_interface_extension(self, wafer_a: Tuple, wafer_b: Tuple, interface_type: str) -> float:
        """Calculate extension needed at interface using bisecting plane geometry."""

        # Extract data from tuples
        start_a, end_a, start_angle_a, end_angle_a, rotation_a, type_a = wafer_a
        start_b, end_b, start_angle_b, end_angle_b, rotation_b, type_b = wafer_b

        # Calculate cylinder axes (chord directions)
        axis_a = (end_a - start_a) / np.linalg.norm(end_a - start_a)
        axis_b = (end_b - start_b) / np.linalg.norm(end_b - start_b)

        # Handle nearly parallel axes
        cross_product = np.cross(axis_a, axis_b)
        if np.linalg.norm(cross_product) < 1e-6:
            return self._minimal_extension()

        # Calculate bisecting direction in the chord plane
        bisector = axis_a + axis_b
        bisector_length = np.linalg.norm(bisector)
        if bisector_length < 1e-6:
            return self._minimal_extension()
        bisector = bisector / bisector_length

        # Get the relevant axis and cut angle for THIS specific extension calculation
        if interface_type == 'end_to_start':
            # Calculate extension for wafer_a's END
            axis = axis_a
            cut_angle = math.radians(end_angle_a)
            wafer_id = "wafer_a_end"
        else:  # 'start_to_end'
            # Calculate extension for wafer_b's START
            axis = axis_b  # FIXED: was axis_a
            cut_angle = math.radians(start_angle_b)  # FIXED: was start_angle_a
            wafer_id = "wafer_b_start"

        # Calculate angle between cylinder axis and bisector
        dot_product = np.clip(np.dot(axis, bisector), -1.0, 1.0)
        half_angle = math.acos(abs(dot_product))

        # Extension based on bisecting plane geometry
        if half_angle < 1e-6:
            extension = self.cylinder_radius * math.tan(abs(cut_angle))
        else:
            extension = self.cylinder_radius * math.tan(abs(cut_angle)) / math.cos(half_angle)

        # Cap at reasonable maximum
        max_extension = self.cylinder_diameter * 2
        extension = min(extension, max_extension)

        logger.debug(f"Bisecting plane extension ({wafer_id}): half_angle={math.degrees(half_angle):.2f}°, "
                     f"cut_angle={math.degrees(cut_angle):.2f}°, extension={extension:.4f}")

        return extension * 2

    def _minimal_extension(self) -> float:
        return 0.001

def validate_ellipse_lcs_alignment(wafer_object, lcs_object, end='start'):
    """Check if the ellipse created by cutting a cylinder aligns with its LCS."""
    # Get the cylinder shape
    cylinder_shape = wafer_object.wafer.Shape

    # Get LCS position and orientation
    lcs_pos = lcs_object.Placement.Base
    lcs_x_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(1, 0, 0))
    lcs_y_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(0, 1, 0))
    lcs_z_axis = lcs_object.Placement.Rotation.multVec(FreeCAD.Vector(0, 0, 1))

    try:
        # Create a thin disk at the LCS position
        disk_thickness = 0.01
        disk_radius = 5.0

        disk = Part.makeCylinder(disk_radius, disk_thickness,
                                 lcs_pos - lcs_z_axis * (disk_thickness / 2),
                                 lcs_z_axis)

        # Get the intersection
        common = cylinder_shape.common(disk)

        if not common or not common.Faces:
            return {'status': 'error', 'message': 'No intersection found', 'aligned': False}

        # Get the largest face
        ellipse_face = max(common.Faces, key=lambda f: f.Area)

        # Sample points on the face boundary using edges
        points = []
        n_samples = 72

        if ellipse_face.OuterWire and ellipse_face.OuterWire.Edges:
            for edge in ellipse_face.OuterWire.Edges:
                # Sample along each edge
                samples_per_edge = max(10, n_samples // len(ellipse_face.OuterWire.Edges))
                try:
                    param_range = edge.ParameterRange
                    for i in range(samples_per_edge):
                        t = param_range[0] + (i / (samples_per_edge - 1)) * (param_range[1] - param_range[0])
                        point = edge.valueAt(t)
                        points.append([point.x, point.y, point.z])
                except Exception as e:
                    logger.debug(f"Error sampling edge: {e}")
                    continue

        if len(points) < 10:
            return {'status': 'error', 'message': f'Insufficient points sampled: {len(points)}', 'aligned': False}

        points = np.array(points)

        # Calculate centroid
        centroid = points.mean(axis=0)
        centroid_vec = FreeCAD.Vector(centroid[0], centroid[1], centroid[2])

        # Project points onto LCS plane
        points_2d = []
        for p in points:
            vec = FreeCAD.Vector(p[0], p[1], p[2]) - lcs_pos
            x_comp = vec.dot(lcs_x_axis)
            y_comp = vec.dot(lcs_y_axis)
            points_2d.append([x_comp, y_comp])

        points_2d = np.array(points_2d)

        # PCA to find principal axes
        centered = points_2d - points_2d.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Major axis in 2D
        major_2d = eigenvectors[:, 0]

        # Convert to 3D
        major_axis_3d = lcs_x_axis * major_2d[0] + lcs_y_axis * major_2d[1]
        major_axis_3d.normalize()

        # Calculate alignment
        major_dot_x = abs(major_axis_3d.dot(lcs_x_axis))
        major_dot_y = abs(major_axis_3d.dot(lcs_y_axis))

        if major_dot_x > major_dot_y:
            aligned_with = 'X'
            alignment_angle = math.degrees(math.acos(min(1.0, major_dot_x)))
        else:
            aligned_with = 'Y'
            alignment_angle = math.degrees(math.acos(min(1.0, major_dot_y)))

        major_radius = math.sqrt(eigenvalues[0]) * 2
        minor_radius = math.sqrt(eigenvalues[1]) * 2
        center_error = (centroid_vec - lcs_pos).Length

        result = {
            'status': 'success',
            'ellipse_found': True,
            'major_radius': major_radius,
            'minor_radius': minor_radius,
            'center_error': center_error,
            'major_axis_aligned_with': aligned_with,
            'alignment_angle_deg': alignment_angle,
            'aligned': alignment_angle < 1.0,
            'num_points': len(points)
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': f'Error: {str(e)}',
            'aligned': False,
            'traceback': traceback.format_exc()
        }

    # Log results
    if result['status'] == 'success':
        logger.debug(f"Ellipse-LCS Alignment ({end}): {result['major_axis_aligned_with']}-axis, "
                     f"{result['alignment_angle_deg']:.2f}°, {'ALIGNED' if result['aligned'] else 'MISALIGNED'}")
        if not result['aligned']:
            logger.warning(f"Ellipse misaligned by {result['alignment_angle_deg']:.2f}°")
    else:
        logger.error(f"Alignment check failed ({end}): {result['message']}")

    return result
