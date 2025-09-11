"""Curves class for generating and manipulating 3D curves.

This module provides functionality to generate various mathematical curves,
apply transformations, and prepare them for use in wafer generation systems.
"""

from core.logging_setup import get_logger, apply_display_levels
# apply_display_levels(["ERROR", "WARNING", "INFO", "COORD", "DEBUG"])
# apply_display_levels(["ERROR", "WARNING", "INFO"])
logger = get_logger(__name__)
import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import FreeCAD
import Part

class Curves:
    """Generates and manipulates 3D curves for manufacturing applications.

    This class provides a library of mathematical curves with comprehensive
    transformation capabilities including rotation, translation, scaling,
    and curve segment selection.

    Attributes:
        doc: FreeCAD document object for visualization
        curve_spec: Dictionary containing curve specifications
        base_curve: Original curve points before transformations
        transformed_curve: Final curve points after all transformations
        available_curves: Dictionary of built-in curve generation methods
    """

    def __init__(self, doc: Any, curve_spec: Dict[str, Any]):
        """Initialize the Curves class.

        Args:
            doc: FreeCAD document object
            curve_spec: Dictionary containing curve specifications with keys:
                - 'type': Curve type name or 'custom'
                - 'parameters': Dict of curve-specific parameters
                - 'transformations': List of transformation operations (optional)
                - 'segment': Dict specifying curve segment selection (optional)
                - 'custom_function': Function for custom curves (optional)

        Example curve_spec:
            {
                'type': 'helical',
                'parameters': {
                    'turns': 4,
                    'radius': 10,
                    'pitch': 2.5,
                    'points': 100
                },
                'transformations': [
                    {'operation': 'scale', 'factor': 2.0},
                    {'operation': 'rotate', 'axis': 'x', 'angle': 45},
                    {'operation': 'translate', 'offset': [0, 0, 10]}
                ],
                'segment': {
                    'start_fraction': 0.1,
                    'end_fraction': 0.9
                }
            }
        """
        self.doc = doc
        self.curve_spec = curve_spec
        self.base_curve = None
        self.transformed_curve = None

        # Available built-in curves
        self.available_curves = {
            'linear': self._generate_linear,
            'helical': self._generate_helical,
            'sinusoidal': self._generate_sinusoidal,
            'overhand_knot': self._generate_overhand_knot,
            'circle': self._generate_circle,
            'spiral': self._generate_spiral,
            'figure_eight': self._generate_figure_eight,
            'trefoil': self._generate_trefoil,
        }

        # Generate the curve
        self._generate_base_curve()
        self._apply_transformations()

    def _generate_base_curve(self) -> None:
        """Generate the base curve from specifications."""
        curve_type = self.curve_spec.get('type')
        parameters = self.curve_spec.get('parameters', {})

        if curve_type == 'custom':
            # Use custom function provided by user
            custom_func = self.curve_spec.get('custom_function')
            if custom_func is None:
                raise ValueError("Custom curve type requires 'custom_function' parameter")
            self.base_curve = np.array(custom_func(**parameters))
        elif curve_type in self.available_curves:
            # Use built-in curve
            self.base_curve = np.array(self.available_curves[curve_type](**parameters))
        else:
            raise ValueError(f"Unknown curve type: {curve_type}. "
                             f"Available: {list(self.available_curves.keys())} or 'custom'")

    def _apply_transformations(self) -> None:
        """Apply all specified transformations to the base curve."""
        self.transformed_curve = self.base_curve.copy()

        # Apply segment selection if specified
        segment_spec = self.curve_spec.get('segment')
        if segment_spec:
            self.transformed_curve = self._select_segment(self.transformed_curve, segment_spec)

        # Apply transformations in order
        transformations = self.curve_spec.get('transformations', [])
        for transform in transformations:
            operation = transform['operation']

            if operation == 'scale':
                self.transformed_curve = self._scale_curve(self.transformed_curve, transform)
            elif operation == 'rotate':
                self.transformed_curve = self._rotate_curve(self.transformed_curve, transform)
            elif operation == 'translate':
                self.transformed_curve = self._translate_curve(self.transformed_curve, transform)
            elif operation == 'mirror':
                self.transformed_curve = self._mirror_curve(self.transformed_curve, transform)
            else:
                logger.info(f"Warning: Unknown transformation operation: {operation}")

    def get_curve_points(self) -> np.ndarray:
        """Get the final transformed curve points.

        Returns:
            Numpy array of curve points with shape (n_points, 3)
        """
        return self.transformed_curve.copy()

    def get_curve_info(self) -> Dict[str, Any]:
        """Get information about the generated curve.

        Returns:
            Dictionary containing curve statistics and properties
        """
        if self.transformed_curve is None:
            return {}

        points = self.transformed_curve
        diffs = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))

        return {
            'num_points': len(points),
            'total_length': np.sum(lengths),
            'bounding_box': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            },
            'start_point': points[0].tolist(),
            'end_point': points[-1].tolist()
        }

    def add_visualization_vertices(self, segment_obj=None, group_name: str = None) -> str:
        """Add visual vertices aligned with FlexSegment coordinate system."""
        if group_name is None:
            curve_type = self.curve_spec.get('type', 'curve')
            group_name = f"{curve_type}_vertices"

        # Remove existing group to avoid conflicts
        existing_groups = self.doc.getObjectsByLabel(group_name)
        for group in existing_groups:
            self.doc.removeObject(group.Name)

        # Create new group with proper internal name
        point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)
        point_group.Label = group_name

        # Add vertices for each curve point
        vertices = []
        for i, point in enumerate(self.transformed_curve):
            vertex_name = f"{group_name}_point_{i}"
            vertex_obj = self.doc.addObject('Part::Vertex', vertex_name)
            vertex_obj.X = float(point[0])
            vertex_obj.Y = float(point[1])
            vertex_obj.Z = float(point[2])
            vertex_obj.Placement = FreeCAD.Placement()
            vertices.append(vertex_obj)

        # Add all vertices to the group
        point_group.addObjects(vertices)

        # Force recompute to ensure objects are created
        self.doc.recompute()

        logger.info(f"Added {len(vertices)} vertices to group '{group_name}'")
        return group_name

    # Transformation methods
    def _select_segment(self, curve: np.ndarray, segment_spec: Dict[str, float]) -> np.ndarray:
        """Select a portion of the curve.

        Args:
            curve: Input curve points
            segment_spec: Dict with 'start_fraction' and 'end_fraction' (0.0 to 1.0)

        Returns:
            Selected segment of the curve
        """
        start_frac = segment_spec.get('start_fraction', 0.0)
        end_frac = segment_spec.get('end_fraction', 1.0)

        n_points = len(curve)
        start_idx = int(start_frac * (n_points - 1))
        end_idx = int(end_frac * (n_points - 1)) + 1

        return curve[start_idx:end_idx]

    def _scale_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Scale the curve uniformly or per-axis.

        Args:
            curve: Input curve points
            transform: Dict with 'factor' (float) or 'factors' (list of 3 floats)

        Returns:
            Scaled curve points
        """
        if 'factor' in transform:
            # Uniform scaling
            factor = transform['factor']
            return curve * factor
        elif 'factors' in transform:
            # Per-axis scaling
            factors = np.array(transform['factors'])
            return curve * factors
        else:
            raise ValueError("Scale transform requires 'factor' or 'factors' parameter")

    def _rotate_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Rotate the curve around a specified axis.

        Args:
            curve: Input curve points
            transform: Dict with 'axis' ('x', 'y', 'z') and 'angle' (degrees)

        Returns:
            Rotated curve points
        """
        axis = transform.get('axis', 'z').lower()
        angle_deg = transform.get('angle', 0.0)
        angle_rad = math.radians(angle_deg)

        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Invalid rotation axis: {axis}. Use 'x', 'y', or 'z'")

        return np.dot(curve, rotation_matrix.T)

    def _translate_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Translate the curve by a specified offset.

        Args:
            curve: Input curve points
            transform: Dict with 'offset' (list of 3 floats)

        Returns:
            Translated curve points
        """
        offset = np.array(transform.get('offset', [0, 0, 0]))
        return curve + offset

    def _mirror_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Mirror the curve across a specified plane.

        Args:
            curve: Input curve points
            transform: Dict with 'plane' ('xy', 'xz', 'yz')

        Returns:
            Mirrored curve points
        """
        plane = transform.get('plane', 'xy').lower()
        mirrored = curve.copy()

        if plane == 'xy':
            mirrored[:, 2] *= -1  # Mirror across XY plane (flip Z)
        elif plane == 'xz':
            mirrored[:, 1] *= -1  # Mirror across XZ plane (flip Y)
        elif plane == 'yz':
            mirrored[:, 0] *= -1  # Mirror across YZ plane (flip X)
        else:
            raise ValueError(f"Invalid mirror plane: {plane}. Use 'xy', 'xz', or 'yz'")

        return mirrored

    # Built-in curve generation methods
    def _generate_linear(self, length: float = 100.0, points: int = 100,
                         direction: List[float] = [0, 0, 1]) -> List[List[float]]:
        """Generate a linear curve.

        Args:
            length: Length of the line
            points: Number of points to generate
            direction: Direction vector [x, y, z]

        Returns:
            List of [x, y, z] coordinate lists
        """
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)  # Normalize

        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            point = t * length * direction
            curve_points.append(point.tolist())

        return curve_points

    def _generate_helical(self, radius: float = 10.0, pitch: float = 2.5,
                          turns: float = 4.0, points: int = 100,
                          start_at_origin: bool = True) -> List[List[float]]:
        """Generate a helical curve.

        Args:
            radius: Radius of the helix
            pitch: Vertical distance per complete turn
            turns: Number of complete turns
            points: Number of points to generate
            start_at_origin: If True, start at [0, 0, 0] instead of [radius, 0, 0]

        Returns:
            List of [x, y, z] coordinate lists
        """
        logger.info(f"SPEC: Radius: {radius}, Pitch: {pitch}, Turns: {turns}, Points: {points}")
        curve_points = []
        total_angle = turns * 2 * math.pi
        total_height = turns * pitch

        for i in range(points):
            t = i / (points - 1)
            angle = t * total_angle
            z = t * total_height

            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            curve_points.append([x, y, z])

        # If starting at origin, translate the curve so first point is at [0, 0, 0]
        if start_at_origin and curve_points:
            offset = curve_points[0]
            curve_points = [[p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
                            for p in curve_points]

        return curve_points


    def _trefoil_param(self, t, R, r, p, q, scale_z, cx, cy, cz):
        cq, sq = np.cos(q * t), np.sin(q * t)
        cp, sp = np.cos(p * t), np.sin(p * t)
        x = (R + r * cq) * cp + cx
        y = (R + r * cq) * sp + cy
        z = (r * sq) * scale_z + cz
        return np.stack((x, y, z), axis=1).astype(float)

    def _moving_avg(self, P, win=0):
        if win and win > 1:
            k = int(win)
            k += (k + 1) % 2  # force odd
            w = np.ones(k, float) / k
            # pad circularly to preserve closure
            Ppad = np.vstack((P[-k // 2:], P, P[:k // 2]))
            S = np.vstack([np.convolve(Ppad[:, i], w, mode="valid") for i in range(3)]).T
            return S
        return P

    def _generate_trefoil(self, **parameters):
        p = generate_woodcut_trefoil(
            slices=150,  # Good balance of detail vs. cutting time
            major_radius=8.0,  # Larger for easier handling
            tube_radius=2.5,
            smooth_factor=0.92,  # Slightly smoother for easier cutting
            jitter=0.02,  # Small natural variation
            optimize_spacing=True)
        return p

    def _generate_sinusoidal(self, length: float = 50.0, amplitude: float = 5.0,
                             frequency: float = 2.0, points: int = 100,
                             axis: str = 'x') -> List[List[float]]:
        """Generate a sinusoidal curve.

        Args:
            length: Length along the primary axis
            amplitude: Amplitude of the sine wave
            frequency: Number of complete cycles
            points: Number of points to generate
            axis: Primary axis ('x', 'y', or 'z')

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            primary_coord = t * length
            sine_value = amplitude * math.sin(frequency * 2 * math.pi * t)

            if axis.lower() == 'x':
                curve_points.append([primary_coord, sine_value, 0])
            elif axis.lower() == 'y':
                curve_points.append([sine_value, primary_coord, 0])
            elif axis.lower() == 'z':
                curve_points.append([0, sine_value, primary_coord])
            else:
                raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")

        return curve_points

    def _generate_overhand_knot(self, scale: float = 1.0, points: int = 100,
                                increment: float = 1.0) -> List[List[float]]:
        """Generate an overhand knot curve.

        Args:
            scale: Scale factor for the knot
            points: Number of points to generate
            increment: Increment parameter for curve density

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        x0 = y0 = z0 = 0  # Will be set from first point

        for t in range(points):
            angle = math.radians(t * increment) - math.pi
            x = (math.cos(angle) + 2 * math.cos(2 * angle)) * scale
            y = (math.sin(angle) - 2 * math.sin(2 * angle)) * scale
            z = (-math.sin(3 * angle)) * scale

            if t == 0:  # Set origin of knot to global origin
                x0, y0, z0 = x, y, z

            curve_points.append([x - x0, y - y0, z - z0])

            # Stop when we complete the knot
            if angle > 2 * math.pi:
                break

        return curve_points

    def _generate_circle(self, radius: float = 10.0, points: int = 100,
                         plane: str = 'xy') -> List[List[float]]:
        """Generate a circular curve.

        Args:
            radius: Radius of the circle
            points: Number of points to generate
            plane: Plane of the circle ('xy', 'xz', 'yz')

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        for i in range(points):
            angle = i / (points - 1) * 2 * math.pi
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            if plane.lower() == 'xy':
                curve_points.append([radius * cos_a, radius * sin_a, 0])
            elif plane.lower() == 'xz':
                curve_points.append([radius * cos_a, 0, radius * sin_a])
            elif plane.lower() == 'yz':
                curve_points.append([0, radius * cos_a, radius * sin_a])
            else:
                raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")

        return curve_points

    def _generate_spiral(self, max_radius: float = 10.0, max_height: float = 10.0, turns: float = 3.0,
                         points: int = 100, plane: str = 'xy') -> List[List[float]]:
        """Generate a spiral curve.

        Args:
            max_radius: Maximum radius of the spiral
            max_height: Maximum height of the spiral
            turns: Number of complete turns
            points: Number of points to generate
            plane: Plane of the spiral ('xy', 'xz', 'yz')

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            h = max_height * t
            angle = t * turns * 2 * math.pi
            radius = t * max_radius
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            if plane.lower() == 'xy':
                curve_points.append([radius * cos_a, radius * sin_a, h])
            elif plane.lower() == 'xz':
                curve_points.append([radius * cos_a, h, radius * sin_a])
            elif plane.lower() == 'yz':
                curve_points.append([h, radius * cos_a, radius * sin_a])
            else:
                raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")

        return curve_points

    def _generate_figure_eight(self, radius: float = 10.0, points: int = 100,
                               plane: str = 'xy') -> List[List[float]]:
        """Generate a figure-eight curve.

        Args:
            radius: Radius of the figure-eight lobes
            points: Number of points to generate
            plane: Plane of the figure-eight ('xy', 'xz', 'yz')

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        for i in range(points):
            t = i / (points - 1) * 2 * math.pi
            x = radius * math.sin(t)
            y = radius * math.sin(t) * math.cos(t)

            if plane.lower() == 'xy':
                curve_points.append([x, y, 0])
            elif plane.lower() == 'xz':
                curve_points.append([x, 0, y])
            elif plane.lower() == 'yz':
                curve_points.append([0, x, y])
            else:
                raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")

        return curve_points

    def add_visualization_vertices_with_lcs(self, lcs_obj, group_name: str = None) -> str:
        """Add vertices using LCS coordinate system as reference."""
        return          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if group_name is None:
            curve_type = self.curve_spec.get('type', 'curve')
            group_name = f"{curve_type}_vertices_lcs"

        # Find or create the point group
        point_groups = self.doc.getObjectsByLabel(group_name)
        if point_groups:
            self.doc.removeObject(point_groups[0].Name)

        point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)

        # Add vertices using LCS placement
        vertices = []
        lcs_placement = lcs_obj.Placement

        for i, point in enumerate(self.transformed_curve):
            # Transform point using LCS coordinate system
            local_vector = FreeCAD.Vector(*point)
            global_position = lcs_placement.multVec(local_vector)

            vertex_shape = Part.Vertex(global_position)
            vertex_obj = self.doc.addObject("Part::Feature", f"{group_name}_point_{i}")
            vertex_obj.Shape = vertex_shape

            vertices.append(vertex_obj)

        point_group.addObjects(vertices)
        logger.info(f"Added {len(vertices)} vertices aligned with LCS to group '{group_name}'")

        return group_name

    def add_curve_visualization(self, group_name: str = None) -> str:
        """Add visual vertices along the curve for debugging/visualization."""
        if group_name is None:
            segment_name = self.segment.get_segment_name()
            group_name = f"{segment_name}_curve_vertices"

        # Create vertices at origin (same as wafer creation)
        vertex_group_name = self.curves.add_visualization_vertices(None, group_name)

        # Register with segment using FreeCAD properties AND grouping
        self.segment.register_curve_vertices_group(vertex_group_name)

        return vertex_group_name

    def validate_curve_sampling(self, min_height: float, max_chord: float) -> Dict[str, Any]:
        """Validate that curve has reasonable sampling for geometry calculations."""

        if len(self.transformed_curve) < 10:
            return {
                'status': 'insufficient_points',
                'current_points': len(self.transformed_curve),
                'message': 'Curve must have at least 10 points for geometry calculations',
                'recommended_points': 50,
                'current_points': len(self.transformed_curve),
            }

        # Check if curve points are so sparse that we miss geometric features
        # This should be based on curve complexity, not wafer constraints
        diffs = np.diff(self.transformed_curve, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        max_segment_length = np.max(segment_lengths)

        # Only fail if curve segments are ridiculously large
        curve_diameter = np.max(self.transformed_curve, axis=0) - np.min(self.transformed_curve, axis=0)
        max_reasonable_segment = np.max(
            curve_diameter) / 20  # Curve should have at least 20 segments across its diameter

        if max_segment_length > max_reasonable_segment:
            recommended_points = int(len(self.transformed_curve) * max_segment_length / max_reasonable_segment)
            return {
                'status': 'insufficient_sampling',
                'current_points': len(self.transformed_curve),
                'recommended_points': recommended_points,
                'message': f'Curve segments too coarse for accurate geometry'
            }

        return {
            'status': 'adequate',
            'current_points': len(self.transformed_curve),
            'message': 'Curve sampling is adequate'
        }

    # In curves.py, add this method to the Curves class:

    def calculate_required_point_density(self, max_chord_error: float) -> Dict[str, Any]:
        """Analyze if current curve points provide adequate density for given chord error tolerance.

        This works with ANY curve points regardless of how they were generated.

        Args:
            max_chord_error: Maximum allowed deviation between chord and curve segment

        Returns:
            Dict with analysis results and recommendations
        """
        if len(self.transformed_curve) < 3:
            return {
                'status': 'insufficient_points',
                'current_points': len(self.transformed_curve),
                'recommended_points': 50,
                'message': 'Need at least 3 points for chord error analysis'
            }

        # Calculate chord errors for all existing segments
        max_observed_error = 0.0
        segment_errors = []

        for i in range(len(self.transformed_curve) - 1):
            start_point = self.transformed_curve[i]
            end_point = self.transformed_curve[i + 1]

            # For adjacent points, chord error is just the deviation from straight line
            # (This is a simplified version - for longer segments you'd need the full calculation)
            segment_length = np.linalg.norm(end_point - start_point)
            segment_errors.append(segment_length)
            max_observed_error = max(max_observed_error, segment_length)

        avg_segment_length = np.mean(segment_errors)

        # If average segment length is much larger than acceptable chord error,
        # we probably need more points
        if avg_segment_length > max_chord_error * 2:  # Rough heuristic
            # Estimate how many more points we'd need
            density_ratio = avg_segment_length / max_chord_error
            recommended_points = int(len(self.transformed_curve) * density_ratio)

            return {
                'status': 'insufficient_density',
                'current_points': len(self.transformed_curve),
                'recommended_points': min(recommended_points, 1000),  # Cap it
                'avg_segment_length': avg_segment_length,
                'max_chord_error_tolerance': max_chord_error,
                'message': f'Average segment length {avg_segment_length:.4f} exceeds tolerance'
            }

        return {
            'status': 'adequate_density',
            'current_points': len(self.transformed_curve),
            'avg_segment_length': avg_segment_length,
            'message': 'Current point density is adequate'
        }

    def calculate_optimal_points(self, max_chord_error: float = 0.2) -> int:
        """Calculate optimal point count based on curve complexity and chord error tolerance.

        Args:
            max_chord_error: Maximum allowed deviation between chord and curve

        Returns:
            Recommended number of points for this curve type
        """
        curve_type = self.curve_spec.get('type')
        parameters = self.curve_spec.get('parameters', {})

        if curve_type == 'helical':
            return self._calculate_helical_points(parameters, max_chord_error)
        elif curve_type == 'circle':
            return self._calculate_circular_points(parameters, max_chord_error)
        elif curve_type == 'linear':
            return max(10, parameters.get('points', 50))  # Linear curves are simple
        else:
            # Generic calculation for other curve types
            return self._calculate_generic_points(parameters, max_chord_error)

    def _calculate_helical_points(self, params: Dict[str, Any], max_chord_error: float) -> int:
        """Calculate optimal points for helical curves."""
        radius = params.get('radius', 10.0)
        pitch = params.get('pitch', 2.5)
        turns = params.get('turns', 1.0)

        # Calculate helical curve properties
        circumference = 2 * math.pi * radius
        total_arc_length = turns * math.sqrt(circumference ** 2 + pitch ** 2)

        def chord_error_for_segment_length(segment_length):
            if segment_length <= 0:
                return 0.0

            half_length = segment_length / 2
            if half_length >= radius:
                return float('inf')

            circular_chord_error = radius - math.sqrt(radius ** 2 - half_length ** 2)
            helix_factor = 1.0 + (pitch / circumference) ** 2
            return circular_chord_error * helix_factor

        min_segments = 20
        max_segments = 1000

        for num_segments in range(min_segments, max_segments):
            segment_length = total_arc_length / num_segments
            chord_error = chord_error_for_segment_length(segment_length)
            if chord_error <= max_chord_error:
                return num_segments
        return max_segments

    # ... add similar methods for other curve types

    @classmethod
    def get_available_curves(cls) -> List[str]:
        """Get list of available built-in curve types.

        Returns:
            List of curve type names
        """
        return ['linear', 'helical', 'sinusoidal', 'overhand_knot',
                'circle', 'spiral', 'figure_eight', 'trefoil', 'custom']


def generate_woodcut_trefoil(slices=180, **parameters):
    """
    Generate a trefoil curve optimized for wood cutting with cylindrical slices.

    Parameters:
        slices (int): Number of cutting slices (affects curve resolution)
        **parameters: Same parameters as original trefoil function

    Returns:
        np.ndarray of shape (slices, 3): Points optimized for woodcutting
    """
    R = float(parameters.get("major_radius", 6.0))
    r = float(parameters.get("tube_radius", 2.0))
    p = int(parameters.get("p", 2))
    q = int(parameters.get("q", 3))
    cx, cy, cz = parameters.get("center", (0.0, 0.0, 0.0))
    phase_deg = float(parameters.get("phase_deg", 0.0))
    scale_z = float(parameters.get("scale_z", 1.0))

    # Ensure we don't exceed 200 points
    n = min(slices, 200)

    # Generate parameter values with strategic spacing
    # Add slight randomization to avoid perfect regularity that can cause grain issues
    t0 = np.deg2rad(phase_deg)
    t_base = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)

    # Add small perturbations for more natural cutting (optional)
    jitter = parameters.get("jitter", 0.0)  # 0.01-0.03 recommended for natural variation
    if jitter > 0:
        t_perturbation = jitter * (np.random.random(n) - 0.5) * (2 * np.pi / n)
        t = t_base + t_perturbation + t0
    else:
        t = t_base + t0

    # Trefoil parametric equations with slight smoothing
    # Use slightly higher precision for manufacturing
    cq = np.cos(q * t)
    sq = np.sin(q * t)
    cp = np.cos(p * t)
    sp = np.sin(p * t)

    # Apply optional smoothing factor for easier cutting
    smooth_factor = parameters.get("smooth_factor", 0.5)  # 0.9-0.95 for gentler curves

    x = (R + r * cq * smooth_factor) * cp + cx
    y = (R + r * cq * smooth_factor) * sp + cy
    z = (r * sq * smooth_factor) * scale_z + cz

    pts = np.stack((x, y, z), axis=1).astype(float)

    # Optional: ensure points are well-distributed for cutting
    if parameters.get("optimize_spacing", True):
        pts = _optimize_cutting_spacing(pts)
    new_pts = np.append(pts, [pts[0]], axis=0)
    return new_pts


def _optimize_cutting_spacing(points):
    """
    Redistribute points to ensure more uniform spacing for cutting operations.
    """
    # Calculate cumulative arc length
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(distances)])

    # Create evenly spaced arc length samples
    total_length = cumulative_length[-1]
    target_lengths = np.linspace(0, total_length, len(points))

    # Interpolate to get evenly spaced points
    x_interp = np.interp(target_lengths, cumulative_length, points[:, 0])
    y_interp = np.interp(target_lengths, cumulative_length, points[:, 1])
    z_interp = np.interp(target_lengths, cumulative_length, points[:, 2])

    return np.column_stack([x_interp, y_interp, z_interp])


def analyze_cutting_requirements(points):
    """
    Analyze the curve to provide cutting guidance.
    """
    # Calculate chord information for each slice
    center = np.mean(points, axis=0)
    radii = np.sqrt(np.sum((points - center) ** 2, axis=1))

    # Z-height variation (important for stacking slices)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_range = z_max - z_min

    # Maximum radius for cylinder stock
    max_radius = np.max(radii)

    # Angle between consecutive cuts
    n_points = len(points)
    angle_per_slice = 360.0 / n_points

    print(f"Cutting Analysis:")
    print(f"  Number of slices: {n_points}")
    print(f"  Cylinder diameter needed: {2 * max_radius:.2f}")
    print(f"  Height range: {z_range:.2f} (from {z_min:.2f} to {z_max:.2f})")
    print(f"  Angle per slice: {angle_per_slice:.2f}°")
    print(f"  Average radius: {np.mean(radii):.2f}")

    return {
        'n_slices': n_points,
        'cylinder_diameter': 2 * max_radius,
        'height_range': z_range,
        'angle_per_slice': angle_per_slice,
        'radii': radii
    }


# Example usage for woodcutting
if __name__ == "__main__":
    # Generate curve optimized for wood cutting
    trefoil_points = generate_woodcut_trefoil(
        slices=150,  # Good balance of detail vs. cutting time
        major_radius=8.0,  # Larger for easier handling
        tube_radius=2.5,
        smooth_factor=0.92,  # Slightly smoother for easier cutting
        jitter=0.02,  # Small natural variation
        optimize_spacing=True
    )

    # Analyze cutting requirements
    cutting_info = analyze_cutting_requirements(trefoil_points)

    # Optional: Save points for CAD/CAM software
    # np.savetxt('trefoil_cutting_points.csv', trefoil_points,
    #           delimiter=',', header='X,Y,Z', comments='')

    print(f"\nGenerated {len(trefoil_points)} points for cutting")