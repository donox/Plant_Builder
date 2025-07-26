"""Curves class for generating and manipulating 3D curves.

This module provides functionality to generate various mathematical curves,
apply transformations, and prepare them for use in wafer generation systems.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import FreeCAD


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
            'figure_eight': self._generate_figure_eight
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
                print(f"Warning: Unknown transformation operation: {operation}")

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

    def add_visualization_vertices(self, group_name: str = None,
                                   base_placement: Any = None) -> str:
        """Add visual vertices at each curve point in FreeCAD.

        Args:
            group_name: Name of the document group (auto-generated if None)
            segment_placement: FlexSegment placement to align coordinates

        Returns:
            Name of the created group
        """
        if group_name is None:
            curve_type = self.curve_spec.get('type', 'curve')
            group_name = f"{curve_type}_vertices"

        # Find or create the point group
        point_groups = self.doc.getObjectsByLabel(group_name)
        if point_groups:
            # Remove existing group to recreate
            self.doc.removeObject(point_groups[0].Name)

        point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)

        # Add vertices for each curve point
        vertices = []
        for i, point in enumerate(self.transformed_curve):
            vertex = self.doc.addObject('Part::Vertex', f"{group_name}_point_{i}")

            if base_placement:
                # Transform curve point to match segment coordinate system
                curve_vector = FreeCAD.Vector(point[0], point[1], point[2])
                transformed_point = base_placement.multiply(FreeCAD.Placement(curve_vector, FreeCAD.Rotation()))
                vertex.X = float(transformed_point.Base.x)
                vertex.Y = float(transformed_point.Base.y)
                vertex.Z = float(transformed_point.Base.z)
            else:
                # Use raw coordinates
                vertex.X = float(point[0])
                vertex.Y = float(point[1])
                vertex.Z = float(point[2])

            vertices.append(vertex)

        # Add all vertices to the group
        point_group.addObjects(vertices)
        print(f"Added {len(vertices)} vertices to group '{group_name}'")

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
                          turns: float = 4.0, points: int = 100) -> List[List[float]]:
        """Generate a helical curve.

        Args:
            radius: Radius of the helix
            pitch: Vertical distance per complete turn
            turns: Number of complete turns
            points: Number of points to generate

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        total_angle = turns * 2 * math.pi  # Total angle to traverse
        total_height = turns * pitch  # Total height of helix

        for i in range(points):
            # Parameter goes from 0 to 1
            t = i / (points - 1)

            # Angle and height are proportional to t
            angle = t * total_angle
            z = t * total_height

            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            curve_points.append([x, y, z])

        return curve_points

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

    def _generate_spiral(self, max_radius: float = 10.0, turns: float = 3.0,
                         points: int = 100, plane: str = 'xy') -> List[List[float]]:
        """Generate a spiral curve.

        Args:
            max_radius: Maximum radius of the spiral
            turns: Number of complete turns
            points: Number of points to generate
            plane: Plane of the spiral ('xy', 'xz', 'yz')

        Returns:
            List of [x, y, z] coordinate lists
        """
        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            angle = t * turns * 2 * math.pi
            radius = t * max_radius
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

    @classmethod
    def get_available_curves(cls) -> List[str]:
        """Get list of available built-in curve types.

        Returns:
            List of curve type names
        """
        return ['linear', 'helical', 'sinusoidal', 'overhand_knot',
                'circle', 'spiral', 'figure_eight', 'custom']