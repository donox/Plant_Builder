"""Curves class for generating and manipulating 3D curves.

This module provides functionality to generate various mathematical curves,
apply transformations, and prepare them for use in wafer generation systems.
"""

from core.logging_setup import get_logger, set_display_levels
# set_display_levels(["ERROR", "WARNING", "INFO"])
logger = get_logger(__name__)

import math
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import FreeCAD
import Part


class Curves:
    """Generates and manipulates 3D curves for manufacturing applications.

    This class provides a library of mathematical curves with comprehensive
    transformation capabilities including rotation, translation, scaling,
    and curve segment selection.
    """

    def __init__(self, doc: Any, curve_spec: Dict[str, Any]):
        """Initialize the Curves class with curve specifications."""
        self.doc = doc
        self.curve_spec = curve_spec
        self.base_curve = None
        self.transformed_curve = None

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

        self._generate_base_curve()
        self._apply_transformations()

    def _generate_base_curve(self) -> None:
        """Generate the base curve from specifications."""
        curve_type = self.curve_spec.get('type')
        parameters = self.curve_spec.get('parameters', {})

        logger.debug(f"Generating base curve: {curve_type}")

        if curve_type == 'custom':
            custom_func = self.curve_spec.get('custom_function')
            if custom_func is None:
                raise ValueError("Custom curve type requires 'custom_function' parameter")
            self.base_curve = np.array(custom_func(**parameters))
            logger.debug(f"Generated custom curve with {len(self.base_curve)} points")
        elif curve_type in self.available_curves:
            self.base_curve = np.array(self.available_curves[curve_type](**parameters))
            logger.debug(f"Generated {curve_type} curve with {len(self.base_curve)} points")
        else:
            raise ValueError(f"Unknown curve type: {curve_type}. "
                             f"Available: {list(self.available_curves.keys())} or 'custom'")

    def _apply_transformations(self) -> None:
        """Apply all specified transformations to the base curve."""
        self.transformed_curve = self.base_curve.copy()

        #DEBUG
        logger.info((f"Curve spec: {self.curve_spec}"))
        segment_spec = self.curve_spec.get('segment')
        logger.info(f"Segment spec: {segment_spec}")
        if segment_spec:
            logger.info(f"Applying segment selection: {segment_spec}")
            self.transformed_curve = self._select_segment(self.transformed_curve, segment_spec)
            logger.info(f"Points after segment selection: {len(self.transformed_curve)}")

        segment_spec = self.curve_spec.get('segment')
        if segment_spec:
            self.transformed_curve = self._select_segment(self.transformed_curve, segment_spec)
            logger.debug(f"Applied segment selection: {len(self.transformed_curve)} points")

        transformations = self.curve_spec.get('transformations', [])
        for transform in transformations:
            operation = transform['operation']

            if operation == 'scale':
                self.transformed_curve = self._scale_curve(self.transformed_curve, transform)
                logger.debug(f"Applied scale transformation")
            elif operation == 'rotate':
                self.transformed_curve = self._rotate_curve(self.transformed_curve, transform)
                logger.debug(f"Applied rotation: {transform.get('axis')} {transform.get('angle')}°")
            elif operation == 'translate':
                self.transformed_curve = self._translate_curve(self.transformed_curve, transform)
                logger.debug(f"Applied translation: {transform.get('offset')}")
            elif operation == 'mirror':
                self.transformed_curve = self._mirror_curve(self.transformed_curve, transform)
                logger.debug(f"Applied mirror: {transform.get('plane')}")
            else:
                logger.warning(f"Unknown transformation operation: {operation}")

    def get_curve_points(self) -> np.ndarray:
        """Get the final transformed curve points."""
        return self.transformed_curve.copy()

    def get_curve_info(self) -> Dict[str, Any]:
        """Get information about the generated curve."""
        if self.transformed_curve is None:
            return {}

        points = self.transformed_curve
        diffs = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))

        info = {
            'num_points': len(points),
            'total_length': np.sum(lengths),
            'bounding_box': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            },
            'start_point': points[0].tolist(),
            'end_point': points[-1].tolist()
        }

        logger.debug(f"Curve info: {len(points)} points, length {info['total_length']:.3f}")
        return info

    def add_visualization_vertices(self, segment_obj=None, group_name: str = None) -> str:
        """Add visual vertices aligned with FlexSegment coordinate system."""
        if group_name is None:
            curve_type = self.curve_spec.get('type', 'curve')
            group_name = f"{curve_type}_vertices"

        existing_groups = self.doc.getObjectsByLabel(group_name)
        for group in existing_groups:
            self.doc.removeObject(group.Name)

        point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)
        point_group.Label = group_name

        vertices = []
        for i, point in enumerate(self.transformed_curve):
            vertex_name = f"{group_name}_point_{i}"
            vertex_obj = self.doc.addObject('Part::Vertex', vertex_name)
            vertex_obj.X = float(point[0])
            vertex_obj.Y = float(point[1])
            vertex_obj.Z = float(point[2])
            vertex_obj.Placement = FreeCAD.Placement()
            vertices.append(vertex_obj)

        point_group.addObjects(vertices)
        self.doc.recompute()

        logger.info(f"Added {len(vertices)} vertices to group '{group_name}'")
        return group_name

    # Transformation methods
    def _select_segment(self, curve: np.ndarray, segment_spec: Dict[str, float]) -> np.ndarray:
        """Select a portion of the curve."""
        start_frac = segment_spec.get('start_fraction', 0.0)
        end_frac = segment_spec.get('end_fraction', 1.0)

        n_points = len(curve)
        start_idx = int(start_frac * (n_points - 1))
        end_idx = int(end_frac * (n_points - 1)) + 1

        return curve[start_idx:end_idx]

    def _scale_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Scale the curve uniformly or per-axis."""
        if 'factor' in transform:
            factor = transform['factor']
            return curve * factor
        elif 'factors' in transform:
            factors = np.array(transform['factors'])
            return curve * factors
        else:
            raise ValueError("Scale transform requires 'factor' or 'factors' parameter")

    def _rotate_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Rotate the curve around a specified axis."""
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
        """Translate the curve by a specified offset."""
        offset = np.array(transform.get('offset', [0, 0, 0]))
        return curve + offset

    def _mirror_curve(self, curve: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Mirror the curve across a specified plane."""
        plane = transform.get('plane', 'xy').lower()
        mirrored = curve.copy()

        if plane == 'xy':
            mirrored[:, 2] *= -1
        elif plane == 'xz':
            mirrored[:, 1] *= -1
        elif plane == 'yz':
            mirrored[:, 0] *= -1
        else:
            raise ValueError(f"Invalid mirror plane: {plane}. Use 'xy', 'xz', or 'yz'")

        return mirrored

    # Built-in curve generation methods
    def _generate_linear(self, length: float = 100.0, points: int = 100,
                         direction: List[float] = [0, 0, 1]) -> List[List[float]]:
        """Generate a linear curve."""
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            point = t * length * direction
            curve_points.append(point.tolist())

        logger.debug(f"Generated linear curve: length={length}, points={points}")
        return curve_points

    def _generate_helical(self, radius: float = 10.0, pitch: float = 2.5,
                          turns: float = 4.0, points: int = 100,
                          start_at_origin: bool = True) -> List[List[float]]:
        """Generate a helical curve."""
        logger.debug(f"Generating helix: radius={radius}, pitch={pitch}, turns={turns}, points={points}")

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

        if start_at_origin and curve_points:
            offset = curve_points[0]
            curve_points = [[p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
                            for p in curve_points]
            logger.debug("Translated helix to start at origin")

        return curve_points

    def _trefoil_param(self, t, R, r, p, q, scale_z, cx, cy, cz):
        """Calculate trefoil parametric points."""
        cq, sq = np.cos(q * t), np.sin(q * t)
        cp, sp = np.cos(p * t), np.sin(p * t)
        x = (R + r * cq) * cp + cx
        y = (R + r * cq) * sp + cy
        z = (r * sq) * scale_z + cz
        return np.stack((x, y, z), axis=1).astype(float)

    def _moving_avg(self, P, win=0):
        """Apply moving average smoothing to points."""
        if win and win > 1:
            k = int(win)
            k += (k + 1) % 2
            w = np.ones(k, float) / k
            Ppad = np.vstack((P[-k // 2:], P, P[:k // 2]))
            S = np.vstack([np.convolve(Ppad[:, i], w, mode="valid") for i in range(3)]).T
            return S
        return P

    def _generate_trefoil(self, **parameters):
        """Generate a trefoil knot curve."""
        logger.debug(f"Generating trefoil with parameters: {parameters}")
        p = generate_woodcut_trefoil(
            slices=150,
            major_radius=8.0,
            tube_radius=2.5,
            smooth_factor=0.92,
            jitter=0.02,
            optimize_spacing=True)
        logger.debug(f"Generated trefoil with {len(p)} points")
        return p

    def _generate_sinusoidal(self, length: float = 50.0, amplitude: float = 5.0,
                             frequency: float = 2.0, points: int = 100,
                             axis: str = 'x') -> List[List[float]]:
        """Generate a sinusoidal curve."""
        logger.debug(f"Generating sinusoidal: length={length}, amplitude={amplitude}, freq={frequency}")

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
        """Generate an overhand knot curve."""
        logger.debug(f"Generating overhand knot: scale={scale}, points={points}")

        curve_points = []
        x0 = y0 = z0 = 0

        for t in range(points):
            angle = math.radians(t * increment) - math.pi
            x = (math.cos(angle) + 2 * math.cos(2 * angle)) * scale
            y = (math.sin(angle) - 2 * math.sin(2 * angle)) * scale
            z = (-math.sin(3 * angle)) * scale

            if t == 0:
                x0, y0, z0 = x, y, z

            curve_points.append([x - x0, y - y0, z - z0])

            if angle > 2 * math.pi:
                break

        return curve_points

    def _generate_circle(self, radius: float = 10.0, points: int = 100,
                         plane: str = 'xy') -> List[List[float]]:
        """Generate a circular curve."""
        logger.debug(f"Generating circle: radius={radius}, plane={plane}")

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

    def _generate_spiral(self, max_radius: float = 10.0, min_radius: float = 5.0,
                         max_height: float = 10.0, turns: float = 2.0,
                         points: int = 100, plane: str = 'xy') -> List[List[float]]:
        """Generate a spiral curve with radius changing based on angle, not height.

        Creates an Archimedean spiral that rises in Z as it spirals outward.
        The radius grows uniformly with angle (each turn adds constant width).
        """
        logger.debug(f"Generating spiral: min_r={min_radius}, max_r={max_radius}, turns={turns}, height={max_height}")

        curve_points = []
        total_angle = turns * 2 * math.pi

        for i in range(points):
            t = i / (points - 1)
            angle = t * total_angle

            # Radius grows linearly with angle (Archimedean spiral)
            angle_fraction = angle / total_angle
            radius = min_radius + angle_fraction * (max_radius - min_radius)

            # Height grows linearly with progress
            h = max_height * t

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
        """Generate a figure-eight curve."""
        logger.debug(f"Generating figure-eight: radius={radius}, plane={plane}")

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

    def validate_curve_sampling(self, min_height: float, max_chord: float) -> Dict[str, Any]:
        """Validate that curve has reasonable sampling for geometry calculations."""
        if len(self.transformed_curve) < 10:
            return {
                'status': 'insufficient_points',
                'current_points': len(self.transformed_curve),
                'message': 'Curve must have at least 10 points for geometry calculations',
                'recommended_points': 50,
            }

        diffs = np.diff(self.transformed_curve, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        max_segment_length = np.max(segment_lengths)

        curve_diameter = np.max(self.transformed_curve, axis=0) - np.min(self.transformed_curve, axis=0)
        max_reasonable_segment = np.max(curve_diameter) / 20

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


def generate_woodcut_trefoil(slices=180, **parameters):
    """Generate a trefoil curve optimized for wood cutting with cylindrical slices."""
    R = float(parameters.get("major_radius", 6.0))
    r = float(parameters.get("tube_radius", 2.0))
    p = int(parameters.get("p", 2))
    q = int(parameters.get("q", 3))
    cx, cy, cz = parameters.get("center", (0.0, 0.0, 0.0))
    phase_deg = float(parameters.get("phase_deg", 0.0))
    scale_z = float(parameters.get("scale_z", 1.0))

    n = min(slices, 200)

    t0 = np.deg2rad(phase_deg)
    t_base = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)

    jitter = parameters.get("jitter", 0.0)
    if jitter > 0:
        t_perturbation = jitter * (np.random.random(n) - 0.5) * (2 * np.pi / n)
        t = t_base + t_perturbation + t0
    else:
        t = t_base + t0

    cq = np.cos(q * t)
    sq = np.sin(q * t)
    cp = np.cos(p * t)
    sp = np.sin(p * t)

    smooth_factor = parameters.get("smooth_factor", 0.5)

    x = (R + r * cq * smooth_factor) * cp + cx
    y = (R + r * cq * smooth_factor) * sp + cy
    z = (r * sq * smooth_factor) * scale_z + cz

    pts = np.stack((x, y, z), axis=1).astype(float)

    if parameters.get("optimize_spacing", True):
        pts = _optimize_cutting_spacing(pts)

    new_pts = np.append(pts, [pts[0]], axis=0)
    logger.debug(f"Generated woodcut trefoil with {len(new_pts)} points")
    return new_pts


def _optimize_cutting_spacing(points):
    """Redistribute points to ensure more uniform spacing for cutting operations."""
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(distances)])

    total_length = cumulative_length[-1]
    target_lengths = np.linspace(0, total_length, len(points))

    x_interp = np.interp(target_lengths, cumulative_length, points[:, 0])
    y_interp = np.interp(target_lengths, cumulative_length, points[:, 1])
    z_interp = np.interp(target_lengths, cumulative_length, points[:, 2])

    return np.column_stack([x_interp, y_interp, z_interp])


def get_available_curves():
    """Return list of available curve types from the Curves class."""
    available = set()

    for name in dir(Curves):
        if name.startswith('_generate_') or name.startswith('generate_'):
            if name.startswith('_generate_'):
                curve_type = name.replace('_generate_', '')
            else:
                curve_type = name.replace('generate_', '')

            if not curve_type.startswith('_'):
                available.add(curve_type)

    return available


def generate_curve(curve_type, doc=None, curve_spec=None, **params):
    """Generate a curve of the specified type."""
    if doc is None:
        import FreeCAD as App
        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("TempDoc")

    if curve_spec is None:
        curve_spec = {'type': curve_type, 'parameters': params}

    curves_instance = Curves(doc, curve_spec)

    func_name = f'generate_{curve_type}'
    private_func_name = f'_generate_{curve_type}'

    if hasattr(curves_instance, func_name):
        func = getattr(curves_instance, func_name)
        return func(**params)
    elif hasattr(curves_instance, private_func_name):
        func = getattr(curves_instance, private_func_name)
        return func(**params)
    else:
        available = get_available_curves()
        raise ValueError(
            f"Curve type '{curve_type}' not available.\n"
            f"Available types: {sorted(available)}\n"
            f"Could not find method: {func_name} or {private_func_name}"
        )