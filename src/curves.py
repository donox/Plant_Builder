"""Curves class for generating and manipulating 3D curves.

This module provides functionality to generate various mathematical curves,
apply transformations, and prepare them for use in wafer generation systems.
"""

from core.logging_setup import get_logger, set_display_levels
# set_display_levels(["ERROR", "WARNING", "INFO"])
logger = get_logger(__name__)

import math
import numpy as np
from typing import List, Dict, Any, Optional
import FreeCAD as App
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
            'closing_curve': self._generate_closing_curve,
        }

        self._generate_base_curve()
        self._apply_transformations()

    def _generate_base_curve(self) -> None:
        """Generate the base curve from specifications."""
        curve_type = self.curve_spec.get('type')
        parameters = self.curve_spec.get('parameters', {})

        # logger.debug(f"Generating base curve: {curve_type}")

        if curve_type == 'custom':
            custom_func = self.curve_spec.get('custom_function')
            if custom_func is None:
                raise ValueError("Custom curve type requires 'custom_function' parameter")
            self.base_curve = np.array(custom_func(**parameters))
            # logger.debug(f"Generated custom curve with {len(self.base_curve)} points")
        elif curve_type in self.available_curves:
            self.base_curve = np.array(self.available_curves[curve_type](**parameters))
            # logger.debug(f"Generated {curve_type} curve with {len(self.base_curve)} points")
        else:
            raise ValueError(f"Unknown curve type: {curve_type}. "
                             f"Available: {list(self.available_curves.keys())} or 'custom'")

    def _apply_transformations(self) -> None:
        """Apply all specified transformations to the base curve."""
        self.transformed_curve = self.base_curve.copy()

        # Apply segment selection if specified
        segment_spec = self.curve_spec.get('segment')
        if segment_spec:
            logger.info(f"Applying segment selection: {segment_spec}")
            self.transformed_curve = self._select_segment(self.transformed_curve, segment_spec)
            # logger.debug(f"Applied segment selection: {len(self.transformed_curve)} points")

        # Apply other transformations
        transformations = self.curve_spec.get('transformations', [])
        for transform in transformations:
            operation = transform['operation']

            if operation == 'scale':
                self.transformed_curve = self._scale_curve(self.transformed_curve, transform)
                # logger.debug(f"Applied scale transformation")
            elif operation == 'rotate':
                self.transformed_curve = self._rotate_curve(self.transformed_curve, transform)
                # logger.debug(f"Applied rotation: {transform.get('axis')} {transform.get('angle')}°")
            elif operation == 'translate':
                self.transformed_curve = self._translate_curve(self.transformed_curve, transform)
                # logger.debug(f"Applied translation: {transform.get('offset')}")
            elif operation == 'mirror':
                self.transformed_curve = self._mirror_curve(self.transformed_curve, transform)
                # logger.debug(f"Applied mirror: {transform.get('plane')}")
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

        # logger.debug(f"Curve info: {len(points)} points, length {info['total_length']:.3f}")
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

        # logger.debug(f"Generated linear curve: length={length}, points={points}")
        return curve_points

    def _generate_helical(self, radius: float = 10.0, pitch: float = 2.5,
                          turns: float = 4.0, points: int = 100,
                          start_at_origin: bool = True) -> List[List[float]]:
        """Generate a helical curve."""
        # logger.debug(f"Generating helix: radius={radius}, pitch={pitch}, turns={turns}, points={points}")

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
            # logger.debug("Translated helix to start at origin")

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

        # Extract parameters with defaults
        slices = parameters.get('points', 150)
        major_radius = parameters.get('major_radius', 4.0)
        tube_radius = parameters.get('tube_radius', 2.5)
        smooth_factor = parameters.get('smooth_factor', 1.0)
        jitter = parameters.get('jitter', 0.02)
        optimize_spacing = parameters.get('optimize_spacing', True)

        # Pass through all parameters to generate_woodcut_trefoil
        p = generate_woodcut_trefoil(
            slices=slices,
            major_radius=major_radius,
            tube_radius=tube_radius,
            smooth_factor=smooth_factor,
            jitter=jitter,
            optimize_spacing=optimize_spacing,
            **{k: v for k, v in parameters.items()
               if k in ['p', 'q', 'center', 'phase_deg', 'scale_z']}
        )

        logger.debug(f"Generated trefoil with {len(p)} points")
        return p

    def _generate_sinusoidal(self, length=50.0, amplitude=5.0,
                             frequency=2.0, points=100,
                             axis='x', **kwargs) -> List[List[float]]:
        """
        Generate a sinusoidal curve.

        Args:
            length: Length along the primary axis
            amplitude: Wave amplitude
            frequency: Either a single frequency or a list of frequencies
                       (superposition of sine waves)
            points: Number of points to generate
            axis: Primary axis ('x', 'y', or 'z')
        """
        # Defensive casts for all numeric parameters
        length = float(length)
        amplitude = float(amplitude)
        points = int(points)

        # Normalize frequency to a list
        if isinstance(frequency, (int, float)):
            frequencies = [float(frequency)]
        else:
            frequencies = [float(f) for f in frequency]

        curve_points = []
        for i in range(points):
            t = i / (points - 1)
            primary_coord = t * length
            # Sum of sine waves for each frequency
            sine_value = amplitude * sum(
                math.sin(freq * 2 * math.pi * t) for freq in frequencies
            ) / len(frequencies)

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
        # logger.debug(f"Generating overhand knot: scale={scale}, points={points}")

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
        # logger.debug(f"Generating circle: radius={radius}, plane={plane}")

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
        # logger.debug(f"Generating spiral: min_r={min_radius}, max_r={max_radius}, turns={turns}, height={max_height}")

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
        # logger.debug(f"Generating figure-eight: radius={radius}, plane={plane}")

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

    def _generate_closing_curve(self, **parameters):
        """
        Generate closing curve using geometric construction from circular arcs.

        Supports both automatic generation and using user-edited curves.
        Delegates to named helpers for each construction phase.
        """
        import FreeCAD as App
        import math

        start_segment = parameters.get('start_segment')
        end_segment = parameters.get('end_segment')

        if start_segment is None or end_segment is None:
            raise ValueError("closing_curve requires start_segment and end_segment parameters")

        # Check for edited curve first
        use_edited_curve = parameters.get('use_edited_curve')
        if use_edited_curve:
            return self._load_edited_curve(
                use_edited_curve, end_segment, parameters.get('points', 50)
            )

        # Automatic geometric generation
        logger.debug("No edited curve specified, using automatic generation")

        geometry = self._compute_closing_geometry(
            start_segment, end_segment, parameters
        )

        P0_world = geometry['P0']
        P3_world = geometry['P3']
        exit_direction = geometry['exit_direction']
        entry_direction = geometry['entry_direction']
        num_points = geometry['num_points']
        construction_radius = geometry['construction_radius']
        min_radius = geometry['min_radius']
        gap_distance = geometry['gap_distance']

        # Direction from P0 toward P3
        to_target = P3_world - P0_world
        to_target.normalize()

        # Calculate turn angles
        exit_turn_angle = math.degrees(exit_direction.getAngle(to_target))
        entry_turn_angle = math.degrees((-to_target).getAngle(entry_direction))

        logger.debug(f"Exit turn needed: {exit_turn_angle:.1f}")
        logger.debug(f"Entry turn needed: {entry_turn_angle:.1f}")

        # Build the curve from geometric pieces
        curve_points_world = []

        # STEP 1: Exit arc
        exit_arc_points, P1_world, dir1_world = self._build_exit_arc(
            P0_world, exit_direction, to_target,
            construction_radius, num_points, exit_turn_angle
        )
        curve_points_world.extend(exit_arc_points)

        # STEP 2: Entry arc
        entry_arc_points, P2_world, dir2_world = self._build_entry_arc(
            P3_world, entry_direction, to_target,
            construction_radius, num_points, entry_turn_angle
        )

        # STEP 3: Bridge between exit and entry arcs
        bridge_points = self._build_bezier_bridge(
            P1_world, dir1_world, P2_world, dir2_world,
            min_radius, num_points, gap_distance
        )
        curve_points_world.extend(bridge_points)

        # Add entry arc (or just the end point)
        if entry_arc_points:
            curve_points_world.extend(entry_arc_points)
        else:
            curve_points_world.append(P3_world)

        # Log total curve length
        total_curve_length = sum(
            curve_points_world[i].distanceToPoint(curve_points_world[i + 1])
            for i in range(len(curve_points_world) - 1)
        )
        logger.info(
            f"Generated geometric closing curve: "
            f"length={total_curve_length:.3f}, points={len(curve_points_world)}"
        )

        # Create editable curve if requested
        if parameters.get('create_editable_curve', False):
            self._create_editable_closing_curve(
                P0_world, P3_world,
                exit_direction, entry_direction,
                exit_turn_angle, entry_turn_angle,
                exit_arc_points if exit_turn_angle > 5 else [],
                entry_arc_points,
                end_segment, start_segment, gap_distance
            )

        # Transform to local coordinates
        end_wafers = end_segment.wafer_list
        last_wafer = end_wafers[-1]
        last_lcs2_world = end_segment.base_placement.multiply(last_wafer.lcs2)
        placement_inv = last_lcs2_world.inverse()

        points_local = []
        for point_world in curve_points_world:
            point_local = placement_inv.multVec(point_world)
            points_local.append([point_local.x, point_local.y, point_local.z])

        logger.debug(f"Curve start (local): {points_local[0]}")
        logger.debug(f"Curve end (local): {points_local[-1]}")

        return np.array(points_local)

    def _load_edited_curve(self, curve_label, end_segment, num_samples):
        """Load and sample points from a user-edited curve object."""
        import FreeCAD as App

        logger.info(f"Using edited curve: {curve_label}")
        curve_obj = App.ActiveDocument.getObject(curve_label)

        if curve_obj is None:
            raise ValueError(f"Could not find curve object: {curve_label}")

        if not (hasattr(curve_obj, 'Shape') and hasattr(curve_obj.Shape, 'Edges')
                and len(curve_obj.Shape.Edges) > 0):
            raise ValueError(
                f"Curve object {curve_label} has no usable geometry (Shape.Edges)"
            )

        edge = curve_obj.Shape.Edges[0]
        curve_points_world = []

        for i in range(num_samples):
            u = i / max(1, num_samples - 1)
            param = edge.FirstParameter + u * (edge.LastParameter - edge.FirstParameter)
            curve_points_world.append(edge.valueAt(param))

        logger.info(f"Sampled {len(curve_points_world)} points from edited curve")

        # Transform to local coordinates
        end_wafers = end_segment.wafer_list
        last_wafer = end_wafers[-1]
        last_lcs2_world = end_segment.base_placement.multiply(last_wafer.lcs2)
        placement_inv = last_lcs2_world.inverse()

        points_local = []
        for point_world in curve_points_world:
            point_local = placement_inv.multVec(point_world)
            points_local.append([point_local.x, point_local.y, point_local.z])

        return np.array(points_local)

    def _compute_closing_geometry(self, start_segment, end_segment, parameters):
        """
        Compute endpoints, directions, and construction parameters for closing curve.

        Returns a dict with keys: P0, P3, exit_direction, entry_direction,
        num_points, construction_radius, min_radius, gap_distance.
        """
        import FreeCAD as App
        import math

        num_lcs = parameters.get('num_lcs_per_end', 3)
        num_points = parameters.get('points', 50)

        start_wafers = start_segment.wafer_list
        end_wafers = end_segment.wafer_list

        if len(start_wafers) < num_lcs or len(end_wafers) < num_lcs:
            logger.warning(f"Not enough wafers for {num_lcs} LCS per end, reducing")
            num_lcs = min(len(start_wafers), len(end_wafers), max(1, num_lcs))

        if len(start_wafers) < 1 or len(end_wafers) < 1:
            raise ValueError("Segments must have at least one wafer to create closing curve")

        # Extract LCS from each end
        end_lcs_list = []
        for i in range(num_lcs):
            wafer_idx = -(num_lcs - i)
            wafer = end_wafers[wafer_idx]
            if wafer.lcs2:
                lcs_world = end_segment.base_placement.multiply(wafer.lcs2)
                end_lcs_list.append(lcs_world)

        start_lcs_list = []
        for i in range(num_lcs):
            wafer = start_wafers[i]
            if wafer.lcs1:
                lcs_world = start_segment.base_placement.multiply(wafer.lcs1)
                start_lcs_list.append(lcs_world)

        if len(end_lcs_list) < 1 or len(start_lcs_list) < 1:
            raise ValueError("Need at least 1 LCS per end to create closing curve")

        logger.debug(f"Using {len(end_lcs_list)} end LCS and {len(start_lcs_list)} start LCS")

        P0_world = end_lcs_list[-1].Base
        P3_world = start_lcs_list[0].Base

        # Calculate exit direction
        exit_direction_world = self._compute_tangent_direction(
            end_lcs_list, reverse=True
        )

        # Calculate entry direction
        entry_direction_world = self._compute_tangent_direction(
            start_lcs_list, reverse=True
        )

        # --- 2A: Angle threshold check ---
        closing_angle_deg = math.degrees(
            (-exit_direction_world).getAngle(-entry_direction_world)
        )
        max_closing_angle = parameters.get('max_closing_angle', 90.0)

        if closing_angle_deg > max_closing_angle:
            logger.warning(
                f"Closing angle ({closing_angle_deg:.1f}) exceeds threshold "
                f"({max_closing_angle:.1f}). The closing curve may be too tight. "
                f"Consider adding helper segments (entry_helper_length / "
                f"exit_helper_length) or inserting intermediate segments to "
                f"reduce the angular gap."
            )
        else:
            logger.debug(
                f"Closing angle: {closing_angle_deg:.1f} "
                f"(threshold: {max_closing_angle:.1f})"
            )

        # --- 2B: Helper segment extensions ---
        exit_helper_length = parameters.get('exit_helper_length')
        entry_helper_length = parameters.get('entry_helper_length')

        if exit_helper_length and float(exit_helper_length) > 0:
            ext = float(exit_helper_length)
            P0_world = P0_world + (-exit_direction_world) * ext
            logger.debug(f"Applied exit helper extension of {ext:.3f}")

        if entry_helper_length and float(entry_helper_length) > 0:
            ext = float(entry_helper_length)
            P3_world = P3_world + (-entry_direction_world) * ext
            logger.debug(f"Applied entry helper extension of {ext:.3f}")

        gap_distance = P0_world.distanceToPoint(P3_world)

        logger.debug(f"=== Geometric closing curve construction ===")
        logger.debug(f"P0_world (exit): {P0_world}")
        logger.debug(f"P3_world (entry): {P3_world}")
        logger.debug(f"exit_direction: {exit_direction_world}")
        logger.debug(f"entry_direction: {entry_direction_world}")
        logger.debug(f"Gap distance: {gap_distance:.3f}")

        # Radius constraints
        cylinder_radius = parameters.get('cylinder_radius', 1.0)
        safety_factor = 4.0
        min_radius = cylinder_radius * safety_factor
        construction_radius = min_radius * 2

        logger.debug(
            f"Minimum safe radius: {min_radius:.3f}, "
            f"using construction radius: {construction_radius:.3f}"
        )

        return {
            'P0': P0_world,
            'P3': P3_world,
            'exit_direction': exit_direction_world,
            'entry_direction': entry_direction_world,
            'num_points': num_points,
            'construction_radius': construction_radius,
            'min_radius': min_radius,
            'gap_distance': gap_distance,
        }

    def _compute_tangent_direction(self, lcs_list, reverse=False):
        """Compute averaged tangent direction from a list of LCS placements."""
        return compute_tangent_direction(lcs_list, reverse=reverse)

    def _build_exit_arc(self, P0, exit_direction, to_target,
                        construction_radius, num_points, exit_turn_angle):
        """
        Build the exit arc from P0 turning toward the target direction.

        Returns:
            (points_to_add, P1_world, dir1_world) where points_to_add
            excludes the last arc point to avoid duplication.
        """
        if exit_turn_angle > 5:
            arc_points = self._create_circular_arc(
                P0, exit_direction, to_target,
                construction_radius,
                max(2, int(num_points * exit_turn_angle / 360))
            )
            points_to_add = arc_points[:-1]
            P1 = arc_points[-1]
            dir1 = to_target
            logger.debug(
                f"Exit arc: {len(points_to_add)} points added, "
                f"turning {exit_turn_angle:.1f}"
            )
        else:
            points_to_add = [P0]
            P1 = P0
            dir1 = exit_direction
            logger.debug("Exit arc: skipped (small angle)")

        return points_to_add, P1, dir1

    def _build_entry_arc(self, P3, entry_direction, to_target,
                         construction_radius, num_points, entry_turn_angle):
        """
        Build the entry arc arriving at P3 from the target direction.

        Returns:
            (entry_arc_points, P2_world, dir2_world) where entry_arc_points
            is in forward order (approach → P3), or empty if no arc needed.
        """
        if entry_turn_angle > 5:
            arc_points = self._create_circular_arc(
                P3, entry_direction, -to_target,
                construction_radius,
                max(2, int(num_points * entry_turn_angle / 360))
            )
            arc_points.reverse()
            P2 = arc_points[0]
            dir2 = -to_target
            logger.debug(
                f"Entry arc: {len(arc_points)} points, "
                f"turning {entry_turn_angle:.1f}"
            )
        else:
            arc_points = []
            P2 = P3
            dir2 = entry_direction
            logger.debug("Entry arc: skipped (small angle)")

        return arc_points, P2, dir2

    def _build_bezier_bridge(self, P1, dir1, P2, dir2,
                             min_radius, num_points, gap_distance):
        """
        Build bridge segment between exit and entry arcs.

        Uses a straight line when directions are nearly parallel,
        otherwise a Bezier-based connecting arc.

        Returns list of bridge points (excluding the last point to
        avoid duplication with the entry arc).
        """
        import math

        bridge_distance = P1.distanceToPoint(P2)
        logger.debug(f"Bridge distance: {bridge_distance:.3f}")

        if bridge_distance <= 0.1:
            logger.debug("Bridge: arcs already meet")
            return []

        angle_between = math.degrees(dir1.getAngle(-dir2))
        logger.debug(f"Angle between arc ends: {angle_between:.1f}")

        if angle_between < 10:
            # Nearly parallel - straight line
            count = max(2, int(num_points * bridge_distance / gap_distance))
            points = []
            for i in range(count - 1):  # exclude last
                t = i / (count - 1)
                points.append(P1 + (P2 - P1) * t)
            logger.debug(f"Bridge: straight line, {len(points)} points added")
            return points
        else:
            # Bezier connecting arc
            bridge_radius = max(min_radius * 2, bridge_distance / 2)
            arc_points = self._create_connecting_arc(
                P1, dir1, P2, dir2,
                bridge_radius, int(num_points * 0.5)
            )
            result = arc_points[:-1]  # exclude last to avoid duplicate
            logger.debug(
                f"Bridge: connecting arc, radius={bridge_radius:.3f}, "
                f"{len(result)} points added"
            )
            return result

    def _create_editable_closing_curve(
            self, P0, P3, exit_direction, entry_direction,
            exit_turn_angle, entry_turn_angle,
            exit_arc_points, entry_arc_points,
            end_segment, start_segment, gap_distance):
        """Create an editable BSpline curve in the FreeCAD document for manual editing."""
        import FreeCAD as App
        import Part

        doc = App.ActiveDocument
        if doc is None:
            logger.warning("No active document - cannot create editable curve")
            return

        curve_name = f"EDIT_Closing_{end_segment.name}_to_{start_segment.name}"

        existing = doc.getObject(curve_name)
        if existing:
            doc.removeObject(curve_name)

        # Build control points
        control_points = [P0]

        P1_actual = P0
        if exit_turn_angle > 5 and exit_arc_points:
            P1_actual = exit_arc_points[-1]
            control_points.append(P1_actual)

        P2_actual = P3
        if entry_turn_angle > 5 and entry_arc_points:
            P2_actual = entry_arc_points[0]
            control_points.append(P2_actual)

        control_points.append(P3)

        # Add 3 intermediate points between P1 and P2 for better control
        if len(control_points) == 4:
            P1 = control_points[1]
            P2 = control_points[2]
            intermediates = []
            for i in range(1, 4):
                t = i / 4.0
                intermediates.append(App.Vector(
                    P1.x + t * (P2.x - P1.x),
                    P1.y + t * (P2.y - P1.y),
                    P1.z + t * (P2.z - P1.z)
                ))
            control_points = [
                control_points[0], control_points[1],
                intermediates[0], intermediates[1], intermediates[2],
                control_points[2], control_points[3]
            ]
            logger.debug("Added 3 intermediate points between exit and entry arcs")

        logger.debug(f"Creating editable curve with {len(control_points)} control points")

        # Calculate tangent vectors
        tangent_start = exit_direction.normalize()
        tangent_end = entry_direction.normalize()

        tangent_length = (
            (P1_actual - P0).Length if len(control_points) > 1
            else gap_distance / 3.0
        )
        tangent_start = tangent_start * tangent_length
        tangent_end = tangent_end * tangent_length

        try:
            bspline_curve = Part.BSplineCurve()
            tangents = (
                [tangent_start]
                + [None] * (len(control_points) - 2)
                + [tangent_end]
            )
            bspline_curve.interpolate(
                Points=control_points,
                PeriodicFlag=False,
                Tolerance=0.001,
                Parameters=None,
                Tangents=tangents
            )
            edge = Part.Edge(bspline_curve)
            bspline = doc.addObject("Part::Feature", curve_name)
            bspline.Shape = edge
            bspline.Label = curve_name
            bspline.ViewObject.LineColor = (1.0, 0.0, 1.0)
            bspline.ViewObject.LineWidth = 4.0

            logger.info(f"Created smooth BSpline with tangent constraints: '{curve_name}'")
        except Exception as e:
            logger.warning(f"Could not create BSpline with tangent constraints: {e}")
            logger.info("Falling back to simple Draft BSpline")

            import Draft
            bspline = Draft.makeBSpline(control_points, closed=False)
            bspline.Label = curve_name
            bspline.ViewObject.LineColor = (1.0, 0.0, 1.0)
            bspline.ViewObject.LineWidth = 4.0

        doc.recompute()

        logger.info(f"Created editable curve: '{curve_name}' with {len(control_points)} control points")
        logger.info(f"  To use: set workflow_mode: 'second_pass' and "
                     f"use_edited_curve: '{curve_name}' in YAML")

    def _create_circular_arc(self, start_point, start_direction, end_direction, radius, num_points):
        """
        Create a circular arc from start_point that begins tangent to start_direction
        and ends tangent to end_direction, with the specified radius.
        """
        import FreeCAD as App
        import math

        angle = start_direction.getAngle(end_direction)

        if angle < 0.01 or num_points < 2:
            return [start_point]

        # Axis of rotation
        axis = start_direction.cross(end_direction)
        if axis.Length < 1e-6:
            return [start_point]
        axis.normalize()

        # Perpendicular to start_direction in the plane of rotation
        # This points toward the center of the arc
        perpendicular = axis.cross(start_direction)
        perpendicular.normalize()

        # CRITICAL FIX: Check if this perpendicular points "into" the turn
        # by seeing if rotating start_direction around axis brings it closer to end_direction
        test_rotation = App.Rotation(axis, 1.0)  # Rotate 1 degree
        test_vector = test_rotation.multVec(start_direction)

        # If test rotation moves away from end_direction, flip the perpendicular
        if test_vector.getAngle(end_direction) > start_direction.getAngle(end_direction):
            perpendicular = -perpendicular

        center = start_point + perpendicular * radius

        # Generate arc points
        points = []
        radius_vector = start_point - center

        for i in range(num_points):
            t = i / max(1, num_points - 1)
            current_angle = t * angle
            rotation = App.Rotation(axis, math.degrees(current_angle))
            rotated_vector = rotation.multVec(radius_vector)
            point = center + rotated_vector
            points.append(point)

        return points

    def _create_connecting_arc(self, P1, dir1, P2, dir2, radius, num_points):
        """
        Create a smooth connecting arc between two points with specified tangent directions.
        Uses a larger radius for a gentle connection.

        This is a simplified version - for now, just use a Bezier curve with controlled arms.
        """

        # Calculate arm length from radius
        gap = P1.distanceToPoint(P2)
        arm_length = min(radius, gap / 3)

        # Control points
        ctrl1 = P1 + dir1 * arm_length
        ctrl2 = P2 + dir2 * arm_length

        # Create Bezier curve
        bspline = Part.BSplineCurve()
        bspline.buildFromPoles([P1, ctrl1, ctrl2, P2], False)

        # Sample points
        points = []
        for i in range(num_points):
            u = i / max(1, num_points - 1)
            points.append(bspline.value(u))

        return points

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

def compute_tangent_direction(lcs_list, reverse=False):
    """Compute averaged tangent direction from a list of LCS placements.

    Args:
        lcs_list: List of App.Placement objects representing local coordinate systems.
        reverse: If True, reverse the computed direction.

    Returns:
        App.Vector: Normalized tangent direction.
    """
    if len(lcs_list) >= 2:
        directions = []
        for i in range(len(lcs_list) - 1):
            direction = lcs_list[i + 1].Base - lcs_list[i].Base
            if direction.Length > 1e-6:
                direction.normalize()
                directions.append(direction)
        if directions:
            avg = App.Vector(
                sum(d.x for d in directions) / len(directions),
                sum(d.y for d in directions) / len(directions),
                sum(d.z for d in directions) / len(directions)
            )
            avg.normalize()
            return -avg if reverse else avg

    forward = lcs_list[-1 if reverse else 0].Rotation.multVec(
        App.Vector(0, 0, 1)
    )
    return -forward if reverse else forward


def generate_woodcut_trefoil(slices=180, **parameters):
    """
    Generate a trefoil knot curve optimized for wood cutting with cylindrical slices.

    A trefoil knot is a (p,q)-torus knot that wraps around a torus p times in one
    direction and q times in the other. This implementation creates a parametric
    trefoil curve suitable for cutting from cylindrical stock.

    Args:
        slices (int): Number of points to generate along the curve (default: 180).
                     More slices = smoother curve but more complex cutting.
                     Maximum is capped at 200 points.

        **parameters: Keyword arguments controlling the trefoil geometry:

            major_radius (float): Radius of the main torus path (default: 6.0).
                                 Controls the overall size of the trefoil.
                                 Larger values = bigger knot.

            tube_radius (float): Radius of the tube that wraps around the torus (default: 2.0).
                                Controls the "thickness" of the knot path.
                                Affects how much the curve deviates from the main circle.

            p (int): Number of times the curve wraps around the torus longitudinally (default: 2).
                    For a trefoil, typically p=2.

            q (int): Number of times the curve wraps around the torus meridionally (default: 3).
                    For a trefoil, typically q=3.
                    The (p,q) pair defines the knot type: (2,3) = trefoil, (3,5) = cinquefoil, etc.

            center (tuple): (x, y, z) coordinates for the center point (default: (0, 0, 0)).
                           Translates the entire curve to this location.

            phase_deg (float): Rotation phase in degrees (default: 0.0).
                              Rotates where the curve starts around the knot.
                              Use to orient the knot optimally for cutting.

            scale_z (float): Vertical scaling factor (default: 1.0).
                            Stretches or compresses the knot in the Z direction.
                            Values < 1.0 flatten the knot, > 1.0 stretch it vertically.

            jitter (float): Random perturbation factor (default: 0.0).
                           Adds randomness to point spacing (0.0 = none, 1.0 = maximum).
                           Can create organic variation but may complicate cutting.

            smooth_factor (float): Smoothing multiplier for tube radius (default: 0.5).
                                  Controls how pronounced the knot's curves are.
                                  Values < 1.0 reduce deviation from the major circle,
                                  creating a gentler, more cuttable curve.

            optimize_spacing (bool): Whether to optimize point spacing for cutting (default: True).
                                    When True, redistributes points for more even spacing,
                                    which can improve cutting accuracy and reduce waste.

    Returns:
        numpy.ndarray: Array of shape (n+1, 3) containing [x, y, z] coordinates.
                      The curve is closed (first point equals last point).

    Example:
        >>> # Basic trefoil with default settings
        >>> points = generate_woodcut_trefoil(slices=120)

        >>> # Larger trefoil with custom dimensions
        >>> points = generate_woodcut_trefoil(
        ...     slices=200,
        ...     major_radius=8.0,
        ...     tube_radius=3.0,
        ...     scale_z=1.5
        ... )

        >>> # Flattened trefoil rotated 45 degrees
        >>> points = generate_woodcut_trefoil(
        ...     slices=150,
        ...     scale_z=0.5,
        ...     phase_deg=45.0,
        ...     smooth_factor=0.3
        ... )

    Notes:
        - The (p, q) = (2, 3) configuration creates the classic trefoil knot
        - Increase smooth_factor for more dramatic knot curves (harder to cut)
        - Decrease smooth_factor for gentler curves (easier to cut from cylinder)
        - Use scale_z < 1.0 to flatten the knot for easier visualization
        - The curve closes on itself (first point = last point)
    """
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
    # logger.debug(f"Generated woodcut trefoil with {len(new_pts)} points")
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