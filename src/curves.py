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

    def _generate_sinusoidal(self, length: float = 50.0, amplitude: float = 5.0,
                             frequency: float = 2.0, points: int = 100,
                             axis: str = 'x') -> List[List[float]]:
        """Generate a sinusoidal curve."""
        # logger.debug(f"Generating sinusoidal: length={length}, amplitude={amplitude}, freq={frequency}")

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
        Generate closing curve using geometric construction from circular arcs

        Supports both automatic generation and using user-edited curves.
        """
        import FreeCAD as App
        import Part
        import math

        start_segment = parameters.get('start_segment')
        end_segment = parameters.get('end_segment')

        if start_segment is None or end_segment is None:
            raise ValueError("closing_curve requires start_segment and end_segment parameters")

        # ============================================================================
        # CHECK FOR EDITED CURVE FIRST - USE IT IF PROVIDED
        # ============================================================================
        use_edited_curve = parameters.get('use_edited_curve')
        if use_edited_curve:
            logger.info(f"📝 Using edited curve: {use_edited_curve}")
            curve_obj = App.ActiveDocument.getObject(use_edited_curve)

            if curve_obj is None:
                raise ValueError(f"Could not find curve object: {use_edited_curve}")

            # Sample points from the edited curve's Shape
            if hasattr(curve_obj, 'Shape') and hasattr(curve_obj.Shape, 'Edges') and len(curve_obj.Shape.Edges) > 0:
                edge = curve_obj.Shape.Edges[0]
                curve_points_world = []
                num_samples = parameters.get('points', 50)

                logger.info(f"Sampling {num_samples} points from edited curve")

                for i in range(num_samples):
                    u = i / max(1, num_samples - 1)
                    param = edge.FirstParameter + u * (edge.LastParameter - edge.FirstParameter)
                    point = edge.valueAt(param)
                    curve_points_world.append(point)

                logger.info(f"✓ Sampled {len(curve_points_world)} points from edited curve")

                # Transform to local coordinates and return
                end_wafers = end_segment.wafer_list
                last_wafer = end_wafers[-1]
                last_lcs2_world = end_segment.base_placement.multiply(last_wafer.lcs2)
                placement_inv = last_lcs2_world.inverse()

                points_local = []
                for point_world in curve_points_world:
                    point_local = placement_inv.multVec(point_world)
                    points_local.append([point_local.x, point_local.y, point_local.z])

                logger.debug(f"Edited curve start (local): {points_local[0]}")
                logger.debug(f"Edited curve end (local): {points_local[-1]}")

                return np.array(points_local)
            else:
                raise ValueError(f"Curve object {use_edited_curve} has no usable geometry (Shape.Edges)")

        # ============================================================================
        # NO EDITED CURVE - CONTINUE WITH AUTOMATIC GEOMETRIC GENERATION
        # ============================================================================
        logger.debug("No edited curve specified, using automatic generation")

        num_lcs = parameters.get('num_lcs_per_end', 3)
        num_points = parameters.get('points', 50)

        # Get wafer lists
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

        # Get connection points and directions in WORLD coordinates
        P0_world = end_lcs_list[-1].Base
        P3_world = start_lcs_list[0].Base

        # Calculate exit direction
        if len(end_lcs_list) >= 2:
            exit_directions = []
            for i in range(len(end_lcs_list) - 1):
                direction = end_lcs_list[i + 1].Base - end_lcs_list[i].Base
                if direction.Length > 1e-6:
                    direction.normalize()
                    exit_directions.append(direction)
            if exit_directions:
                exit_direction_world = App.Vector(
                    sum(d.x for d in exit_directions) / len(exit_directions),
                    sum(d.y for d in exit_directions) / len(exit_directions),
                    sum(d.z for d in exit_directions) / len(exit_directions)
                )
                exit_direction_world.normalize()
                exit_direction_world = -exit_direction_world
            else:
                forward = end_lcs_list[-1].Rotation.multVec(App.Vector(0, 0, 1))
                exit_direction_world = -forward
        else:
            forward = end_lcs_list[-1].Rotation.multVec(App.Vector(0, 0, 1))
            exit_direction_world = -forward

        exit_direction_world.normalize()

        # Calculate entry direction
        if len(start_lcs_list) >= 2:
            entry_directions = []
            for i in range(len(start_lcs_list) - 1):
                direction = start_lcs_list[i + 1].Base - start_lcs_list[i].Base
                if direction.Length > 1e-6:
                    direction.normalize()
                    entry_directions.append(direction)
            if entry_directions:
                forward_direction = App.Vector(
                    sum(d.x for d in entry_directions) / len(entry_directions),
                    sum(d.y for d in entry_directions) / len(entry_directions),
                    sum(d.z for d in entry_directions) / len(entry_directions)
                )
                forward_direction.normalize()
                entry_direction_world = -forward_direction
            else:
                forward = start_lcs_list[0].Rotation.multVec(App.Vector(0, 0, 1))
                entry_direction_world = -forward
        else:
            forward = start_lcs_list[0].Rotation.multVec(App.Vector(0, 0, 1))
            entry_direction_world = -forward

        entry_direction_world.normalize()

        gap_distance = P0_world.distanceToPoint(P3_world)

        logger.debug(f"=== Geometric closing curve construction ===")
        logger.debug(f"P0_world (exit): {P0_world}")
        logger.debug(f"P3_world (entry): {P3_world}")
        logger.debug(f"exit_direction: {exit_direction_world}")
        logger.debug(f"entry_direction: {entry_direction_world}")
        logger.debug(f"Gap distance: {gap_distance:.3f}")

        # Get minimum radius constraint
        cylinder_radius = parameters.get('cylinder_radius', 1.0)
        safety_factor = 4.0
        min_radius = cylinder_radius * safety_factor

        # Use LARGER radius for arc construction (more gentle curves)
        construction_radius = min_radius * 2  # 8× cylinder radius instead of 4×

        logger.debug(f"Minimum safe radius: {min_radius:.3f}, using construction radius: {construction_radius:.3f}")

        # Direction from P0 toward P3
        to_target = P3_world - P0_world
        to_target.normalize()

        # Calculate turn angles
        exit_turn_angle = math.degrees(exit_direction_world.getAngle(to_target))
        entry_turn_angle = math.degrees((-to_target).getAngle(entry_direction_world))

        logger.debug(f"Exit turn needed: {exit_turn_angle:.1f}°")
        logger.debug(f"Entry turn needed: {entry_turn_angle:.1f}°")

        # Build the curve from geometric pieces
        curve_points_world = []

        # STEP 1: Exit arc (if turn needed)
        if exit_turn_angle > 5:  # Need to turn
            exit_arc_points = self._create_circular_arc(
                P0_world,
                exit_direction_world,
                to_target,
                construction_radius,
                int(num_points * exit_turn_angle / 360)  # Proportional sampling
            )
            # Add all but the last point (to avoid duplicate at connection)
            curve_points_world.extend(exit_arc_points[:-1])
            P1_world = exit_arc_points[-1]
            dir1_world = to_target
            logger.debug(f"Exit arc: {len(exit_arc_points) - 1} points added, turning {exit_turn_angle:.1f}°")
        else:
            # No turn needed, start at P0
            curve_points_world.append(P0_world)
            P1_world = P0_world
            dir1_world = exit_direction_world
            logger.debug("Exit arc: skipped (small angle)")

        # STEP 2: Entry arc (if turn needed)
        if entry_turn_angle > 5:
            # Build entry arc backwards from P3
            entry_arc_points = self._create_circular_arc(
                P3_world,
                entry_direction_world,
                -to_target,
                construction_radius,
                int(num_points * entry_turn_angle / 360)
            )
            # Reverse so it goes from approach direction to P3
            entry_arc_points.reverse()
            P2_world = entry_arc_points[0]
            dir2_world = -to_target
            logger.debug(f"Entry arc: {len(entry_arc_points)} points, turning {entry_turn_angle:.1f}°")
        else:
            P2_world = P3_world
            dir2_world = entry_direction_world
            entry_arc_points = []
            logger.debug("Entry arc: skipped (small angle)")

        # STEP 3: Bridge between exit and entry arcs
        bridge_distance = P1_world.distanceToPoint(P2_world)
        logger.debug(f"Bridge distance: {bridge_distance:.3f}")

        if bridge_distance > 0.1:  # Need a bridge
            # Check if directions are parallel
            angle_between = math.degrees(dir1_world.getAngle(-dir2_world))
            logger.debug(f"Angle between arc ends: {angle_between:.1f}°")

            if angle_between < 10:  # Nearly parallel - use straight line
                bridge_points_count = max(2, int(num_points * bridge_distance / gap_distance))
                for i in range(bridge_points_count):
                    t = i / (bridge_points_count - 1)
                    point = P1_world + (P2_world - P1_world) * t
                    # Exclude last point to avoid duplicate with entry arc
                    if i < bridge_points_count - 1:
                        curve_points_world.append(point)
                logger.debug(f"Bridge: straight line, {bridge_points_count - 1} points added")
            else:  # Need a connecting arc
                # Use a larger radius arc for the bridge (gentler curve)
                bridge_radius = max(min_radius * 2, bridge_distance / 2)
                bridge_arc_points = self._create_connecting_arc(
                    P1_world, dir1_world,
                    P2_world, dir2_world,
                    bridge_radius,
                    int(num_points * 0.5)  # Use about half the points for bridge
                )
                # Exclude last point to avoid duplicate with entry arc
                curve_points_world.extend(bridge_arc_points[:-1])
                logger.debug(
                    f"Bridge: connecting arc, radius={bridge_radius:.3f}, {len(bridge_arc_points) - 1} points added")
        else:
            logger.debug("Bridge: arcs already meet")

        # Add entry arc
        if len(entry_arc_points) > 0:
            curve_points_world.extend(entry_arc_points)
        else:
            # No entry arc, just add P3
            curve_points_world.append(P3_world)

        # Calculate total curve length
        total_curve_length = 0
        for i in range(len(curve_points_world) - 1):
            total_curve_length += curve_points_world[i].distanceToPoint(curve_points_world[i + 1])

        logger.info(
            f"Generated geometric closing curve: length={total_curve_length:.3f}, points={len(curve_points_world)}")

        # CREATE EDITABLE CURVE IF REQUESTED
        create_editable = parameters.get('create_editable_curve', False)

        if create_editable:
            import Draft

            # Get document
            doc = App.ActiveDocument
            if doc is None:
                logger.warning("No active document - cannot create editable curve")
            else:
                # Create unique name based on segment names
                end_seg_name = end_segment.name
                start_seg_name = start_segment.name
                curve_name = f"EDIT_Closing_{end_seg_name}_to_{start_seg_name}"

                # Check if it already exists and remove it
                existing = doc.getObject(curve_name)
                if existing:
                    doc.removeObject(curve_name)

                # Build control points with intermediate points for better editing
                control_points = [P0_world]

                # Add exit arc end point if arc was created
                P1_world_actual = P0_world  # Default if no arc
                if exit_turn_angle > 5 and 'exit_arc_points' in locals() and len(exit_arc_points) > 0:
                    P1_world_actual = exit_arc_points[-1]
                    control_points.append(P1_world_actual)

                # Add entry arc start point if arc was created
                P2_world_actual = P3_world  # Default if no arc
                if entry_turn_angle > 5 and 'entry_arc_points' in locals() and len(entry_arc_points) > 0:
                    P2_world_actual = entry_arc_points[0]
                    control_points.append(P2_world_actual)

                # Add end point
                control_points.append(P3_world)

                # Add 3 intermediate points between P1 and P2 for better control
                if len(control_points) == 4:
                    P1 = control_points[1]
                    P2 = control_points[2]

                    intermediate_points = []
                    for i in range(1, 4):  # Create points at 1/4, 2/4, 3/4
                        t = i / 4.0
                        intermediate = App.Vector(
                            P1.x + t * (P2.x - P1.x),
                            P1.y + t * (P2.y - P1.y),
                            P1.z + t * (P2.z - P1.z)
                        )
                        intermediate_points.append(intermediate)

                    # Rebuild: P0, P1, intermediate1-3, P2, P3
                    control_points = [
                        control_points[0],
                        control_points[1],
                        intermediate_points[0],
                        intermediate_points[1],
                        intermediate_points[2],
                        control_points[2],
                        control_points[3]
                    ]
                    logger.debug(f"Added 3 intermediate points between exit and entry arcs")

                logger.debug(f"Creating editable curve with {len(control_points)} control points")

                # Calculate tangent vectors at endpoints for smooth interpolation
                tangent_start = exit_direction_world.normalize()
                tangent_end = entry_direction_world.normalize()

                # Scale tangents by approximate segment length for better curve shape
                tangent_length = (P1_world_actual - P0_world).Length if len(control_points) > 1 else gap_distance / 3.0
                tangent_start = tangent_start * tangent_length
                tangent_end = tangent_end * tangent_length

                # Create BSpline with tangent constraints using Part.BSplineCurve
                try:
                    bspline_curve = Part.BSplineCurve()

                    # Build tangent list: tangent at start, None for interior points, tangent at end
                    tangents = [tangent_start] + [None] * (len(control_points) - 2) + [tangent_end]

                    # Interpolate through points with tangent constraints
                    bspline_curve.interpolate(
                        Points=control_points,
                        PeriodicFlag=False,
                        Tolerance=0.001,
                        Parameters=None,  # Auto-calculate
                        Tangents=tangents
                    )

                    # Create Part feature from curve
                    edge = Part.Edge(bspline_curve)
                    bspline = doc.addObject("Part::Feature", curve_name)
                    bspline.Shape = edge
                    bspline.Label = curve_name
                    bspline.ViewObject.LineColor = (1.0, 0.0, 1.0)  # Magenta
                    bspline.ViewObject.LineWidth = 4.0

                    logger.info(f"✏️  Created smooth BSpline with tangent constraints: '{curve_name}'")
                    logger.debug(f"    Tangent at start: {tangent_start}")
                    logger.debug(f"    Tangent at end: {tangent_end}")

                except Exception as e:
                    logger.warning(f"Could not create BSpline with tangent constraints: {e}")
                    logger.info(f"Falling back to simple Draft BSpline")

                    # Fall back to Draft.makeBSpline
                    bspline = Draft.makeBSpline(control_points, closed=False)
                    bspline.Label = curve_name
                    bspline.ViewObject.LineColor = (1.0, 0.0, 1.0)
                    bspline.ViewObject.LineWidth = 4.0

                doc.recompute()

                logger.info(f"✏️  Created editable curve: '{curve_name}' with {len(control_points)} control points")
                logger.info(f"    Control points breakdown:")
                logger.info(f"    - Point 1: Exit from last segment (tangent constrained)")
                if len(control_points) >= 7:
                    logger.info(f"    - Point 2: End of exit arc")
                    logger.info(f"    - Points 3-5: Bridge section (freely adjustable)")
                    logger.info(f"    - Point 6: Start of entry arc")
                    logger.info(f"    - Point 7: Entry to first segment (tangent constrained)")
                logger.info(f"    To edit:")
                logger.info(f"    1. Select '{curve_name}' in the tree")
                logger.info(f"    2. Use Draft → Modify → Edit (or press E key)")
                logger.info(f"    3. Drag the {len(control_points)} control points to adjust shape")
                logger.info(f"    4. Interior points can be moved freely while endpoints maintain tangency")
                logger.info(f"    5. Press ESC or click 'Close' when done")
                logger.info(f"    6. In YAML: set workflow_mode: 'second_pass'")
                logger.info(f"    7. In YAML: add use_edited_curve: '{curve_name}'")
                logger.info(f"    8. Re-run the workflow")

        # Transform to local coordinates
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