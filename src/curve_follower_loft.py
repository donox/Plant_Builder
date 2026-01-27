"""
curve_follower_loft.py - Curve utilities for loft-based wafer generation

This module provides:
- Curve point loading from curves module
- Chord-distance sampling algorithm
- Curve manipulation utilities

Works with LoftWaferGenerator to create wafers along curved paths.
Uses the existing curves module for curve generation.
"""

import FreeCAD as App
import Part
import math
import curves
from wafer_loft import LoftWaferGenerator, simple_chord_distance_sampler
from core.logging_setup import get_logger
from core.core_utils import is_identity_placement

logger = get_logger(__name__)


class CurveFollowerLoft:
    """Manages 3D curves for loft-based wafer generation"""

    def __init__(self, wafer_settings=None):
        """
        Initialize curve follower

        Args:
            wafer_settings: Dictionary with wafer configuration (cylinder_diameter, etc.)
        """
        self.curve_points = []
        self.spine_curve = None
        self.generator = None
        self.wafer_settings = wafer_settings or {}
        # logger.debug("CurveFollowerLoft initialized")

    def load_curve_from_file(self, filename):
        """
        Load curve points from a file

        File format: x y z (one point per line)
        Lines starting with # are comments

        Args:
            filename: Path to file containing points

        Returns:
            List of App.Vector points
        """
        logger.info(f"Loading curve from {filename}")

        points = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                try:
                    coords = [float(x) for x in line.split()]
                    if len(coords) != 3:
                        logger.warning(f"Line {line_num} does not have 3 coordinates, skipping")
                        continue

                    points.append(App.Vector(coords[0], coords[1], coords[2]))

                except ValueError as e:
                    logger.warning(f"Could not parse line {line_num}: {e}")
                    continue

        self.curve_points = points
        logger.info(f"Loaded {len(points)} points from {filename}")
        return points

    def load_curve_from_curves_module(self, curve_type, **params):
        """
        Generate curve points using the existing curves module

        Args:
            curve_type: Type of curve ('helix', 'spiral', 'sine', etc.)
            **params: Parameters to pass to the curve generator

        Returns:
            List of App.Vector points
        """
        logger.info(f"Generating {curve_type} curve using curves module")
        # logger.debug(f"Curve parameters: {params}")

        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("CurveGen")
            # logger.debug("Created new document for curve generation")

        points = curves.generate_curve(curve_type, doc=doc, **params)

        self.curve_points = points
        logger.info(f"Generated {len(points)} points")
        return points

    def set_curve_points(self, points):
        """
        Set curve points directly

        Args:
            points: List of App.Vector points
        """
        self.curve_points = points
        # logger.debug(f"Set {len(points)} curve points")

    def create_spine_curve(self, points=None):
        """
        Create a B-spline curve from points

        Args:
            points: List of App.Vector points (uses self.curve_points if None)

        Returns:
            Part.Edge representing the spine curve
        """
        if points is None:
            points = self.curve_points

        if len(points) < 2:
            raise ValueError("Need at least 2 points to create a curve")

        # logger.debug(f"Creating spine curve from {len(points)} points")

        spline = Part.BSplineCurve()
        spline.interpolate(points)
        self.spine_curve = spline.toShape()

        logger.info(f"Spine curve created, length: {self.spine_curve.Length:.3f}")
        return self.spine_curve

    def chord_distance_sampler(self, spine_edge, max_chord_deviation):
        """
        Sample points along curve using maximum chord deviation

        Places sample points such that the maximum perpendicular distance
        from the chord (straight line between points) to the curve does not
        exceed the target deviation.

        Args:
            spine_edge: Part.Edge representing the spine
            max_chord_deviation: Maximum allowed deviation from chord to curve

        Returns:
            List of parameters along the curve
        """
        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter

        # logger.debug(f"Chord deviation sampling: max_deviation={max_chord_deviation:.3f}")

        params = [first_param]
        current_param = first_param

        while current_param < last_param:
            param_low = current_param
            param_high = last_param
            best_param = current_param

            for iteration in range(25):
                test_param = (param_low + param_high) / 2

                if test_param <= current_param or test_param >= last_param:
                    best_param = last_param
                    break

                max_deviation = self._calculate_max_chord_deviation(
                    curve, current_param, test_param
                )

                if max_deviation <= max_chord_deviation:
                    best_param = test_param
                    param_low = test_param
                else:
                    param_high = test_param

                if abs(param_high - param_low) < (last_param - first_param) / 10000:
                    break

            if best_param <= current_param:
                best_param = min(current_param + (last_param - first_param) / 100, last_param)

            params.append(best_param)
            current_param = best_param

            if best_param >= last_param:
                break

        if params[-1] < last_param:
            params[-1] = last_param

        # logger.debug(f"Sampled {len(params)} points")
        return params

    def _calculate_max_chord_deviation(self, curve, param_start, param_end, num_samples=20):
        """
        Calculate maximum perpendicular distance from chord to curve segment

        Args:
            curve: The B-spline curve
            param_start: Starting parameter
            param_end: Ending parameter
            num_samples: Number of points to check along the segment

        Returns:
            Maximum perpendicular deviation
        """
        start_point = curve.value(param_start)
        end_point = curve.value(param_end)

        chord_vector = end_point - start_point
        chord_length = chord_vector.Length

        if chord_length < 1e-10:
            return 0.0

        chord_direction = chord_vector.normalize()
        max_deviation = 0.0

        for i in range(1, num_samples):
            t = param_start + (param_end - param_start) * i / num_samples
            curve_point = curve.value(t)

            to_curve = curve_point - start_point
            projection_length = to_curve.dot(chord_direction)

            if projection_length <= 0:
                closest_on_chord = start_point
            elif projection_length >= chord_length:
                closest_on_chord = end_point
            else:
                closest_on_chord = start_point + chord_direction * projection_length

            deviation = (curve_point - closest_on_chord).Length
            max_deviation = max(max_deviation, deviation)

        return max_deviation

    def uniform_sampler(self, spine_edge, num_samples):
        """
        Sample points uniformly by parameter

        Args:
            spine_edge: Part.Edge representing the spine
            num_samples: Number of samples to generate

        Returns:
            List of parameters along the curve
        """
        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter

        # logger.debug(f"Uniform sampling: {num_samples} samples")

        params = []
        for i in range(num_samples):
            t = first_param + (last_param - first_param) * i / (num_samples - 1)
            params.append(t)

        return params

    def arc_length_sampler(self, spine_edge, target_arc_distance):
        """
        Sample points at equal arc length intervals

        Args:
            spine_edge: Part.Edge representing the spine
            target_arc_distance: Target arc length between samples

        Returns:
            List of parameters along the curve
        """
        total_length = spine_edge.Length
        num_samples = max(2, int(total_length / target_arc_distance) + 1)

        # logger.debug(f"Arc-length sampling: target={target_arc_distance:.3f}, {num_samples} samples")

        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter

        params = []

        for i in range(num_samples):
            target_length = total_length * i / (num_samples - 1)

            param_low = first_param
            param_high = last_param

            for _ in range(20):
                param_mid = (param_low + param_high) / 2

                edge_segment = Part.Edge(curve, first_param, param_mid)
                current_length = edge_segment.Length

                if abs(current_length - target_length) < 0.001:
                    break
                elif current_length < target_length:
                    param_low = param_mid
                else:
                    param_high = param_mid

            params.append(param_mid)

        return params

    def generate_loft_wafers(self, curve_spec, wafer_settings, base_placement=None):
        """
        Generate wafers using loft-based approach

        Args:
            curve_spec: Dictionary with curve type and parameters
            wafer_settings: Dictionary with wafer configuration
            base_placement: Optional placement to transform curve points
        """
        self.wafer_settings = wafer_settings

        cylinder_diameter = wafer_settings.get('cylinder_diameter', 1.875)
        cylinder_radius = cylinder_diameter / 2.0

        self.generator = LoftWaferGenerator(wafer_settings=wafer_settings , cylinder_radius=cylinder_radius)
        # logger.debug(f"Created LoftWaferGenerator with radius {cylinder_radius:.3f}")

        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("CurveGen")

        # Use Curves class to handle full curve_spec with transformations and segments
        from curves import Curves
        curves_instance = Curves(doc, curve_spec)
        points_array = curves_instance.get_curve_points()
        # logger.debug(f"Points from Curves class: {len(points_array)}")

        # Convert numpy array to list of App.Vector
        points = [App.Vector(float(p[0]), float(p[1]), float(p[2])) for p in points_array]

        # Transform points by base_placement if provided
        if base_placement is not None and not self._is_identity_placement(base_placement):
            # logger.debug(f"Transforming {len(points)} curve points by base_placement")
            points = [base_placement.multVec(p) for p in points]

        # logger.debug(f"Generated {len(points)} curve points with transformations and segments applied")

        self.generator.create_spine_from_points(points)
        self.generator.create_loft_along_spine(self)

        if self.generator.loft is None:
            raise ValueError("Loft creation returned None")

        logger.info(f"Loft created successfully, volume: {self.generator.loft.Volume:.4f}")

        target_chord = wafer_settings.get('max_chord', 0.5)
        max_wafer_count = wafer_settings.get('max_wafer_count', None)

        def limited_sampler(spine):
            params = self.chord_distance_sampler(spine, target_chord)

            if max_wafer_count and len(params) > max_wafer_count + 1:
                # logger.debug(f"Limiting to {max_wafer_count} wafers (from {len(params) - 1} total samples)")
                params = params[:max_wafer_count + 1]

            return params

        self.generator.sample_points_along_loft(limited_sampler)

        # logger.debug(f"Creating {len(self.generator.sample_points) - 1} wafers")

        self.generator.calculate_cutting_planes()

        wafers = self.generator.generate_wafers()

        if wafers and len(wafers) > 0:
            x_coords = []
            for wafer in wafers:
                if wafer.wafer is not None:
                    bbox = wafer.wafer.BoundBox
                    x_coords.extend([bbox.XMin, bbox.XMax])

            if x_coords:
                self.x_min = min(x_coords)
                self.x_max = max(x_coords)
                logger.coord(f"Wafer X range: {self.x_min:.2f} to {self.x_max:.2f}")
            else:
                self.x_min = None
                self.x_max = None

        return wafers

    def get_wafer_list(self):
        """
        Get the list of generated wafers

        Returns:
            List of Wafer objects
        """
        if self.generator is None:
            return []
        return self.generator.wafers

    def get_curve_info(self):
        """
        Get information about the current curve

        Returns:
            Dictionary with curve statistics
        """
        if not self.curve_points:
            return {'num_points': 0}

        info = {
            'num_points': len(self.curve_points),
            'start_point': self.curve_points[0],
            'end_point': self.curve_points[-1]
        }

        if self.spine_curve:
            info['curve_length'] = self.spine_curve.Length

            bbox = self.spine_curve.BoundBox
            info['bbox_min'] = App.Vector(bbox.XMin, bbox.YMin, bbox.ZMin)
            info['bbox_max'] = App.Vector(bbox.XMax, bbox.YMax, bbox.ZMax)
            info['bbox_size'] = App.Vector(bbox.XLength, bbox.YLength, bbox.ZLength)

        return info

    def visualize_curve(self, doc, show_points=True, show_curve=True):
        """
        Visualize the curve in FreeCAD

        Args:
            doc: FreeCAD document
            show_points: Show individual curve points
            show_curve: Show the interpolated curve
        """
        logger.info("Visualizing curve")

        if show_points and self.curve_points:
            for i, point in enumerate(self.curve_points):
                sphere = Part.makeSphere(0.1, point)
                point_obj = doc.addObject("Part::Feature", f"CurvePoint_{i:03d}")
                point_obj.Shape = sphere
                point_obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)
            # logger.debug(f"Added {len(self.curve_points)} curve points")

        if show_curve and self.spine_curve:
            curve_obj = doc.addObject("Part::Feature", "Spine_Curve")
            curve_obj.Shape = self.spine_curve
            curve_obj.ViewObject.LineColor = (0.0, 0.5, 1.0)
            curve_obj.ViewObject.LineWidth = 3
            # logger.debug("Added spine curve")

        doc.recompute()
        logger.info("Curve visualization complete")

    def _is_identity_placement(self, placement):
        """Check if placement is identity (no transformation)"""
        return is_identity_placement(placement)

    def visualize_wafers(self, doc, show_lcs=True, show_cutting_planes=True):
        """
        Visualize the generated wafers in FreeCAD

        Args:
            doc: FreeCAD document
            show_lcs: Show local coordinate systems
            show_cutting_planes: Show cutting plane discs
        """
        if self.generator:
            self.generator.visualize_in_freecad(doc, show_lcs, show_cutting_planes)


def create_sampler_function(method='chord_distance', target_distance=0.5, num_samples=None):
    """
    Create a sampler function for use with LoftWaferGenerator

    Args:
        method: Sampling method ('chord_distance', 'uniform', 'arc_length')
        target_distance: Target distance for chord_distance or arc_length methods
        num_samples: Number of samples for uniform method

    Returns:
        Function that takes spine_edge and returns list of parameters
    """
    follower = CurveFollowerLoft()

    if method == 'chord_distance':
        return lambda spine: follower.chord_distance_sampler(spine, target_distance)
    elif method == 'uniform':
        if num_samples is None:
            raise ValueError("num_samples required for uniform sampling")
        return lambda spine: follower.uniform_sampler(spine, num_samples)
    elif method == 'arc_length':
        return lambda spine: follower.arc_length_sampler(spine, target_distance)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def get_curve_points_from_curves_module(curve_config):
    """
    Helper function to get curve points from curves module based on config

    Args:
        curve_config: Dictionary with 'type' and curve parameters

    Returns:
        List of App.Vector points
    """
    curve_type = curve_config.get('type')
    if not curve_type:
        raise ValueError("curve_config must have 'type' key")

    params = {k: v for k, v in curve_config.items() if k != 'type'}

    follower = CurveFollowerLoft()
    return follower.load_curve_from_curves_module(curve_type, **params)