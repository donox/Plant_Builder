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
from core.wafer_settings import WaferSettings

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
        self.wafer_settings = wafer_settings if wafer_settings is not None else WaferSettings()
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

    def chord_distance_sampler(self, spine_edge, max_chord_length, max_chord_deviation=None):
        """
        Sample points along curve using chord length and optional deviation limits.

        Places sample points such that:
        - The 3D distance between consecutive points does not exceed max_chord_length
        - If max_chord_deviation is set, the perpendicular deviation from chord
          to curve does not exceed that value either (adaptive: tighter on curves)

        Args:
            spine_edge: Part.Edge representing the spine
            max_chord_length: Maximum allowed 3D distance between consecutive points
            max_chord_deviation: Optional max perpendicular deviation from chord to curve

        Returns:
            List of parameters along the curve
        """
        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter

        params = [first_param]
        current_param = first_param
        current_point = curve.value(current_param)

        convergence_threshold = (last_param - first_param) / 10000

        while current_param < last_param:
            # If remaining range is below convergence threshold, snap to end
            if (last_param - current_param) < convergence_threshold:
                params.append(last_param)
                break

            param_low = current_param
            param_high = last_param
            best_param = current_param

            for iteration in range(25):
                test_param = (param_low + param_high) / 2

                if test_param <= current_param or test_param >= last_param:
                    best_param = last_param
                    break

                test_point = curve.value(test_param)
                chord_length = (test_point - current_point).Length

                # Check chord length constraint
                accept = chord_length <= max_chord_length

                # Additionally check deviation constraint if specified
                if accept and max_chord_deviation is not None:
                    deviation = self._calculate_max_chord_deviation(
                        curve, current_param, test_param
                    )
                    accept = deviation <= max_chord_deviation

                if accept:
                    best_param = test_param
                    param_low = test_param
                else:
                    param_high = test_param

                if abs(param_high - param_low) < convergence_threshold:
                    break

            if best_param <= current_param:
                best_param = min(current_param + (last_param - first_param) / 100, last_param)

            # Snap to end if close enough
            if (last_param - best_param) < convergence_threshold:
                best_param = last_param

            params.append(best_param)
            current_param = best_param
            current_point = curve.value(current_param)

            if best_param >= last_param:
                break

        if params[-1] < last_param:
            params[-1] = last_param

        return params

    def _estimate_curvature(self, curve, param, delta_fraction=0.001):
        """
        Estimate curvature κ at a parameter using tangent angle change.

        κ ≈ |ΔT| / Δs where ΔT is the change in unit tangent direction
        and Δs is the arc length over a small parameter interval.

        Args:
            curve: The B-spline curve
            param: Parameter at which to estimate curvature
            delta_fraction: Fraction of total parameter range for finite difference

        Returns:
            Estimated curvature (1/radius_of_curvature)
        """
        param_range = curve.LastParameter - curve.FirstParameter
        delta = param_range * delta_fraction

        p0 = max(curve.FirstParameter, param - delta / 2)
        p1 = min(curve.LastParameter, p0 + delta)
        if p1 - p0 < 1e-15:
            return 0.0

        t0 = App.Vector(*curve.tangent(p0)[0])
        t1 = App.Vector(*curve.tangent(p1)[0])
        t0.normalize()
        t1.normalize()

        dt = t1 - t0
        angle = dt.Length  # for small angles, |ΔT| ≈ angle in radians

        # Arc length over [p0, p1]
        edge_segment = Part.Edge(curve, p0, p1)
        ds = edge_segment.Length

        if ds < 1e-15:
            return 0.0

        return angle / ds

    def check_curvature_compatibility(self, spine_edge, cylinder_radius, max_chord_length,
                                       min_inner_chord=0.25):
        """
        Check if the curve's curvature is compatible with the cylinder diameter
        by verifying that wafer inner chord stays above a minimum.

        The inner (concave-side) height of a wafer is approximately:
            inner_chord ≈ L × (1 - r × κ)
        where L is chord length, r is cylinder radius, κ is local curvature.

        At peak curvature κ_max, the best-case inner chord (using max chord length)
        must meet the minimum: max_chord × (1 - r × κ_max) ≥ min_inner_chord.

        Args:
            spine_edge: Part.Edge representing the spine
            cylinder_radius: Radius of the cylinder
            max_chord_length: Maximum chord length (after max_wafer_count adjustment)
            min_inner_chord: Minimum acceptable inner chord width (default 0.25)
        """
        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter
        num_samples = 200

        max_kappa = 0.0
        max_kappa_param = first_param

        for i in range(num_samples + 1):
            param = first_param + (last_param - first_param) * i / num_samples
            kappa = self._estimate_curvature(curve, param)
            if kappa > max_kappa:
                max_kappa = kappa
                max_kappa_param = param

        if max_kappa < 1e-10:
            return  # Straight or nearly straight — no issues

        min_radius_of_curvature = 1.0 / max_kappa
        rk = cylinder_radius * max_kappa
        best_inner_chord = max_chord_length * (1.0 - rk)

        if rk >= 1.0 or best_inner_chord < min_inner_chord:
            problem_point = curve.value(max_kappa_param)
            # Compute the max κ that would be compatible
            # max_chord * (1 - r*κ) >= min_inner → κ <= (1 - min_inner/max_chord) / r
            max_compatible_kappa = (1.0 - min_inner_chord / max_chord_length) / cylinder_radius
            min_compatible_radius = 1.0 / max_compatible_kappa if max_compatible_kappa > 0 else float('inf')

            raise RuntimeError(
                f"Curve curvature is incompatible with cylinder diameter.\n"
                f"  Peak curvature κ = {max_kappa:.4f} "
                f"(radius of curvature = {min_radius_of_curvature:.3f})\n"
                f"  Cylinder radius: {cylinder_radius:.3f}\n"
                f"  Best inner chord at peak: {best_inner_chord:.3f} "
                f"(minimum required: {min_inner_chord:.3f})\n"
                f"  Tightest bend at: ({problem_point.x:.2f}, {problem_point.y:.2f}, "
                f"{problem_point.z:.2f})\n"
                f"\n"
                f"Suggestions:\n"
                f"  - Reduce the curve amplitude (decreases peak curvature)\n"
                f"  - Increase the curve length/scale (spreads the bend)\n"
                f"  - Use a lower frequency (gentler curves)\n"
                f"  - Need radius of curvature >= {min_compatible_radius:.2f} "
                f"(currently {min_radius_of_curvature:.3f})"
            )

        logger.info(f"Curvature check passed: inner chord {best_inner_chord:.3f} "
                     f">= min {min_inner_chord:.3f} "
                     f"(κ_max={max_kappa:.4f}, R_min={min_radius_of_curvature:.3f})")

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

        cylinder_radius = wafer_settings.cylinder_radius

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

        target_chord = wafer_settings.max_chord
        max_wafer_count = wafer_settings.max_wafer_count
        min_inner_chord = wafer_settings.min_inner_chord

        # Compute effective max_chord_length (may be adjusted by max_wafer_count)
        spine_length = self.generator.spine_curve.Length
        max_chord_length = target_chord
        if max_wafer_count:
            min_chord_for_limit = spine_length / max_wafer_count
            if min_chord_for_limit > max_chord_length:
                max_chord_length = min_chord_for_limit

        # Check curvature compatibility before expensive loft creation
        self.check_curvature_compatibility(
            self.generator.spine_curve, cylinder_radius,
            max_chord_length, min_inner_chord
        )

        self.generator.create_loft_along_spine(self)

        if self.generator.loft is None:
            raise ValueError("Loft creation returned None")

        logger.info(f"Loft created successfully, volume: {self.generator.loft.Volume:.4f}")

        def limited_sampler(spine):
            chord_limit = target_chord
            if max_wafer_count:
                min_chord_for_limit = spine.Length / max_wafer_count
                if min_chord_for_limit > chord_limit:
                    logger.info(f"Adjusting max_chord_length from {chord_limit:.4f} to "
                                f"{min_chord_for_limit:.4f} to fit {max_wafer_count} wafers")
                    chord_limit = min_chord_for_limit

            return self.chord_distance_sampler(spine, chord_limit, target_chord)

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