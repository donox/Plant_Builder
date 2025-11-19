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
import curves  # Use existing curves module
from wafer_loft import LoftWaferGenerator, simple_chord_distance_sampler


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
        print(f"Loading curve from {filename}...")

        points = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                try:
                    coords = [float(x) for x in line.split()]
                    if len(coords) != 3:
                        print(f"  Warning: Line {line_num} does not have 3 coordinates, skipping")
                        continue

                    points.append(App.Vector(coords[0], coords[1], coords[2]))

                except ValueError as e:
                    print(f"  Warning: Could not parse line {line_num}: {e}")
                    continue

        self.curve_points = points
        print(f"✓ Loaded {len(points)} points from {filename}")
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
        print(f"Generating {curve_type} curve using curves module...")

        # Get or create document
        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("CurveGen")

        # Use the curves module's unified interface
        points = curves.generate_curve(curve_type, doc=doc, **params)

        self.curve_points = points
        print(f"✓ Generated {len(points)} points")
        return points

    def set_curve_points(self, points):
        """
        Set curve points directly

        Args:
            points: List of App.Vector points
        """
        self.curve_points = points
        print(f"Set {len(points)} curve points")

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

        print(f"Creating spine curve from {len(points)} points...")

        spline = Part.BSplineCurve()
        spline.interpolate(points)
        self.spine_curve = spline.toShape()

        print(f"✓ Spine curve created, length: {self.spine_curve.Length:.3f}")
        return self.spine_curve

    # In curve_follower_loft.py

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

        print(f"Chord deviation sampling: max_deviation={max_chord_deviation:.3f}")

        params = [first_param]
        current_param = first_param

        while current_param < last_param:
            # Binary search for the farthest parameter where deviation <= max
            param_low = current_param
            param_high = last_param

            best_param = current_param

            for iteration in range(25):  # Binary search iterations
                test_param = (param_low + param_high) / 2

                if test_param <= current_param or test_param >= last_param:
                    best_param = last_param
                    break

                # Calculate maximum deviation between current_param and test_param
                max_deviation = self._calculate_max_chord_deviation(
                    curve, current_param, test_param
                )

                if max_deviation <= max_chord_deviation:
                    # Deviation is acceptable, try going farther
                    best_param = test_param
                    param_low = test_param
                else:
                    # Deviation too large, need to go closer
                    param_high = test_param

                # Stop if range is very small
                if abs(param_high - param_low) < (last_param - first_param) / 10000:
                    break

            if best_param <= current_param:
                # Couldn't find a valid next point, take a small step
                best_param = min(current_param + (last_param - first_param) / 100, last_param)

            params.append(best_param)
            current_param = best_param

            if best_param >= last_param:
                break

        # Ensure we end at the last parameter
        if params[-1] < last_param:
            params[-1] = last_param

        print(f"  ✓ Sampled {len(params)} points")
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

        # Chord vector
        chord_vector = end_point - start_point
        chord_length = chord_vector.Length

        if chord_length < 1e-10:
            return 0.0

        chord_direction = chord_vector.normalize()

        max_deviation = 0.0

        # Sample points along the curve segment
        for i in range(1, num_samples):
            t = param_start + (param_end - param_start) * i / num_samples
            curve_point = curve.value(t)

            # Vector from start to curve point
            to_curve = curve_point - start_point

            # Project onto chord direction
            projection_length = to_curve.dot(chord_direction)

            # Find closest point on chord
            if projection_length <= 0:
                closest_on_chord = start_point
            elif projection_length >= chord_length:
                closest_on_chord = end_point
            else:
                closest_on_chord = start_point + chord_direction * projection_length

            # Perpendicular deviation
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

        print(f"Uniform sampling: {num_samples} samples")

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

        print(f"Arc-length sampling: target={target_arc_distance:.3f}, {num_samples} samples")

        curve = spine_edge.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter

        params = []

        for i in range(num_samples):
            target_length = total_length * i / (num_samples - 1)

            # Binary search for parameter at target arc length
            param_low = first_param
            param_high = last_param

            for _ in range(20):  # Binary search iterations
                param_mid = (param_low + param_high) / 2

                # Get arc length from start to mid
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

    # In curve_follower_loft.py, in the generate_loft_wafers() method

    def generate_loft_wafers(self, curve_spec, wafer_settings):
        """
        Generate wafers using loft-based approach

        Args:
            curve_spec: Dictionary with curve type and parameters
            wafer_settings: Dictionary with wafer configuration
        """
        # Store wafer settings
        self.wafer_settings = wafer_settings

        # Extract cylinder diameter from wafer_settings
        cylinder_diameter = wafer_settings.get('cylinder_diameter', 1.875)
        cylinder_radius = cylinder_diameter / 2.0

        # Create the loft generator with just the cylinder radius
        self.generator = LoftWaferGenerator(cylinder_radius=cylinder_radius)

        # Generate curve points using the curve_spec
        curve_type = curve_spec.get('type')
        curve_params = curve_spec.get('parameters', {})

        # Get or create document
        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("CurveGen")

        points = self.load_curve_from_curves_module(curve_type, **curve_params)

        # Create spine and loft with ADAPTIVE profile count
        self.generator.create_spine_from_points(points)
        self.generator.create_loft_along_spine()

        # Verify loft was created
        if self.generator.loft is None:
            raise ValueError("Loft creation returned None")

        print(f"✓ Loft created successfully, volume: {self.generator.loft.Volume:.4f}")

        # Sample and create wafers
        target_chord = wafer_settings.get('max_chord', 0.5)
        max_wafer_count = wafer_settings.get('max_wafer_count', None)

        # Create sampler with limits
        def limited_sampler(spine):
            params = self.chord_distance_sampler(spine, target_chord)

            # Apply max wafer count limit - just take the first N wafers
            if max_wafer_count and len(params) > max_wafer_count + 1:
                print(f"  Limiting to {max_wafer_count} wafers (from {len(params) - 1} total samples)")
                params = params[:max_wafer_count + 1]  # First N+1 points = N wafers

            return params

        self.generator.sample_points_along_loft(limited_sampler)

        print(f"  Creating {len(self.generator.sample_points) - 1} wafers")

        self.generator.calculate_cutting_planes()

        # Generate wafers
        wafers = self.generator.generate_wafers()

        # Calculate bounding box for output files (if needed by driver)
        if wafers and len(wafers) > 0:
            x_coords = []
            for wafer in wafers:
                if wafer.wafer is not None:
                    bbox = wafer.wafer.BoundBox
                    x_coords.extend([bbox.XMin, bbox.XMax])

            if x_coords:
                self.x_min = min(x_coords)
                self.x_max = max(x_coords)
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

            # Calculate bounding box
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
        print("Visualizing curve...")

        if show_points and self.curve_points:
            # Show points as small spheres
            for i, point in enumerate(self.curve_points):
                sphere = Part.makeSphere(0.1, point)
                point_obj = doc.addObject("Part::Feature", f"CurvePoint_{i:03d}")
                point_obj.Shape = sphere
                point_obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)

        if show_curve and self.spine_curve:
            # Show the curve
            curve_obj = doc.addObject("Part::Feature", "Spine_Curve")
            curve_obj.Shape = self.spine_curve
            curve_obj.ViewObject.LineColor = (0.0, 0.5, 1.0)
            curve_obj.ViewObject.LineWidth = 3

        doc.recompute()
        print("✓ Curve visualization complete")

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

    Example:
        config = {
            'type': 'helix',
            'radius': 5.0,
            'pitch': 2.0,
            'num_turns': 3,
            'points_per_turn': 50
        }
        points = get_curve_points_from_curves_module(config)
    """
    curve_type = curve_config.get('type')
    if not curve_type:
        raise ValueError("curve_config must have 'type' key")

    # Remove 'type' from params
    params = {k: v for k, v in curve_config.items() if k != 'type'}

    follower = CurveFollowerLoft()
    return follower.load_curve_from_curves_module(curve_type, **params)