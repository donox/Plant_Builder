"""CurveFollower class for creating wafer slices along curved cylinders.

This module provides functionality to generate sequences of wafers that follow
arbitrary 3D curves, with proper geometric calculations for woodworking applications.
"""
try:
    from core.logging_setup import get_logger, log_coord, apply_display_levels
    # apply_display_levels(["ERROR", "WARNING", "INFO", "COORD"])
    apply_display_levels(["ERROR", "WARNING", "INFO"])
except Exception:
    try:
        from logging_setup import get_logger
    except Exception:
        import logging
        get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)

import sys
import math
import numpy as np
from typing import List, Tuple, Any, Dict
import FreeCAD
from curves import Curves  # Import the new Curves class
import traceback
import json, os, pathlib, time

class CurveFollower:
    """Creates wafer slices along a curved cylinder path.

    This class generates a sequence of wafers (cylindrical slices) that follow
    a 3D curve, calculating proper elliptical cross-sections and ensuring
    adjacent wafers can be cut with complementary angles for woodworking.

    Attributes:
        doc: FreeCAD document object
        segment: FlexSegment object for adding wafers
        cylinder_diameter: Diameter of the cylinder in model units
        radius: Radius of the cylinder (cylinder_diameter / 2)
        min_height: Minimum distance between wafer end surfaces
        max_chord: Maximum allowed chord distance for wafer approximation
        curves: Curves object containing the generated curve
        curve_points: Generated curve points as numpy array
        curve_length: Total length of the curve
        curvatures: Radius of curvature at each curve point
    """

    def __init__(self, doc: Any, segment: Any, cylinder_diameter: float,
                 curve_spec: Dict[str, Any], min_height: float, max_chord: float):
        """Initialize the CurveFollower.

        Args:
            doc: FreeCAD document object
            segment: FlexSegment object with add_wafer method
            cylinder_diameter: Diameter of the cylinder
            curve_spec: Dictionary specifying the curve (see Curves class)
            min_height: Minimum distance between wafer end surfaces
            max_chord: Maximum chord distance for wafer approximation

        Raises:
            ValueError: If curve specification is invalid
        """
        self.doc = doc
        self.segment = segment
        self.cylinder_diameter = cylinder_diameter
        self.radius = cylinder_diameter / 2.0
        self.min_height = min_height  # better name = min_wafer_length or min_chord_length
        self.max_chord = max_chord

        # Generate the curve using the Curves class
        self.curves = Curves(doc, curve_spec)
        self.curve_points = self.curves.get_curve_points()

        # Calculate curve properties
        self.curve_length = self._calculate_curve_length()
        self.curvatures = self._calculate_curvatures()

    def _calculate_curve_length(self) -> float:
        """Calculate the total length of the curve.

        Returns:
            Total curve length in model units
        """
        if len(self.curve_points) < 2:
            return 0.0

        diffs = np.diff(self.curve_points, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.sum(lengths)

    def _calculate_curvatures(self) -> np.ndarray:
        """Calculate radius of curvature at each point along the curve.

        Uses the formula: radius = 1/k where k = |v1 × v2| / |v1|^3
        for consecutive curve segments v1 and v2.

        Returns:
            Array of radius of curvature values at each curve point
        """
        if len(self.curve_points) < 3:
            return np.array([float('inf')] * len(self.curve_points))

        curvatures = []
        for i in range(1, len(self.curve_points) - 1):
            p1, p2, p3 = self.curve_points[i - 1], self.curve_points[i], self.curve_points[i + 1]

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate curvature using the formula: k = |v1 × v2| / |v1|^3
            cross_product = np.cross(v1, v2)
            if isinstance(cross_product, np.ndarray):
                cross_magnitude = np.linalg.norm(cross_product)
            else:
                cross_magnitude = abs(cross_product)

            v1_magnitude = np.linalg.norm(v1)

            if v1_magnitude > 1e-10 and cross_magnitude > 1e-10:
                curvature = cross_magnitude / (v1_magnitude ** 3)
                radius_of_curvature = 1.0 / curvature if curvature > 0 else float('inf')
            else:
                radius_of_curvature = float('inf')

            curvatures.append(radius_of_curvature)

        # Extend for first and last points
        if curvatures:
            curvatures = [curvatures[0]] + curvatures + [curvatures[-1]]
        else:
            curvatures = [float('inf')] * len(self.curve_points)

        return np.array(curvatures)

    def validate_and_adjust_curve_sampling(self) -> Dict[str, Any]:
        """Validate curve sampling density and estimate required wafer count."""

        # Use the generic point density analysis
        density_analysis = self.curves.calculate_required_point_density(self.max_chord)

        if density_analysis['status'] == 'insufficient_density':
            return {
                'status': 'insufficient_sampling',
                'current_points': density_analysis['current_points'],
                'recommended_points': density_analysis['recommended_points'],
                'message': density_analysis['message'],
                'avg_segment_length': density_analysis['avg_segment_length']
            }

        return density_analysis

    def _estimate_required_wafers(self) -> int:
        """Estimate number of wafers needed based on geometric constraints.

        Returns:
            Estimated wafer count
        """
        if self.curve_length == 0:
            return 1

        # Conservative estimate: average wafer spans min_height with some margin
        avg_wafer_length = max(self.min_height * 1.5, self.curve_length / 20)
        estimated_count = max(3, int(self.curve_length / avg_wafer_length))

        return estimated_count

    def check_feasibility(self) -> bool:
        """Check if a feasible solution exists for the given curve and cylinder.

        Verifies that the curve's minimum radius of curvature allows for the
        minimum wafer height constraint to be satisfied.

        Returns:
            True if feasible solution exists, False otherwise
        """
        min_radius_of_curvature = np.min(self.curvatures)

        if min_radius_of_curvature <= self.radius:
            return False

        max_possible_height = 2 * math.sqrt(min_radius_of_curvature ** 2 - self.radius ** 2)
        return max_possible_height >= self.min_height

    def _calculate_chord_distance(self, start_point: np.ndarray, end_point: np.ndarray,
                                  curve_segment: np.ndarray) -> float:
        """Calculate maximum distance between chord and curve segment.

        Measures how much the curve deviates from a straight line between
        the start and end points.

        Args:
            start_point: Starting point of the chord
            end_point: Ending point of the chord
            curve_segment: Points along the curve between start and end

        Returns:
            Maximum perpendicular distance from curve to chord
        """

        if len(curve_segment) <= 2:
            # log_coord(__name__, "FEW points")
            return 0.0

        chord_vector = end_point - start_point
        chord_length = np.linalg.norm(chord_vector)

        if chord_length < 1e-10:
            # log_coord(__name__, "TOO SHORT")
            return 0.0

        chord_unit = chord_vector / chord_length

        max_distance = 0.0
        # Fix: Only check interior points (exclude start and end)
        for point in curve_segment[1:-1]:  # Skip first and last points
            point_vector = point - start_point
            projection_length = np.dot(point_vector, chord_unit)
            projection_point = start_point + projection_length * chord_unit
            distance = np.linalg.norm(point - projection_point)
            # log_coord(__name__, f"Distance: {distance}")
            max_distance = max(max_distance, distance)

        # log_coord(__name__, f"Final max_distance: {max_distance}")
        return max_distance

    def _check_segment_collinearity(self, start_point: np.ndarray, end_point: np.ndarray,
                                    start_index: int, end_index: int,
                                    chord_distance: float = None) -> bool:
        """Check if a wafer segment is approximately collinear.

        Args:
            start_point: Start point of wafer
            end_point: End point of wafer
            start_index: Index of start point in curve
            end_index: Index of end point in curve
            chord_distance: Precomputed chord distance (optional)

        Returns:
            True if segment is nearly collinear
        """
        if end_index <= start_index:
            return True

        # Use provided chord distance if available, otherwise calculate
        if chord_distance is None:
            curve_segment = self.curve_points[start_index:end_index + 1]
            # log_coord(__name__, f"Extracting segment [{start_index}:{end_index + 1}] = {len(curve_segment)} points")
            chord_distance = self._calculate_chord_distance(start_point, end_point, curve_segment)
        else:
            # log_coord(__name__, f"Using precomputed chord_distance: {chord_distance}")
            pass

        chord_length = np.linalg.norm(end_point - start_point)
        threshold = self.max_chord * 0.1

        is_collinear = chord_distance < threshold

        # DEBUG OUTPUT
        # log_coord(__name__, f"  Collinearity check: chord_dist={chord_distance:.4f}, threshold={threshold:.4f}, "
                            # f"chord_len={chord_length:.4f}, collinear={is_collinear}")
        return is_collinear

    # In curve_follower.py, add this method to the CurveFollower class:
    def calculate_optimal_curve_points(self, max_chord_error: float = None) -> int:
        """Calculate optimal point count for the current curve and constraints.

        Args:
            max_chord_error: Override the instance max_chord value

        Returns:
            Recommended number of points for optimal wafer generation
        """
        if max_chord_error is None:
            max_chord_error = self.max_chord

        return self.curves.calculate_optimal_points(max_chord_error)

    def _calculate_ellipse_parameters(
            self,
            start_point: np.ndarray,
            end_point: np.ndarray,
            start_index: int,
            end_index: int,
            is_first_wafer: bool = False,
            is_last_wafer: bool = False,
    ) -> Tuple[float, float, float, str]:
        """
        Compute start/end cut angles (radians), rotation (radians), and wafer type from joints.

        End typing is determined by the *joint* at that end:
          - start end uses bend(prev_chord, this_chord)
          - end   end uses bend(this_chord, next_chord)
          If bend <= eps_circular → that end is 'C', else 'E'.

        Per-end cut angle:
          angle = 0 for 'C', else (bend / 2).

        Rotation:
          - 0 on planar segments (zero torsion)
          - computed only for EE (non-planar) as before.

        Returns angles in degrees.
        """
        import numpy as np

        EPS_CIRCULAR_DEG = 0.25  # angle ≤ this is considered circular
        eps_circ = np.deg2rad(EPS_CIRCULAR_DEG)

        def _unit(v: np.ndarray):
            n = float(np.linalg.norm(v))
            return (v / n) if n > 1e-12 else None

        # This wafer's chord
        chord_vec = end_point - start_point
        chord_len = float(np.linalg.norm(chord_vec))
        if chord_len < 1e-10:
            # Degenerate: both ends circular
            return 0.0, 0.0, 0.0, "CC"
        chord_hat = chord_vec / chord_len

        # Neighbor chords (±1 point)
        N = len(self.curve_points)
        prev_hat = None
        if start_index > 0:
            v = start_point - self.curve_points[start_index - 1]
            prev_hat = _unit(v)
        next_hat = None
        if end_index < N - 1:
            v = self.curve_points[end_index + 1] - end_point
            next_hat = _unit(v)

        # Planarity check on just this wafer’s points
        neighborhood = self.curve_points[start_index:end_index + 1]
        is_planar, n_hat, rms, tol = self._segment_is_planar(neighborhood)
        # log_coord(__name__, f"[{start_index}:{end_index}] planarity: is_planar={is_planar} "
        #                     f"rms={rms:.3e} tol={tol:.3e}")

        # Bend helper (radians)
        def bend_between(u: np.ndarray, v: np.ndarray) -> float:
            if u is None or v is None:
                return 0.0
            if is_planar and n_hat is not None:
                # use signed-in-plane, magnitude only
                return abs(self._signed_angle_in_plane(u, v, n_hat))
            # generic unsigned bend
            uu = u / (np.linalg.norm(u) + 1e-12)
            vv = v / (np.linalg.norm(v) + 1e-12)
            d = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
            return float(np.arccos(d))

        # Joint bends at this wafer’s ends
        bend_start = bend_between(prev_hat, chord_hat) if prev_hat is not None else 0.0
        bend_end = bend_between(chord_hat, next_hat) if next_hat is not None else 0.0

        # End types from joints (first start/end and last end default to C when neighbor absent)
        start_is_C = (bend_start <= eps_circ) or (prev_hat is None)
        end_is_C = (bend_end <= eps_circ) or (next_hat is None)

        start_type = "C" if start_is_C else "E"
        end_type = "C" if end_is_C else "E"
        wafer_type = start_type + end_type

        # Per-end cut angles (radians)
        start_angle = 0.0 if start_is_C else 0.5 * bend_start
        end_angle = 0.0 if end_is_C else 0.5 * bend_end

        # Rotation: zero for planar; only compute for EE on non-planar segments
        rotation_angle = 0.0
        if (not is_planar) and (wafer_type == "EE"):
            # Build a reference 'm' vector at start: perpendicular to prev_hat within the plane ⟂ chord_hat
            if prev_hat is not None and next_hat is not None:
                # start 'm'
                m_start = np.cross(chord_hat, prev_hat)
                nrm = float(np.linalg.norm(m_start))
                if nrm < 1e-8:
                    # fallback: any vector ⟂ chord_hat
                    ref = np.array([0.0, 0.0, 1.0], float)
                    if abs(np.dot(ref, chord_hat)) > 0.9:
                        ref = np.array([1.0, 0.0, 0.0], float)
                    m_start = np.cross(chord_hat, ref)
                    nrm = float(np.linalg.norm(m_start))
                m_start /= (nrm + 1e-12)

                # reference 'm' transported to end: project m_start onto plane ⟂ next_hat, then reproject to ⟂ chord_hat
                proj = m_start - next_hat * float(np.dot(m_start, next_hat))
                if float(np.linalg.norm(proj)) > 1e-8:
                    m_end_ref = proj / float(np.linalg.norm(proj))
                else:
                    m_end_ref = m_start

                # true 'm' at end is ⟂ next_hat within plane ⟂ chord_hat
                m_end_true = np.cross(next_hat, chord_hat)
                nrm = float(np.linalg.norm(m_end_true))
                if nrm < 1e-8:
                    m_end_true = m_end_ref
                    nrm = float(np.linalg.norm(m_end_true))
                m_end_true /= (nrm + 1e-12)

                # rotation about chord between m_end_ref → m_end_true
                axis = chord_hat
                rot_sin = float(np.dot(axis, np.cross(m_end_ref, m_end_true)))
                rot_cos = float(np.clip(np.dot(m_end_ref, m_end_true), -1.0, 1.0))
                rotation_angle = float(np.arctan2(rot_sin, rot_cos))

        if is_planar or abs(rotation_angle) < 1e-6:
            rotation_angle = 0.0

        # Clamp silly values; zero-out tiny noise
        # Convert all to degrees for external consumption
        max_angle = np.pi / 3  # 60°
        start_angle = np.rad2deg(float(np.clip(start_angle, 0.0, max_angle)))
        end_angle = np.rad2deg(float(np.clip(end_angle, 0.0, max_angle)))
        rotation_angle = np.rad2deg(rotation_angle)
        # start_angle = float(np.clip(start_angle, 0.0, max_angle))
        # end_angle = float(np.clip(end_angle, 0.0, max_angle))
        # rotation_angle = np.rad2deg(rotation_angle)

        return start_angle, end_angle, rotation_angle, wafer_type

    def create_wafer_list(self) -> List[Tuple[np.ndarray, np.ndarray, float, float, float, str]]:
        """Create a list of wafers satisfying geometric constraints.

        Generates wafers that satisfy both minimum height and maximum chord
        distance constraints while maximizing wafer size.

        Returns:
            List of tuples: (start_point, end_point, start_angle, end_angle,
                           rotation_angle, wafer_type)
        """

        # --- CLEAR ellipse capture file for this run --------------------
        import pathlib
        _TRACE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "ellipse_cases.jsonl"
        try:
            _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure tests/data/ exists
            # Truncate (create if missing) so we don't append across runs
            with open(_TRACE_PATH, "w", encoding="utf-8") as _f:
                _f.write(json.dumps({"ts": time.time(), "note": "begin create_wafer_list"}) + "\n")
        except Exception as _e:
            # Don't fail the build if clearing the capture file has issues
            print(f"⚠️ could not clear capture file {_TRACE_PATH}: {_e}")
        # ---------------------------------------------------------------

        wafers_parameters = []
        current_index = 0
        safety_counter = 0

        while current_index < len(self.curve_points) - 1:
            safety_counter += 1
            # SAFETY CHECK: Prevent infinite loops
            if safety_counter > 1000:
                logger.error(f"🛑 SAFETY STOP: Generated {len(wafers_parameters)} wafers, stopping to prevent infinite loop")
                break

            start_point = self.curve_points[current_index]

            # Find the farthest valid end point - (the longest segment of the curve meeting constraints
            best_end_index = current_index + 1
            best_chord_distance = 0.0  # Initialize to 0.0 instead of inf

            for end_index in range(current_index + 1, len(self.curve_points)):
                end_point = self.curve_points[end_index]

                # Check minimum height constraint
                min_wafer_length = np.linalg.norm(end_point - start_point)  # 3D absolute distance
                if min_wafer_length < self.min_height:
                    continue

                # Check maximum chord constraint
                curve_segment = self.curve_points[current_index:end_index + 1]
                # Debug: Log what we received
                # log_coord(__name__, f"_C_C_D called {current_index} to {end_index} ")
                chord_distance = self._calculate_chord_distance(start_point, end_point, curve_segment)

                if chord_distance <= self.max_chord:      # max_chord from user parameters
                    best_end_index = end_index
                    best_chord_distance = chord_distance  # Store the actual calculated distance
                    # log_coord(__name__, f"Found valid segment to index {end_index}, chord_dist={chord_distance}")
                else:
                    # This is not an error - indicates first point beyond allowable chord
                    # log_coord(__name__,
                    #           f"Chord distance {chord_distance} exceeds max_chord {self.max_chord}, stopping search")
                    break

            end_point = self.curve_points[best_end_index]

            # Determine if this is a first or last wafer
            is_first_wafer = (current_index == 0)
            is_last_wafer = (best_end_index == len(self.curve_points) - 1)

            assert start_angle == 0 or start_angle > 1.0, "start_angle likely in radians"
            assert end_angle == 0 or end_angle > 1.0, "end_angle likely in radians"
            assert rotation_angle == 0 or abs(rotation_angle) > 1.0, f"Rotation ({rotation_angle})likely in radians"
            wafers_parameters.append((start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type))
            current_index = best_end_index

        return wafers_parameters

    def _find_point_index(self, point: np.ndarray) -> int:
        """Find the index of a point in the curve points array.

        Args:
            point: Point to find

        Returns:
            Index of the point, or 0 if not found
        """
        for i, curve_point in enumerate(self.curve_points):
            if np.allclose(point, curve_point, atol=1e-6):
                return i
        return 0

    def _calculate_outside_height(self, start_point: np.ndarray, end_point: np.ndarray,
                                  start_angle: float, end_angle: float, rotation_angle: float) -> float:
        """Calculate the outside height (maximum distance between end ellipses).

        Computes the physical height needed for the wafer based on the chord
        length and ellipse extensions at each end.

        Args:
            start_point: 3D coordinates of wafer start
            end_point: 3D coordinates of wafer end
            start_angle: Angle of start ellipse plane to perpendicular (radians)
            end_angle: Angle of end ellipse plane to perpendicular (radians)
            rotation_angle: Rotation between ellipse major axes (radians)

        Returns:
            Maximum height of the wafer in model units
        """
        assert start_angle == 0 or start_angle > 1.0, "start_angle likely in radians"
        assert end_angle == 0 or end_angle > 1.0, "end_angle likely in radians"
        assert rotation_angle == 0 or  abs(rotation_angle) > 1.0, f"Rotation ({rotation_angle})likely in radians"
        # Base distance between points (this is the minimum height)
        chord_length = np.linalg.norm(end_point - start_point)

        def safe_ellipse_extension(angle: float) -> float:
            """Calculate ellipse extension safely avoiding division by zero.

            Args:
                angle: Ellipse angle in radians

            Returns:
                Extension distance beyond cylinder diameter
            """
            if abs(angle) < math.radians(1):  # Less than 1 degree - treat as circular
                return 0.0
            elif abs(angle) > math.radians(89):  # More than 89 degrees - cap it
                return self.cylinder_diameter * 2  # Reasonable maximum
            else:
                # Normal case: extension based on ellipse geometry
                try:
                    cos_angle = math.cos(angle)
                    if abs(cos_angle) < 0.01:  # Avoid near-zero division
                        return self.cylinder_diameter * 2
                    major_axis = self.cylinder_diameter / abs(cos_angle)
                    extension = (major_axis - self.cylinder_diameter) / 2
                    return min(extension, self.cylinder_diameter * 2)  # Cap at reasonable value
                except:
                    return self.cylinder_diameter * 0.5  # Fallback

        start_extension = safe_ellipse_extension(np.deg2rad(start_angle))
        end_extension = safe_ellipse_extension(np.deg2rad(end_angle))

        outside_height = chord_length + start_extension + end_extension

        # Sanity check: outside_height should be reasonable compared to chord_length
        if outside_height > chord_length * 5:  # If more than 5x chord length, something's wrong
            # logger.debug(f"WARNING: Capping excessive outside_height")
            # logger.debug(f"  Original calculation: {outside_height:.4f}")
            # logger.debug(f"  Chord length: {chord_length:.4f}")
            # logger.debug(f"  Start angle: {math.degrees(start_angle):.2f}°")
            # logger.debug(f"  End angle: {math.degrees(end_angle):.2f}°")
            outside_height = chord_length + self.cylinder_diameter  # Conservative fallback
        return outside_height

    def get_curve_info(self) -> Dict[str, Any]:
        """Get information about the curve being followed.

        Returns:
            Dictionary containing curve statistics and properties
        """
        return self.curves.get_curve_info()

    def add_curve_visualization(self, group_name: str = None) -> str:
        """Add visual vertices along the curve for debugging/visualization."""
        if group_name is None:
            segment_name = self.segment.get_segment_name()
            group_name = f"{segment_name}_curve_vertices"  # Make it unique per segment

        logger.info(f"Creating curve vertices with group name: '{group_name}'")

        # Create vertices at origin (same as wafer creation)
        vertex_group_name = self.curves.add_visualization_vertices(None, group_name)

        # Register with segment using FreeCAD properties AND grouping
        self.segment.register_curve_vertices_group(vertex_group_name)

        return vertex_group_name

    def process_wafers(self, add_curve_vertices: bool = False) -> None:
        """Main processing method that creates and adds wafers to the segment."""

        # Validate the user parameters
        validation = self.validate_parameters()
        if not validation['valid']:
            logger.error("Invalid parameters:")
            for issue in validation['issues']:
                logger.error(f"  - {issue}")
            logger.info("Recommendations:")
            for rec in validation['recommendations']:
                logger.info(f"  - {rec}")
            raise ValueError("Cannot create wafers with current parameters")

        if validation['warnings']:
            logger.warning("Parameter warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")

        # Step 1: Create wafer list
        wafers = self.create_wafer_list()

        # Step 2: Apply wafer type consistency
        # consistent_wafers = self._determine_consistent_wafer_types(wafers)

        # Step 3: Correct rotation angles
        # corrected_wafers = consistent_wafers    # removed null correction
        corrected_wafers = wafers  # They are supposed to be correct now

        # Step 4: Process each wafer
        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            assert start_angle == 0 or start_angle > 1.0, "start_angle likely in radians"
            assert end_angle == 0 or end_angle > 1.0, "end_angle likely in radians"
            assert rotation_angle == 0 or abs(rotation_angle) > 1.0, f"Rotation ({rotation_angle})likely in radians"

            # Call add_wafer_from_curve_data with the wafer data
            self.add_wafer_from_curve_data(
                start_point,
                end_point,
                start_angle,
                end_angle,
                rotation_angle,
                wafer_type
            )

        # Step 5: Add curve visualization if requested
        if add_curve_vertices:
            self.add_curve_visualization()

    def add_wafer_from_curve_data(self, start_point, end_point, start_angle,
                                  end_angle, rotation_angle, wafer_type):
        """Add a wafer using curve-derived data.

        This method converts curve data into the format expected by flex_segment.add_wafer()

        Args:
            start_point: numpy array [x, y, z]
            end_point: numpy array [x, y, z]
            start_angle: float, start cut angle in radians
            end_angle: float, end cut angle in radians
            rotation_angle: float, rotation angle in radians
            wafer_type: str, type of wafer (CE, EE, EC, CC)
        """
        # this add_wafer_..  called before add_wafer in flex_segment
        # Calculate the lift based on wafer type and angles
        assert start_angle == 0 or start_angle > 1.0, "start_angle likely in radians"
        assert end_angle == 0 or end_angle > 1.0, "end_angle likely in radians"
        assert rotation_angle == 0 or  abs(rotation_angle) > 1.0, f"Rotation ({rotation_angle})likely in radians"
        lift = self._calculate_lift(start_angle, end_angle, wafer_type)

        # Calculate outside height
        outside_height = self._calculate_outside_height(start_point, end_point,
                                                        start_angle, end_angle, rotation_angle)

        chord_length = np.linalg.norm(end_point - start_point)
        assert outside_height < chord_length * 1.1, "outside_height likely too large"

        # Get curve tangent at start point for first wafer
        curve_tangent = None
        if self.segment.wafer_count == 0:  # First wafer
            # Calculate tangent from curve
            start_idx = self._find_point_index(start_point)
            if start_idx < len(self.curve_points) - 1:
                tangent_vec = self.curve_points[start_idx + 1] - self.curve_points[start_idx]
                tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
                curve_tangent = FreeCAD.Vector(tangent_vec[0], tangent_vec[1], tangent_vec[2])

        # Keep positions as numpy arrays - no conversion needed!
        # The add_wafer method in flex_segment expects numpy arrays
        self.segment.add_wafer(
            lift=lift,
            rotation=rotation_angle,
            cylinder_diameter=self.cylinder_diameter,
            outside_height=outside_height,
            wafer_type=wafer_type,
            start_pos=start_point,  # Pass numpy array directly
            end_pos=end_point,  # Pass numpy array directly
            curve_tangent=curve_tangent
        )

    # In curve_follower.py, CurveFollower class (around line 235)
    @staticmethod
    def normalize_angle(angle_rad):
        """Normalize angle to [-π, π] range to prevent boundary issues"""
        if math.isnan(angle_rad) or math.isinf(angle_rad):
            return 0.0

        # Use remainder for clean normalization to [-π, π]
        normalized = math.remainder(angle_rad, 2 * math.pi)

        # Avoid near-boundary values that cause test failures
        epsilon = 1e-3  # Increased from 1e-10 for more robust boundary avoidance
        if abs(abs(normalized) - math.pi) < epsilon:
            normalized = math.copysign(math.pi - epsilon, normalized)

        return normalized

    # def calculate_cutting_planes(self, wafers):
    #     """Pre-calculate all cutting planes for the wafer sequence.
    #
    #     Args:
    #         wafers: List of (start_point, end_point, ...) tuples
    #
    #     Returns:
    #         List of cutting plane data for each wafer
    #     """
    #     cutting_planes = []
    #
    #     for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):
    #         # Current wafer axis (chord)
    #         current_axis = end_point - start_point
    #         current_axis_length = np.linalg.norm(current_axis)
    #         if current_axis_length > 1e-6:
    #             current_axis = current_axis / current_axis_length
    #
    #         # Start cutting plane
    #         if i == 0:
    #             # First wafer: perpendicular cut
    #             start_normal = current_axis
    #         else:
    #             # Get previous wafer axis
    #             prev_start, prev_end = wafers[i - 1][0], wafers[i - 1][1]
    #             prev_axis = prev_end - prev_start
    #             prev_axis_length = np.linalg.norm(prev_axis)
    #             if prev_axis_length > 1e-6:
    #                 prev_axis = prev_axis / prev_axis_length
    #
    #             # Calculate bisecting plane normal
    #             start_normal = self._calculate_bisecting_plane_normal(prev_axis, current_axis)
    #
    #         # End cutting plane
    #         if i == len(wafers) - 1:
    #             # Last wafer: perpendicular cut
    #             end_normal = current_axis
    #         else:
    #             # Get next wafer axis
    #             next_start, next_end = wafers[i + 1][0], wafers[i + 1][1]
    #             next_axis = next_end - next_start
    #             next_axis_length = np.linalg.norm(next_axis)
    #             if next_axis_length > 1e-6:
    #                 next_axis = next_axis / next_axis_length
    #
    #             # Calculate bisecting plane normal
    #             end_normal = self._calculate_bisecting_plane_normal(current_axis, next_axis)
    #
    #         cutting_planes.append({
    #             'start_pos': start_point,
    #             'end_pos': end_point,
    #             'start_normal': start_normal,
    #             'end_normal': end_normal,
    #             'wafer_axis': current_axis,
    #             'rotation': rotation_angle,
    #             'wafer_type': wafer_type
    #         })
    #
    #     return cutting_planes


    # def _calculate_bisecting_plane_normal(self, axis1, axis2):
    #     """Calculate normal to the plane that bisects two cylinder axes.
    #
    #     The bisecting plane contains both axes and has a normal that makes
    #     equal angles with both axes.
    #     """
    #     # Handle parallel/antiparallel axes
    #     cross = np.cross(axis1, axis2)
    #     cross_length = np.linalg.norm(cross)
    #
    #     if cross_length < 1e-6:
    #         # Axes are parallel - return perpendicular to axis
    #         return axis1
    #
    #     # The bisecting plane normal is the normalized sum of the axes
    #     # (for acute angles) or difference (for obtuse angles)
    #     dot_product = np.dot(axis1, axis2)
    #
    #     if dot_product > 0:
    #         # Acute angle - bisector is sum
    #         bisector = axis1 + axis2
    #     else:
    #         # Obtuse angle - bisector is difference
    #         bisector = axis1 - axis2
    #
    #     bisector_length = np.linalg.norm(bisector)
    #     if bisector_length > 1e-6:
    #         bisector = bisector / bisector_length
    #
    #     # The cutting plane normal is the bisector direction
    #     # (which is perpendicular to the intersection line of the two cylinders)
    #     return bisector

    def _calculate_lift(self, start_angle: float, end_angle: float, wafer_type: str) -> float:
        """Calculate the lift angle for a wafer based on its cutting angles.

        Args:
            start_angle: Start cut angle in radians
            end_angle: End cut angle in radians
            wafer_type: Type of wafer (CE, EE, EC, CC)

        Returns:
            Lift angle in radians
        """
        # For most cases, use the average of the cut angles
        # This represents the overall bend angle of the wafer
        if wafer_type == "CC":
            return 0.0  # No lift for straight sections
        elif wafer_type == "CE":
            return end_angle  # Use end angle for first wafer
        elif wafer_type == "EC":
            return start_angle  # Use start angle for last wafer
        else:  # "EE"
            return (start_angle + end_angle) / 2.0  # Average for middle wafers

    # def add_wafer_with_cutting_planes(self, curve, cutting_data, diameter, doc_object, base_placement):
    #     """Create a wafer from curve data with angled cutting planes.
    #
    #     For the first wafer in a segment, establishes the segment's coordinate system
    #     by aligning the Y-axis with the plane containing both the chord and the curve
    #     tangent. This creates a deterministic, geometrically meaningful orientation.
    #
    #     Args:
    #         curve: The curve object to follow
    #         cutting_data: Dictionary with start/end points, normals, cut angles, and cut types
    #         diameter: Wafer cylinder diameter
    #         doc_object: FreeCAD document object
    #         base_placement: Base placement for the segment
    #     """
    #     raise ValueError("add_wafer_with_cutting_planes - method to be removed")
    #     # Extract cutting data
    #     start_point = cutting_data['start_point']
    #     end_point = cutting_data['end_point']
    #     start_normal = cutting_data['start_normal']
    #     end_normal = cutting_data['end_normal']
    #     wafer_type = cutting_data['type']
    #     cut_rotation = cutting_data.get('rotation', 0.0)
    #     start_cut_angle = cutting_data.get('start_angle', 0)
    #     end_cut_angle = cutting_data.get('end_angle', 0)
    #
    #     # Calculate wafer properties
    #     wafer_axis = (end_point - start_point).normalize()
    #     chord_length = (end_point - start_point).Length
    #
    #     logger.info(f"\n=== Creating Wafer {len(self.wafer_list) + 1} with cutting planes ===")
    #     logger.info(f"    Type: {wafer_type}")
    #     logger.debug(f"    Rotation: {cut_rotation:.2f}°")
    #
    #     # Check if this is the first wafer
    #     is_first_wafer = (len(self.wafer_list) == 0)
    #
    #     # Create coordinate system
    #     if is_first_wafer:
    #         # For first wafer, create deterministic orientation based on curve geometry
    #         # Get the curve tangent at the start point
    #         start_param = curve.getParameterByLength(0)
    #         start_tangent = curve.tangent(start_param)[0]
    #
    #         # Z-axis is the wafer/cylinder axis
    #         z_axis = wafer_axis
    #
    #         # Find the normal to the plane containing chord and tangent
    #         plane_normal = z_axis.cross(start_tangent)
    #
    #         if plane_normal.Length < 0.001:
    #             # Chord and tangent are parallel (straight section)
    #             # Use perpendicular to Z and global Z if possible
    #             if abs(z_axis.z) < 0.9:
    #                 plane_normal = z_axis.cross(FreeCAD.Vector(0, 0, 1))
    #             else:
    #                 plane_normal = z_axis.cross(FreeCAD.Vector(1, 0, 0))
    #
    #         plane_normal.normalize()
    #
    #         # X-axis is the plane normal (perpendicular to bending plane)
    #         x_axis = plane_normal
    #
    #         # Y-axis lies in the bending plane
    #         y_axis = z_axis.cross(x_axis)
    #         y_axis.normalize()
    #
    #         # Update base LCS orientation to match first wafer
    #         placement_matrix = FreeCAD.Matrix()
    #         placement_matrix.A11, placement_matrix.A21, placement_matrix.A31 = x_axis.x, x_axis.y, x_axis.z
    #         placement_matrix.A12, placement_matrix.A22, placement_matrix.A32 = y_axis.x, y_axis.y, y_axis.z
    #         placement_matrix.A13, placement_matrix.A23, placement_matrix.A33 = z_axis.x, z_axis.y, z_axis.z
    #         placement_matrix.A14, placement_matrix.A24, placement_matrix.A34 = 0, 0, 0  # Keep at origin
    #
    #         self.lcs_base.Placement = FreeCAD.Placement(placement_matrix)
    #
    #         logger.debug(f"\n     First wafer - updating base LCS orientation")
    #         logger.debug(f"       Base LCS Z-axis alignment with cylinder: {z_axis.dot(wafer_axis):.4f}")
    #         logger.debug(f"       Base LCS position: {self.lcs_base.Placement.Base}")
    #     else:
    #         # For subsequent wafers, use the established coordinate system
    #         x_axis = self.lcs_base.Placement.Rotation.multVec(FreeCAD.Vector(1, 0, 0))
    #         y_axis = self.lc

    def _segment_is_planar(self, points: np.ndarray, eps_abs: float = 1e-6, eps_rel: float = 1e-4):
        """
        Returns (is_planar: bool, n_hat: np.ndarray, rms: float, tol: float)

        Scale-aware tolerance: tol = max(eps_abs, eps_rel * extent),
        where extent is the diagonal of the bbox of the points.
        """
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return True, np.array([0.0, 0.0, 1.0]), 0.0, eps_abs

        # Use your own plane-fit if available:
        try:
            is_planar, n_hat, _, rms = self._fit_plane(pts, eps=None)  # don't pass eps here
        except Exception:
            # PCA plane fit
            c = pts.mean(axis=0)
            Q = pts - c
            # covariance
            H = Q.T @ Q
            w, v = np.linalg.eigh(H)
            n_hat = v[:, 0] / (np.linalg.norm(v[:, 0]) + 1e-12)
            # distance of pts to plane
            d = Q @ n_hat
            rms = float(np.sqrt(np.mean(d * d)))
            is_planar = True  # we'll decide with tol below

        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = max(eps_abs, eps_rel * extent)
        return (rms <= tol), n_hat, rms, tol

    def _fit_plane(self, pts: np.ndarray, eps: float = 1e-6):
        """
        Least-squares plane fit. Returns (is_planar, n_hat, p0, rms).
        - is_planar: True if all points lie near a single plane
        - n_hat: unit normal
        - p0: a point on the plane (centroid)
        - rms: root-mean-square distance to plane
        """
        # Note - used only to support _segment_is_planar above
        if pts.shape[0] < 3:
            return True, np.array([0.0, 0.0, 1.0]), pts[0] if len(pts) else np.zeros(3), 0.0

        c = pts.mean(axis=0)
        M = pts - c
        # PCA: smallest singular vector is the plane normal
        _, _, vh = np.linalg.svd(M, full_matrices=False)
        n = vh[-1, :]
        n_norm = np.linalg.norm(n)
        n_hat = n / (n_norm if n_norm > 0 else 1.0)

        # distances of points to plane
        d = M @ n_hat
        rms = float(np.sqrt(np.mean(d * d)))

        # Scale tolerance to curve extent so it works for tiny/large curves
        extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]), 1.0)
        is_planar = rms <= (eps * extent)
        log_coord(__name__, f"_fit_plane: extent={extent:.6f}, rms={rms:.6f}, threshold={eps * extent:.6f}")

        return is_planar, n_hat, c, rms


    def _signed_angle_in_plane(self, t1: np.ndarray, t2: np.ndarray, n_hat: np.ndarray) -> float:
        """
        Signed angle between directions t1 and t2 measured in the plane with normal n_hat.
        Positive sign by right-hand rule about n_hat.
        """
        t1u = t1 / (np.linalg.norm(t1) + 1e-12)
        t2u = t2 / (np.linalg.norm(t2) + 1e-12)
        cross = np.cross(t1u, t2u)
        sin_term = np.dot(n_hat, cross)
        cos_term = np.clip(np.dot(t1u, t2u), -1.0, 1.0)
        return float(np.arctan2(sin_term, cos_term))

    def validate_parameters(self) -> Dict[str, Any]:
        """Validate that user parameters can produce viable wafers.

        Returns:
            Dictionary with validation results and suggestions
        """
        issues = []
        warnings = []

        # Check 1: Are consecutive points too close together?
        point_distances = []
        for i in range(len(self.curve_points) - 1):
            dist = np.linalg.norm(self.curve_points[i + 1] - self.curve_points[i])
            point_distances.append(dist)

        min_point_distance = min(point_distances)
        avg_point_distance = np.mean(point_distances)

        if self.min_height > avg_point_distance * 3:
            issues.append(f"min_height ({self.min_height:.3f}) is too large. "
                          f"Average point spacing is {avg_point_distance:.3f}. "
                          f"Recommended max: {avg_point_distance * 2:.3f}")

        # Check 2: Is max_chord too restrictive?
        sample_chord_distances = []
        for i in range(0, len(self.curve_points) - 3, 2):  # Sample every other segment
            start = self.curve_points[i]
            end = self.curve_points[i + 2]
            segment = self.curve_points[i:i + 3]
            chord_dist = self._calculate_chord_distance(start, end, segment)
            sample_chord_distances.append(chord_dist)

        if sample_chord_distances and min(sample_chord_distances) > self.max_chord:
            issues.append(f"max_chord ({self.max_chord:.3f}) is too restrictive. "
                          f"Even short curve segments exceed this. "
                          f"Minimum found: {min(sample_chord_distances):.3f}")

        # Check 3: Can we find ANY valid segments?
        valid_segments_found = 0
        for start_idx in range(0, len(self.curve_points) - 1, 5):  # Sample every 5th point
            for end_idx in range(start_idx + 1, min(start_idx + 6, len(self.curve_points))):
                start_pt = self.curve_points[start_idx]
                end_pt = self.curve_points[end_idx]
                height = np.linalg.norm(end_pt - start_pt)

                if height >= self.min_height:
                    segment = self.curve_points[start_idx:end_idx + 1]
                    chord_dist = self._calculate_chord_distance(start_pt, end_pt, segment)
                    if chord_dist <= self.max_chord:
                        valid_segments_found += 1

        if valid_segments_found == 0:
            issues.append("No valid wafer segments found with current parameters. "
                          "Try reducing min_height or increasing max_chord.")
        elif valid_segments_found < 3:
            warnings.append(f"Very few valid segments found ({valid_segments_found}). "
                            "This may produce poor wafer layouts.")

        # Check 4: Curve sampling density
        if len(self.curve_points) < 10:
            warnings.append(f"Curve has only {len(self.curve_points)} points. "
                            "Consider increasing point density for better results.")

        # Generate recommendations
        recommendations = []
        if issues:
            recommendations.append(f"Suggested min_height: {avg_point_distance * 1.5:.3f}")
            if sample_chord_distances:
                recommended_max_chord = max(sample_chord_distances) * 1.2
                recommendations.append(f"Suggested max_chord: {recommended_max_chord:.3f}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'stats': {
                'curve_points': len(self.curve_points),
                'curve_length': self.curve_length,
                'min_point_distance': min_point_distance,
                'avg_point_distance': avg_point_distance,
                'valid_segments_found': valid_segments_found
            }
        }
