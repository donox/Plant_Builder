"""CurveFollower class for creating wafer slices along curved cylinders.

This module provides functionality to generate sequences of wafers that follow
arbitrary 3D curves, with proper geometric calculations for woodworking applications.
"""
try:
    from core.logging_setup import get_logger, log_coord, apply_display_levels
    apply_display_levels(["ERROR", "WARNING", "INFO"])
    # apply_display_levels(["ERROR", "WARNING", "INFO"])
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
        Compute start/end cut angles (DEGREES), rotation (DEGREES), and wafer type.

        End typing is determined by the joint bend at each end:
          - start end uses bend(prev_tangent, chord_hat)
          - end   end uses bend(chord_hat, next_tangent)
          If bend <= eps_circ -> that end is 'C', else 'E'.

        Per-end cut angle:
          angle = 0 for 'C', else (bend / 2).

        Rotation:
          - 0 on planar segments (zero torsion)
          - otherwise, for EE only, signed twist about the chord between
            a transported in-plane reference at the end and the true end frame.

        Returns: (start_deg, end_deg, rotation_deg, wafer_type)
        """
        import numpy as np

        # ---------- helpers ----------
        def _unit(v: np.ndarray):
            n = float(np.linalg.norm(v))
            return (v / n) if n > 1e-12 else None

        def _angle(u: np.ndarray, v: np.ndarray) -> float:
            """Unsigned angle between u and v in RADIANS."""
            if u is None or v is None:
                return 0.0
            uu = u / (np.linalg.norm(u) + 1e-12)
            vv = v / (np.linalg.norm(v) + 1e-12)
            c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
            return float(np.arccos(c))

        def _tangent(points: np.ndarray, idx: int, k: int = 3):
            """
            Symmetric, windowed tangent at idx using k points each side (clamped).
            Returns unit vector or None if undefined.
            """
            n = len(points)
            i0 = max(0, idx - k)
            i1 = min(n - 1, idx + k)
            if i1 == i0:
                return None
            v = points[i1] - points[i0]
            nv = float(np.linalg.norm(v))
            if nv > 0:
                return v / nv
            # gentle fallbacks
            if idx + 1 < n:
                v = points[idx + 1] - points[idx]
                nv = float(np.linalg.norm(v))
                if nv > 0:
                    return v / nv
            if idx - 1 >= 0:
                v = points[idx] - points[idx - 1]
                nv = float(np.linalg.norm(v))
                if nv > 0:
                    return v / nv
            return None

        # ---------- chord and degenerate guard ----------
        chord_vec = end_point - start_point
        chord_len = float(np.linalg.norm(chord_vec))
        if chord_len < 1e-10:
            if 'log_coord' in globals():
                log_coord(__name__, "ELL_PARMS: degenerate chord -> CC, angles 0")
            return 0.0, 0.0, 0.0, "CC"
        chord_hat = chord_vec / chord_len

        # ---------- windowed tangents at ends ----------
        prev_hat = _tangent(self.curve_points, start_index, k=3)
        next_hat = _tangent(self.curve_points, end_index, k=3)

        # ---------- stable planarity neighborhood ----------
        wpad = 3
        N = len(self.curve_points)
        i0 = max(0, start_index - wpad)
        i1 = min(N - 1, end_index + wpad)
        neighborhood = self.curve_points[i0:i1 + 1]

        is_planar, n_hat, rms, tol = self._segment_is_planar(neighborhood)

        # ---------- adaptive circular threshold (in RADIANS) ----------
        # Estimate local turn near start/end and set eps as a small fraction.
        t_s_l = _tangent(self.curve_points, max(0, start_index - 1), k=3)
        t_s_r = _tangent(self.curve_points, min(N - 1, start_index + 1), k=3)
        t_e_l = _tangent(self.curve_points, max(0, end_index - 1), k=3)
        t_e_r = _tangent(self.curve_points, min(N - 1, end_index + 1), k=3)
        turn_start = _angle(t_s_l, t_s_r)
        turn_end = _angle(t_e_l, t_e_r)
        typ_turn = max(turn_start, turn_end, np.deg2rad(0.03))  # never below 0.03°
        eps_circ = max(typ_turn * 0.15, np.deg2rad(0.05))  # 5–15% of local turn, min 0.05°

        # ---------- joint bends (in RADIANS) ----------
        def bend_between(u: np.ndarray, v: np.ndarray) -> float:
            if u is None or v is None:
                return 0.0
            if is_planar and n_hat is not None:
                # use signed-in-plane, magnitude only
                return abs(self._signed_angle_in_plane(u, v, n_hat))
            return _angle(u, v)

        bend_start = bend_between(prev_hat, chord_hat) if prev_hat is not None else 0.0
        bend_end = bend_between(chord_hat, next_hat) if next_hat is not None else 0.0

        # ---------- end typing ----------
        start_is_C = (bend_start <= eps_circ) or (prev_hat is None)
        end_is_C = (bend_end <= eps_circ) or (next_hat is None)
        start_type = "C" if start_is_C else "E"
        end_type = "C" if end_is_C else "E"
        wafer_type = start_type + end_type

        # ---------- per-end cut angles (in RADIANS) ----------
        start_angle = 0.0 if start_is_C else 0.5 * bend_start
        end_angle = 0.0 if end_is_C else 0.5 * bend_end

        # ---------- rotation (about the chord) ----------
        # Compute a single, self-consistent twist measure and snap tiny noise.
        rotation_deg = 0.0
        if (not is_planar) and (wafer_type == "EE") and (prev_hat is not None) and (next_hat is not None):
            # start reference m ⟂ chord_hat and ⟂ prev_hat
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

            # transport m_start to end: remove next_hat component
            tmp = m_start - next_hat * float(np.dot(m_start, next_hat))
            m_end_ref = (tmp / float(np.linalg.norm(tmp))) if float(np.linalg.norm(tmp)) > 1e-10 else m_start

            # true end 'm' ⟂ next_hat within plane ⟂ chord_hat
            m_end_true = np.cross(next_hat, chord_hat)
            nrm = float(np.linalg.norm(m_end_true))
            if nrm < 1e-8:
                m_end_true = m_end_ref
                nrm = float(np.linalg.norm(m_end_true))
            m_end_true /= (nrm + 1e-12)

            # signed angle about chord
            sin_term = float(np.dot(chord_hat, np.cross(m_end_ref, m_end_true)))
            cos_term = float(np.clip(np.dot(m_end_ref, m_end_true), -1.0, 1.0))
            angle_rad = float(np.arctan2(sin_term, cos_term))
            rotation_deg = float(np.rad2deg(angle_rad))

            # snap tiny noise only; do NOT mix with any other torsion proxy
            ROT_EPS_DEG = 0.10
            if abs(rotation_deg) < ROT_EPS_DEG:
                rotation_deg = 0.0

            log_coord(__name__,
                      (f"ROT about chord: start_idx={start_index} end_idx={end_index} "
                       f"type={wafer_type} planar={is_planar} "
                       f"sin={sin_term:.6f} cos={cos_term:.6f} "
                       f"angle_deg={rotation_deg:.3f}"))

        if 'log_coord' in globals():
            log_coord(__name__,
                      f"HATS: prev_hat: {prev_hat}, next_hat: {next_hat}, planar: {is_planar}, type: {wafer_type}")

        # ---------- clamp, convert to DEGREES, return ----------
        max_angle = np.pi / 3  # 60°
        start_deg = float(np.rad2deg(np.clip(start_angle, 0.0, max_angle)))
        end_deg = float(np.rad2deg(np.clip(end_angle, 0.0, max_angle)))
        rotation_out = float(rotation_deg)  # already degrees

        # ---------- diagnostics ----------
        # if 'log_coord' in globals():
        #     log_coord(__name__,
        #               (f"ELL_PARMS i[{start_index}:{end_index}] type={wafer_type} "
        #                f"bendS={np.rad2deg(bend_start):.3f}° bendE={np.rad2deg(bend_end):.3f}° "
        #                f"epsC={np.rad2deg(eps_circ):.3f}° "
        #                f"planar={is_planar} rms={rms:.3e} tol={tol:.3e} "
        #                f"rot={rotation_out:.3f}°"))
        # elif hasattr(self, "logger"):
        #     try:
        #         self.logger.debug(
        #             "ELL_PARMS i[%d:%d] type=%s bendS=%.3f° bendE=%.3f° epsC=%.3f° planar=%s rms=%.3e tol=%.3e rot=%.3f°",
        #             start_index, end_index, wafer_type,
        #             float(np.rad2deg(bend_start)), float(np.rad2deg(bend_end)),
        #             float(np.rad2deg(eps_circ)),
        #             bool(is_planar), float(rms), float(tol), float(rotation_out)
        #         )
        #     except Exception:
        #         pass

        return start_deg, end_deg, rotation_out, wafer_type

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
            if safety_counter > 200:
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

            # Call with hard try/except so exceptions can’t disappear
            try:
                # --- BEGIN capture wrapper for _calculate_ellipse_parameters ---    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # Choose a stable file under your repo (tests/data/)
                _TRACE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "ellipse_cases.jsonl"

                def _dump_case(payload: dict):
                    try:
                        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                        import json, os, pathlib, time, numpy as _np
                        with open(_TRACE_PATH, "a", encoding="utf-8") as _f:
                            _f.write(json.dumps(payload) + "\n")
                    except Exception:
                        # don't break the build if logging fails
                        pass

                # Build the input payload (include full curve_points so the test can replay exactly)
                _case_in = {
                    "ts": time.time(),
                    "start_point": list(map(float, np.asarray(start_point, dtype=float).tolist())),
                    "end_point": list(map(float, np.asarray(end_point, dtype=float).tolist())),
                    "start_index": int(current_index),
                    "end_index": int(best_end_index),
                    "is_first_wafer": bool(is_first_wafer),
                    "is_last_wafer": bool(is_last_wafer),
                    "curve_points": np.asarray(self.curve_points, dtype=float).tolist(),
                }

                try:
                    start_angle, end_angle, rotation_angle, wafer_type = self._calculate_ellipse_parameters(
                        start_point, end_point, current_index, best_end_index, is_first_wafer, is_last_wafer
                    )
                    _case_out = {
                        "ok": True,
                        "start_angle": float(start_angle),
                        "end_angle": float(end_angle),
                        "rotation_angle": float(rotation_angle),
                        "wafer_type": str(wafer_type),
                    }
                except Exception as _e:
                    # capture failure so the test can surface it explicitly
                    _case_out = {
                        "ok": False,
                        "error": f"{type(_e).__name__}: {str(_e)}",
                    }
                    # re-raise so normal behavior stays the same
                    _dump_case({**_case_in, **_case_out})
                    raise
                finally:
                    # Write only once per call (success or failure)
                    if "ok" in locals() or "_case_out" in locals():
                        _dump_case({**_case_in, **_case_out})
                # --- END capture wrapper ---                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

            except Exception :
                # Always print the traceback to real stderr so you see it even if logging is filtered
                print("EXC in _calculate_ellipse_parameters:", file=sys.__stderr__)
                traceback.print_exc(file=sys.stderr)
                raise

            # assert start_angle == 0 or start_angle > 1.0, "start_angle likely in radians"
            # assert end_angle == 0 or end_angle > 1.0, "end_angle likely in radians"
            # assert rotation_angle == 0 or abs(rotation_angle) > 1.0, f"Rotation ({rotation_angle})likely in radians"
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
                                  start_angle: float, end_angle: float, rotation_angle: float,
                                  wafer_type: str,
                                  start_bend_angle: float = None, end_bend_angle: float = None) -> Tuple[
        float, float, float]:
        """Calculate the outside height and individual extensions using bisecting plane geometry.

        Args:
            start_point: 3D coordinates of wafer start
            end_point: 3D coordinates of wafer end
            start_angle: Angle of start ellipse (degrees) - NOT USED with bisecting plane
            end_angle: Angle of end ellipse (degrees) - NOT USED with bisecting plane
            rotation_angle: Rotation between ellipses (degrees) - NOT USED
            wafer_type: Type of wafer (CE, EE, EC, CC)
            start_bend_angle: Bend angle at start interface (degrees)
            end_bend_angle: Bend angle at end interface (degrees)

        Returns:
            Tuple of (outside_height, start_extension, end_extension)
        """

        chord_length = np.linalg.norm(end_point - start_point)

        def calculate_bisecting_extension(bend_angle_rad: float, is_circular: bool) -> float:
            """Calculate extension for a bisecting plane cut.

            Args:
                bend_angle_rad: Bend angle in RADIANS (not degrees!)
                is_circular: If True, this is a circular cut (perpendicular), return 0
            """
            if is_circular:
                return 0.0

            if abs(bend_angle_rad) < 0.001:  # Nearly straight
                return 0.0

            half_angle_rad = bend_angle_rad / 2.0

            # Extension = R / tan(θ/2)
            try:
                tan_half = math.tan(half_angle_rad)
                if abs(tan_half) < 0.001:  # Avoid division by very small numbers
                    return self.cylinder_diameter * 2  # Large extension

                extension = self.radius * tan_half

                # Sanity cap
                max_extension = self.cylinder_diameter * 3
                return min(abs(extension), max_extension)
            except:
                return self.radius * 0.5

        # Calculate extensions based on wafer type and bend angles
        start_is_circular = (wafer_type[0] == 'C')
        end_is_circular = (wafer_type[1] == 'C')

        if start_bend_angle is not None:
            start_extension = calculate_bisecting_extension(start_bend_angle, start_is_circular)
            logger.debug(
                f"Start: bend={start_bend_angle:.2f}°, type={'C' if start_is_circular else 'E'}, ext={start_extension:.4f}")
        else:
            start_extension = 0.0

        if end_bend_angle is not None:
            end_extension = calculate_bisecting_extension(end_bend_angle, end_is_circular)
            logger.debug(
                f"End: bend={end_bend_angle:.2f}°, type={'C' if end_is_circular else 'E'}, ext={end_extension:.4f}")
        else:
            end_extension = 0.0

        outside_height = chord_length + start_extension + end_extension

        # Sanity check
        if outside_height > chord_length * 5:
            logger.warning(f"WARNING: Excessive outside_height: {outside_height:.4f} vs chord: {chord_length:.4f}")
            logger.warning(f"  Capping extensions")
            start_extension = min(start_extension, chord_length * 0.5)
            end_extension = min(end_extension, chord_length * 0.5)
            outside_height = chord_length + start_extension + end_extension

        return outside_height, start_extension, end_extension

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

        # Step 2: Pre-calculate all LCS orientations and bend angles
        lcs_orientations = self._precalculate_lcs_orientations(wafers)

        # Step 3: Process each wafer with pre-calculated orientations and bend angles
        for i, ((start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type), lcs_data) in enumerate(
                zip(wafers, lcs_orientations)):
            if i > 2:  # Remove this limit when testing is complete
                pass

            # Call add_wafer_from_curve_data with pre-calculated data
            self.add_wafer_from_curve_data(
                start_point,
                end_point,
                start_angle,
                end_angle,
                rotation_angle,
                wafer_type,
                lcs1_rotation=lcs_data['lcs1_rotation'],
                lcs2_rotation=lcs_data['lcs2_rotation'],
                start_bend_angle=lcs_data['start_bend_angle'],  # NEW
                end_bend_angle=lcs_data['end_bend_angle']  # NEW
            )

        # Step 4: Add curve visualization if requested
        if add_curve_vertices:
            self.add_curve_visualization()

    def add_wafer_from_curve_data(self, start_point, end_point, start_angle,
                                  end_angle, rotation_angle, wafer_type,
                                  lcs1_rotation=None, lcs2_rotation=None,
                                  start_bend_angle=None, end_bend_angle=None):  # NEW parameters
        """Add a wafer using curve-derived data with pre-calculated LCS orientations and bend angles."""

        # Calculate the lift based on wafer type and angles
        lift = self._calculate_lift(start_angle, end_angle, wafer_type)

        # Calculate outside height and individual extensions using bend angles
        outside_height, start_extension, end_extension = self._calculate_outside_height(
            start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type,
            start_bend_angle, end_bend_angle)  # NEW parameters

        logger.debug(
            f"curve_follower: add_wafer: outside_height: {outside_height}, start_ext: {start_extension}, end_ext: {end_extension}")

        # Get curve tangent at start point for first wafer
        curve_tangent = None
        if self.segment.wafer_count == 0:
            start_idx = self._find_point_index(start_point)
            if start_idx < len(self.curve_points) - 1:
                tangent_vec = self.curve_points[start_idx + 1] - self.curve_points[start_idx]
                tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
                curve_tangent = FreeCAD.Vector(tangent_vec[0], tangent_vec[1], tangent_vec[2])

        # Call segment.add_wafer with pre-calculated data
        self.segment.add_wafer(
            lift=lift,
            rotation=0.0,
            cylinder_diameter=self.cylinder_diameter,
            outside_height=outside_height,
            wafer_type=wafer_type,
            start_pos=start_point,
            end_pos=end_point,
            curve_tangent=curve_tangent,
            start_extension=start_extension,
            end_extension=end_extension,
            lcs1_rotation=lcs1_rotation,
            lcs2_rotation=lcs2_rotation
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
            log_coord(__name__, f"(try) is_planar: {is_planar}, rms: {rms:.4f}, n_hat: {n_hat}")
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
        log_coord(__name__, f"(at return)  rms: {rms:.4f}, tol: {tol:.4f},n_hat: {n_hat}")
        return (rms <= tol), n_hat, rms, tol

    def _fit_plane(self, pts: np.ndarray, eps: float = 1e-6):
        """
        Least-squares plane fit with robust guards.
        Returns: (is_planar, n_hat, p0, rms)

        Logs reasons for any fallback so you can see *why* a section failed:
          - non_finite_rows
          - too_few_points
          - zero_or_tiny_spread
          - svd_failed (eigen fallback used)
          - degenerate_normal
        """
        import numpy as np

        # --- tiny logger shim ----------------------------------------------
        def _log(level, *a):
            # Prefer your existing logger if present; else use print.
            if hasattr(self, "logger") and self.logger:
                getattr(self.logger, level, self.logger.info)(" ".join(str(x) for x in a))
            else:
                print(" ".join(str(x) for x in a))

        def _lc(*a):  # coordinate/debug log (if you have log_coord)
            try:
                from logging import getLogger
                if 'log_coord' in globals():
                    log_coord(__name__, " ".join(str(x) for x in a))
                else:
                    _log("debug", *a)
            except Exception:
                _log("debug", *a)

        # -------------------------------------------------------------------

        # 0) Coerce -> float (handles FreeCAD Vectors, lists, etc.)
        reason = None
        try:
            P = np.asarray(pts, dtype=float)
        except Exception:
            try:
                P = np.array([
                    [float(getattr(p, "x", p[0])),
                     float(getattr(p, "y", p[1])),
                     float(getattr(p, "z", p[2]))] for p in pts
                ], dtype=float)
                reason = "object_coercion"
            except Exception as e:
                _log("warning", f"_fit_plane: conversion_failed: {e}; returning default plane")
                return True, np.array([0.0, 0.0, 1.0]), (np.zeros(3)), 0.0

        n_in = int(P.shape[0])
        if n_in == 0:
            _log("warning", "_fit_plane: too_few_points (0); returning default plane")
            return True, np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0

        # 1) Drop non-finite rows (NaN/Inf)
        finite_rows = np.all(np.isfinite(P), axis=1)
        if not np.all(finite_rows):
            bad = int((~finite_rows).sum())
            bad_idx = np.where(~finite_rows)[0]
            reason = (reason + "+non_finite_rows") if reason else "non_finite_rows"
            _log("warning", f"_fit_plane: non_finite_rows={bad}/{n_in}; "
                            f"first_bad_idx={int(bad_idx[0]) if bad_idx.size else -1}")
            P = P[finite_rows]

        # 2) Need at least 3 points
        if P.shape[0] < 3:
            _log("warning", f"_fit_plane: too_few_points ({P.shape[0]}); returning default plane")
            p0 = P[0] if P.shape[0] else np.zeros(3)
            return True, np.array([0.0, 0.0, 1.0]), p0, 0.0

        # 3) Center for stability
        c = P.mean(axis=0)
        M = P - c

        # 4) Guard degenerate spread (all points equal or extremely small extent)
        ext_x, ext_y, ext_z = np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])
        extent = float(max(ext_x, ext_y, ext_z, 1.0))
        if extent <= 1e-15:
            reason = (reason + "+zero_or_tiny_spread") if reason else "zero_or_tiny_spread"
            _log("warning", f"_fit_plane: zero_or_tiny_spread; returning plane through centroid")
            return True, np.array([0.0, 0.0, 1.0]), c, 0.0

        # 5) Try SVD; fallback to eigen if SVD fails
        n = None
        try:
            # PCA via SVD: smallest singular vector is the plane normal
            _, svals, vh = np.linalg.svd(M, full_matrices=False)
            n = vh[-1, :]
            # Optional: if almost rank-1 (nearly a line), SVD can be numerically touchy.
            # We'll still proceed; distances will tell us if it's 'planar enough'.
        except Exception as e:
            reason = (reason + "+svd_failed") if reason else "svd_failed"
            _log("warning", f"_fit_plane: svd_failed ({e}); using eigen fallback")
            try:
                H = M.T @ M  # symmetric 3x3
                w, v = np.linalg.eigh(H)
                n = v[:, 0]  # smallest eigenvalue -> normal
            except Exception as e2:
                _log("warning", f"_fit_plane: eigen_failed ({e2}); returning default plane")
                return True, np.array([0.0, 0.0, 1.0]), c, 0.0

        # 6) Normalize normal; guard degeneracy
        n_norm = float(np.linalg.norm(n)) if n is not None else 0.0
        if not np.isfinite(n_norm) or n_norm == 0.0:
            reason = (reason + "+degenerate_normal") if reason else "degenerate_normal"
            _log("warning", "_fit_plane: degenerate_normal; returning default normal")
            n_hat = np.array([0.0, 0.0, 1.0])
        else:
            n_hat = n / n_norm

        # 7) Distances and planarity metric
        d = M @ n_hat
        rms = float(np.sqrt(np.mean(d * d))) if d.size else 0.0
        thr = float(eps * extent)
        is_planar = bool(rms <= thr)

        # 8) Final summary log
        why = f" reason={reason}" if reason else ""
        _lc(f"_fit_plane: n_in={n_in}, kept={P.shape[0]}, extent={extent:.6g}, "
            f"rms={rms:.6g}, thr={thr:.6g}, is_planar={is_planar}{why}")

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

    def _precalculate_lcs_orientations(self, wafers_parameters: List[Tuple]) -> List[Dict[str, Any]]:
        """Pre-calculate LCS orientations and interface bend angles for all wafer interfaces."""
        # Calculate all chord directions first
        chord_axes = []
        for start_point, end_point, _, _, _, _ in wafers_parameters:
            chord_vec = end_point - start_point
            chord_axis = FreeCAD.Vector(chord_vec[0], chord_vec[1], chord_vec[2])
            chord_axis.normalize()
            chord_axes.append(chord_axis)

        # Calculate interface orientations and bend angles
        interface_orientations = []
        interface_bend_angles = []  # Bend angle AT each interface

        for i in range(len(chord_axes) + 1):  # Need len+1 interfaces for len wafers
            # Interface i is between wafer i-1 and wafer i
            # Special cases: interface 0 is before first wafer, interface len is after last wafer

            if i == 0:
                # Before first wafer - no bend angle
                prev_chord = None
                current_chord = chord_axes[0]
                next_chord = chord_axes[1] if len(chord_axes) > 1 else None
                interface_rot = self._calculate_interface_orientation(None, current_chord, next_chord)
                bend_angle = 0.0  # First interface has no previous chord

            elif i == len(chord_axes):
                # After last wafer - no bend angle for next
                prev_chord = chord_axes[i - 1]
                current_chord = chord_axes[i - 1]  # Use last chord
                next_chord = None
                interface_rot = self._calculate_interface_orientation(prev_chord, current_chord, None)
                bend_angle = 0.0  # Last interface has no next chord

            else:
                # Middle interfaces - between wafer i-1 and wafer i
                prev_chord = chord_axes[i - 1]
                current_chord = chord_axes[i]
                next_chord = chord_axes[i + 1] if i + 1 < len(chord_axes) else None
                interface_rot = self._calculate_interface_orientation(prev_chord, current_chord, next_chord)

                # Bend angle AT THIS INTERFACE is between the two chords meeting here
                bend_angle_rad = prev_chord.getAngle(current_chord)
                bend_angle = bend_angle_rad  # Already in radians!
                logger.debug(
                    f"Interface {i}: prev_chord={i - 1}, current_chord={i}, bend_angle_rad={bend_angle_rad:.4f}, bend_angle_deg={math.degrees(bend_angle_rad):.2f}°")
                logger.debug(f"  prev_chord: [{prev_chord.x:.3f}, {prev_chord.y:.3f}, {prev_chord.z:.3f}]")
                logger.debug(f"  current_chord: [{current_chord.x:.3f}, {current_chord.y:.3f}, {current_chord.z:.3f}]")
                logger.debug(f"  prev length: {prev_chord.Length:.3f}, current length: {current_chord.Length:.3f}")

            interface_orientations.append(interface_rot)
            interface_bend_angles.append(bend_angle)

        # Now assign orientations to wafer ends and store bend angles
        lcs_orientations = []

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                wafers_parameters):
            # LCS1 uses the orientation from interface i
            lcs1_rotation = interface_orientations[i]

            # Get bend angle at start (interface i)
            start_bend_angle = interface_bend_angles[i]

            # LCS2 uses the orientation from interface i+1
            if i + 1 < len(interface_orientations):
                lcs2_base_rotation = interface_orientations[i + 1]
                end_bend_angle = interface_bend_angles[i + 1]
            else:
                lcs2_base_rotation = interface_orientations[i]
                end_bend_angle = 0.0  # Last wafer end

            # Apply lift angle tilt to LCS2 for elliptical end
            if wafer_type[1] == 'E' and abs(end_angle) > 0.1:
                tilt_matrix = FreeCAD.Matrix()
                tilt_matrix.rotateX(end_angle)
                combined_matrix = lcs2_base_rotation.toMatrix().multiply(tilt_matrix)
                lcs2_rotation = FreeCAD.Rotation(combined_matrix)
            else:
                lcs2_rotation = lcs2_base_rotation

            lcs_orientations.append({
                'lcs1_rotation': lcs1_rotation,
                'lcs2_rotation': lcs2_rotation,
                'start_point': start_point,
                'end_point': end_point,
                'start_bend_angle': start_bend_angle,  # NEW
                'end_bend_angle': end_bend_angle,  # NEW
                'wafer_index': i  # NEW: for debugging
            })

        # Validate that adjacent LCS orientations match
        for i in range(len(lcs_orientations) - 1):
            if i + 1 < len(interface_orientations):
                lcs2_base_rot = interface_orientations[i + 1]
            else:
                lcs2_base_rot = interface_orientations[i]

            lcs1_next_rot = lcs_orientations[i + 1]['lcs1_rotation']

            try:
                diff_matrix = lcs2_base_rot.toMatrix().multiply(lcs1_next_rot.toMatrix().inverse())
                diff_rotation = FreeCAD.Rotation(diff_matrix)
                angle_diff = math.degrees(abs(diff_rotation.Angle))

                if angle_diff > 0.1:
                    logger.error(f"BUG: Interface {i}-{i + 1} base orientations don't match!")
                    logger.error(f"  Angle difference: {angle_diff:.3f}°")
                    raise ValueError(f"LCS orientation mismatch at interface {i}-{i + 1}")
                else:
                    logger.debug(f"Interface {i}-{i + 1} orientations match (diff={angle_diff:.4f}°)")
            except Exception as e:
                logger.error(f"Error validating interface {i}-{i + 1}: {e}")

        logger.debug(f"Pre-calculated LCS orientations for {len(lcs_orientations)} wafers")
        return lcs_orientations

    def _calculate_interface_orientation(self, prev_chord, current_chord, next_chord):
        """Calculate orientation for an interface between two cylinders.

        The X-axis should lie in the plane containing the two chords.

        Args:
            prev_chord: Previous chord axis (or None)
            current_chord: Current chord axis
            next_chord: Next chord axis (or None)
        """
        import FreeCAD

        z_axis = current_chord.normalize()

        # Determine plane normal from the two chords meeting at this interface
        if next_chord is not None:
            # Use current and next chord
            plane_normal = current_chord.cross(next_chord)
        elif prev_chord is not None:
            # Use previous and current chord
            plane_normal = prev_chord.cross(current_chord)
        else:
            # Single wafer case - arbitrary
            if abs(z_axis.z) < 0.9:
                plane_normal = z_axis.cross(FreeCAD.Vector(0, 0, 1))
            else:
                plane_normal = z_axis.cross(FreeCAD.Vector(1, 0, 0))

        if plane_normal.Length < 1e-6:
            # Parallel chords - use perpendicular to z-axis
            if abs(z_axis.z) < 0.9:
                plane_normal = z_axis.cross(FreeCAD.Vector(0, 0, 1))
            else:
                plane_normal = z_axis.cross(FreeCAD.Vector(1, 0, 0))

        plane_normal.normalize()

        # X-axis perpendicular to plane normal and z-axis
        x_axis = plane_normal.cross(z_axis)
        x_axis.normalize()

        # Y-axis completes right-handed system
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize()

        # Create rotation matrix
        matrix = FreeCAD.Matrix(
            x_axis.x, y_axis.x, z_axis.x, 0,
            x_axis.y, y_axis.y, z_axis.y, 0,
            x_axis.z, y_axis.z, z_axis.z, 0,
            0, 0, 0, 1
        )

        return FreeCAD.Rotation(matrix)
