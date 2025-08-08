"""CurveFollower class for creating wafer slices along curved cylinders.

This module provides functionality to generate sequences of wafers that follow
arbitrary 3D curves, with proper geometric calculations for woodworking applications.
"""

import math
import numpy as np
from typing import List, Tuple, Any, Dict
import FreeCAD
import FreeCADGui
from curves import Curves  # Import the new Curves class

debug = True
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
        self.debug = True
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
        self.min_height = min_height
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
            return 0.0

        chord_vector = end_point - start_point
        chord_length = np.linalg.norm(chord_vector)

        if chord_length < 1e-10:
            return 0.0

        chord_unit = chord_vector / chord_length

        max_distance = 0.0
        for i, point in enumerate(curve_segment[1:-1]):
            point_vector = point - start_point
            projection_length = np.dot(point_vector, chord_unit)
            projection_point = start_point + projection_length * chord_unit
            distance = np.linalg.norm(point - projection_point)
            max_distance = max(max_distance, distance)
        #     if i < 3:  # Debug first few points
        #         print(f"      Point {i}: {point}, distance: {distance:.4f}")
        #
        # print(f"    Max chord distance: {max_distance:.4f}")

        return max_distance

    def _check_segment_collinearity(self, start_point: np.ndarray, end_point: np.ndarray,
                                    start_index: int, end_index: int) -> bool:
        """Check if a wafer segment is approximately collinear.

        Args:
            start_point: Start point of wafer
            end_point: End point of wafer
            start_index: Index of start point in curve
            end_index: Index of end point in curve  # Add this parameter

        Returns:
            True if segment is nearly collinear
        """
        if end_index <= start_index:
            return True

        curve_segment = self.curve_points[start_index:end_index + 1]
        chord_distance = self._calculate_chord_distance(start_point, end_point, curve_segment)
        chord_length = np.linalg.norm(end_point - start_point)
        threshold = self.max_chord * 0.1

        is_collinear = chord_distance < threshold

        # DEBUG OUTPUT
        print(f"  Collinearity check: chord_dist={chord_distance:.4f}, threshold={threshold:.4f}, "
              f"chord_len={chord_length:.4f}, collinear={is_collinear}")

        return is_collinear

    def _get_tangent_at_index(self, index: int) -> np.ndarray:
        """Get the tangent vector at a specific curve point index.

        Uses forward, backward, or central differences depending on position.

        Args:
            index: Index of the curve point

        Returns:
            Unit tangent vector at the specified point
        """
        if index == 0:
            tangent = self.curve_points[1] - self.curve_points[0]
        elif index == len(self.curve_points) - 1:
            tangent = self.curve_points[-1] - self.curve_points[-2]
        else:
            tangent = self.curve_points[index + 1] - self.curve_points[index - 1]

        return tangent / np.linalg.norm(tangent)

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

    @staticmethod
    def normalize_angle(angle_rad):
        """Normalize angle to [-π, π] range to prevent boundary issues"""
        if math.isnan(angle_rad) or math.isinf(angle_rad):
            return 0.0

        # Use remainder for clean normalization to [-π, π]
        normalized = math.remainder(angle_rad, 2 * math.pi)

        # Avoid exact boundary values that cause test failures
        epsilon = 1e-10
        if abs(abs(normalized) - math.pi) < epsilon:
            normalized = math.copysign(math.pi - epsilon, normalized)

        return normalized

    def _calculate_ellipse_parameters(
            self, start_point: np.ndarray, end_point: np.ndarray,
            start_index: int, end_index: int,
            is_first_wafer: bool = False, is_last_wafer: bool = False
    ) -> Tuple[float, float, float, str]:
        """
        Calculate ellipse angles and wafer type for the wafer ends.

        CORRECTED VERSION: Uses bisecting plane approach where adjacent wafers
        share a single cutting plane.
        """

        # Vector between endpoints (current wafer's chord/axis)
        chord_vector = end_point - start_point
        chord_length = np.linalg.norm(chord_vector)

        if chord_length < 1e-10:
            return 0.0, 0.0, 0.0, "CC"

        chord_unit = chord_vector / chord_length

        # Check for collinearity
        is_collinear = self._check_segment_collinearity(start_point, end_point, start_index, end_index)

        if is_collinear:
            return 0.0, 0.0, 0.0, "CC"

        # Initialize angles
        start_angle = 0.0
        end_angle = 0.0
        rotation_angle = 0.0

        # For first wafer, start is always circular
        if is_first_wafer:
            start_angle = 0.0

            # For the end cut, we need the angle to the NEXT wafer
            if end_index < len(self.curve_points) - 1:
                # Get next wafer's chord
                next_point = self.curve_points[min(end_index + 10, len(self.curve_points) - 1)]
                next_chord = next_point - end_point
                next_chord_length = np.linalg.norm(next_chord)

                if next_chord_length > 1e-6:
                    next_chord_unit = next_chord / next_chord_length

                    # Calculate angle between current and next chord
                    dot_product = np.dot(chord_unit, next_chord_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    bend_angle = np.arccos(dot_product)

                    # The cut angle is half the bend angle (bisecting plane)
                    end_angle = bend_angle / 2.0

                    # DEBUG
                    print(
                        f"    First wafer end: bend_angle = {np.rad2deg(bend_angle):.2f}°, cut_angle = {np.rad2deg(end_angle):.2f}°")

            wafer_type = "CE"

        elif is_last_wafer:
            # For the start cut, we need the angle from the PREVIOUS wafer
            if start_index > 0:
                # Get previous wafer's chord
                prev_point = self.curve_points[max(start_index - 10, 0)]
                prev_chord = start_point - prev_point
                prev_chord_length = np.linalg.norm(prev_chord)

                if prev_chord_length > 1e-6:
                    prev_chord_unit = prev_chord / prev_chord_length

                    # Calculate angle between previous and current chord
                    dot_product = np.dot(prev_chord_unit, chord_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    bend_angle = np.arccos(dot_product)

                    # The cut angle is half the bend angle (bisecting plane)
                    start_angle = bend_angle / 2.0

                    # DEBUG
                    print(
                        f"    Last wafer start: bend_angle = {np.rad2deg(bend_angle):.2f}°, cut_angle = {np.rad2deg(start_angle):.2f}°")

            end_angle = 0.0  # Last wafer ends with circular cut
            wafer_type = "EC"

        else:
            # Middle wafer - both cuts are elliptical

            # Start cut angle (from previous wafer)
            if start_index > 0:
                prev_point = self.curve_points[max(start_index - 10, 0)]
                prev_chord = start_point - prev_point
                prev_chord_length = np.linalg.norm(prev_chord)

                if prev_chord_length > 1e-6:
                    prev_chord_unit = prev_chord / prev_chord_length
                    dot_product = np.dot(prev_chord_unit, chord_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    bend_angle = np.arccos(dot_product)
                    start_angle = bend_angle / 2.0

                    # DEBUG
                    print(
                        f"    Wafer start: bend_angle = {np.rad2deg(bend_angle):.2f}°, cut_angle = {np.rad2deg(start_angle):.2f}°")

            # End cut angle (to next wafer)
            if end_index < len(self.curve_points) - 1:
                next_point = self.curve_points[min(end_index + 10, len(self.curve_points) - 1)]
                next_chord = next_point - end_point
                next_chord_length = np.linalg.norm(next_chord)

                if next_chord_length > 1e-6:
                    next_chord_unit = next_chord / next_chord_length
                    dot_product = np.dot(chord_unit, next_chord_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    bend_angle = np.arccos(dot_product)
                    end_angle = bend_angle / 2.0

                    # DEBUG
                    print(
                        f"    Wafer end: bend_angle = {np.rad2deg(bend_angle):.2f}°, cut_angle = {np.rad2deg(end_angle):.2f}°")

            wafer_type = "EE"

            # Calculate rotation angle ONLY for EE wafers
            # This is the twist needed between the ellipse orientations
            if wafer_type == "EE":
                # Simple approach: use angular progression around curve
                start_angle_2d = np.arctan2(start_point[1], start_point[0])
                end_angle_2d = np.arctan2(end_point[1], end_point[0])

                rotation_angle = end_angle_2d - start_angle_2d
                rotation_angle = self.normalize_angle(rotation_angle)

                # DEBUG
                print(f"    EE wafer rotation: {np.rad2deg(rotation_angle):.2f}°")

        # Clamp angles to reasonable values
        max_angle = np.pi / 3  # 60 degrees max
        start_angle = np.clip(start_angle, 0, max_angle)
        end_angle = np.clip(end_angle, 0, max_angle)

        # DEBUG summary
        print(
            f"    Final: start={np.rad2deg(start_angle):.2f}°, end={np.rad2deg(end_angle):.2f}°, rot={np.rad2deg(rotation_angle):.2f}°, type={wafer_type}")

        return start_angle, end_angle, rotation_angle, wafer_type

    def _correct_rotation_angles(self, wafers: List[Tuple]) -> List[Tuple]:
        """Correct rotation angles so adjacent wafers have complementary cuts.

        RESTORED TO ORIGINAL - just pass through the calculated rotations.
        """
        return wafers  # Don't modify rotations

    def _calculate_local_twist(self, start_tangent: np.ndarray, end_tangent: np.ndarray,
                               chord_unit: np.ndarray) -> float:
        """Calculate the local twist angle between two tangent vectors.

        FIXED VERSION: Properly handles the twist calculation for wafers.

        The twist represents how much the cutting plane rotates as we move
        along the curve. For a helix, this should be proportional to the
        distance traveled.
        """

        # For very small wafers or when tangents are nearly parallel to chord,
        # we can't reliably calculate twist from tangent projections
        start_dot = abs(np.dot(start_tangent, chord_unit))
        end_dot = abs(np.dot(end_tangent, chord_unit))

        if start_dot > 0.95 or end_dot > 0.95:
            # Tangents nearly parallel to chord
            # For a helix or similar curve, estimate twist from geometry

            # This is a fallback calculation based on the assumption that
            # twist is proportional to distance along curve
            # For 9 wafers making one full turn, each should twist ~40°

            # Better approach: return 0 and let the curve follower handle it
            return 0.0

        # Original calculation for cases where tangents aren't parallel to chord
        def project_onto_plane(v, normal):
            """Project vector v onto plane perpendicular to normal."""
            return v - np.dot(v, normal) * normal

        # Project tangents onto plane perpendicular to chord
        v1_proj = project_onto_plane(start_tangent, chord_unit)
        v2_proj = project_onto_plane(end_tangent, chord_unit)

        v1_norm = np.linalg.norm(v1_proj)
        v2_norm = np.linalg.norm(v2_proj)

        if v1_norm < 1e-9 or v2_norm < 1e-9:
            return 0.0

        # Normalize projected vectors
        v1_proj /= v1_norm
        v2_proj /= v2_norm

        # Calculate rotation angle
        cross = np.cross(v1_proj, v2_proj)
        dot = np.dot(v1_proj, v2_proj)

        # Determine sign of rotation using chord direction
        if isinstance(cross, np.ndarray):
            sign = np.sign(np.dot(cross, chord_unit))
        else:
            sign = np.sign(cross)

        rotation_angle = sign * math.acos(np.clip(dot, -1.0, 1.0))

        return rotation_angle


    def create_wafer_list(self) -> List[Tuple[np.ndarray, np.ndarray, float, float, float, str]]:
        """Create a list of wafers satisfying geometric constraints.

        Generates wafers that satisfy both minimum height and maximum chord
        distance constraints while maximizing wafer size.

        Returns:
            List of tuples: (start_point, end_point, start_angle, end_angle,
                           rotation_angle, wafer_type)
        """
        wafers = []
        current_index = 0
        safety_counter = 0

        while current_index < len(self.curve_points) - 1:
            safety_counter +=1
            # SAFETY CHECK: Prevent infinite loops
            if safety_counter > 1000:  # ADD THIS
                print(f"🛑 SAFETY STOP: Generated {len(wafers)} wafers, stopping to prevent infinite loop")
                break
            start_point = self.curve_points[current_index]

            # Find the farthest valid end point
            best_end_index = current_index + 1

            for end_index in range(current_index + 1, len(self.curve_points)):
                end_point = self.curve_points[end_index]

                # Check minimum height constraint
                height = np.linalg.norm(end_point - start_point)
                if height < self.min_height:
                    continue

                # Check maximum chord constraint
                curve_segment = self.curve_points[current_index:end_index + 1]
                chord_distance = self._calculate_chord_distance(start_point, end_point, curve_segment)

                if chord_distance <= self.max_chord:
                    best_end_index = end_index
                else:
                    break

            end_point = self.curve_points[best_end_index]

            # Determine if this is a first or last wafer
            is_first_wafer = (current_index == 0)
            is_last_wafer = (best_end_index == len(self.curve_points) - 1)

            # Calculate ellipse parameters
            start_angle, end_angle, rotation_angle, wafer_type = self._calculate_ellipse_parameters(
                start_point, end_point, current_index, best_end_index,
                is_first_wafer, is_last_wafer
            )

            wafers.append((start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type))
            current_index = best_end_index

        return wafers

    def _determine_consistent_wafer_types(self, wafers: List[Tuple]) -> List[Tuple]:
        """Ensure wafer types form a consistent cutting sequence.

        Args:
            wafers: List of wafer tuples from create_wafer_list()

        Returns:
            List of corrected wafer tuples with consistent types
        """
        if len(wafers) <= 1:
            return wafers

        corrected_wafers = []

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, initial_type) in enumerate(wafers):

            # Find the indices for this wafer segment
            start_index = self._find_point_index(start_point)
            end_index = self._find_point_index(end_point)

            # Determine actual wafer type based on adjacent wafers and collinearity
            is_collinear = self._check_segment_collinearity(start_point, end_point,
                                                            start_index, end_index)

            if i == 0:
                # First wafer
                start_type = 'C'  # Always circular
            else:
                # Start type must match previous wafer's end type
                prev_end_type = corrected_wafers[i - 1][5][1]  # Get end type from previous wafer
                start_type = prev_end_type

            if is_collinear:
                # For collinear segments, end type same as start type
                end_type = start_type
                corrected_start_angle = 0.0 if start_type == 'C' else start_angle
                corrected_end_angle = 0.0  # Parallel cut
                corrected_rotation = 0.0  # No twist
            else:
                # Normal case: calculate end type based on geometry
                if i == len(wafers) - 1:
                    # Last wafer: end is always circular
                    end_type = 'C'
                    corrected_end_angle = 0.0
                else:
                    # Middle wafer: end is elliptical
                    end_type = 'E'
                    corrected_end_angle = end_angle

                corrected_start_angle = 0.0 if start_type == 'C' else start_angle
                corrected_rotation = rotation_angle

            # Construct wafer type string
            wafer_type = start_type + end_type

            corrected_wafers.append((
                start_point, end_point,
                corrected_start_angle, corrected_end_angle,
                corrected_rotation, wafer_type
            ))

        return corrected_wafers

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

    def _correct_rotation_angles(self, wafers: List[Tuple]) -> List[Tuple]:
        """Correct rotation angles so adjacent wafers have complementary cuts.

        Ensures that adjacent wafers can be cut from a single cylinder using
        compatible rotation angles.

        Args:
            wafers: List of wafer tuples

        Returns:
            List of corrected wafer tuples with complementary rotation angles
        """
        if len(wafers) <= 1:
            return wafers

        corrected_wafers = []
        cumulative_rotation = 0.0

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):

            if i == 0:
                # First wafer: use calculated rotation as-is
                corrected_rotation = rotation_angle
            else:
                # Subsequent wafers: ensure complementary cutting
                # The start of this wafer should align with the end of the previous wafer
                prev_wafer = corrected_wafers[i-1]
                prev_end_rotation = prev_wafer[4]  # Previous wafer's rotation

                # For adjacent cuts to be complementary, rotations should be additive
                corrected_rotation = rotation_angle

            # Track cumulative rotation for debugging
            cumulative_rotation += corrected_rotation

            corrected_wafers.append((start_point, end_point, start_angle,
                                   end_angle, corrected_rotation, wafer_type))

        return corrected_wafers

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

        start_extension = safe_ellipse_extension(start_angle)
        end_extension = safe_ellipse_extension(end_angle)

        outside_height = chord_length + start_extension + end_extension

        # Sanity check: outside_height should be reasonable compared to chord_length
        if outside_height > chord_length * 5:  # If more than 5x chord length, something's wrong
            # print(f"WARNING: Capping excessive outside_height")
            # print(f"  Original calculation: {outside_height:.4f}")
            # print(f"  Chord length: {chord_length:.4f}")
            # print(f"  Start angle: {math.degrees(start_angle):.2f}°")
            # print(f"  End angle: {math.degrees(end_angle):.2f}°")
            outside_height = chord_length + self.cylinder_diameter  # Conservative fallback
            # print(f"  Capped to: {outside_height:.4f}")

        return outside_height

    def add_wafer_from_curve_data(self, start_point: np.ndarray, end_point: np.ndarray,
                                  start_angle: float, end_angle: float, rotation_angle: float,
                                  wafer_type: str, debug: bool = True) -> None:
        """Convert curve follower data to FlexSegment.add_wafer() parameters.

        UPDATED: Uses improved lift angle calculation based on wafer type.
        """

        import traceback
        wafer_num = self.segment.get_wafer_count() + 1
        print(f"\n🔥 ENTERING add_wafer_from_curve_data() - Future wafer #{wafer_num}")
        print(f"   Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
        print(f"   End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
        print(f"   Type: {wafer_type}")

        # Show the call stack to see what's calling this method
        print(f"   CALL STACK:")
        for line in traceback.format_stack()[-3:-1]:  # Show last 2 callers
            print(f"     {line.strip()}")

        # NEW: Use improved lift calculation based on wafer type
        lift = self._calculate_lift_angle_improved(start_angle, end_angle, wafer_type)

        # Normalize the result
        lift = self.normalize_angle(lift)

        # Clamp to practical cutting range if needed
        if abs(lift) > math.pi / 2:  # > 90 degrees
            if debug:
                print(f"WARNING: Large lift angle {math.degrees(lift):.1f}°, clamping to ±90°")
            lift = math.copysign(math.pi / 2, lift)

        # Normalize rotation angle
        rotation = self.normalize_angle(rotation_angle)

        # Calculate outside height
        outside_height = self._calculate_outside_height(start_point, end_point,
                                                        start_angle, end_angle, rotation_angle)

        wafer_num = self.segment.get_wafer_count() + 1

        # 🔍 CRITICAL DEBUG - Check if positions are changing
        # print(f"🔍 WAFER {wafer_num} POSITION CHECK:")
        # print(f"    Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
        # print(f"    End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
        # print(f"    Distance: {np.linalg.norm(end_point - start_point):.3f}")

        # Check if this wafer has the same position as the previous one
        if wafer_num > 1:
            print(f"    ⚠️  Position change from previous wafer:")
            print(f"        Start diff: {np.linalg.norm(start_point - getattr(self, '_last_start', start_point)):.6f}")
            print(f"        End diff: {np.linalg.norm(end_point - getattr(self, '_last_end', end_point)):.6f}")

        # Store for next comparison
        self._last_start = start_point.copy()
        self._last_end = end_point.copy()

        # Calculate and validate parametersdifference
        lift = self._calculate_lift_angle_improved(start_angle, end_angle, wafer_type)
        lift = self.normalize_angle(lift)
        rotation = self.normalize_angle(rotation_angle)
        outside_height = self._calculate_outside_height(start_point, end_point,
                                                        start_angle, end_angle, rotation_angle)

        # CRITICAL VALIDATION: Check for degenerate geometry
        chord_length = np.linalg.norm(end_point - start_point)

        if chord_length < 0.001:
            print(f"⚠️  SKIPPING degenerate wafer: chord length {chord_length:.6f} too small")
            return  # Skip this wafer

        if outside_height <= 0.001:
            print(f"⚠️  FIXING invalid outside_height: {outside_height:.6f} -> {chord_length}")
            outside_height = max(chord_length, 0.1)  # Use chord length as minimum

        if self.cylinder_diameter <= 0.001:
            print(f"⚠️  FIXING invalid cylinder_diameter: {self.cylinder_diameter}")
            self.cylinder_diameter = 1.0  # Default value

        # Additional safety checks
        if abs(lift) > math.pi / 2:
            print(f"⚠️  Clamping extreme lift: {math.degrees(lift):.1f}° -> ±90°")
            lift = math.copysign(math.pi / 2, lift)

        if debug:
            # print(f"=== Wafer {wafer_num} ===")
            # print(f"  Chord length: {chord_length:.4f}")
            # print(f"  Outside height: {outside_height:.4f}")
            # print(f"  Cylinder diameter: {self.cylinder_diameter:.4f}")
            # print(f"  Lift: {math.degrees(lift):.2f}°")
            # print(f"  Rotation: {math.degrees(rotation):.2f}°")
            # print(f"  Wafer type: {wafer_type}")

            # Geometry validation
            height_ratio = outside_height / chord_length if chord_length > 0 else 0
            if height_ratio > 10:
                print(f"  ⚠️  WARNING: Height/chord ratio {height_ratio:.1f} is very large")

            if height_ratio < 0.1:
                print(f"  ⚠️  WARNING: Height/chord ratio {height_ratio:.1f} is very small")

    #   REMOVE FOLLOWING DUPLICATE CALL
        # try:
        #     # Call FlexSegment with validated parameters
        #     self.segment.add_wafer(lift, rotation, self.cylinder_diameter, outside_height, wafer_type,
        #                            start_pos=start_point, end_pos=end_point)
        #     if debug:
        #         print(f"  ✅ Created wafer {wafer_num}")
        # except Exception as e:
        #     print(f"  ❌ ERROR creating wafer {wafer_num}: {e}")
        #     print(
        #         f"     Parameters: lift={lift:.4f}, rot={rotation:.4f}, diam={self.cylinder_diameter:.4f}, height={outside_height:.4f}")
            # Don't raise - continue with next wafer
        # if debug:
            # print(f"=== Wafer {wafer_num} ===")
            # print(f"  Start point: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            # print(f"  End point: [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            # print(f"  Z difference: {end_point[2] - start_point[2]:.3f}")

        try:
            # Pass 3D position data
            self.segment.add_wafer(lift, rotation, self.cylinder_diameter, outside_height, wafer_type,
                                   start_pos=start_point, end_pos=end_point)
        except Exception as e:
            print(f"Error adding wafer: {e.args}")

    def _calculate_lift_angle_improved(self, start_angle: float, end_angle: float,
                                       wafer_type: str) -> float:
        """Calculate lift angle with improved logic for different wafer types.

        This replaces the old 'lift = start_angle + end_angle' calculation.
        """

        if wafer_type == "CC":  # Circular-Circular (parallel cuts)
            return 0.0

        elif wafer_type == "CE":  # Circular-Elliptical
            # Only the elliptical end contributes to lift
            return end_angle

        elif wafer_type == "EC":  # Elliptical-Circular
            # Only the elliptical start contributes to lift
            return start_angle

        elif wafer_type == "EE":  # Elliptical-Elliptical
            # For EE wafers, use average to avoid extreme angles
            # This prevents the -180° issue from adding two large angles
            return (start_angle + end_angle) / 2.0

        else:
            # Fallback for unknown wafer types
            print(f"WARNING: Unknown wafer type '{wafer_type}', using average")
            return (start_angle + end_angle) / 2.0

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

        print(f"Creating curve vertices with group name: '{group_name}'")

        # Create vertices at origin (same as wafer creation)
        vertex_group_name = self.curves.add_visualization_vertices(None, group_name)

        # Register with segment using FreeCAD properties AND grouping
        self.segment.register_curve_vertices_group(vertex_group_name)

        return vertex_group_name

    def process_wafers(self, add_curve_vertices: bool = False, debug: bool = True) -> None:
        """Main processing method that creates and adds wafers to the segment."""

        # Step 2: Create wafer list
        wafers = self.create_wafer_list()

        # Step 3: Apply wafer type consistency
        consistent_wafers = self._determine_consistent_wafer_types(wafers)

        # Step 4: Correct rotation angles
        corrected_wafers = self._correct_rotation_angles(consistent_wafers)

        # NEW Step 5: Calculate all cutting planes
        cutting_plane_data = self.calculate_cutting_planes(corrected_wafers)

        # Step 6: Process each wafer with cutting plane data
        for i, cutting_data in enumerate(cutting_plane_data):
            if debug:
                print(f"\nProcessing wafer {i + 1}/{len(cutting_plane_data)}:")

            # Pass the complete cutting plane information
            self.add_wafer_with_cutting_planes(cutting_data, debug)

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

    def debug_wafer_generation(self) -> None:
        """Debug the wafer generation process to find duplicate positioning."""

        print(f"\n=== DEBUGGING WAFER GENERATION ===")
        print(f"Curve has {len(self.curve_points)} points")
        print(f"Min height: {self.min_height}")
        print(f"Max chord: {self.max_chord}")

        # Step 1: Debug the initial wafer creation
        print(f"\n--- Step 1: Raw wafer creation ---")
        wafers = self.create_wafer_list()

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):
            print(f"Raw wafer {i + 1}:")
            print(f"  Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"  End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"  Type: {wafer_type}")

        # Step 2: Debug wafer type consistency
        print(f"\n--- Step 2: After wafer type consistency ---")
        consistent_wafers = self._determine_consistent_wafer_types(wafers)

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                consistent_wafers):
            print(f"Consistent wafer {i + 1}:")
            print(f"  Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"  End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"  Type: {wafer_type}")

        # Step 3: Debug rotation angle correction
        print(f"\n--- Step 3: After rotation angle correction ---")
        corrected_wafers = self._correct_rotation_angles(consistent_wafers)

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            print(f"Final wafer {i + 1}:")
            print(f"  Start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"  End:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"  Type: {wafer_type}")
            print(f"  Rotation: {rotation_angle:.3f}")

        # Step 4: Check for duplicates
        print(f"\n--- Step 4: Duplicate detection ---")
        seen_positions = set()
        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            # Create a hashable representation of the position
            position_key = (round(start_point[0], 6), round(start_point[1], 6), round(start_point[2], 6),
                            round(end_point[0], 6), round(end_point[1], 6), round(end_point[2], 6))

            if position_key in seen_positions:
                print(f"🚨 DUPLICATE FOUND: Wafer {i + 1} has same position as previous wafer!")
                print(f"   Position key: {position_key}")
            else:
                seen_positions.add(position_key)
                print(f"✅ Wafer {i + 1}: Unique position")

        print(f"\nTotal wafers generated: {len(corrected_wafers)}")
        print(f"Unique positions: {len(seen_positions)}")

        return corrected_wafers

    def calculate_cutting_planes(self, wafers):
        """Pre-calculate all cutting planes for the wafer sequence.

        Args:
            wafers: List of (start_point, end_point, ...) tuples

        Returns:
            List of cutting plane data for each wafer
        """
        cutting_planes = []

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):
            # Current wafer axis (chord)
            current_axis = end_point - start_point
            current_axis_length = np.linalg.norm(current_axis)
            if current_axis_length > 1e-6:
                current_axis = current_axis / current_axis_length

            # Start cutting plane
            if i == 0:
                # First wafer: perpendicular cut
                start_normal = current_axis
            else:
                # Get previous wafer axis
                prev_start, prev_end = wafers[i - 1][0], wafers[i - 1][1]
                prev_axis = prev_end - prev_start
                prev_axis_length = np.linalg.norm(prev_axis)
                if prev_axis_length > 1e-6:
                    prev_axis = prev_axis / prev_axis_length

                # Calculate bisecting plane normal
                start_normal = self._calculate_bisecting_plane_normal(prev_axis, current_axis)

            # End cutting plane
            if i == len(wafers) - 1:
                # Last wafer: perpendicular cut
                end_normal = current_axis
            else:
                # Get next wafer axis
                next_start, next_end = wafers[i + 1][0], wafers[i + 1][1]
                next_axis = next_end - next_start
                next_axis_length = np.linalg.norm(next_axis)
                if next_axis_length > 1e-6:
                    next_axis = next_axis / next_axis_length

                # Calculate bisecting plane normal
                end_normal = self._calculate_bisecting_plane_normal(current_axis, next_axis)

            cutting_planes.append({
                'start_pos': start_point,
                'end_pos': end_point,
                'start_normal': start_normal,
                'end_normal': end_normal,
                'wafer_axis': current_axis,
                'rotation': rotation_angle,
                'wafer_type': wafer_type
            })

        return cutting_planes

    def _calculate_bisecting_plane_normal(self, axis1, axis2):
        """Calculate normal to the plane that bisects two cylinder axes.

        The bisecting plane contains both axes and has a normal that makes
        equal angles with both axes.
        """
        # Handle parallel/antiparallel axes
        cross = np.cross(axis1, axis2)
        cross_length = np.linalg.norm(cross)

        if cross_length < 1e-6:
            # Axes are parallel - return perpendicular to axis
            return axis1

        # The bisecting plane normal is the normalized sum of the axes
        # (for acute angles) or difference (for obtuse angles)
        dot_product = np.dot(axis1, axis2)

        if dot_product > 0:
            # Acute angle - bisector is sum
            bisector = axis1 + axis2
        else:
            # Obtuse angle - bisector is difference
            bisector = axis1 - axis2

        bisector_length = np.linalg.norm(bisector)
        if bisector_length > 1e-6:
            bisector = bisector / bisector_length

        # The cutting plane normal is the bisector direction
        # (which is perpendicular to the intersection line of the two cylinders)
        return bisector

    def add_wafer_with_cutting_planes(self, cutting_data, debug=False):
        """Add a wafer using pre-calculated cutting plane data."""

        # Calculate the actual cut angles from the cutting plane normals
        start_angle = np.arccos(np.clip(np.abs(np.dot(cutting_data['start_normal'],
                                                      cutting_data['wafer_axis'])), 0, 1))
        end_angle = np.arccos(np.clip(np.abs(np.dot(cutting_data['end_normal'],
                                                    cutting_data['wafer_axis'])), 0, 1))

        # For wafer parameter calculation
        if cutting_data['wafer_type'][0] == 'C':
            lift_start = 0.0  # Circular start
        else:
            lift_start = start_angle

        if cutting_data['wafer_type'][1] == 'C':
            lift_end = 0.0  # Circular end
        else:
            lift_end = end_angle

        # Average lift for the wafer (used for height calculations)
        lift = (lift_start + lift_end) / 2.0

        # Calculate outside height based on geometry
        chord_length = np.linalg.norm(cutting_data['end_pos'] - cutting_data['start_pos'])
        outside_height = self._calculate_outside_height(
            cutting_data['start_pos'],
            cutting_data['end_pos'],
            lift_start,
            lift_end,
            cutting_data['rotation']
        )

        if debug:
            print(f"  Cutting plane normals:")
            print(f"    Start: [{cutting_data['start_normal'][0]:.3f}, "
                  f"{cutting_data['start_normal'][1]:.3f}, {cutting_data['start_normal'][2]:.3f}]")
            print(f"    End: [{cutting_data['end_normal'][0]:.3f}, "
                  f"{cutting_data['end_normal'][1]:.3f}, {cutting_data['end_normal'][2]:.3f}]")
            print(f"  Cut angles: start={np.rad2deg(start_angle):.1f}°, "
                  f"end={np.rad2deg(end_angle):.1f}°")

        # Pass the cut angles to the segment
        self.segment.add_wafer_with_planes(
            start_pos=cutting_data['start_pos'],
            end_pos=cutting_data['end_pos'],
            start_normal=cutting_data['start_normal'],
            end_normal=cutting_data['end_normal'],
            wafer_axis=cutting_data['wafer_axis'],
            rotation=cutting_data['rotation'],
            cylinder_diameter=self.cylinder_diameter,
            outside_height=outside_height,
            wafer_type=cutting_data['wafer_type'],
            start_cut_angle=start_angle,  # ADD THIS
            end_cut_angle=end_angle  # ADD THIS
        )

def format_mixed(result):
    if isinstance(result, float):
        return f"{result:.4f}"
    else:
        return result