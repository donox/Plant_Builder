"""CurveFollower class for creating wafer slices along curved cylinders.

This module provides functionality to generate sequences of wafers that follow
arbitrary 3D curves, with proper geometric calculations for woodworking applications.
"""

import math
import numpy as np
from typing import List, Tuple, Any, Dict
import FreeCAD
import FreeCADGui
from .curves import Curves  # Import the new Curves class


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
        for point in curve_segment[1:-1]:
            point_vector = point - start_point
            projection_length = np.dot(point_vector, chord_unit)
            projection_point = start_point + projection_length * chord_unit
            distance = np.linalg.norm(point - projection_point)
            max_distance = max(max_distance, distance)

        return max_distance

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

    def _calculate_ellipse_parameters(self, start_point: np.ndarray, end_point: np.ndarray,
                                      start_index: int, end_index: int, is_first_wafer: bool = False,
                                      is_last_wafer: bool = False) -> Tuple[float, float, float, str]:
        """Calculate ellipse angles and wafer type for the wafer ends.

        Determines the cutting angles needed to create proper elliptical or
        circular cross-sections at each end of the wafer.

        Args:
            start_point: 3D coordinates of wafer start
            end_point: 3D coordinates of wafer end
            start_index: Index of start point in curve
            end_index: Index of end point in curve
            is_first_wafer: True if this is the first wafer in sequence
            is_last_wafer: True if this is the last wafer in sequence

        Returns:
            Tuple of (start_angle, end_angle, rotation_angle, wafer_type)
            where angles are in radians and wafer_type is "CE", "EC", "EE", or "CC"
        """
        # Calculate tangent vectors at start and end points
        start_tangent = self._get_tangent_at_index(start_index)
        end_tangent = self._get_tangent_at_index(end_index)

        # Vector between endpoints
        chord_vector = end_point - start_point
        chord_unit = chord_vector / np.linalg.norm(chord_vector)

        # Determine wafer type and angles
        if is_first_wafer:
            # First wafer: start is circular (perpendicular to curve), end is elliptical
            start_angle = 0.0  # 0 degrees - perpendicular to curve gives circular cross-section
            end_angle = math.acos(np.clip(np.abs(np.dot(end_tangent, chord_unit)), 0, 1))
            wafer_type = "CE"  # Circle-Ellipse
        elif is_last_wafer:
            # Last wafer: start is elliptical, end is circular (perpendicular to curve)
            start_angle = math.acos(np.clip(np.abs(np.dot(start_tangent, chord_unit)), 0, 1))
            end_angle = 0.0  # 0 degrees - perpendicular to curve gives circular cross-section
            wafer_type = "EC"  # Ellipse-Circle
        else:
            # Middle wafers: both ends are elliptical
            start_angle = math.acos(np.clip(np.abs(np.dot(start_tangent, chord_unit)), 0, 1))
            end_angle = math.acos(np.clip(np.abs(np.dot(end_tangent, chord_unit)), 0, 1))
            wafer_type = "EE"  # Ellipse-Ellipse

        # Calculate rotation angle between ellipse major axes
        cross_product = np.cross(start_tangent, end_tangent)
        if isinstance(cross_product, np.ndarray):
            rotation_angle = math.atan2(np.linalg.norm(cross_product),
                                        np.dot(start_tangent, end_tangent))
        else:
            rotation_angle = math.atan2(abs(cross_product),
                                        np.dot(start_tangent, end_tangent))

        return start_angle, end_angle, rotation_angle, wafer_type


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

        while current_index < len(self.curve_points) - 1:
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


    def _correct_lift_angles(self, wafers: List[Tuple]) -> List[Tuple]:
        """Correct lift angles so adjacent wafers have complementary angles.

        Ensures that adjacent wafers can be cut from a single cylinder using
        a single cutting plane by making their adjoining angles equal.

        Args:
            wafers: List of wafer tuples from create_wafer_list()

        Returns:
            List of corrected wafer tuples with same structure
        """
        if len(wafers) <= 1:
            return wafers

        corrected_wafers = []

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):
            corrected_start_angle = start_angle
            corrected_end_angle = end_angle

            # For adjacent wafers, make angles complementary
            if i > 0:  # Not the first wafer
                # Current wafer's start should complement previous wafer's end
                prev_end_angle = corrected_wafers[i - 1][3]  # Previous wafer's corrected end angle
                corrected_start_angle = prev_end_angle  # Make them equal for complementary cutting

            corrected_wafers.append((start_point, end_point, corrected_start_angle,
                                     corrected_end_angle, rotation_angle, wafer_type))

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
            print(f"WARNING: Capping excessive outside_height")
            print(f"  Original calculation: {outside_height:.4f}")
            print(f"  Chord length: {chord_length:.4f}")
            print(f"  Start angle: {math.degrees(start_angle):.2f}°")
            print(f"  End angle: {math.degrees(end_angle):.2f}°")
            outside_height = chord_length + self.cylinder_diameter  # Conservative fallback
            print(f"  Capped to: {outside_height:.4f}")

        return outside_height


    def add_wafer_from_curve_data(self, start_point: np.ndarray, end_point: np.ndarray,
                                  start_angle: float, end_angle: float, rotation_angle: float,
                                  wafer_type: str, debug: bool = True) -> None:
        """Convert curve follower data to FlexSegment.add_wafer() parameters.

        Adapter method that translates the geometric curve data into the
        parameters expected by the FlexSegment wafer creation system.

        Args:
            start_point: 3D coordinates of wafer start
            end_point: 3D coordinates of wafer end
            start_angle: Angle of start ellipse plane to perpendicular (radians)
            end_angle: Angle of end ellipse plane to perpendicular (radians)
            rotation_angle: Rotation between ellipse major axes (radians)
            wafer_type: Type string ("EE", "CE", "EC", or "CC")
            debug: Whether to print conversion details

        Raises:
            Exception: If wafer creation fails
        """
        # Calculate lift angle (sum of the two end angles)
        lift = start_angle + end_angle

        # Rotation is the rotation between major axes
        rotation = rotation_angle

        # Calculate outside height
        outside_height = self._calculate_outside_height(start_point, end_point,
                                                        start_angle, end_angle, rotation_angle)

        wafer_num = self.segment.get_wafer_count() + 1

        if debug:
            print(f"=== Wafer {wafer_num} ===")
            print(f"  Start point: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"  End point: [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"  Start angle: {math.degrees(start_angle):.2f}°")
            print(f"  End angle: {math.degrees(end_angle):.2f}°")
            print(f"  Lift (sum): {math.degrees(lift):.2f}°")
            print(f"  Rotation: {math.degrees(rotation):.2f}°")
            print(f"  Outside height: {outside_height:.4f}")
            print(f"  Wafer type: {wafer_type}")

            # Validation checks
            if outside_height <= 0:
                print(f"  ERROR: Invalid outside height!")
            if math.isnan(lift) or math.isnan(rotation):
                print(f"  ERROR: NaN values detected!")

        try:
            # Call the FlexSegment add_wafer method
            self.segment.add_wafer(lift, rotation, self.cylinder_diameter, outside_height, wafer_type)
            if debug:
                print(f"  Successfully created wafer {wafer_num}")
        except Exception as e:
            print(f"  ERROR creating wafer {wafer_num}: {e}")
            raise


    def get_curve_info(self) -> Dict[str, Any]:
        """Get information about the curve being followed.

        Returns:
            Dictionary containing curve statistics and properties
        """
        return self.curves.get_curve_info()

    def add_curve_visualization(self, group_name: str = None) -> str:
        """Add visual vertices along the curve aligned with wafer geometry."""
        if group_name is None:
            group_name = "curve_vertices"

        # Get the segment object to extract coordinate transformations
        segment_obj = self.segment.get_segment_object()

        if segment_obj:
            # Use segment coordinate system for alignment
            return self.curves.add_visualization_vertices(segment_obj, group_name)
        else:
            # Fallback: try using segment base LCS coordinate system
            segment_base = self.segment.get_lcs_base()
            if segment_base:
                print("Using segment base LCS for coordinate alignment")
                return self.curves.add_visualization_vertices_with_lcs(segment_base, group_name)
            else:
                print("Warning: No coordinate reference found, using raw coordinates")
                return self.curves.add_visualization_vertices(None, group_name)

    def process_wafers(self, add_curve_vertices: bool = False, debug: bool = True) -> None:
        """Main processing method that creates and adds wafers to the segment."""
        # Step 1: Check feasibility
        if not self.check_feasibility():
            raise ValueError("No feasible solution exists - curve has too tight curvature")

        # Step 2: Create wafer list
        wafers = self.create_wafer_list()
        if debug:
            print(f"Created {len(wafers)} wafers before lift angle correction")

        # Step 3: Correct lift angles for complementary cutting
        corrected_wafers = self._correct_lift_angles(wafers)
        if debug:
            print(f"Applied lift angle corrections for complementary cutting")

        # Step 4: Process each wafer using the adapter
        if debug:
            print(f"\n=== Processing {len(corrected_wafers)} wafers ===")

        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            if debug:
                print(f"\nProcessing wafer {i + 1}/{len(corrected_wafers)}:")
            self.add_wafer_from_curve_data(start_point, end_point, start_angle,
                                           end_angle, rotation_angle, wafer_type, debug)

        # Step 5: Add curve vertices AFTER wafer creation so we can use wafer coordinate system
        if add_curve_vertices:
            if debug:
                print(f"\nAdding aligned curve visualization vertices...")
            self.add_curve_visualization()

        if debug:
            print(f"\n=== Finished processing all wafers ===")
