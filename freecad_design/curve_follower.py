import math
import numpy as np
from typing import List, Tuple, Callable, Any
import FreeCADGui


class CurveFollower:  # Renamed as requested
    def __init__(self, doc, segment: Any, cylinder_diameter: float, curve: str,
                 nbr_wafers: int, scale: float, min_height: float, max_chord: float):
        """
        Initialize the CurveFollower class for creating wafer slices along a curved cylinder.

        Args:
            segment: FlexSegment object with add_wafer method
            cylinder_diameter: Diameter of the cylinder
            curve: Name of the curve generation method in this class
            nbr_wafers: Number of wafers to create
            scale: Multiplier for all x,y,z curve values
            min_height: Minimum distance between wafer end surfaces
            max_chord: Maximum chord distance for wafer approximation
        """
        self.doc = doc
        self.segment = segment
        self.cylinder_diameter = cylinder_diameter
        self.radius = cylinder_diameter / 2.0
        self.curve_name = curve
        self.nbr_wafers = nbr_wafers
        self.scale = scale
        self.min_height = min_height
        self.max_chord = max_chord

        # Get the curve generation function
        self.curve_func = getattr(self, curve)

        # Generate and scale the curve points
        self.curve_points = self._generate_scaled_curve()

        # Calculate curve properties
        self.curve_length = self._calculate_curve_length()
        self.curvatures = self._calculate_curvatures()

    def _generate_scaled_curve(self) -> np.ndarray:
        """Generate the curve points and apply scaling."""
        points = self.curve_func()
        return np.array(points) * self.scale

    def _calculate_curve_length(self) -> float:
        """Calculate the total length of the curve."""
        if len(self.curve_points) < 2:
            return 0.0

        diffs = np.diff(self.curve_points, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.sum(lengths)

    def _calculate_curvatures(self) -> np.ndarray:
        """Calculate radius of curvature at each point along the curve."""
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
        """
        Check if a feasible solution exists by ensuring minimum radius of curvature
        allows for minimum wafer height.
        """
        min_radius_of_curvature = np.min(self.curvatures)

        if min_radius_of_curvature <= self.radius:
            return False

        max_possible_height = 2 * math.sqrt(min_radius_of_curvature ** 2 - self.radius ** 2)
        return max_possible_height >= self.min_height

    def _calculate_chord_distance(self, start_point: np.ndarray, end_point: np.ndarray,
                                  curve_segment: np.ndarray) -> float:
        """Calculate maximum distance between chord and curve segment."""
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
        """Get the tangent vector at a specific curve point index."""
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
        """Calculate ellipse angles and wafer type for the wafer ends."""

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
        """
        Create a list of wafers satisfying min_height and max_chord constraints.

        Returns:
            List of tuples: (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type)
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
        """
        Correct lift angles so adjacent wafers have complementary angles.
        Returns corrected wafer list with same structure.
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

            if i < len(wafers) - 1:  # Not the last wafer
                # Keep the calculated end angle for now - will be used by next wafer
                pass

            corrected_wafers.append((start_point, end_point, corrected_start_angle,
                                     corrected_end_angle, rotation_angle, wafer_type))

        return corrected_wafers

    def _calculate_outside_height(self, start_point: np.ndarray, end_point: np.ndarray,
                                  start_angle: float, end_angle: float, rotation_angle: float) -> float:
        """
        Calculate the outside height using a more robust method.
        """
        # Base distance between points (this is the minimum height)
        chord_length = np.linalg.norm(end_point - start_point)

        # For very small angles (near 0°), treat as circular - no extension needed
        # For larger angles, add a reasonable extension based on geometry

        def safe_ellipse_extension(angle):
            """Calculate ellipse extension safely avoiding division by zero."""
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
        """
        Adapter method to convert curve follower data to FlexSegment.add_wafer() parameters.
        """
        # Calculate lift angle (sum of the two end angles)
        lift = start_angle + end_angle

        # Rotation is the rotation between major axes
        rotation = rotation_angle

        # Calculate outside height
        outside_height = self._calculate_outside_height(start_point, end_point,
                                                        start_angle, end_angle, rotation_angle)

        # Add validation for the final wafer
        wafer_num = self.segment.get_wafer_count() + 1
        is_final = (wafer_num == len(self.create_wafer_list()))
        debug = True
        if debug:
            print(f"=== Wafer {wafer_num} ===")
            print(f"  Is final wafer: {is_final}")
            print(f"  Start point: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"  End point: [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"  Start angle: {math.degrees(start_angle):.2f}°")
            print(f"  End angle: {math.degrees(end_angle):.2f}°")
            print(f"  Lift (sum): {math.degrees(lift):.2f}°")
            print(f"  Rotation: {math.degrees(rotation):.2f}°")
            print(f"  Outside height: {outside_height:.4f}")
            print(f"  Wafer type: {wafer_type}")

            # Additional validation
            if outside_height <= 0:
                print(f"  ERROR: Invalid outside height!")
            if math.isnan(lift) or math.isnan(rotation):
                print(f"  ERROR: NaN values detected!")

            print(f"  About to call segment.add_wafer...")

        try:
            # Call the FlexSegment add_wafer method
            self.segment.add_wafer(lift, rotation, self.cylinder_diameter, outside_height, wafer_type)
            print(f"  Successfully created wafer {wafer_num}")
        except Exception as e:
            print(f"  ERROR creating wafer {wafer_num}: {e}")
            raise

    def process_wafers(self):
        """Main processing method that creates and adds wafers to the segment."""
        # Step 1: Check feasibility
        if not self.check_feasibility():
            raise ValueError("No feasible solution exists - curve has too tight curvature")

        # Step 2: Create wafer list
        wafers = self.create_wafer_list()
        print(f"Created {len(wafers)} wafers before lift angle correction")

        # Debug: Print all wafer info
        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(wafers):
            print(
                f"Raw Wafer {i + 1}: type={wafer_type}, start_angle={math.degrees(start_angle):.1f}°, end_angle={math.degrees(end_angle):.1f}°")

        # Step 3: Correct lift angles for complementary cutting
        corrected_wafers = self._correct_lift_angles(wafers)
        print(f"Applied lift angle corrections for complementary cutting")

        # Debug: Print corrected wafer info
        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            print(
                f"Corrected Wafer {i + 1}: type={wafer_type}, start_angle={math.degrees(start_angle):.1f}°, end_angle={math.degrees(end_angle):.1f}°")

        # Step 4: Process each wafer using the adapter
        print(f"\n=== Processing {len(corrected_wafers)} wafers ===")
        for i, (start_point, end_point, start_angle, end_angle, rotation_angle, wafer_type) in enumerate(
                corrected_wafers):
            print(f"\nProcessing wafer {i + 1}/{len(corrected_wafers)}:")
            self.add_wafer_from_curve_data(start_point, end_point, start_angle,
                                           end_angle, rotation_angle, wafer_type, debug=True)

        print(f"\n=== Finished processing all wafers ===")

    # Example curve functions - implement your specific curves here
    def linear_curve(self) -> List[List[float]]:
        """Generate a simple linear curve for testing."""
        points = []
        for i in range(100):
            t = i / 99.0
            points.append([0, 0, t * 100])
        return points

    def helical_curve(self) -> List[List[float]]:
        """Generate a helical curve."""
        points = []
        for i in range(100):
            t = i / 99.0 * 4 * math.pi
            x = 10 * math.cos(t)
            y = 10 * math.sin(t)
            z = t * 2.5
            points.append([x, y, z])
        return points

    def sinusoidal_curve(self) -> List[List[float]]:
        """Generate a sinusoidal curve."""
        points = []
        for i in range(100):
            t = i / 99.0 * 4 * math.pi
            x = t * 2
            y = 5 * math.sin(t)
            z = 2 * math.cos(t * 0.5)
            points.append([x, y, z])
        return points

# Example usage:
# curve_follower = CurveFollower(
#     segment=my_flex_segment,
#     cylinder_diameter=10.0,
#     curve="helical_curve",
#     nbr_wafers=20,
#     scale=1.5,
#     min_height=2.0,
#     max_chord=0.5
# )
# curve_follower.process_wafers()