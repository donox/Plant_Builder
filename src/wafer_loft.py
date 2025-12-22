"""
wafer_loft.py - Loft-based wafer creation

This module creates wafers by:
1. Creating a 3D loft from a curved path (for reference)
2. Sampling points along the loft
3. Creating straight cylinder wafers between cutting planes
4. Each wafer is a straight cylinder approximating the curved section
"""

import FreeCAD as App
import Part
import math
from core.logging_setup import get_logger

logger = get_logger(__name__)


class Wafer:
    """Wafer object with straight cylinder geometry and LCS information"""
    def __init__(self, solid, index, plane1, plane2, geometry, lcs1=None, lcs2=None):
        self.wafer = solid
        self.index = index
        self.plane1 = plane1
        self.plane2 = plane2
        self.geometry = geometry
        self.lcs1 = lcs1
        self.lcs2 = lcs2

    @property
    def volume(self):
        """Get wafer volume"""
        return self.wafer.Volume if self.wafer else 0.0

    @property
    def lift_angle(self):
        """Get lift angle in degrees"""
        return self.geometry['lift_angle_deg']

    @property
    def rotation_angle(self):
        """Get rotation angle in degrees"""
        return self.geometry['rotation_angle_deg']


class LoftWaferGenerator:
    """Generate wafers by creating straight cylinders between cutting planes"""

    def __init__(self, cylinder_radius=1.0, wafer_settings=None):
        """
        Initialize loft wafer generator

        Args:
            cylinder_radius: Radius of the cylinder
            wafer_settings: Dictionary with wafer configuration (optional)
        """
        self.cylinder_radius = cylinder_radius
        self.wafer_settings = wafer_settings if wafer_settings else {}
        self.spine_curve = None
        self.loft = None
        self.sample_points = []
        self.cutting_planes = []
        self.wafers = []
        self.spine = None
        self.radius = cylinder_radius
        logger.debug(f"LoftWaferGenerator initialized with radius {self.radius:.3f}")

    def create_spine_from_points(self, points):
        """
        Create a smooth 3D curve (spine) from a list of points

        Args:
            points: List of App.Vector points defining the curve

        Returns:
            Part.Edge representing the spine
        """
        logger.debug(f"Creating spine from {len(points)} points")

        if len(points) < 2:
            raise ValueError("Need at least 2 points to create a spine")

        self.spine = Part.BSplineCurve()
        self.spine.interpolate(points)
        self.spine_curve = self.spine.toShape()

        logger.info(f"Spine created, length: {self.spine_curve.Length:.3f}")
        return self.spine_curve

    def _create_profiles(self, num_profiles):
        """Create circular profiles perpendicular to spine at evenly spaced parameters"""
        profiles = []
        spine_edge = Part.Edge(self.spine)

        for i in range(num_profiles):
            # Calculate parameter along spine (0.0 to 1.0)
            u = i / (num_profiles - 1) if num_profiles > 1 else 0.0

            # Get point and tangent at this parameter
            point = self.spine.value(u)
            tangent = self.spine.tangent(u)[0]  # Returns tuple (tangent, )
            tangent.normalize()

            # Create a plane perpendicular to the tangent
            # Use arbitrary perpendicular vector for the plane's normal direction
            if abs(tangent.z) < 0.9:
                perp = App.Vector(0, 0, 1).cross(tangent)
            else:
                perp = App.Vector(1, 0, 0).cross(tangent)
            perp.normalize()

            # Create circle in the plane
            circle = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), self.radius)
            circle_edge = Part.Edge(circle)
            circle_wire = Part.Wire(circle_edge)

            # Create placement: translate to point, rotate to align with tangent
            # Z-axis should align with tangent
            rotation = App.Rotation(App.Vector(0, 0, 1), tangent)
            placement = App.Placement(point, rotation)

            # Transform the circle
            profile = circle_wire.transformGeometry(placement.toMatrix())
            profiles.append(profile)

        return profiles

    def create_loft_along_spine(self, curve_follower):
        """Generate loft along spine and sample wafers"""
        try:
            logger.info(f"Creating loft along spine")

            # Get spine length - self.spine is a BSplineCurve, we need to convert to Edge to get Length
            if hasattr(self, 'spine') and self.spine:
                spine_edge = Part.Edge(self.spine)
                spine_length = spine_edge.Length
            else:
                spine_length = 0

            logger.debug(f"Spine length: {spine_length:.3f}")

            # Calculate number of profiles based on spine length
            # Read profile_density from wafer_settings
            profiles_per_unit = self.wafer_settings.get('profile_density', 0.89)
            num_profiles = max(10, int(spine_length * profiles_per_unit))

            logger.debug(f"Auto-calculated {num_profiles} profiles ({profiles_per_unit:.2f} per unit)")

            # Create profiles
            profiles = self._create_profiles(num_profiles)
            logger.debug(f"Created {len(profiles)} profiles")

            # Check for potential issues before lofting
            self._diagnose_profiles(profiles, spine_length)

            # Try to create solid loft
            try:
                self.loft = Part.makeLoft(profiles, True, True)
                logger.info(f"Loft created as solid, volume: {self.loft.Volume:.4f}")
            except Part.OCCError as e:
                logger.warning(f"Could not create solid loft: {e}")
                logger.info("Attempting to create shell loft")
                self.loft = Part.makeLoft(profiles, False, True)
                logger.info(f"Loft created as shell")

        except Part.OCCError as e:
            # Provide diagnostic guidance
            error_msg = f"Failed to create loft: {e}"
            logger.error(error_msg)

            # Add helpful suggestions
            try:
                suggestions = self._get_loft_failure_suggestions(curve_follower, num_profiles)
            except:
                suggestions = ["Could not generate detailed diagnostics"]

            logger.error("LOFT FAILURE DIAGNOSTICS:")
            for i, suggestion in enumerate(suggestions, 1):
                logger.error(f"  {i}. {suggestion}")

            raise RuntimeError(error_msg)

        def _create_profiles(self, num_profiles):
            """Create circular profiles perpendicular to spine at evenly spaced parameters"""
            profiles = []
            spine_edge = Part.Edge(self.spine)

            for i in range(num_profiles):
                # Calculate parameter along spine (0.0 to 1.0)
                u = i / (num_profiles - 1) if num_profiles > 1 else 0.0

                # Get point and tangent at this parameter
                point = self.spine.value(u)
                tangent = self.spine.tangent(u)[0]  # Returns tuple (tangent, )
                tangent.normalize()

                # Create a plane perpendicular to the tangent
                # Use arbitrary perpendicular vector for the plane's normal direction
                if abs(tangent.z) < 0.9:
                    perp = App.Vector(0, 0, 1).cross(tangent)
                else:
                    perp = App.Vector(1, 0, 0).cross(tangent)
                perp.normalize()

                # Create circle in the plane
                circle = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), self.radius)
                circle_edge = Part.Edge(circle)
                circle_wire = Part.Wire(circle_edge)

                # Create placement: translate to point, rotate to align with tangent
                # Z-axis should align with tangent
                rotation = App.Rotation(App.Vector(0, 0, 1), tangent)
                placement = App.Placement(point, rotation)

                # Transform the circle
                profile = circle_wire.transformGeometry(placement.toMatrix())
                profiles.append(profile)

            return profiles

    def _diagnose_profiles(self, profiles, spine_length):
        """Check profiles for potential loft issues"""
        issues = []

        # Check for profile intersections
        for i in range(len(profiles) - 1):
            # Use BoundBox center instead of CenterOfMass
            center1 = profiles[i].BoundBox.Center
            center2 = profiles[i + 1].BoundBox.Center
            dist = center1.distanceToPoint(center2)
            if dist < 0.01:
                issues.append(f"Profiles {i} and {i + 1} are very close (dist={dist:.4f})")

        # Check profile sizes
        radii = [p.BoundBox.DiagonalLength / 2 for p in profiles]
        max_radius = max(radii)
        min_spacing = spine_length / len(profiles)

        if max_radius > min_spacing:
            issues.append(
                f"Profile diameter ({max_radius * 2:.3f}) larger than spacing ({min_spacing:.3f}) - profiles may intersect")

        # Check for degenerate profiles
        for i, profile in enumerate(profiles):
            if profile.Length < 0.01:
                issues.append(f"Profile {i} is degenerate (length={profile.Length:.4f})")

        if issues:
            logger.warning("Profile diagnostic warnings:")
            for issue in issues:
                logger.warning(f"  ⚠ {issue}")

    def _get_loft_failure_suggestions(self, curve_follower, num_profiles):
        """Generate helpful suggestions when loft fails"""
        suggestions = []

        # Get current settings
        diameter = self.radius * 2
        spine_length = curve_follower.spine.Length if curve_follower.spine else 0

        # Suggestion 1: Reduce cylinder diameter
        suggested_diameter = diameter * 0.6
        suggestions.append(
            f"Reduce cylinder_diameter from {diameter:.2f} to {suggested_diameter:.2f} "
            f"(smaller diameter reduces self-intersection risk)"
        )

        # Suggestion 2: Increase max_chord
        current_chord = self.max_chord if hasattr(self, 'max_chord') else 0.2
        suggested_chord = min(current_chord * 1.5, 0.5)
        suggestions.append(
            f"Increase max_chord from {current_chord:.2f} to {suggested_chord:.2f} "
            f"(fewer, larger wafers = simpler geometry)"
        )

        # Suggestion 3: Reduce point count
        if num_profiles > 50:
            suggested_points = int(num_profiles * 0.5)
            suggestions.append(
                f"Reduce curve points from {num_profiles} to ~{suggested_points} "
                f"(fewer profiles = less complex loft)"
            )

        # Suggestion 4: Check for tight curves
        suggestions.append(
            "Check if curve has very tight bends or self-intersections - "
            "increase scale parameter or simplify curve geometry"
        )

        # Suggestion 5: Profile density
        if spine_length > 0:
            density = num_profiles / spine_length
            if density > 1.0:
                suggestions.append(
                    f"Profile density is high ({density:.2f} per unit) - "
                    f"this can cause profile overlap in tight curves"
                )

        return suggestions

    def sample_points_along_loft(self, chord_distance_algorithm):
        """
        Sample points along the loft spine using chord-distance algorithm

        Args:
            chord_distance_algorithm: Function that takes spine curve and returns sample parameters

        Returns:
            List of dictionaries with 'point', 'tangent', 'parameter'
        """
        logger.info("Sampling points along loft")

        if self.spine_curve is None:
            raise ValueError("Must create spine first")

        sample_params = chord_distance_algorithm(self.spine_curve)

        curve = self.spine_curve.Curve
        self.sample_points = []

        for param in sample_params:
            point = curve.value(param)
            tangent = curve.tangent(param)[0]
            tangent.normalize()

            self.sample_points.append({
                'point': point,
                'tangent': tangent,
                'parameter': param
            })

        logger.info(f"{len(self.sample_points)} sample points generated")
        return self.sample_points

    def calculate_cutting_planes(self):
        """
        Create cutting planes perpendicular to spine at each sample point

        Returns:
            List of cutting plane data dictionaries
        """
        # logger.debug("Calculating cutting planes")

        self.cutting_planes = []

        for sample in self.sample_points:
            self.cutting_planes.append({
                'point': sample['point'],
                'normal': sample['tangent'],
                'parameter': sample['parameter']
            })

        logger.info(f"{len(self.cutting_planes)} cutting planes created")
        return self.cutting_planes

    def _calculate_basic_geometry(self, plane1, plane2, chord_vector):
        """Calculate basic wafer geometry without rotation angle"""
        n1 = plane1['normal']
        n2 = plane2['normal']

        cos_dihedral = n1.dot(n2)
        cos_dihedral = max(-1.0, min(1.0, cos_dihedral))
        lift_angle_rad = math.acos(cos_dihedral)

        chord_length = chord_vector.Length
        if chord_length < 1e-9:
            chord_dir = App.Vector(1, 0, 0)
        else:
            chord_dir = App.Vector(chord_vector.x, chord_vector.y, chord_vector.z)
            chord_dir.normalize()

        major1_dir = chord_dir - n1 * (chord_dir.dot(n1))
        if major1_dir.Length < 1e-9:
            if abs(n1.z) < 0.9:
                major1_dir = App.Vector(0, 0, 1).cross(n1)
            else:
                major1_dir = App.Vector(1, 0, 0).cross(n1)
            major1_dir.normalize()
            major1_length = self.radius
        else:
            major1_dir.normalize()
            cos_angle = abs(n1.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major1_length = self.radius / cos_angle

        major2_dir = chord_dir - n2 * (chord_dir.dot(n2))
        if major2_dir.Length < 1e-9:
            if abs(n2.z) < 0.9:
                major2_dir = App.Vector(0, 0, 1).cross(n2)
            else:
                major2_dir = App.Vector(1, 0, 0).cross(n2)
            major2_dir.normalize()
            major2_length = self.radius
        else:
            major2_dir.normalize()
            cos_angle = abs(n2.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major2_length = self.radius / cos_angle

        bisector_normal = (n1 + n2)
        if bisector_normal.Length > 0:
            bisector_normal.normalize()
        else:
            bisector_normal = n1

        bisector_point = (plane1['point'] + plane2['point']) * 0.5

        minor1_dir = n1.cross(major1_dir)
        if minor1_dir.Length > 0:
            minor1_dir.normalize()
        else:
            minor1_dir = chord_dir.cross(major1_dir).normalize()

        minor2_dir = n2.cross(major2_dir)
        if minor2_dir.Length > 0:
            minor2_dir.normalize()
        else:
            minor2_dir = chord_dir.cross(major2_dir).normalize()

        geometry = {
            'lift_angle_rad': lift_angle_rad,
            'lift_angle_deg': math.degrees(lift_angle_rad),
            'rotation_angle_rad': 0.0,
            'rotation_angle_deg': 0.0,
            'center1': plane1['point'],
            'center2': plane2['point'],
            'chord_vector': chord_vector,
            'chord_length': chord_length,
            'bisector_normal': bisector_normal,
            'bisector_point': bisector_point,
            'ellipse1': {
                'center': plane1['point'],
                'normal': n1,
                'major_axis': major1_length,
                'minor_axis': self.radius,
                'major_axis_vector': major1_dir,
                'minor_axis_vector': minor1_dir
            },
            'ellipse2': {
                'center': plane2['point'],
                'normal': n2,
                'major_axis': major2_length,
                'minor_axis': self.radius,
                'major_axis_vector': major2_dir,
                'minor_axis_vector': minor2_dir
            }
        }

        return geometry

    def generate_all_wafers_by_slicing(self):
        """Generate wafers as straight cylinders between cutting planes"""
        if len(self.cutting_planes) < 2:
            raise ValueError(f"Need at least 2 cutting planes, got {len(self.cutting_planes)}")

        logger.info(f"Creating straight cylinder wafers at {len(self.cutting_planes)} planes")

        wafer_data_list = []
        chord_vectors = []

        min_chord_threshold = self.radius * 0.01
        min_volume_threshold = 0.001

        def get_face_at_position(solid, target_point, tolerance=0.5):
            """Find face whose center is closest to target_point"""
            best_face = None
            min_dist = float('inf')
            for face in solid.Faces:
                face_center = face.CenterOfMass
                dist = (face_center - target_point).Length
                if dist < min_dist:
                    min_dist = dist
                    best_face = face
            if min_dist < tolerance:
                return best_face
            else:
                logger.warning(f"No face found near {target_point}, closest was {min_dist:.4f} away")
                return None

        def get_ellipse_from_face(face, target_center):
            """Extract ellipse parameters from actual face geometry"""
            if face is None:
                return None

            # Get face normal (actual, from geometry)
            normal = face.normalAt(0, 0)
            center = face.CenterOfMass
            # logger.debug(
            #     f"Face CenterOfMass={center}, target was {target_center}, distance={(center - target_center).Length:.6f}")

            # Find the ellipse or circle edge
            for edge in face.Edges:
                curve = edge.Curve
                # Check if it's an ellipse or circle
                if hasattr(curve, 'Radius'):
                    # It's a circle
                    radius = curve.Radius
                    major_radius = radius
                    minor_radius = radius

                    # For circles, choose a consistent major axis direction
                    # Use the face normal to pick perpendicular direction
                    axis = curve.Axis if hasattr(curve, 'Axis') else normal
                    if abs(axis.z) < 0.9:
                        major_axis_dir = App.Vector(0, 0, 1).cross(axis)
                    else:
                        major_axis_dir = App.Vector(1, 0, 0).cross(axis)
                    major_axis_dir.normalize()

                    # Ensure major_axis_dir is in the plane
                    major_axis_dir = major_axis_dir - normal * (major_axis_dir.dot(normal))
                    if major_axis_dir.Length > 1e-9:
                        major_axis_dir.normalize()
                    # logger.debug(f"Extracted circle: center={center}, normal={normal}, radius={self.radius}")

                elif hasattr(curve, 'MajorRadius'):
                    # It's an ellipse
                    major_radius = curve.MajorRadius
                    minor_radius = curve.MinorRadius

                    # Get major axis direction from the curve
                    if hasattr(curve, 'XAxis'):
                        major_axis_dir = curve.XAxis
                    else:
                        # Fallback
                        if abs(normal.z) < 0.9:
                            major_axis_dir = App.Vector(0, 0, 1).cross(normal)
                        else:
                            major_axis_dir = App.Vector(1, 0, 0).cross(normal)
                        major_axis_dir.normalize()

                    # Ensure major_axis_dir is in the plane (perpendicular to normal)
                    major_axis_dir = major_axis_dir - normal * (major_axis_dir.dot(normal))
                    if major_axis_dir.Length > 1e-9:
                        major_axis_dir.normalize()
                    else:
                        # Fallback if parallel
                        if abs(normal.z) < 0.9:
                            major_axis_dir = App.Vector(0, 0, 1).cross(normal)
                        else:
                            major_axis_dir = App.Vector(1, 0, 0).cross(normal)
                        major_axis_dir.normalize()
                    # logger.debug(
                    #     f"Extracted ellipse: center={center}, normal={normal}, major_axis_dir={major_axis_dir}")
                else:
                    continue

                return {
                    'center': center,
                    'normal': normal,
                    'major_radius': major_radius,
                    'minor_radius': minor_radius,
                    'major_axis_vector': major_axis_dir
                }

            # No ellipse or circle found
            logger.warning(f"No ellipse/circle edge found on face at {center}")
            return None

        for i in range(len(self.cutting_planes) - 1):
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            # logger.debug(f"Creating wafer {i}")

            try:
                center1 = plane1['point']
                center2 = plane2['point']
                chord_vector = center2 - center1
                chord_length = chord_vector.Length

                if chord_length < min_chord_threshold:
                    logger.info(f"Wafer {i} stopped (chord {chord_length:.4f} below threshold)")
                    break

                chord_direction = App.Vector(chord_vector.x, chord_vector.y, chord_vector.z)
                chord_direction.normalize()

                cylinder = Part.makeCylinder(
                    self.radius,
                    chord_length * 1.5,
                    center1 - chord_direction * chord_length * 0.25,
                    chord_direction
                )

                plane_size = self.radius * 100

                circle1 = Part.Circle(plane1['point'], plane1['normal'], plane_size)
                wire1 = Part.Wire([circle1.toShape()])
                face1 = Part.Face(wire1)
                half_space1 = face1.extrude(-plane1['normal'] * plane_size)
                wafer = cylinder.cut(half_space1)

                circle2 = Part.Circle(plane2['point'], plane2['normal'], plane_size)
                wire2 = Part.Wire([circle2.toShape()])
                face2 = Part.Face(wire2)
                half_space2 = face2.extrude(plane2['normal'] * plane_size)
                wafer = wafer.cut(half_space2)

                if wafer.ShapeType == 'Compound' and len(wafer.Solids) > 0:
                    wafer = wafer.Solids[0]

                if wafer.Volume > 1e-6:
                    if wafer.Volume < min_volume_threshold:
                        logger.info(f"Wafer {i} stopped (volume {wafer.Volume:.6f} below threshold)")
                        break

                    chord_vectors.append(chord_vector)

                    # Extract ACTUAL ellipse geometry from the wafer faces
                    actual_face1 = get_face_at_position(wafer, center1)
                    actual_face2 = get_face_at_position(wafer, center2)

                    actual_ellipse1 = get_ellipse_from_face(actual_face1, center1)
                    actual_ellipse2 = get_ellipse_from_face(actual_face2, center2)

                    if actual_ellipse1 is None or actual_ellipse2 is None:
                        logger.error(f"Wafer {i} failed to extract ellipse geometry")
                        wafer_data_list.append(None)
                        continue

                    # Use actual geometry for the wafer data
                    geometry = {
                        'chord_vector': chord_vector,
                        'chord_length': chord_length,
                        'ellipse1': actual_ellipse1,
                        'ellipse2': actual_ellipse2,
                        'rotation_angle_rad': 0.0,
                        'rotation_angle_deg': 0.0,
                    }

                    wafer_data_list.append({
                        'solid': wafer,
                        'geometry': geometry,
                        'plane1': plane1,
                        'plane2': plane2
                    })
                    # logger.debug(f"Wafer {i} created, volume: {wafer.Volume:.4f}")
                else:
                    logger.warning(f"Wafer {i} failed (zero volume)")
                    wafer_data_list.append(None)

            except Exception as e:
                logger.error(f"Wafer {i} error: {e}", exc_info=True)
                wafer_data_list.append(None)

        # Rotation angle calculation remains the same...
        # logger.debug("Calculating rotation angles")

        chord_0a = None
        chord_0b = None

        if len(wafer_data_list) >= 2 and self.spine_curve is not None:
            param0 = self.cutting_planes[0]['parameter']
            param1 = self.cutting_planes[1]['parameter']
            param_mid = param0 + (param1 - param0) * 0.5

            curve = self.spine_curve.Curve
            midpoint = curve.value(param_mid)

            center0 = self.cutting_planes[0]['point']
            center1 = self.cutting_planes[1]['point']

            chord_0a = midpoint - center0
            chord_0b = center1 - midpoint

        for i, wafer_data in enumerate(wafer_data_list):
            if wafer_data is None:
                continue

            collinearity_angle = 0.0
            chord_azimuth = 0.0

            chord_vec = chord_vectors[i]
            chord_azimuth = math.degrees(math.atan2(chord_vec.x, chord_vec.y))

            if i == 0:
                rotation_angle_deg = 0.0
                collinearity_angle = 0.0

            elif i == 1:
                if chord_0a is not None and chord_0b is not None:
                    chord_0a_dir = App.Vector(chord_0a.x, chord_0a.y, chord_0a.z)
                    chord_0b_dir = App.Vector(chord_0b.x, chord_0b.y, chord_0b.z)
                    chord_1_dir = App.Vector(chord_vectors[1].x, chord_vectors[1].y, chord_vectors[1].z)

                    if chord_0a_dir.Length > 1e-9:
                        chord_0a_dir.normalize()
                    if chord_0b_dir.Length > 1e-9:
                        chord_0b_dir.normalize()
                    if chord_1_dir.Length > 1e-9:
                        chord_1_dir.normalize()

                    cos_collin = chord_0b_dir.dot(chord_1_dir)
                    cos_collin = max(-1.0, min(1.0, cos_collin))
                    collinearity_angle = math.degrees(math.acos(cos_collin))

                    plane_A_normal = chord_0a_dir.cross(chord_0b_dir)
                    plane_B_normal = chord_0b_dir.cross(chord_1_dir)

                    if plane_A_normal.Length < 1e-9 or plane_B_normal.Length < 1e-9:
                        rotation_angle_deg = 0.0
                    else:
                        plane_A_normal.normalize()
                        plane_B_normal.normalize()

                        cos_angle = plane_A_normal.dot(plane_B_normal)
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        rotation_angle_rad = math.acos(cos_angle)

                        cross = plane_A_normal.cross(plane_B_normal)
                        if cross.dot(chord_0b_dir) < 0:
                            rotation_angle_rad = -rotation_angle_rad

                        rotation_angle_deg = math.degrees(rotation_angle_rad) * 2.0
                else:
                    rotation_angle_deg = 0.0

            else:
                chord_im2_dir = App.Vector(chord_vectors[i - 2].x, chord_vectors[i - 2].y, chord_vectors[i - 2].z)
                chord_im1_dir = App.Vector(chord_vectors[i - 1].x, chord_vectors[i - 1].y, chord_vectors[i - 1].z)
                chord_i_dir = App.Vector(chord_vectors[i].x, chord_vectors[i].y, chord_vectors[i].z)

                if chord_im2_dir.Length > 1e-9:
                    chord_im2_dir.normalize()
                if chord_im1_dir.Length > 1e-9:
                    chord_im1_dir.normalize()
                if chord_i_dir.Length > 1e-9:
                    chord_i_dir.normalize()

                cos_collin = chord_im1_dir.dot(chord_i_dir)
                cos_collin = max(-1.0, min(1.0, cos_collin))
                collinearity_angle = math.degrees(math.acos(cos_collin))

                plane_A_normal = chord_im2_dir.cross(chord_im1_dir)
                plane_B_normal = chord_im1_dir.cross(chord_i_dir)

                if plane_A_normal.Length < 1e-9 or plane_B_normal.Length < 1e-9:
                    rotation_angle_deg = 0.0
                else:
                    plane_A_normal.normalize()
                    plane_B_normal.normalize()

                    cos_angle = plane_A_normal.dot(plane_B_normal)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    rotation_angle_rad = math.acos(cos_angle)

                    cross = plane_A_normal.cross(plane_B_normal)
                    if cross.dot(chord_im1_dir) < 0:
                        rotation_angle_rad = -rotation_angle_rad

                    rotation_angle_deg = math.degrees(rotation_angle_rad)

            wafer_data['geometry']['rotation_angle_rad'] = math.radians(rotation_angle_deg)
            wafer_data['geometry']['rotation_angle_deg'] = rotation_angle_deg
            wafer_data['geometry']['collinearity_angle_deg'] = collinearity_angle
            wafer_data['geometry']['chord_azimuth_deg'] = chord_azimuth

            logger.coord(f"Wafer {i}: rotation = {rotation_angle_deg:.2f}°, "
                         f"collinearity = {collinearity_angle:.4f}°, azimuth = {chord_azimuth:.2f}°")

        successful = sum(1 for w in wafer_data_list if w is not None)
        total_volume = sum(w['solid'].Volume for w in wafer_data_list if w is not None)

        logger.info(f"Created {successful} wafers")
        if successful < len(self.cutting_planes) - 1:
            logger.info(f"Stopped early - {len(self.cutting_planes) - 1 - successful} degenerate wafers skipped")
        logger.info(f"Total volume: {total_volume:.4f}")

        return wafer_data_list

    def _create_lcs(self, center, normal, major_axis_dir):
        """
        Create LCS placement from actual ellipse geometry

        Args:
            center: Center point of the ellipse
            normal: Ellipse normal vector (becomes Z-axis direction)
            major_axis_dir: Ellipse major axis vector (becomes X-axis direction)

        Returns:
            App.Placement: LCS with X along major axis, Z along normal, Y completing right-hand rule
        """
        # logger.debug(f"_create_lcs called:")
        # logger.debug(f"  center={center}")
        # logger.debug(f"  normal (target Z)={normal}")
        # logger.debug(f"  major_axis_dir (target X)={major_axis_dir}")

        # Normalize Z-axis (normal)
        z_axis = App.Vector(normal.x, normal.y, normal.z)
        z_axis_length = z_axis.Length
        # logger.debug(f"  Z-axis length before normalization: {z_axis_length:.6f}")

        if z_axis_length < 1e-9:
            logger.error(f"  Z-axis has zero length! Using fallback (0,0,1)")
            z_axis = App.Vector(0, 0, 1)
        else:
            z_axis.normalize()

        # Normalize X-axis (major axis)
        x_axis = App.Vector(major_axis_dir.x, major_axis_dir.y, major_axis_dir.z)
        x_axis_length = x_axis.Length
        # logger.debug(f"  X-axis length before normalization: {x_axis_length:.6f}")

        if x_axis_length < 1e-9:
            logger.error(f"  X-axis has zero length! Using fallback")
            if abs(z_axis.z) < 0.9:
                x_axis = App.Vector(0, 0, 1).cross(z_axis)
            else:
                x_axis = App.Vector(1, 0, 0).cross(z_axis)
            x_axis.normalize()
        else:
            x_axis.normalize()

        # Ensure X is perpendicular to Z (project X onto plane perpendicular to Z)
        dot_product = x_axis.dot(z_axis)
        # logger.debug(f"  X·Z dot product: {dot_product:.9f}")

        if abs(dot_product) > 1e-6:
            logger.info(f"  X and Z are not perpendicular, projecting X onto Z's perpendicular plane")
            x_axis = x_axis - z_axis * dot_product
            x_axis_length_after_projection = x_axis.Length
            # logger.debug(f"  X-axis length after projection: {x_axis_length_after_projection:.6f}")

            if x_axis_length_after_projection > 1e-9:
                x_axis.normalize()
            else:
                logger.warning(f"  X-axis became zero after projection! Using fallback")
                if abs(z_axis.z) < 0.9:
                    x_axis = App.Vector(0, 0, 1).cross(z_axis)
                else:
                    x_axis = App.Vector(1, 0, 0).cross(z_axis)
                x_axis.normalize()
        else:
            # logger.debug(f"  X and Z are already perpendicular")
            pass

        # Calculate Y = Z × X (right-hand rule)
        y_axis = z_axis.cross(x_axis)
        y_axis_length = y_axis.Length
        # logger.debug(f"  Y-axis (Z×X) length: {y_axis_length:.6f}")

        if y_axis_length < 1e-9:
            logger.error(f"  Y-axis has zero length! X and Z might be parallel")
            # Emergency fallback
            y_axis = App.Vector(0, 1, 0)
        else:
            y_axis.normalize()

        # Verify orthogonality
        xy_dot = x_axis.dot(y_axis)
        yz_dot = y_axis.dot(z_axis)
        zx_dot = z_axis.dot(x_axis)
        # logger.debug(f"  Orthogonality check: X·Y={xy_dot:.9f}, Y·Z={yz_dot:.9f}, Z·X={zx_dot:.9f}")

        if abs(xy_dot) > 1e-6 or abs(yz_dot) > 1e-6 or abs(zx_dot) > 1e-6:
            logger.warning(f"  Axes are not orthogonal!")

        # logger.debug(f"  Final LCS axes:")
        # logger.debug(f"    X={x_axis} (length={x_axis.Length:.6f})")
        # logger.debug(f"    Y={y_axis} (length={y_axis.Length:.6f})")
        # logger.debug(f"    Z={z_axis} (length={z_axis.Length:.6f})")

        # Create placement from axes
        placement = App.Placement()
        placement.Base = center

        # Build rotation matrix: columns are X, Y, Z axes
        rotation_matrix = App.Matrix()
        rotation_matrix.A11, rotation_matrix.A12, rotation_matrix.A13 = x_axis.x, y_axis.x, z_axis.x
        rotation_matrix.A21, rotation_matrix.A22, rotation_matrix.A23 = x_axis.y, y_axis.y, z_axis.y
        rotation_matrix.A31, rotation_matrix.A32, rotation_matrix.A33 = x_axis.z, y_axis.z, z_axis.z

        placement.Rotation = App.Rotation(rotation_matrix)
        # logger.debug(f"  Created LCS placement: {placement}")

        return placement

    def generate_wafers(self):
        """Generate all wafers with geometry and LCS information"""
        logger.info("Generating wafers with LCS")

        if len(self.cutting_planes) < 2:
            raise ValueError("Need at least 2 cutting planes to create wafers")

        wafer_data_list = self.generate_all_wafers_by_slicing()

        self.wafers = []

        for i, wafer_data in enumerate(wafer_data_list):
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            # Skip degenerate wafers (None from generate_all_wafers_by_slicing)
            if wafer_data is None:
                logger.debug(f"Skipping degenerate wafer {i}")
                continue  # ← Don't add to wafer list at all

            solid = wafer_data['solid']
            geometry = wafer_data['geometry']

            ellipse1 = geometry['ellipse1']
            ellipse2 = geometry['ellipse2']

            # Get chord direction for consistent orientation
            chord_direction = App.Vector(
                geometry['chord_vector'].x,
                geometry['chord_vector'].y,
                geometry['chord_vector'].z
            )
            chord_direction.normalize()

            # For lcs1 (entry): ensure Z-axis points along the chord direction
            normal1 = App.Vector(
                ellipse1['normal'].x,
                ellipse1['normal'].y,
                ellipse1['normal'].z
            )
            major_axis1 = App.Vector(
                ellipse1['major_axis_vector'].x,
                ellipse1['major_axis_vector'].y,
                ellipse1['major_axis_vector'].z
            )

            # Check if normal points opposite to chord direction
            if normal1.dot(chord_direction) < 0:
                normal1 = -normal1
                major_axis1 = -major_axis1  # Flip major axis to maintain right-hand rule

            lcs1 = self._create_lcs(
                ellipse1['center'],
                normal1,
                major_axis1
            )

            # For lcs2 (exit): ensure Z-axis points along the chord direction
            normal2 = App.Vector(
                ellipse2['normal'].x,
                ellipse2['normal'].y,
                ellipse2['normal'].z
            )
            major_axis2 = App.Vector(
                ellipse2['major_axis_vector'].x,
                ellipse2['major_axis_vector'].y,
                ellipse2['major_axis_vector'].z
            )

            # Check if normal points opposite to chord direction
            if normal2.dot(chord_direction) < 0:
                normal2 = -normal2
                major_axis2 = -major_axis2  # Flip major axis to maintain right-hand rule

            lcs2 = self._create_lcs(
                ellipse2['center'],
                normal2,
                major_axis2
            )

            wafer_obj = Wafer(
                solid=solid,
                index=i,
                plane1=plane1,
                plane2=plane2,
                geometry=geometry,
                lcs1=lcs1,
                lcs2=lcs2
            )

            self.wafers.append(wafer_obj)

        successful = len(self.wafers)
        logger.info(f"Packaged {successful} wafers with LCS")

        return self.wafers
    def visualize_in_freecad(self, doc, show_lcs=True, show_cutting_planes=True, lcs_size=None):
        """Create FreeCAD objects to visualize wafers, loft, and LCS"""
        logger.info("Creating FreeCAD visualization")

        if lcs_size is None:
            lcs_size = self.radius * 2

        if self.spine_curve:
            spine_obj = doc.addObject("Part::Feature", "Loft_Spine")
            spine_obj.Shape = self.spine_curve
            spine_obj.ViewObject.LineColor = (1.0, 0.5, 0.0)
            spine_obj.ViewObject.LineWidth = 4
            # logger.debug("Added spine curve")

        if self.loft:
            loft_obj = doc.addObject("Part::Feature", "Loft_Reference")
            loft_obj.Shape = self.loft
            loft_obj.ViewObject.ShapeColor = (0.7, 0.7, 0.5)
            loft_obj.ViewObject.Transparency = 85
            # logger.debug("Added loft reference")

        if show_cutting_planes:
            cutting_planes_group = doc.addObject("App::DocumentObjectGroup", "CuttingPlanes")

            for i, plane in enumerate(self.cutting_planes):
                plane_radius = self.radius * 2
                circle = Part.Circle(plane['point'], plane['normal'], plane_radius)
                wire = Part.Wire([circle.toShape()])
                face = Part.Face(wire)

                plane_obj = doc.addObject("Part::Feature", f"Plane_{i:03d}")
                plane_obj.Shape = face
                plane_obj.ViewObject.ShapeColor = (0.8, 0.9, 1.0)
                plane_obj.ViewObject.Transparency = 85

                cutting_planes_group.addObject(plane_obj)
            # logger.debug(f"Added {len(self.cutting_planes)} cutting planes")

        wafer_count = 0
        for wafer_obj in self.wafers:
            if wafer_obj.wafer is not None:
                wafer_group = doc.addObject("App::DocumentObjectGroup", f"Wafer_{wafer_obj.index:03d}")

                wafer_solid_obj = doc.addObject("Part::Feature", f"Wafer_{wafer_obj.index:03d}_Solid")
                wafer_solid_obj.Shape = wafer_obj.wafer

                if wafer_obj.index % 2 == 0:
                    wafer_solid_obj.ViewObject.ShapeColor = (0.9, 0.8, 0.4)
                else:
                    wafer_solid_obj.ViewObject.ShapeColor = (0.8, 0.9, 0.5)

                wafer_solid_obj.ViewObject.Transparency = 0
                wafer_group.addObject(wafer_solid_obj)

                if show_lcs and wafer_obj.lcs1 and wafer_obj.lcs2:
                    lcs_group = doc.addObject("App::DocumentObjectGroup", f"Wafer_{wafer_obj.index:03d}_LCS")

                    lcs1_obj = self._create_lcs_object(
                        doc,
                        wafer_obj.lcs1,
                        f"Wafer_{wafer_obj.index:03d}_LCS_Face1",
                        lcs_size
                    )

                    lcs2_obj = self._create_lcs_object(
                        doc,
                        wafer_obj.lcs2,
                        f"Wafer_{wafer_obj.index:03d}_LCS_Face2",
                        lcs_size
                    )

                    lcs_group.addObject(lcs1_obj)
                    lcs_group.addObject(lcs2_obj)

                    wafer_group.addObject(lcs_group)

                wafer_count += 1

        # logger.debug(f"Added {wafer_count} wafers to visualization")
        doc.recompute()
        logger.info("Visualization complete")

    def _create_lcs_object(self, doc, placement, name, size):
        """Create a proper FreeCAD Local Coordinate System object"""
        try:
            lcs = doc.addObject("PartDesign::CoordinateSystem", name)
            lcs.Placement = placement

            if hasattr(lcs.ViewObject, 'AxisLength'):
                lcs.ViewObject.AxisLength = size

            return lcs

        except:
            origin = placement.Base
            rotation = placement.Rotation

            x_dir = rotation.multVec(App.Vector(1, 0, 0))
            x_line = Part.makeLine(origin, origin + x_dir * size)

            y_dir = rotation.multVec(App.Vector(0, 1, 0))
            y_line = Part.makeLine(origin, origin + y_dir * size)

            z_dir = rotation.multVec(App.Vector(0, 0, 1))
            z_line = Part.makeLine(origin, origin + z_dir * size)

            compound = Part.makeCompound([x_line, y_line, z_line])

            lcs = doc.addObject("Part::Feature", name)
            lcs.Shape = compound

            return lcs


def simple_chord_distance_sampler(spine_edge, target_chord_distance=0.5):
    """Simple chord-distance algorithm for sampling points along curve"""
    curve = spine_edge.Curve
    first_param = curve.FirstParameter
    last_param = curve.LastParameter
    total_length = spine_edge.Length

    num_samples = max(2, int(total_length / target_chord_distance) + 1)

    params = []
    for i in range(num_samples):
        t = first_param + (last_param - first_param) * i / (num_samples - 1)
        params.append(t)
    # logger.debug(f"Chord distance sampler: {num_samples} samples for length {total_length:.3f}")
    return params