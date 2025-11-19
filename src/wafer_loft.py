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


class Wafer:
    """Wafer object with straight cylinder geometry and LCS information"""
    def __init__(self, solid, index, plane1, plane2, geometry, lcs1=None, lcs2=None):
        self.wafer = solid  # Part.Solid - the straight cylinder wafer
        self.index = index
        self.plane1 = plane1
        self.plane2 = plane2
        self.geometry = geometry
        self.lcs1 = lcs1  # LCS at first elliptical face
        self.lcs2 = lcs2  # LCS at second elliptical face

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

    def __init__(self, cylinder_radius=1.0):
        """
        Initialize loft wafer generator

        Args:
            cylinder_radius: Radius of the cylindrical wafers
        """
        self.cylinder_radius = cylinder_radius
        self.spine_curve = None
        self.loft = None
        self.sample_points = []
        self.cutting_planes = []
        self.wafers = []

    def create_spine_from_points(self, points):
        """
        Create a smooth 3D curve (spine) from a list of points

        Args:
            points: List of App.Vector points defining the curve

        Returns:
            Part.Edge representing the spine
        """
        print(f"Creating spine from {len(points)} points...")

        if len(points) < 2:
            raise ValueError("Need at least 2 points to create a spine")

        # Create B-spline through points
        spline = Part.BSplineCurve()
        spline.interpolate(points)
        self.spine_curve = spline.toShape()

        print(f"✓ Spine created, length: {self.spine_curve.Length:.3f}")
        return self.spine_curve

    def create_loft_along_spine(self, num_profiles=None, profiles_per_unit=None):
        """
        Create a cylindrical loft along the spine (for reference visualization)

        Args:
            num_profiles: Number of circular profiles (if None, calculated automatically)
            profiles_per_unit: Profiles per unit length (default: ~2 per cylinder diameter)

        Returns:
            Part.Shape representing the loft
        """
        print(f"Creating loft along spine...")

        if self.spine_curve is None:
            raise ValueError("Must create spine first")

        # Get spine parameters
        curve = self.spine_curve.Curve
        first_param = curve.FirstParameter
        last_param = curve.LastParameter
        spine_length = self.spine_curve.Length

        print(f"  Spine length: {spine_length:.3f}")

        # Calculate adaptive number of profiles if not specified
        if num_profiles is None:
            if profiles_per_unit is None:
                # Default: about 2 profiles per cylinder diameter
                profiles_per_unit = 2.0 / (2 * self.cylinder_radius)

            num_profiles = max(10, int(spine_length * profiles_per_unit))
            num_profiles = min(num_profiles, 200)  # Cap at 200 for performance
            print(f"  Auto-calculated {num_profiles} profiles ({profiles_per_unit:.2f} per unit)")
        else:
            print(f"  Using specified {num_profiles} profiles")

        # Create circular profiles at regular intervals along spine
        profiles = []

        for i in range(num_profiles):
            # Parameter along curve
            t = first_param + (last_param - first_param) * i / (num_profiles - 1)

            # Get point and tangent at this parameter
            point = curve.value(t)
            tangent = curve.tangent(t)[0]
            tangent.normalize()

            # Create a circular profile perpendicular to tangent
            # Find perpendicular vector to tangent for circle plane
            if abs(tangent.z) < 0.9:
                perp1 = App.Vector(0, 0, 1).cross(tangent)
            else:
                perp1 = App.Vector(1, 0, 0).cross(tangent)
            perp1.normalize()

            # Create circle in the plane perpendicular to tangent
            circle = Part.Circle()
            circle.Center = point
            circle.Axis = tangent
            circle.Radius = self.cylinder_radius

            wire = Part.Wire([circle.toShape()])
            profiles.append(wire)

        print(f"  Created {len(profiles)} profiles")

        # Create loft through all profiles as a solid
        try:
            self.loft = Part.makeLoft(profiles, True, True)  # solid=True, ruled=True
            print(f"✓ Loft created as solid, volume: {self.loft.Volume:.4f}")
        except Exception as e:
            print(f"Warning: Could not create solid loft: {e}")
            print("Attempting to create shell loft...")
            try:
                self.loft = Part.makeLoft(profiles, False, True)
                print(f"✓ Loft created as shell")
            except Exception as e2:
                raise ValueError(f"Failed to create loft: {e2}")

        # Validate the loft
        if self.loft is None:
            raise ValueError("Loft creation returned None")

        if not self.loft.isValid():
            raise ValueError("Created loft is not valid")

        return self.loft

    def sample_points_along_loft(self, chord_distance_algorithm):
        """
        Sample points along the loft spine using chord-distance algorithm

        Args:
            chord_distance_algorithm: Function that takes spine curve and returns sample parameters

        Returns:
            List of dictionaries with 'point', 'tangent', 'parameter'
        """
        print("Sampling points along loft...")

        if self.spine_curve is None:
            raise ValueError("Must create spine first")

        # Use the provided chord-distance algorithm
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

        print(f"✓ {len(self.sample_points)} sample points generated")
        return self.sample_points

    def calculate_cutting_planes(self):
        """
        Create cutting planes perpendicular to spine at each sample point

        Returns:
            List of cutting plane data dictionaries
        """
        print("Calculating cutting planes...")

        self.cutting_planes = []

        for sample in self.sample_points:
            self.cutting_planes.append({
                'point': sample['point'],
                'normal': sample['tangent'],
                'parameter': sample['parameter']
            })

        print(f"✓ {len(self.cutting_planes)} cutting planes created")
        return self.cutting_planes

    def calculate_wafer_geometry(self, plane1, plane2, chord_vector, prev_chord_vector=None):
        """
        Calculate all wafer geometry analytically from planes and chord

        Args:
            plane1: First cutting plane (with 'point' and 'normal')
            plane2: Second cutting plane
            chord_vector: Vector from plane1 to plane2 (cylinder axis)
            prev_chord_vector: Previous wafer's chord vector (for rotation calculation)

        Returns:
            Dictionary with lift_angle, rotation_angle, ellipse parameters, and LCS data
        """
        # Normals
        n1 = plane1['normal']
        n2 = plane2['normal']

        # Lift angle = dihedral angle between planes
        cos_dihedral = n1.dot(n2)
        cos_dihedral = max(-1.0, min(1.0, cos_dihedral))
        lift_angle_rad = math.acos(cos_dihedral)

        # Chord (cylinder axis) direction
        chord_length = chord_vector.Length
        if chord_length < 1e-9:
            chord_dir = App.Vector(1, 0, 0)
        else:
            chord_dir = App.Vector(chord_vector.x, chord_vector.y, chord_vector.z)
            chord_dir.normalize()

        # Calculate rotation angle using dihedral between consecutive chord planes
        if prev_chord_vector is None:
            # First wafer - no rotation (starting from global XY plane)
            rotation_angle_rad = 0.0
        else:
            # Calculate dihedral angle between:
            # Plane A: contains prev_chord and current chord
            # Plane B: contains current chord and next chord (approximated by n2 direction)

            prev_chord_dir = App.Vector(prev_chord_vector.x, prev_chord_vector.y, prev_chord_vector.z)
            if prev_chord_dir.Length > 1e-9:
                prev_chord_dir.normalize()
            else:
                prev_chord_dir = App.Vector(1, 0, 0)

            # Normal to plane containing prev_chord and current_chord
            plane_A_normal = prev_chord_dir.cross(chord_dir)

            if plane_A_normal.Length < 1e-9:
                # Chords are parallel - no rotation
                rotation_angle_rad = 0.0
            else:
                plane_A_normal.normalize()

                # For the next plane, we need the next chord direction
                # We approximate this using the change in tangent direction (n2 vs n1)
                # The plane containing current_chord would have normal perpendicular to chord
                # and in the direction of the curve's bending

                # Alternative: use the plane perpendicular to chord that contains the
                # "bending direction" of the curve
                # The bending direction is approximated by (n2 - n1)

                bend_direction = n2 - n1
                if bend_direction.Length < 1e-9:
                    rotation_angle_rad = 0.0
                else:
                    # Project bend direction onto plane perpendicular to chord
                    bend_in_plane = bend_direction - chord_dir * bend_direction.dot(chord_dir)

                    if bend_in_plane.Length < 1e-9:
                        rotation_angle_rad = 0.0
                    else:
                        bend_in_plane.normalize()

                        # Similarly for previous chord
                        prev_bend = n1 - prev_chord_dir * n1.dot(prev_chord_dir)
                        if prev_bend.Length < 1e-9:
                            rotation_angle_rad = 0.0
                        else:
                            prev_bend.normalize()

                            # Rotation is the angle between these bend directions
                            # as measured around the chord axis
                            cos_rot = prev_bend.dot(bend_in_plane)
                            cos_rot = max(-1.0, min(1.0, cos_rot))
                            rotation_angle_rad = math.acos(cos_rot)

                            # Determine sign
                            cross = prev_bend.cross(bend_in_plane)
                            if cross.dot(chord_dir) < 0:
                                rotation_angle_rad = -rotation_angle_rad

        # Calculate ellipse major axis directions (for LCS)
        major1_dir = chord_dir - n1 * (chord_dir.dot(n1))
        if major1_dir.Length < 1e-9:
            if abs(n1.z) < 0.9:
                major1_dir = App.Vector(0, 0, 1).cross(n1)
            else:
                major1_dir = App.Vector(1, 0, 0).cross(n1)
            major1_dir.normalize()
            major1_length = self.cylinder_radius
        else:
            major1_dir.normalize()
            cos_angle = abs(n1.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major1_length = self.cylinder_radius / cos_angle

        major2_dir = chord_dir - n2 * (chord_dir.dot(n2))
        if major2_dir.Length < 1e-9:
            if abs(n2.z) < 0.9:
                major2_dir = App.Vector(0, 0, 1).cross(n2)
            else:
                major2_dir = App.Vector(1, 0, 0).cross(n2)
            major2_dir.normalize()
            major2_length = self.cylinder_radius
        else:
            major2_dir.normalize()
            cos_angle = abs(n2.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major2_length = self.cylinder_radius / cos_angle

        # Calculate bisector for reference
        bisector_normal = (n1 + n2)
        if bisector_normal.Length > 0:
            bisector_normal.normalize()
        else:
            bisector_normal = n1

        bisector_point = (plane1['point'] + plane2['point']) * 0.5

        # Minor axis vectors
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

        # Package geometry
        geometry = {
            'lift_angle_rad': lift_angle_rad,
            'lift_angle_deg': math.degrees(lift_angle_rad),
            'rotation_angle_rad': rotation_angle_rad,
            'rotation_angle_deg': math.degrees(rotation_angle_rad),
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
                'minor_axis': self.cylinder_radius,
                'major_axis_vector': major1_dir,
                'minor_axis_vector': minor1_dir
            },
            'ellipse2': {
                'center': plane2['point'],
                'normal': n2,
                'major_axis': major2_length,
                'minor_axis': self.cylinder_radius,
                'major_axis_vector': major2_dir,
                'minor_axis_vector': minor2_dir
            }
        }

        return geometry

    def _create_lcs(self, center, z_axis, major_axis_vector):
        """
        Create Local Coordinate System at ellipse center

        Args:
            center: Origin point (center of ellipse)
            z_axis: Normal to plane (perpendicular to ellipse)
            major_axis_vector: Major axis direction of ellipse

        Returns:
            App.Placement representing the LCS
        """
        # Z-axis is the normal
        z = z_axis.normalize()

        # X-axis is along major axis
        x = major_axis_vector.normalize()

        # Y-axis completes right-hand system
        y = z.cross(x)
        if y.Length < 1e-9:
            # Degenerate case - choose arbitrary perpendicular
            if abs(z.z) < 0.9:
                y = App.Vector(0, 0, 1).cross(z)
            else:
                y = App.Vector(1, 0, 0).cross(z)
        y.normalize()

        # Re-orthogonalize x to ensure perfect perpendicularity
        x = y.cross(z)
        x.normalize()

        # Create rotation matrix
        rotation = App.Rotation(
            App.Matrix(
                x.x, y.x, z.x, 0,
                x.y, y.y, z.y, 0,
                x.z, y.z, z.z, 0,
                0, 0, 0, 1
            )
        )

        # Create placement
        placement = App.Placement(center, rotation)

        return placement

    def generate_all_wafers_by_slicing(self):
        """
        Generate wafers as straight cylinders between cutting planes

        Calculates rotation using dihedral angle between consecutive chord planes.
        For wafer 0: rotation = 0 (no previous chord)
        For wafer 1: uses midpoint of wafer 0's curve to create reference plane, doubled
        For wafer 2+: dihedral between plane(chord_i-2, chord_i-1) and plane(chord_i-1, chord_i)

        Returns:
            List of dictionaries with 'solid' and geometry
        """
        if len(self.cutting_planes) < 2:
            raise ValueError(f"Need at least 2 cutting planes, got {len(self.cutting_planes)}")

        print(f"\nCreating straight cylinder wafers at {len(self.cutting_planes)} planes...")

        wafer_data_list = []
        chord_vectors = []

        min_chord_threshold = self.cylinder_radius * 0.01
        min_volume_threshold = 0.001

        # First pass: create all wafer solids and collect chord vectors
        for i in range(len(self.cutting_planes) - 1):
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            print(f"  Creating wafer {i}...", end='')

            try:
                center1 = plane1['point']
                center2 = plane2['point']
                chord_vector = center2 - center1
                chord_length = chord_vector.Length

                if chord_length < min_chord_threshold:
                    print(f" STOPPED (chord {chord_length:.4f} below threshold)")
                    break

                chord_direction = App.Vector(chord_vector.x, chord_vector.y, chord_vector.z)
                chord_direction.normalize()

                cylinder = Part.makeCylinder(
                    self.cylinder_radius,
                    chord_length * 1.5,
                    center1 - chord_direction * chord_length * 0.25,
                    chord_direction
                )

                plane_size = self.cylinder_radius * 100

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
                        print(f" STOPPED (volume {wafer.Volume:.6f} below threshold)")
                        break

                    chord_vectors.append(chord_vector)

                    geometry = self._calculate_basic_geometry(plane1, plane2, chord_vector)

                    wafer_data_list.append({
                        'solid': wafer,
                        'geometry': geometry,
                        'plane1': plane1,
                        'plane2': plane2
                    })
                    print(f" volume: {wafer.Volume:.4f}")
                else:
                    print(f" FAILED (zero volume)")
                    wafer_data_list.append(None)

            except Exception as e:
                print(f" ERROR: {e}")
                import traceback
                traceback.print_exc()
                wafer_data_list.append(None)

        # Second pass: calculate rotation angles using consecutive chords
        print(f"\n  Calculating rotation angles...")

        # For wafer 1, we need the midpoint of wafer 0's curve segment
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

            # Calculate chord azimuth (angle from +Y axis in XY plane)
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

                    # Collinearity: angle between chord_0b and chord[1]
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

                        # Double the rotation for wafer 1 (we only capture half the torsion)
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

                # Collinearity: angle between chord[i-1] and chord[i]
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

            # Update geometry
            wafer_data['geometry']['rotation_angle_rad'] = math.radians(rotation_angle_deg)
            wafer_data['geometry']['rotation_angle_deg'] = rotation_angle_deg
            wafer_data['geometry']['collinearity_angle_deg'] = collinearity_angle
            wafer_data['geometry']['chord_azimuth_deg'] = chord_azimuth

            print(
                f"    Wafer {i}: rotation = {rotation_angle_deg:.2f}°, collinearity = {collinearity_angle:.4f}°, azimuth = {chord_azimuth:.2f}°")

        # Summary
        successful = sum(1 for w in wafer_data_list if w is not None)
        total_volume = sum(w['solid'].Volume for w in wafer_data_list if w is not None)

        print(f"\n✓ Created {successful} wafers")
        if successful < len(self.cutting_planes) - 1:
            print(f"  (Stopped early - {len(self.cutting_planes) - 1 - successful} degenerate wafers skipped)")
        print(f"  Total volume: {total_volume:.4f}")

        return wafer_data_list

    def _calculate_basic_geometry(self, plane1, plane2, chord_vector):
        """
        Calculate basic wafer geometry without rotation angle
        (rotation is calculated separately using 3 consecutive chords)
        """
        n1 = plane1['normal']
        n2 = plane2['normal']

        # Lift angle = dihedral angle between planes
        cos_dihedral = n1.dot(n2)
        cos_dihedral = max(-1.0, min(1.0, cos_dihedral))
        lift_angle_rad = math.acos(cos_dihedral)

        # Chord length
        chord_length = chord_vector.Length
        if chord_length < 1e-9:
            chord_dir = App.Vector(1, 0, 0)
        else:
            chord_dir = App.Vector(chord_vector.x, chord_vector.y, chord_vector.z)
            chord_dir.normalize()

        # Calculate ellipse major axis directions (for LCS)
        major1_dir = chord_dir - n1 * (chord_dir.dot(n1))
        if major1_dir.Length < 1e-9:
            if abs(n1.z) < 0.9:
                major1_dir = App.Vector(0, 0, 1).cross(n1)
            else:
                major1_dir = App.Vector(1, 0, 0).cross(n1)
            major1_dir.normalize()
            major1_length = self.cylinder_radius
        else:
            major1_dir.normalize()
            cos_angle = abs(n1.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major1_length = self.cylinder_radius / cos_angle

        major2_dir = chord_dir - n2 * (chord_dir.dot(n2))
        if major2_dir.Length < 1e-9:
            if abs(n2.z) < 0.9:
                major2_dir = App.Vector(0, 0, 1).cross(n2)
            else:
                major2_dir = App.Vector(1, 0, 0).cross(n2)
            major2_dir.normalize()
            major2_length = self.cylinder_radius
        else:
            major2_dir.normalize()
            cos_angle = abs(n2.dot(chord_dir))
            cos_angle = max(1e-9, min(1.0, cos_angle))
            major2_length = self.cylinder_radius / cos_angle

        # Calculate bisector
        bisector_normal = (n1 + n2)
        if bisector_normal.Length > 0:
            bisector_normal.normalize()
        else:
            bisector_normal = n1

        bisector_point = (plane1['point'] + plane2['point']) * 0.5

        # Minor axis vectors
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

        # Package geometry (rotation will be added later)
        geometry = {
            'lift_angle_rad': lift_angle_rad,
            'lift_angle_deg': math.degrees(lift_angle_rad),
            'rotation_angle_rad': 0.0,  # Will be set in second pass
            'rotation_angle_deg': 0.0,  # Will be set in second pass
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
                'minor_axis': self.cylinder_radius,
                'major_axis_vector': major1_dir,
                'minor_axis_vector': minor1_dir
            },
            'ellipse2': {
                'center': plane2['point'],
                'normal': n2,
                'major_axis': major2_length,
                'minor_axis': self.cylinder_radius,
                'major_axis_vector': major2_dir,
                'minor_axis_vector': minor2_dir
            }
        }

        return geometry

    def generate_wafers(self):
        """
        Generate all wafers with geometry and LCS information

        Returns:
            List of Wafer objects
        """
        print("\nGenerating wafers with LCS...")

        if len(self.cutting_planes) < 2:
            raise ValueError("Need at least 2 cutting planes to create wafers")

        # Get the wafer solids with geometry
        wafer_data_list = self.generate_all_wafers_by_slicing()

        # Package into Wafer objects with LCS
        self.wafers = []

        # Iterate only over the wafers that were actually created
        for i, wafer_data in enumerate(wafer_data_list):
            # Get corresponding planes
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            if wafer_data is None:
                # Failed wafer
                wafer_obj = Wafer(
                    solid=None,
                    index=i,
                    plane1=plane1,
                    plane2=plane2,
                    geometry={},
                    lcs1=None,
                    lcs2=None
                )
            else:
                solid = wafer_data['solid']
                geometry = wafer_data['geometry']

                ellipse1 = geometry['ellipse1']
                ellipse2 = geometry['ellipse2']

                # Create LCS for each face
                lcs1 = self._create_lcs(
                    ellipse1['center'],
                    ellipse1['normal'],
                    ellipse1['major_axis_vector']
                )

                lcs2 = self._create_lcs(
                    ellipse2['center'],
                    ellipse2['normal'],
                    ellipse2['major_axis_vector']
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

        successful = sum(1 for w in self.wafers if w.wafer is not None)
        print(f"✓ Packaged {successful} wafers with LCS\n")

        return self.wafers

    def visualize_in_freecad(self, doc, show_lcs=True, show_cutting_planes=True, lcs_size=None):
        """
        Create FreeCAD objects to visualize wafers, loft, and LCS

        Args:
            doc: FreeCAD document
            show_lcs: Show local coordinate systems (default: True)
            show_cutting_planes: Show cutting plane discs (default: True)
            lcs_size: Size of LCS display (default: 2 * cylinder_radius)
        """
        print("Creating FreeCAD visualization...")

        if lcs_size is None:
            lcs_size = self.cylinder_radius * 2

        # Show spine (orange curve)
        if self.spine_curve:
            spine_obj = doc.addObject("Part::Feature", "Loft_Spine")
            spine_obj.Shape = self.spine_curve
            spine_obj.ViewObject.LineColor = (1.0, 0.5, 0.0)
            spine_obj.ViewObject.LineWidth = 4

        # Show loft (transparent - for reference)
        if self.loft:
            loft_obj = doc.addObject("Part::Feature", "Loft_Reference")
            loft_obj.Shape = self.loft
            loft_obj.ViewObject.ShapeColor = (0.7, 0.7, 0.5)
            loft_obj.ViewObject.Transparency = 85

        # Show cutting planes if requested
        if show_cutting_planes:
            # Create group for cutting planes
            cutting_planes_group = doc.addObject("App::DocumentObjectGroup", "CuttingPlanes")

            # Show cutting plane discs in group
            for i, plane in enumerate(self.cutting_planes):
                plane_radius = self.cylinder_radius * 2
                circle = Part.Circle(plane['point'], plane['normal'], plane_radius)
                wire = Part.Wire([circle.toShape()])
                face = Part.Face(wire)

                plane_obj = doc.addObject("Part::Feature", f"Plane_{i:03d}")
                plane_obj.Shape = face
                plane_obj.ViewObject.ShapeColor = (0.8, 0.9, 1.0)
                plane_obj.ViewObject.Transparency = 85

                cutting_planes_group.addObject(plane_obj)

        # Create wafer groups
        for wafer_obj in self.wafers:
            if wafer_obj.wafer is not None:
                # Create group for this wafer
                wafer_group = doc.addObject("App::DocumentObjectGroup", f"Wafer_{wafer_obj.index:03d}")

                # Add the wafer solid
                wafer_solid_obj = doc.addObject("Part::Feature", f"Wafer_{wafer_obj.index:03d}_Solid")
                wafer_solid_obj.Shape = wafer_obj.wafer

                # Color wafers alternately
                if wafer_obj.index % 2 == 0:
                    wafer_solid_obj.ViewObject.ShapeColor = (0.9, 0.8, 0.4)
                else:
                    wafer_solid_obj.ViewObject.ShapeColor = (0.8, 0.9, 0.5)

                wafer_solid_obj.ViewObject.Transparency = 0

                wafer_group.addObject(wafer_solid_obj)

                # Add LCS if requested
                if show_lcs and wafer_obj.lcs1 and wafer_obj.lcs2:
                    # Create subgroup for coordinate systems
                    lcs_group = doc.addObject("App::DocumentObjectGroup", f"Wafer_{wafer_obj.index:03d}_LCS")

                    # Create LCS objects
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

        doc.recompute()
        print("✓ Visualization complete")

    def _create_lcs_object(self, doc, placement, name, size):
        """
        Create a proper FreeCAD Local Coordinate System object

        Args:
            doc: FreeCAD document
            placement: App.Placement for the LCS
            name: Name for the LCS object
            size: Display size for the LCS

        Returns:
            The created LCS object
        """
        # Create an AxisCross object (standard FreeCAD coordinate system display)
        try:
            # Try to create PartDesign CoordinateSystem if available
            lcs = doc.addObject("PartDesign::CoordinateSystem", name)
            lcs.Placement = placement

            # Set display properties if available
            if hasattr(lcs.ViewObject, 'AxisLength'):
                lcs.ViewObject.AxisLength = size

            return lcs

        except:
            # Fallback: create using compound of three colored lines
            origin = placement.Base
            rotation = placement.Rotation

            # X-axis (red)
            x_dir = rotation.multVec(App.Vector(1, 0, 0))
            x_line = Part.makeLine(origin, origin + x_dir * size)

            # Y-axis (green)
            y_dir = rotation.multVec(App.Vector(0, 1, 0))
            y_line = Part.makeLine(origin, origin + y_dir * size)

            # Z-axis (blue)
            z_dir = rotation.multVec(App.Vector(0, 0, 1))
            z_line = Part.makeLine(origin, origin + z_dir * size)

            # Create compound
            compound = Part.makeCompound([x_line, y_line, z_line])

            lcs = doc.addObject("Part::Feature", name)
            lcs.Shape = compound

            return lcs


def simple_chord_distance_sampler(spine_edge, target_chord_distance=0.5):
    """
    Simple chord-distance algorithm for sampling points along curve

    Args:
        spine_edge: Part.Edge representing the spine
        target_chord_distance: Target chord distance between samples

    Returns:
        List of parameters along the curve
    """
    curve = spine_edge.Curve
    first_param = curve.FirstParameter
    last_param = curve.LastParameter
    total_length = spine_edge.Length

    # Approximate number of samples based on chord distance
    num_samples = max(2, int(total_length / target_chord_distance) + 1)

    # Uniform sampling by parameter (simplified)
    params = []
    for i in range(num_samples):
        t = first_param + (last_param - first_param) * i / (num_samples - 1)
        params.append(t)

    print(f"  Chord distance sampler: {num_samples} samples for length {total_length:.3f}")
    return params