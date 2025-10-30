"""
wafer_loft.py - Loft-based wafer creation

This module creates wafers by:
1. Creating a 3D loft from a curved path
2. Sampling points along the loft
3. Slicing the loft at all cutting planes
4. Extracting wafer solids from the slices
"""

import FreeCAD as App
import Part
import math


class Wafer:
    """Simple wafer object wrapper"""
    def __init__(self, solid, index, plane1, plane2, geometry):
        self.wafer = solid  # Part.Solid - the actual wafer shape
        self.index = index
        self.plane1 = plane1
        self.plane2 = plane2
        self.geometry = geometry

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
    """Generate wafers by slicing a loft"""

    def __init__(self, cylinder_radius=1.0):
        """
        Initialize loft wafer generator

        Args:
            cylinder_radius: Radius of the cylindrical loft
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
        Create a cylindrical loft along the spine

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

    def calculate_wafer_geometry(self, plane1, plane2):
        """
        Calculate lift angle and rotation angle for wafer between two planes

        Args:
            plane1: First cutting plane data
            plane2: Second cutting plane data

        Returns:
            Dictionary with lift_angle, rotation_angle, and other geometry
        """
        # Dihedral angle between planes = lift angle
        n1 = plane1['normal']
        n2 = plane2['normal']

        cos_dihedral = n1.dot(n2)
        cos_dihedral = max(-1.0, min(1.0, cos_dihedral))  # Clamp to [-1, 1]
        dihedral_angle = math.acos(cos_dihedral)
        lift_angle_rad = dihedral_angle

        # Calculate bisector plane
        bisector_normal = (n1 + n2)
        if bisector_normal.Length > 0:
            bisector_normal.normalize()
        else:
            bisector_normal = n1  # Planes are parallel

        bisector_point = (plane1['point'] + plane2['point']) * 0.5

        # Distance along spine
        spine_distance = (plane2['point'] - plane1['point']).Length

        # TODO: For rotation angle calculation:
        # 1. Find the ellipse intersections at each plane
        # 2. Find the major axes of each ellipse
        # 3. Project onto bisector plane
        # 4. Measure angle between projections

        geometry = {
            'lift_angle_rad': lift_angle_rad,
            'lift_angle_deg': math.degrees(lift_angle_rad),
            'rotation_angle_rad': 0.0,  # To be calculated from ellipse major axes
            'rotation_angle_deg': 0.0,
            'center1': plane1['point'],
            'center2': plane2['point'],
            'bisector_normal': bisector_normal,
            'bisector_point': bisector_point,
            'spine_distance': spine_distance
        }

        return geometry

    def generate_all_wafers_by_slicing(self):
        """
        Slice the loft at all cutting planes to create wafers

        Returns:
            List of wafer solids (Part.Solid objects)
        """
        # Better error checking
        if self.loft is None:
            raise ValueError(
                "Loft is None - loft creation must have failed.\n"
                "Check create_loft_along_spine() output for errors."
            )
        if self.loft is None or len(self.cutting_planes) < 2:
            raise ValueError("Need loft and at least 2 cutting planes")

        print(f"\nSlicing loft at {len(self.cutting_planes)} planes...")
        print(f"Original loft volume: {self.loft.Volume:.4f}")

        wafer_solids = []
        plane_size = self.cylinder_radius * 100

        # Create each wafer independently from the original loft
        for i in range(len(self.cutting_planes) - 1):
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            print(f"  Creating wafer {i}...", end='')

            try:
                # Start with a fresh copy of the loft
                wafer = self.loft.copy()

                # Create circular face for plane 1
                plane1_circle = Part.Circle(
                    plane1['point'],
                    plane1['normal'],
                    plane_size
                )
                plane1_wire = Part.Wire([plane1_circle.toShape()])
                plane1_face = Part.Face(plane1_wire)

                # Extrude backward to create half-space to cut away
                half_space1 = plane1_face.extrude(-plane1['normal'] * plane_size)
                wafer = wafer.cut(half_space1)

                # Create circular face for plane 2
                plane2_circle = Part.Circle(
                    plane2['point'],
                    plane2['normal'],
                    plane_size
                )
                plane2_wire = Part.Wire([plane2_circle.toShape()])
                plane2_face = Part.Face(plane2_wire)

                # Extrude forward to create half-space to cut away
                half_space2 = plane2_face.extrude(plane2['normal'] * plane_size)
                wafer = wafer.cut(half_space2)

                # Handle compound
                if wafer.ShapeType == 'Compound' and len(wafer.Solids) > 0:
                    wafer = wafer.Solids[0]

                if wafer.Volume > 1e-6:
                    wafer_solids.append(wafer)
                    print(f" volume: {wafer.Volume:.4f}")
                else:
                    wafer_solids.append(None)
                    print(f" FAILED (zero volume)")

            except Exception as e:
                print(f" ERROR: {e}")
                wafer_solids.append(None)

        # Summary
        successful = sum(1 for w in wafer_solids if w is not None)
        total_volume = sum(w.Volume for w in wafer_solids if w is not None)

        print(f"\n✓ Created {successful}/{len(self.cutting_planes)-1} wafers")
        print(f"  Total volume: {total_volume:.4f} ({total_volume/self.loft.Volume*100:.1f}% of loft)")

        return wafer_solids

    def generate_wafers(self):
        """
        Generate all wafers by slicing the loft at cutting planes

        Returns:
            List of Wafer objects (with .wafer attribute containing Part.Solid)
        """
        print("\nGenerating wafers from loft slices...")

        if len(self.cutting_planes) < 2:
            raise ValueError("Need at least 2 cutting planes to create wafers")

        # Get the wafer solids from slicing
        wafer_solids = self.generate_all_wafers_by_slicing()

        # Package into Wafer objects
        self.wafers = []
        for i in range(len(self.cutting_planes) - 1):
            plane1 = self.cutting_planes[i]
            plane2 = self.cutting_planes[i + 1]

            geometry = self.calculate_wafer_geometry(plane1, plane2)

            wafer_obj = Wafer(
                solid=wafer_solids[i] if i < len(wafer_solids) else None,
                index=i,
                plane1=plane1,
                plane2=plane2,
                geometry=geometry
            )

            self.wafers.append(wafer_obj)

        successful = sum(1 for w in self.wafers if w.wafer is not None)
        print(f"✓ Packaged {successful} wafers\n")

        return self.wafers

    def visualize_in_freecad(self, doc, show_profiles=False, show_spine=True, show_loft=True, show_planes=True):
        """
        Create FreeCAD objects to visualize the loft and wafers

        Args:
            doc: FreeCAD document
            show_profiles: Show the individual circular profiles used to create loft (default: False)
            show_spine: Show the spine curve (default: True)
            show_loft: Show the full loft (default: True)
            show_planes: Show cutting plane discs (default: True)
        """
        print("Creating FreeCAD visualization...")

        # Show spine
        if show_spine and self.spine_curve:
            spine_obj = doc.addObject("Part::Feature", "Loft_Spine")
            spine_obj.Shape = self.spine_curve
            spine_obj.ViewObject.LineColor = (1.0, 0.5, 0.0)
            spine_obj.ViewObject.LineWidth = 4

        # Show loft (transparent)
        if show_loft and self.loft:
            loft_obj = doc.addObject("Part::Feature", "Loft")
            loft_obj.Shape = self.loft
            loft_obj.ViewObject.ShapeColor = (0.7, 0.7, 0.5)
            loft_obj.ViewObject.Transparency = 80  # Make more transparent

        # Show cutting planes as discs
        if show_planes:
            for i, plane in enumerate(self.cutting_planes):
                plane_radius = self.cylinder_radius * 2
                circle = Part.Circle(plane['point'], plane['normal'], plane_radius)
                wire = Part.Wire([circle.toShape()])
                face = Part.Face(wire)

                plane_obj = doc.addObject("Part::Feature", f"CuttingPlane_{i:03d}")
                plane_obj.Shape = face
                plane_obj.ViewObject.ShapeColor = (0.8, 0.9, 1.0)
                plane_obj.ViewObject.Transparency = 70

        # Show wafers (MAIN FOCUS)
        for wafer_obj in self.wafers:
            if wafer_obj.wafer is not None:
                fc_obj = doc.addObject("Part::Feature", f"Wafer_{wafer_obj.index:03d}")
                fc_obj.Shape = wafer_obj.wafer

                # Color wafers alternately for easy identification
                if wafer_obj.index % 2 == 0:
                    fc_obj.ViewObject.ShapeColor = (0.9, 0.8, 0.4)
                else:
                    fc_obj.ViewObject.ShapeColor = (0.8, 0.9, 0.5)

                fc_obj.ViewObject.Transparency = 0  # Wafers are solid

        doc.recompute()
        print("✓ Visualization complete")

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