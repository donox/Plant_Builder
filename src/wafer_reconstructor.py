"""
wafer_reconstructor.py - Reconstruct wafer structure from cut list parameters

This module builds wafers by applying the cutting angles (blade tilt, cylinder rotation)
to verify that the cut list parameters correctly reproduce the FreeCAD model.
"""

import FreeCAD as App
import Part
import math


class WaferReconstructor:
    """Reconstruct wafer structure from cut list parameters"""

    def __init__(self, cylinder_radius=0.9375):
        """
        Initialize reconstructor

        Args:
            cylinder_radius: Radius of the cylinder being cut
        """
        self.cylinder_radius = cylinder_radius
        self.wafers = []
        self.positions = []  # Track position of each wafer center

    def build_from_cut_list(self, cut_data, rotation_multiplier=1.0, initial_orientation=None):
        """
        Reconstruct using ONLY physical cutting parameters: length, blade angle, rotation angle.
        Cylinder angle is ignored - it's only for cutting instructions.

        Args:
            cut_data: List of dicts with cutting parameters
            rotation_multiplier: Factor to multiply rotation angles (for testing)
            initial_orientation: Dict with 'azimuth' to position first wafer (optional, for alignment)
        """
        print(f"\nReconstructing wafer structure...")
        print(f"  Rotation multiplier: {rotation_multiplier}")

        self.wafers = []
        self.positions = []

        face_center = App.Vector(0, 0, 0)

        # Initial face1 normal (tangent to curve at start)
        if initial_orientation and 'azimuth' in initial_orientation:
            azimuth_rad = math.radians(initial_orientation['azimuth'])
            face1_normal = App.Vector(
                math.sin(azimuth_rad),
                math.cos(azimuth_rad),
                0
            )
        else:
            face1_normal = App.Vector(0, 1, 0)

        # Stable up reference for determining bend axis
        up_ref = App.Vector(0, 0, 1)

        for i, cut in enumerate(cut_data):
            chord_length = cut['chord_length']
            blade_angle_deg = cut['blade_angle']
            rotation_deg = cut['rotation_angle'] * rotation_multiplier

            lift_angle_rad = math.radians(blade_angle_deg * 2.0)
            rotation_rad = math.radians(rotation_deg)

            # Bend axis: perpendicular to face1_normal and up_ref
            # This is the axis around which the path bends (lift)
            bend_axis = face1_normal.cross(up_ref)
            if bend_axis.Length < 1e-9:
                # If face1_normal is parallel to up_ref, choose perpendicular
                if abs(face1_normal.x) < 0.9:
                    bend_axis = App.Vector(1, 0, 0)
                else:
                    bend_axis = App.Vector(0, 1, 0)
            else:
                bend_axis.normalize()

            # Rotate bend_axis by rotation angle around face1_normal
            # This tilts the plane of bending, causing Z-rise
            if abs(rotation_deg) > 0.01:
                bend_axis = self._rotate_vector_around_axis(
                    bend_axis, face1_normal, rotation_rad
                )

            # Face 2 normal: rotate face1 by lift_angle around bend_axis
            # Negated to get correct upward bending
            face2_normal = self._rotate_vector_around_axis(
                face1_normal, bend_axis, -lift_angle_rad
            )
            face2_normal.normalize()

            # Chord direction bisects the normals
            chord_dir = (face1_normal + face2_normal)
            if chord_dir.Length < 1e-9:
                chord_dir = face1_normal
            else:
                chord_dir.normalize()

            face2_center = face_center + chord_dir * chord_length

            # Create wafer
            wafer = Part.makeCylinder(
                self.cylinder_radius,
                chord_length * 1.2,
                face_center - chord_dir * chord_length * 0.1,
                chord_dir
            )

            c1 = Part.Circle(face_center, face1_normal, self.cylinder_radius * 10)
            f1 = Part.Face(Part.Wire([c1.toShape()]))
            wafer = wafer.cut(f1.extrude(-face1_normal * self.cylinder_radius * 10))

            c2 = Part.Circle(face2_center, face2_normal, self.cylinder_radius * 10)
            f2 = Part.Face(Part.Wire([c2.toShape()]))
            wafer = wafer.cut(f2.extrude(face2_normal * self.cylinder_radius * 10))

            if wafer.ShapeType == 'Compound' and wafer.Solids:
                wafer = wafer.Solids[0]

            self.wafers.append(wafer)
            self.positions.append((face_center + face2_center) * 0.5)

            print(f"  Wafer {i}: rot={rotation_deg:.1f}° "
                  f"face2=({face2_center.x:.2f},{face2_center.y:.2f},{face2_center.z:.2f})")

            # Update for next wafer
            face_center = face2_center
            face1_normal = face2_normal

        print(f"\n  Final Z: {face_center.z:.3f}")
        return self.wafers

    def _create_wafer_solid(self, chord_length, blade_angle_deg):
        """
        Create a single wafer solid at origin

        The wafer is a cylinder section with angled end cuts.
        Blade angle is the tilt from perpendicular (lift/2).
        """
        # Create cylinder along X-axis
        cylinder = Part.makeCylinder(
            self.cylinder_radius,
            chord_length * 1.5,
            App.Vector(-chord_length * 0.25, 0, 0),
            App.Vector(1, 0, 0)
        )

        # Cut at both ends with angled planes
        blade_angle_rad = math.radians(blade_angle_deg)

        # Plane 1 at x=0, tilted by blade_angle
        # Normal points in -X direction, tilted toward +Z
        normal1 = App.Vector(-math.cos(blade_angle_rad), 0, math.sin(blade_angle_rad))
        plane1_center = App.Vector(0, 0, 0)

        circle1 = Part.Circle(plane1_center, normal1, self.cylinder_radius * 10)
        wire1 = Part.Wire([circle1.toShape()])
        face1 = Part.Face(wire1)
        half_space1 = face1.extrude(-normal1 * self.cylinder_radius * 10)
        wafer = cylinder.cut(half_space1)

        # Plane 2 at x=chord_length, tilted by blade_angle (opposite direction)
        # Normal points in +X direction, tilted toward +Z
        normal2 = App.Vector(math.cos(blade_angle_rad), 0, math.sin(blade_angle_rad))
        plane2_center = App.Vector(chord_length, 0, 0)

        circle2 = Part.Circle(plane2_center, normal2, self.cylinder_radius * 10)
        wire2 = Part.Wire([circle2.toShape()])
        face2 = Part.Face(wire2)
        half_space2 = face2.extrude(normal2 * self.cylinder_radius * 10)
        wafer = wafer.cut(half_space2)

        # Extract solid
        if wafer.ShapeType == 'Compound' and len(wafer.Solids) > 0:
            wafer = wafer.Solids[0]

        return wafer

    def _rotate_vector_around_axis(self, vector, axis, angle_rad):
        """
        Rotate a vector around an axis by an angle (Rodrigues' rotation formula)
        """
        axis = App.Vector(axis)
        axis.normalize()

        v = App.Vector(vector)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rodrigues' formula: v_rot = v*cos(a) + (axis × v)*sin(a) + axis*(axis·v)*(1-cos(a))
        term1 = v * cos_a
        term2 = axis.cross(v) * sin_a
        term3 = axis * (axis.dot(v)) * (1 - cos_a)

        return term1 + term2 + term3

    def _transform_wafer(self, wafer_solid, center, direction, up_vector):
        """
        Transform wafer from origin to specified position and orientation
        """
        # Create rotation to align X-axis with direction
        x_axis = direction
        z_axis = up_vector
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize()

        # Re-orthogonalize
        z_axis = x_axis.cross(y_axis)
        z_axis.normalize()

        # Create rotation matrix
        rot_matrix = App.Matrix()
        rot_matrix.A11 = x_axis.x
        rot_matrix.A21 = x_axis.y
        rot_matrix.A31 = x_axis.z
        rot_matrix.A12 = y_axis.x
        rot_matrix.A22 = y_axis.y
        rot_matrix.A32 = y_axis.z
        rot_matrix.A13 = z_axis.x
        rot_matrix.A23 = z_axis.y
        rot_matrix.A33 = z_axis.z

        # Apply rotation
        rotated = wafer_solid.transformGeometry(rot_matrix)

        # Translate to center position
        # The wafer was created centered at (chord/2, 0, 0), now at origin after rotation
        # We need to move it to 'center'
        translated = rotated.translated(center)

        return translated

    def visualize_in_freecad(self, doc, name_prefix="Recon"):
        """
        Create FreeCAD objects to visualize reconstructed wafers
        """
        print(f"\nCreating FreeCAD visualization of reconstruction...")

        # Create group for reconstruction
        recon_group = doc.addObject("App::DocumentObjectGroup", f"{name_prefix}_Wafers")

        for i, wafer in enumerate(self.wafers):
            wafer_obj = doc.addObject("Part::Feature", f"{name_prefix}_Wafer_{i:03d}")
            wafer_obj.Shape = wafer

            # Alternate colors
            if i % 2 == 0:
                wafer_obj.ViewObject.ShapeColor = (0.4, 0.8, 0.4)  # Green
            else:
                wafer_obj.ViewObject.ShapeColor = (0.4, 0.4, 0.8)  # Blue

            wafer_obj.ViewObject.Transparency = 20
            recon_group.addObject(wafer_obj)

        # Add path line connecting centers
        if len(self.positions) > 1:
            path_points = [pos for pos in self.positions]
            path_wire = Part.makePolygon(path_points)
            path_obj = doc.addObject("Part::Feature", f"{name_prefix}_Path")
            path_obj.Shape = path_wire
            path_obj.ViewObject.LineColor = (1.0, 0.0, 0.0)
            path_obj.ViewObject.LineWidth = 3
            recon_group.addObject(path_obj)

        doc.recompute()
        print(f"✓ Reconstruction visualization complete")


def extract_cut_data_from_segment(segment):
    """
    Extract cut data from a segment's wafer list

    Returns:
        List of dicts with chord_length, blade_angle, rotation_angle, cylinder_angle_deg
    """
    cut_data = []

    cumulative_rotation = 0.0

    for i, wafer in enumerate(segment.wafer_list):
        if wafer.wafer is None:
            continue

        geom = wafer.geometry
        rotation_deg = geom.get('rotation_angle_deg', 0)

        # Calculate cylinder angle (includes 180° flips)
        if i == 0:
            cylinder_angle_deg = 0.0
        else:
            cumulative_rotation += rotation_deg
            if i % 2 == 1:
                cylinder_angle_deg = cumulative_rotation + 180.0
            else:
                cylinder_angle_deg = cumulative_rotation
            cylinder_angle_deg = cylinder_angle_deg % 360.0

        cut_data.append({
            'chord_length': geom.get('chord_length', 0),
            'blade_angle': geom.get('lift_angle_deg', 0) / 2.0,
            'rotation_angle': rotation_deg,
            'cylinder_angle_deg': cylinder_angle_deg,
            'chord_azimuth_deg': geom.get('chord_azimuth_deg', 0)
        })

    return cut_data


def reconstruct_from_segment(segment, doc, rotation_multiplier=1.0, name_prefix="Recon"):
    """
    Convenience function to reconstruct and visualize from a segment

    Args:
        segment: Segment with wafer_list
        doc: FreeCAD document
        rotation_multiplier: Factor to multiply rotation angles
        name_prefix: Prefix for created objects

    Returns:
        WaferReconstructor instance
    """
    # Get cylinder radius from first wafer
    if segment.wafer_list and segment.wafer_list[0].geometry:
        radius = segment.wafer_list[0].geometry.get('ellipse1', {}).get('minor_axis', 0.9375)
    else:
        radius = 0.9375

    reconstructor = WaferReconstructor(cylinder_radius=radius)

    cut_data = extract_cut_data_from_segment(segment)
    reconstructor.build_from_cut_list(cut_data, rotation_multiplier=rotation_multiplier)
    reconstructor.visualize_in_freecad(doc, name_prefix=name_prefix)

    return reconstructor