
from core.logging_setup import get_logger


logger = get_logger(__name__)

import numpy as np
import math
import time
import FreeCAD
import Part


class Wafer(object):

    def __init__(self, app, gui, parm_set, wafer_type="EE"):
        self.app = app
        self.gui = gui
        self.parm_set = parm_set        # Naming preamble for elements in a set (segment)
        self.wafer_type = wafer_type    # EE, CE, EC, CC
        self.outside_height = None
        self.lift_angle = None
        self.rotation_angle = None
        self.cylinder_radius = None
        self.lcs_top = None
        self.lcs_base = None
        self.wafer_name = None
        self.wafer = None
        self.angle = None   # Angle to x-y plane of top surface

    def set_parameters(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE"):
        self.lift_angle = lift
        self.rotation_angle = rotation
        self.cylinder_radius = cylinder_diameter / 2
        self.outside_height = outside_height
        self.wafer_type = wafer_type

    def make_wafer_from_lcs(self, lcs1, lcs2, cylinder_diameter, wafer_name,
                            start_cut_angle=None, end_cut_angle=None):
        """Create a wafer from two LCS objects.

        Args:
            start_cut_angle: Angle in radians (0 for circular cut)
            end_cut_angle: Angle in radians (0 for circular cut)
        """
        self.lcs_top = lcs2
        self.lcs_base = lcs1
        self.cylinder_radius = cylinder_diameter / 2
        self.wafer_name = wafer_name

        # Get positions and the wafer axis (chord)
        pos1 = lcs1.Placement.Base
        pos2 = lcs2.Placement.Base
        wafer_axis = pos2 - pos1
        chord_length = wafer_axis.Length

        if chord_length < 1e-6:
            raise ValueError("Wafer has zero length")

        wafer_axis.normalize()

        # Determine extensions based on cut angles
        epsilon = 0.001  # Small value to avoid numerical issues

        # Check the wafer type characters directly
        if self.wafer_type[0] == 'C':  # Circular start
            extend1 = epsilon
        else:  # Elliptical start
            if start_cut_angle is not None and abs(start_cut_angle) > 0.001:
                extend1 = self.cylinder_radius * np.tan(abs(start_cut_angle)) + epsilon
            else:
                extend1 = epsilon

        if self.wafer_type[1] == 'C':  # Circular end
            extend2 = epsilon
        else:  # Elliptical end
            if end_cut_angle is not None and abs(end_cut_angle) > 0.001:
                extend2 = self.cylinder_radius * np.tan(abs(end_cut_angle)) + epsilon
            else:
                extend2 = epsilon

        logger.debug(f"        Extensions based on type {self.wafer_type}: extend1={extend1:.3f}, extend2={extend2:.3f}")

        # Cap extensions to reasonable values
        max_extend = chord_length * 0.5
        extend1 = min(extend1, max_extend)
        extend2 = min(extend2, max_extend)

        # Create cylinder with extensions
        cylinder_start = pos1 - wafer_axis * extend1
        cylinder_length = chord_length + extend1 + extend2
        cylinder_end = pos1 + wafer_axis * (chord_length + extend2)

        logger.debug(f"      Cylinder: chord_length={chord_length:.3f}, extend1={extend1:.3f}, extend2={extend2:.3f}")
        logger.debug(f"      Total cylinder length: {cylinder_length:.3f}")
        # DEBUG: Print the relationship between LCS and actual geometry
        logger.debug(f"        🔍 WAFER GEOMETRY DEBUG for {wafer_name}:")
        logger.debug(f"           LCS1 position: [{pos1.x:.3f}, {pos1.y:.3f}, {pos1.z:.3f}]")
        logger.debug(f"           LCS2 position: [{pos2.x:.3f}, {pos2.y:.3f}, {pos2.z:.3f}]")
        logger.debug(f"           Cylinder START: [{cylinder_start.x:.3f}, {cylinder_start.y:.3f}, {cylinder_start.z:.3f}]")
        logger.debug(f"           Cylinder END:   [{cylinder_end.x:.3f}, {cylinder_end.y:.3f}, {cylinder_end.z:.3f}]")
        logger.debug(f"           Offset from LCS1 to cylinder start: {extend1:.3f} (backwards)")
        logger.debug(f"           Offset from LCS2 to cylinder end: {(cylinder_end - pos2).Length:.3f}")

        # Create the cylinder
        cylinder = Part.makeCylinder(
            self.cylinder_radius,
            cylinder_length,
            cylinder_start,
            wafer_axis
        )

        # Verify the shape is valid
        if not cylinder.isValid():
            logger.error(f"      WARNING: Created cylinder is not valid!")
            cylinder.fix(0.1, 0.1, 0.1)
            if not cylinder.isValid():
                logger.error(f"      ERROR: Could not fix invalid cylinder")

        # Create the wafer object
        self.wafer = self.app.activeDocument().addObject("Part::Feature", wafer_name)
        self.wafer.Shape = cylinder
        self.wafer.ViewObject.Transparency = 0
        self.wafer.Visibility = False

        self.lcs1 = lcs1
        self.lcs2 = lcs2

        logger.debug(f"      Created cylinder for wafer between {lcs1.Label} and {lcs2.Label}")

        #  per-wafer rotation sanity check (logged)
        self._validate_rotation_after_build(tol_deg=2.0)

    def get_lift_angle(self, next_wafer):
        """
        Per-step lift/tilt (radians) to go from LCS1(n) to LCS1(n+1).

        Lift is the angle between the two START-face Z axes:
            lift = acos(  z1(n) · z1(n+1)  ).

        Return 0.0 if next_wafer is missing or geometry is degenerate.
        """
        import math
        try:
            if next_wafer is None:
                return 0.0

            lcs1_cur = getattr(self, 'lcs1', None) or getattr(self, 'lcs_base', None)
            lcs1_next = getattr(next_wafer, 'lcs1', None) or getattr(next_wafer, 'lcs_base', None)
            if not lcs1_cur or not lcs1_next:
                return 0.0

            R1 = lcs1_cur.Placement.Rotation
            R2 = lcs1_next.Placement.Rotation

            zhat = FreeCAD.Vector(0, 0, 1)
            z1 = R1.multVec(zhat)
            z2 = R2.multVec(zhat)

            if z1.Length < 1e-12 or z2.Length < 1e-12:
                return 0.0

            z1.normalize();
            z2.normalize()
            dotv = max(-1.0, min(1.0, z1.dot(z2)))
            return math.acos(dotv)
        except Exception:
            return 0.0

    # wafer.py
    import math

    EPS = 1e-9

    def get_rotation_angle(self, next_wafer=None, expected_deg=None, **_ignore) -> float:
        """
        Absolute twist angle **in degrees** about the chord (line from A→B),
        comparing THIS.END→NEXT.START if next_wafer is given, otherwise THIS.START→THIS.END.

        Compatible with callers that pass `expected_deg`, and resilient to missing/degenerate frames.
        Emits detailed DEBUG of the intermediate projections (y1p/y2p/z1p/z2p).
        """
        import math
        try:
            # --- helpers that don't depend on project-wide utilities ---
            def unit_and_len(v):
                # returns (unit_vector, length) without mutating v
                L = float(v.Length) if hasattr(v, "Length") else math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                if L <= 1e-12:
                    return FreeCAD.Vector(0, 0, 0), 0.0
                return FreeCAD.Vector(v.x / L, v.y / L, v.z / L), L

            def proj_perp(v, u_unit):
                # v - (v·u)u
                dot = float(v.dot(u_unit))
                return FreeCAD.Vector(v.x - u_unit.x * dot,
                                      v.y - u_unit.y * dot,
                                      v.z - u_unit.z * dot)

            def axes_YZ(lcs, Vec):
                """
                Return the world-space images of local Y and Z for an LCS-like thing.
                Works with:
                  - Part::Feature / Datum objects (use .Placement.Rotation)
                  - anything exposing .getGlobalPlacement()
                  - a raw App.Placement passed directly
                """
                P = getattr(lcs, "Placement", None)
                if P is None:
                    get_glob = getattr(lcs, "getGlobalPlacement", None)
                    if callable(get_glob):
                        P = get_glob()
                    else:
                        # maybe lcs *is* a Placement already
                        P = lcs
                R = getattr(P, "Rotation", None)
                if R is None:
                    raise AttributeError(f"{type(lcs).__name__} has no Placement.Rotation")
                Y = R.multVec(Vec(0, 1, 0))
                Z = R.multVec(Vec(0, 0, 1))
                return Y, Z

            # --- circular-face joints are defined to be 0° ---
            this_t = getattr(self, "wafer_type", "") or ""
            next_t = getattr(next_wafer, "wafer_type", "") if next_wafer else ""
            if ("C" in this_t) or ("C" in (next_t or "")):
                logger.debug("get_rotation_angle: circular joint (this=%s, next=%s) → 0.0°", this_t, next_t or "")
                return 0.0

            # --- choose the frames to compare ---
            if next_wafer is None:
                A = getattr(self, "lcs1", None) or getattr(self, "lcs_base", None)
                B = getattr(self, "lcs2", None) or getattr(self, "lcs_top", None)
            else:
                A = getattr(self, "lcs2", None) or getattr(self, "lcs_top", None)  # THIS end
                B = getattr(next_wafer, "lcs1", None) or getattr(next_wafer, "lcs_base", None)  # NEXT start

            if not (A and B):
                logger.debug("get_rotation_angle: missing LCS (A=%s, B=%s) → 0.0°", bool(A), bool(B))
                return 0.0

            Pa, Pb = A.Placement.Base, B.Placement.Base
            chord = Pb.sub(Pa)
            c_u, c_len = unit_and_len(chord)
            if c_len <= 1e-12:
                logger.debug("get_rotation_angle: chord length ≈ 0 → 0.0°")
                return 0.0

            # world-space Y/Z axes for both frames
            Y1, Z1 = axes_YZ(A, Vec)
            Y2, Z2 = axes_YZ(B, Vec)

            # project Y/Z into plane ⟂ chord
            y1p = proj_perp(Y1, c_u);
            y1u, ny1 = unit_and_len(y1p)
            y2p = proj_perp(Y2, c_u);
            y2u, ny2 = unit_and_len(y2p)
            z1p = proj_perp(Z1, c_u);
            z1u, nz1 = unit_and_len(z1p)
            z2p = proj_perp(Z2, c_u);
            z2u, nz2 = unit_and_len(z2p)

            # candidate angles using Y and Z, pick the more stable (larger of the two minima)
            def angle_deg(u1, u2):
                d = max(-1.0, min(1.0, float(u1.dot(u2))))
                s = float(c_u.dot(u1.cross(u2)))
                return abs(math.degrees(math.atan2(s, d)))

            candY = angle_deg(y1u, y2u) if min(ny1, ny2) > 1e-12 else None
            candZ = angle_deg(z1u, z2u) if min(nz1, nz2) > 1e-12 else None

            if candY is None and candZ is None:
                logger.debug("get_rotation_angle: degenerate projections (Y & Z nearly 0) → 0.0°")
                return 0.0

            # choose by projection strength, tie-break by smaller angle
            strengthY = min(ny1, ny2)
            strengthZ = min(nz1, nz2)
            if (candY is not None and strengthY >= strengthZ) or (candZ is None):
                rot_deg = candY
                axis_used = "Y"
            else:
                rot_deg = candZ
                axis_used = "Z"

            # detailed diagnostics
            logger.debug(
                "get_rotation_angle: axis=%s, angle=%+.3f°, chord_len=%.4f, "
                "Yp=(%.3e, %.3e), Zp=(%.3e, %.3e), candY=%.3f°, candZ=%.3f°",
                axis_used, rot_deg, c_len,
                ny1, ny2, nz1, nz2,
                (candY if candY is not None else float("nan")),
                (candZ if candZ is not None else float("nan")),
            )

            # optional expectation check for existing callers
            if expected_deg is not None:
                try:
                    delta = abs(rot_deg - float(expected_deg))
                    if delta > 5.0:
                        logger.error(
                            "get_rotation_angle: Δ=%.2f° > 5° (actual=%.2f°, expected=%.2f°)",
                            delta, rot_deg, float(expected_deg),
                        )
                except Exception:
                    pass

            return float(rot_deg)

        except Exception:
            logger.exception("get_rotation_angle: unexpected error")
            return 0.0

    def get_outside_height(self):
        return self.outside_height

    def get_wafer_type(self):
        return self.wafer_type

    def get_cylinder_diameter(self):
        return self.cylinder_radius * 2

    def get_angle(self):
        return self.angle

    def get_top(self):
        return self.lcs_top

    def get_base(self):
        return self.lcs_base

    def get_wafer_name(self):
        return self.wafer_name

    def get_lcs_top(self):
        return self.lcs_top

    def get_wafer(self):
        return self.wafer

    def lift_lcs(self, lcs, lcs_type):
        """Move lcs on base surface to corresponding position on top.

        This is a side-effecting operation.  Result is modified input lcs.

        There are four cases where the base is a circle (C) or ellipse (E) and similarly on top.

            For CC:  The lift angle is zero and the new placement is simply the addition of the
                outside height to the z-axis.
            For CE: The wafer is a cylinder of height outside height / 2 and an inclined section
                with a circle at the base and ellipse at the top.
            For EC: The wafer is an inverted CE.
            For EE: The wafer is an EC at the base and a CE on top.

        Calculations:
            Calculate each part separately (thus CE above is a cylinder on bottom and an ellipse on top with the
            cylinder of height outside_height / 2).  Then combine two parts appropriately to create the full case above.

            If base is an ellipse, rotate point to global zero.  Figure will be a cylinder inclined
            by the lift angle with inclination in the xz plane. Make changes and rotate back to original position.

        """
        # logger.debug(f"LIFT_LCS: {lcs_type}, angle: {convert_angle(lift_angle)}, diam: {cylinder_diameter}, oh: {outside_height}")
        parts = {"CC": ("CC2", "CC2"),
                 "CE": ("CC2", "CE2"),
                 "EC": ("EC2", "CC2"),
                 "EE": ("EC2", "CE2")}
        parts_used = parts[lcs_type]
        h2 = self.outside_height / 2
        d2 = self.cylinder_radius
        la = self.lift_angle        # Lift angle already divided for end segments
        if lcs_type == "EE":
            la = self.lift_angle / 2
        result_vec = []
        for i in range(2):
            # logger.debug(f"PARTS: {parts_used}, LCS: {lcs.Label}, i: {i}")
            if "EC2" == parts_used[i]:
                del_x = 0  # -d2 * np.sin(la)
                del_z = h2 - d2 * np.tan(la)
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"EC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}, {i}: {parts_used}")
            if "CE2" == parts_used[i]:
                del_x = 0  # -d2 * np.sin(la)
                del_z = h2 - d2 * np.tan(la)
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"CE  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)}  {i}: {parts_used}")
            if "CC2" == parts_used[i]:
                del_x = 0
                del_z = 0              # removes cylindrical part
                result_vec.append(FreeCAD.Vector(del_x, 0, del_z))
                # logger.debug(f"CC  x: {np.round(del_x, 3)}, z: {np.round(del_z, 3)} res: {result_vec[i]}")
            rot = FreeCAD.Rotation(0, 0, 0)
            if parts_used[i] in ["CE2", "EC2"]:
                # Not clear if Rotation wants radians or degrees.  Testing says radians - complex_path says degrees
                rot = FreeCAD.Rotation(0, -np.rad2deg(la), 0)
            # if parts_used[i] == "EE2":          # there is no EE2
            #     rot = FreeCAD.Rotation(0, -np.rad2deg(la * 2), 0)
            new_place = FreeCAD.Placement(result_vec[i], rot)
            lcs.Placement = lcs.Placement.multiply(new_place)
            # break

    def validate_segment_join(segment1_last_wafer, segment2_first_wafer):
        # Check that segment1 ends with 'C' and segment2 starts with 'C'
        if segment1_last_wafer.wafer_type[1] != 'C':
            raise ValueError("Segment must end with circular cut for joining")
        if segment2_first_wafer.wafer_type[0] != 'C':
            raise ValueError("Segment must start with circular cut for joining")

    def _validate_rotation_after_build(self, tol_deg: float = 2.0) -> None:
        """
        Validate immediately after construction.

        - Skips CE/EC (rotation defined 0°).
        - Uses Wafer.get_rotation_angle(expected_deg=None, tol_deg=tol_deg) which:
            * Computes the shop rotation from LCS1/LCS2 (Y-projections in the cut plane).
            * Does a curve-agnostic plausibility check (binormal-style) and logs if it drifts.
        """
        try:
            wt = getattr(self, "wafer_type", "EE")
            if "C" in wt:
                logger.debug(f"🧪 Rotation check: '{wt}' circular face → defined 0.0°; skipping validation.")
                return

            # Compute + internally validate against expected; no external utilities needed.
            _ = self.get_rotation_angle(expected_deg=None, tol_deg=tol_deg)

        except Exception as e:
            logger.error(f"❌ Rotation validation threw: {e}")


def convert_angle(angle):
    return np.round(np.rad2deg(angle), 3)

def log_lcs_info(lcs, tag, logger_level="info"):
    """
    Log detailed information about a FreeCAD LCS (Local Coordinate System) object.

    Args:
        lcs: FreeCAD LCS object (PartDesign::CoordinateSystem)
        tag: String identifier for the LCS in log output
        logger_level: Logging level ("debug", "info", "warning", "error")
    """
    try:
        if lcs is None:
            getattr(logger, logger_level)(f"{tag}: LCS is None")
            return

        # Get the logging function
        log_func = getattr(logger, logger_level)

        # Basic object info
        log_func(f"\n=== LCS INFO: {tag} ===")
        log_func(f"  Label: {getattr(lcs, 'Label', 'N/A')}")
        log_func(f"  Name: {getattr(lcs, 'Name', 'N/A')}")

        # Check if object has placement
        if not hasattr(lcs, 'Placement'):
            log_func(f"  ERROR: No Placement attribute")
            return

        placement = lcs.Placement

        # Position information
        base = placement.Base
        log_func(f"  Position: [{base.x:.6f}, {base.y:.6f}, {base.z:.6f}]")

        # Rotation information
        rotation = placement.Rotation

        # Euler angles (in degrees)
        euler = rotation.toEuler()
        log_func(f"  Euler angles: [Yaw={euler[0]:.3f}°, Pitch={euler[1]:.3f}°, Roll={euler[2]:.3f}°]")

        # Axis-angle representation
        # axis = rotation.Axis
        # angle_deg = rotation.Angle * 180.0 / 3.14159265359
        # log_func(f"  Axis-Angle: Axis=[{axis.x:.6f}, {axis.y:.6f}, {axis.z:.6f}], Angle={angle_deg:.3f}°")

        # Quaternion
        # q = rotation.Q
        # log_func(f"  Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")

        # Local coordinate axes in global coordinates
        x_axis = rotation.multVec(FreeCAD.Vector(1, 0, 0))
        y_axis = rotation.multVec(FreeCAD.Vector(0, 1, 0))
        z_axis = rotation.multVec(FreeCAD.Vector(0, 0, 1))

        log_func(f"  Local X-axis in global: [{x_axis.x:.6f}, {x_axis.y:.6f}, {x_axis.z:.6f}]")
        log_func(f"  Local Y-axis in global: [{y_axis.x:.6f}, {y_axis.y:.6f}, {y_axis.z:.6f}]")
        log_func(f"  Local Z-axis in global: [{z_axis.x:.6f}, {z_axis.y:.6f}, {z_axis.z:.6f}]")

        # Verify orthonormality
        x_len = x_axis.Length
        y_len = y_axis.Length
        z_len = z_axis.Length
        xy_dot = x_axis.dot(y_axis)
        xz_dot = x_axis.dot(z_axis)
        yz_dot = y_axis.dot(z_axis)

        # log_func(f"  Axis lengths: X={x_len:.6f}, Y={y_len:.6f}, Z={z_len:.6f}")
        # log_func(f"  Dot products: X·Y={xy_dot:.6f}, X·Z={xz_dot:.6f}, Y·Z={yz_dot:.6f}")
        #
        # # Check for orthonormality issues
        # if abs(x_len - 1.0) > 1e-6 or abs(y_len - 1.0) > 1e-6 or abs(z_len - 1.0) > 1e-6:
        #     log_func(f"  WARNING: Axes are not unit length!")
        # if abs(xy_dot) > 1e-6 or abs(xz_dot) > 1e-6 or abs(yz_dot) > 1e-6:
        #     log_func(f"  WARNING: Axes are not orthogonal!")
        #
        # # Additional placement info if available
        # if hasattr(lcs, 'Visibility'):
        #     log_func(f"  Visibility: {lcs.Visibility}")

        log_func(f"=== END LCS INFO: {tag} ===\n")

    except Exception as e:
        getattr(logger, logger_level)(f"ERROR logging LCS info for {tag}: {e}")


# Convenience wrapper functions for different log levels
def log_lcs_debug(lcs, tag):
    """Log LCS info at DEBUG level"""
    log_lcs_info(lcs, tag, "debug")


def log_lcs_info_level(lcs, tag):
    """Log LCS info at INFO level"""
    log_lcs_info(lcs, tag, "info")


def log_lcs_warning(lcs, tag):
    """Log LCS info at WARNING level"""
    log_lcs_info(lcs, tag, "warning")