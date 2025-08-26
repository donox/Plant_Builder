# tests/test_get_rotation_angle.py
import math
import types
import sys
import numpy as np
from fixtures.freecad_mocks import install_freecad_stubs
install_freecad_stubs(sys_modules=sys.modules)

from fixtures.freecad_mocks import Vector, make_lcs
import sys
assert 'FreeCAD' in sys.modules, "FreeCAD stubs were not installed before imports"
from src.wafer import Wafer

# # Import your project module AFTER conftest has installed stubs
# import src.wafer  # your repo’s wafer.py

# ---- Helper: independent “reference” rotation using only tangents ----
def ref_rotation_deg(lcs1, lcs2, eps=1e-12):
    """
    Reference measure (no Y-axis dependency):
      1) chord = P2 - P1
      2) project Z1 and Z2 (tangents) onto plane normal to chord
      3) signed angle between those projections, sign via chord·(Z1×Z2)
    """
    p1 = lcs1.Placement.Base
    p2 = lcs2.Placement.Base
    z1 = lcs1.z_axis
    z2 = lcs2.z_axis

    chord = Vector(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
    n = chord.normalized() if chord.Length > eps else Vector(0,0,1)

    def proj_perp(v, axis):
        # remove component along axis
        return (v - axis * v.dot(axis)).normalized() if v.Length > eps else Vector(0,0,0)

    z1p = proj_perp(z1, n)
    z2p = proj_perp(z2, n)

    # If either projection collapsed (rare but possible at near-parallel), define 0
    if z1p.Length < eps or z2p.Length < eps:
        return 0.0

    angle = z1p.getAngle(z2p)
    sign = 1.0
    tri = z1p.cross(z2p)
    if n.dot(tri) < 0.0:
        sign = -1.0
    return math.degrees(sign * angle)

def deg_close_mod(a, b, tol=1e-3):
    """
    True if angles a and b (deg) are equivalent up to flipping the direction
    (i.e. modulo 180). Use this when the reference may be based on an
    unoriented normal and your implementation uses an oriented frame.
    """
    # map difference to [-180, 180)
    d = (a - b + 180.0) % 360.0 - 180.0
    # also consider the 180°-flipped case (equate v and -v)
    d_alt = (a - (b + 180.0) + 180.0) % 360.0 - 180.0
    return min(abs(d), abs(d_alt)) <= tol

def shortest_diff_mod180(a_deg: float, b_deg: float) -> float:
    """Shortest distance between two angles in deg, treating a and a±180 as equivalent."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    d_flip = (a_deg - (b_deg + 180.0) + 180.0) % 360.0 - 180.0
    return min(abs(d), abs(d_flip))

# ---- Test fixtures: parametric curves to generate LCS pairs ----
def sample_helix(radius=6.0, pitch=4.0, turns=1.0, N=12):
    """Return [(lcs1, lcs2), ...] pairs along a helix arc with unit-speed-ish tangents."""
    pts = []
    for k in range(N+1):
        t = 2.0 * math.pi * turns * (k / N)
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        z = pitch * (t / (2.0 * math.pi))
        # tangent w.r.t. parameter t (not arc length – good enough for direction)
        tx = -radius * math.sin(t)
        ty =  radius * math.cos(t)
        tz =  pitch / (2.0 * math.pi)
        pts.append(((x,y,z), (tx,ty,tz)))
    pairs = []
    for i in range(N):
        o1, t1 = pts[i]
        o2, t2 = pts[i+1]
        l1 = make_lcs(o1, t1)
        l2 = make_lcs(o2, t2)
        pairs.append((l1, l2))
    return pairs

def sample_line(length=5.0, N=6):
    """Straight line along +Z: all rotations should be ~0."""
    pairs = []
    for i in range(N):
        z1 = length * i / N
        z2 = length * (i+1) / N
        l1 = make_lcs((0,0,z1), (0,0,1))
        l2 = make_lcs((0,0,z2), (0,0,1))
        pairs.append((l1, l2))
    return pairs

def sample_circle(radius=6.0, N=12):
    """Flat circle in XY: lift=0, rotation should be ~0 by the tangent-only definition."""
    pairs = []
    for i in range(N):
        t1 = 2.0*math.pi*(i/N)
        t2 = 2.0*math.pi*((i+1)/N)
        o1 = (radius*math.cos(t1), radius*math.sin(t1), 0.0)
        o2 = (radius*math.cos(t2), radius*math.sin(t2), 0.0)
        # 2D circle tangents
        l1 = make_lcs(o1, (-radius*math.sin(t1), radius*math.cos(t1), 0.0))
        l2 = make_lcs(o2, (-radius*math.sin(t2), radius*math.cos(t2), 0.0))
        pairs.append((l1, l2))
    return pairs

# ---- A wafer-like shim so we can call the instance method directly ----
class DummyWafer:
    """Carries just the bits get_rotation_angle() touches: lcs1, lcs2, wafer_type."""
    def __init__(self, lcs1, lcs2, wafer_type="EE"):
        self.lcs1 = lcs1
        self.lcs2 = lcs2
        self.wafer_type = wafer_type
    def get_wafer_type(self):
        return self.wafer_type

# ---- Tests ----
def test_line_rotation_zeroish():
    for l1, l2 in sample_line():
        w = DummyWafer(l1, l2, "EE")
        got_rad = Wafer.get_rotation_angle(w, expected_deg=None)  # method on your class
        got_deg = math.degrees(got_rad)
        ref_deg = ref_rotation_deg(l1, l2)
        assert abs(got_deg) < 1e-6, f"Line should have ~0° rotation; got {got_deg:.6f}°"
        assert abs(got_deg - ref_deg) < 1e-6, f"Mismatch vs reference: {got_deg:.6f} vs {ref_deg:.6f}"

def test_circle_rotation_zeroish():
    for l1, l2 in sample_circle():
        w = DummyWafer(l1, l2, "EE")
        got_rad = Wafer.get_rotation_angle(w, expected_deg=None)
        got_deg = math.degrees(got_rad)
        ref_deg = ref_rotation_deg(l1, l2)
        # Expect ~0° numerically
        assert abs(got_deg) < 1e-3, f"Circle step should be ~0°; got {got_deg:.6f}°"
        assert deg_close_mod(got_deg, ref_deg), f"Mismatch vs reference: {got_deg:.6f} vs {ref_deg:.6f}"

def test_helix_rotation_matches_reference():
    pairs = sample_helix(radius=6.0, pitch=4.0, turns=1.0, N=15)

    gots = []
    refs = []
    for l1, l2 in pairs:
        w = DummyWafer(l1, l2, "EE")
        gots.append(math.degrees(Wafer.get_rotation_angle(w, expected_deg=None)))
        refs.append(ref_rotation_deg(l1, l2))

    # Compare with 180°-flip tolerance
    diffs = [shortest_diff_mod180(g, r) for g, r in zip(gots, refs)]
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))

    # Pick a tolerance that reflects the definition you want
    assert mean_diff < 1.0 and max_diff < 1.0, \
        f"Helix rotation deviates: mean {mean_diff:.3f}°, max {max_diff:.3f}°"

def test_ce_ec_defined_zero():
    # Ends around a circle/helix, you may define rotation as 0.0 for CE/EC
    l1, l2 = sample_line(N=2)[0]
    w = DummyWafer(l1, l2, "CE")
    got_deg = math.degrees(Wafer.get_rotation_angle(w, expected_deg=None))
    assert abs(got_deg) < 1e-9
