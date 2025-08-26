
# test_get_rotation_angle_numpy.py
# -----------------------------------------------------------------------------
# Pure-numpy reproduction of the rotation-angle math used by get_rotation_angle.
# No FreeCAD dependency. Use this to isolate whether the math is correct.
#
# Run:
#   python3 test_get_rotation_angle_numpy.py
#
# What it does:
#   * Implements a signed angle between projections of two Y-axes onto the plane
#     perpendicular to the chord (segment direction), matching the shop definition.
#   * Builds a short synthetic helix and LCS-like frames and compares the computed
#     angles to the azimuth change about the helix axis.
#   * Includes degeneracy checks.
# -----------------------------------------------------------------------------

import math
import numpy as np

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3), 0.0
    return v / n, n

def _project_onto_plane(v, normal):
    # v_proj = v - (v·n) n
    v = np.asarray(v, float)
    n, _ = _normalize(normal)
    return v - np.dot(v, n) * n

def signed_angle_about(u, v, normal, eps=1e-12):
    '''
    Signed angle from u to v about 'normal' using atan2:
        sin = ((normal x u) · v)
        cos = (u · v)
        angle = atan2(sin, cos)
    Inputs are assumed to already lie in the plane perpendicular to 'normal'.
    '''
    un, unlen = _normalize(u, eps)
    vn, vnlen = _normalize(v, eps)
    if unlen < eps or vnlen < eps:
        return 0.0
    cross = np.cross(normal, un)
    sin_a = np.dot(cross, vn)
    cos_a = np.dot(un, vn)
    if abs(sin_a) < eps and abs(cos_a) < eps:
        # ambiguous; treat as 0 to match robust behavior
        return 0.0
    return math.atan2(sin_a, cos_a)

def rotation_angle_from_LCS(y1, y2, chord):
    '''
    Mirror of get_rotation_angle's core math:
      - Project y1 and y2 into the plane perpendicular to 'chord'
      - Return signed angle from y1p to y2p about +chord
    '''
    y1p = _project_onto_plane(y1, chord)
    y2p = _project_onto_plane(y2, chord)
    return signed_angle_about(y1p, y2p, chord)

def helix_points(radius=6.0, pitch=4.0, turns=1.0, N=15):
    '''
    Return points P_i and tangents T_i for a right-handed helix about global Z.
    θ in [0, 2π·turns], equally spaced at N+1 points.
    '''
    thetas = np.linspace(0, 2*math.pi*turns, N+1)
    k = pitch/(2*math.pi)  # dz/dθ
    P = []
    T = []
    for th in thetas:
        x = radius*math.cos(th)
        y = radius*math.sin(th)
        z = k*th
        P.append(np.array([x, y, z]))
        # derivative wrt θ: (-r sinθ, r cosθ, k)
        t = np.array([-radius*math.sin(th), radius*math.cos(th), k], dtype=float)
        t, _ = _normalize(t)
        T.append(t)
    return np.asarray(P), np.asarray(T), thetas

def make_LCS_like(P, T):
    '''
    Build "LCS-like" (y_i, z_i) at each point:
       - z_i = T_i (tangent)
       - y_i = projection of global X onto plane perpendicular to z_i
    '''
    Y = []
    Z = []
    Xg = np.array([1.0, 0.0, 0.0], float)
    for z in T:
        y = _project_onto_plane(Xg, z)
        y, _ = _normalize(y)
        Y.append(y)
        Z.append(z/np.linalg.norm(z))
    return np.asarray(Y), np.asarray(Z)

def run_numpy_helix_suite(radius=6.0, pitch=4.0, turns=1.0, N=15):
    P, T, thetas = helix_points(radius, pitch, turns, N)
    Y, Z = make_LCS_like(P, T)

    results = []
    for i in range(N):
        chord = P[i+1] - P[i]
        chord, _ = _normalize(chord)
        angle = rotation_angle_from_LCS(Y[i], Y[i+1], chord)
        # Reference: azimuth change about global Z
        dtheta = math.atan2(P[i+1][1], P[i+1][0]) - math.atan2(P[i][1], P[i][0])
        # Wrap to (-π,π]
        while dtheta <= -math.pi:
            dtheta += 2*math.pi
        while dtheta > math.pi:
            dtheta -= 2*math.pi
        results.append((math.degrees(angle), math.degrees(dtheta)))

    degs = np.array([r[0] for r in results])
    refs = np.array([r[1] for r in results])
    print("=== Pure-numpy helix test ===")
    print(f"N={N}, turns={turns}, nominal step ≈ {360.0*turns/N:.2f}°")
    print(" idx |  angle(deg)  ref_dθ(deg)  err(deg)")
    for i, (a, r) in enumerate(results):
        print(f"{i:>4} | {a:>10.3f}   {r:>10.3f}   {a-r:>8.3f}")
    print(f"mean(angle) = {degs.mean():.3f}°, std(angle) = {degs.std():.3f}°")
    print(f"mean(|err|) = {np.mean(np.abs(degs-refs)):.3f}°")



def _unit_tests():
    # 1) chord along +Z; y1 = +X, y2 = +X rotated +30° in XY -> expect ≈ +30°
    chord = np.array([0.0, 0.0, 1.0])
    y1 = np.array([1.0, 0.0, 0.0])
    ang = math.radians(30.0)
    y2 = np.array([math.cos(ang), math.sin(ang), 0.0])
    a = math.degrees(rotation_angle_from_LCS(y1, y2, chord))
    print('unit1 expected ~30°, got', round(a,3), '°')

    # 2) chord tilted; projections still work
    chord = np.array([0.0, 1.0, 1.0]); chord, _ = _normalize(chord)
    y1 = np.array([1.0, 0.0, 0.0])
    y2 = np.array([0.0, 1.0, 0.0])
    a = math.degrees(rotation_angle_from_LCS(y1, y2, chord))
    print('unit2 (tilted chord) angle =', round(a,3), '°')

    # 3) degenerate: projections vanish -> 0.0
    chord = np.array([0.0, 1.0, 0.0])
    y1 = np.array([0.0, 1.0, 0.0])  # parallel to chord
    y2 = np.array([0.0, 1.0, 0.0])  # parallel to chord
    a = math.degrees(rotation_angle_from_LCS(y1, y2, chord))
    print('unit3 (degenerate) angle =', round(a,6), '°')

if __name__ == '__main__':
    _unit_tests()
    run_numpy_helix_suite(N=15, turns=1.05)
    print('\n-- sanity variants --')
    run_numpy_helix_suite(N=12, turns=1.0)
    run_numpy_helix_suite(N=20, turns=0.5)
