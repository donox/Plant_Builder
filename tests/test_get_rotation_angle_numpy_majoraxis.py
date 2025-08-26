
# test_get_rotation_angle_numpy_majoraxis.py
# -----------------------------------------------------------------------------
# Pure-numpy diagnostic modeling wafer Y-axis as ellipse major axis.
# See header comments in previous version for details.
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

def helix_points(radius=6.0, pitch=4.0, turns=1.0, N=15):
    thetas = np.linspace(0, 2*math.pi*turns, N+1)
    k = pitch/(2*math.pi)
    P = []
    T = []
    for th in thetas:
        P.append(np.array([radius*math.cos(th), radius*math.sin(th), k*th]))
        t = np.array([-radius*math.sin(th), radius*math.cos(th), k], dtype=float)
        t, _ = _normalize(t)
        T.append(t)
    return np.asarray(P), np.asarray(T), thetas

def ellipse_major_axis_direction(plane_normal, cylinder_axis):
    # In the plane with normal n, major axis is perpendicular to proj of cylinder axis a:
    #   p = a - (a·n) n
    #   y = unit( n cross p )  # in-plane, 90 deg from p
    n, _ = _normalize(plane_normal)
    a, _ = _normalize(cylinder_axis)
    p = a - np.dot(a, n) * n
    p, plen = _normalize(p)
    if plen == 0.0:
        return np.zeros(3)
    y = np.cross(n, p)
    y, _ = _normalize(y)
    return y

def rotation_angle_about_chord(y1, y2, chord, eps=1e-12):
    # project y1, y2 into plane perpendicular to chord and take signed angle:
    # atan2( (c x y1p)·y2p, y1p·y2p )
    y1p = _project_onto_plane(y1, chord)
    y2p = _project_onto_plane(y2, chord)
    y1p, n1 = _normalize(y1p, eps)
    y2p, n2 = _normalize(y2p, eps)
    if n1 < eps or n2 < eps:
        return 0.0
    c, _ = _normalize(chord, eps)
    sin_a = np.dot(np.cross(c, y1p), y2p)
    cos_a = np.dot(y1p, y2p)
    if abs(sin_a) < eps and abs(cos_a) < eps:
        return 0.0
    return math.atan2(sin_a, cos_a)

def run_suite(radius=6.0, pitch=4.0, turns=1.05, N=15):
    P, T, thetas = helix_points(radius, pitch, turns, N)
    chords = P[1:] - P[:-1]   # shape (N, 3)

    # build y_i = major axis at face i for i=0..N-1
    Y = []
    for i in range(N):
        n = T[i]          # plane normal at face i
        a = chords[i]     # cylinder axis for wafer starting at face i
        y = ellipse_major_axis_direction(n, a)
        Y.append(y)
    Y = np.asarray(Y)

    print("=== Major-axis model (pure numpy) ===")
    print(f"N={N}, turns={turns}, nominal azimuth step ≈ {360.0*turns/N:.2f}°")
    print(" idx |  rot(deg)  ref_dθ(deg)")
    for i in range(N-1):
        c = chords[i]
        rot = rotation_angle_about_chord(Y[i], Y[i+1], c)
        rot_deg = math.degrees(rot)
        dtheta = math.atan2(P[i+1][1], P[i+1][0]) - math.atan2(P[i][1], P[i][0])
        while dtheta <= -math.pi: dtheta += 2*math.pi
        while dtheta >  math.pi: dtheta -= 2*math.pi
        print(f"{i:>4} | {rot_deg:>9.3f}   {math.degrees(dtheta):>10.3f}")

if __name__ == '__main__':
    run_suite(N=15, turns=1.05)
    print('\n-- variants --')
    run_suite(N=12, turns=1.0)
    run_suite(N=20, turns=0.5)
