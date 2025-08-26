#!/usr/bin/env python3
import math
import numpy as np

def v(x,y,z): return np.array([x,y,z], dtype=float)
def nrm(a):
    L = np.linalg.norm(a)
    return a/L if L > 1e-12 else a*0.0

def proj_perp(vec, n):
    n = nrm(n)
    return vec - np.dot(vec, n) * n

def signed_angle_about(n_hat, a_hat, b_hat):
    n = nrm(n_hat); a = nrm(a_hat); b = nrm(b_hat)
    s = np.dot(np.cross(a, b), n)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.atan2(s, c)  # (-pi, pi]

def helix_pts_and_tangents(R=6.0, pitch=4.0, turns=1.05, N=15):
    pts, tangents = [], []
    for i in range(N+1):
        t = i / N
        th = 2*math.pi*turns*t
        # point
        pts.append(v(R*math.cos(th), R*math.sin(th), pitch*turns*t))
        # tangent (up to scale): d/dθ with dz/dθ = pitch/(2π)
        tangents.append(v(-R*math.sin(th), R*math.cos(th), pitch/(2*math.pi)))
    return np.array(pts), np.array(tangents)

def major_axis_in_face(z_axis, face_normal):
    """
    For a cylinder with axis z_axis cut by a plane whose normal is face_normal:
      - Project z_axis into the plane
      - Major axis is any unit vector in the plane ⟂ that projection
      - One convenient choice:  m = n × (z - (z·n)n)
    """
    n = nrm(face_normal)
    z = nrm(z_axis)
    z_in_plane = proj_perp(z, n)
    # circular/near-circular guard
    if np.linalg.norm(z_in_plane) < 1e-9:
        return v(1,0,0)  # arbitrary; caller should treat as 0-rotation case
    m = np.cross(n, z_in_plane)
    return nrm(m)

def angle_between_major_axes_in_face(z1, z2, chord):
    """
    Angle between the two *lines* (major axes) measured in the cut plane ⟂ chord.
    Treat m and -m as the same physical axis (flip for closest alignment).
    """
    n = nrm(chord)
    if np.linalg.norm(n) < 1e-12:
        return 0.0
    m1 = major_axis_in_face(z1, n)
    m2 = major_axis_in_face(z2, n)

    # sign-agnostic: flip m2 if needed so it's as close as possible to m1
    if np.dot(m1, m2) < 0:
        m2 = -m2

    # If either axis became near-zero (degenerate), define rotation as 0
    if np.linalg.norm(m1) < 1e-9 or np.linalg.norm(m2) < 1e-9:
        return 0.0

    return signed_angle_about(n, m1, m2)

def run_case(N, turns, R=6.0, pitch=4.0):
    pts, tangents = helix_pts_and_tangents(R=R, pitch=pitch, turns=turns, N=N)
    nominal = 360.0*turns/N
    print(f"\n=== Pure-NumPy golden test (fixed) ===")
    print(f"N={N}, turns={turns}, nominal azimuth step ≈ {nominal:.2f}°")
    print(" idx | rot(deg)")
    for i in range(1, N):  # rotation for face between wafer i and i+1
        z1 = tangents[i-1]
        z2 = tangents[i]
        chord = pts[i] - pts[i-1]  # face normal
        ang = math.degrees(angle_between_major_axes_in_face(z1, z2, chord))
        print(f"{i:4d} | {ang:9.3f}")
    print("  CE/EC (ends): defined to be 0.0°")

if __name__ == "__main__":
    for args in [(15, 1.05), (12, 1.0), (20, 0.5)]:
        run_case(*args)
