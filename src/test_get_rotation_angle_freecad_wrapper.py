
# test_get_rotation_angle_freecad_wrapper.py
# -----------------------------------------------------------------------------
# FreeCAD-aware wrapper: builds a tiny synthetic helix *segment* using your
# FlexSegment.add_wafer calls, then exercises wafer.get_rotation_angle(next_wafer)
# on consecutive wafer faces. This avoids running the full driver.
#
# Run inside your FreeCAD Python / macro console (with your modules on sys.path):
#   python3 test_get_rotation_angle_freecad_wrapper.py
# -----------------------------------------------------------------------------

import sys, os, math, csv
import numpy as np

try:
    import FreeCAD, FreeCADGui
    import Part
except Exception as e:
    print('❌ FreeCAD not available in this environment.')
    raise

try:
    from wafer import Wafer
    from flex_segment import FlexSegment
except Exception as e:
    print('❌ Could not import your Wafer/FlexSegment:', e)
    raise

def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return (v/n if n>1e-12 else v), n

def helix_points(radius=6.0, pitch=4.0, turns=1.0, N=15):
    thetas = np.linspace(0.0, 2*math.pi*turns, N+1)
    k = pitch/(2*math.pi)
    P = []
    for th in thetas:
        P.append([radius*math.cos(th), radius*math.sin(th), k*th])
    return np.asarray(P), thetas

def build_synthetic_helix_segment(doc, name='testseg', radius=6.0, pitch=4.0, turns=1.0, N=15, dia=2.25):
    seg = FlexSegment(name, show_lcs=True, temp_file='temp.dat', to_build=True, rotate_segment=0.0)
    P, thetas = helix_points(radius, pitch, turns, N)

    for i in range(N):
        start = P[i]
        end   = P[i+1]
        # small non-zero lift to ensure elliptical ends on interior wafers
        lift = math.radians(7.5)
        wafer_type = 'CE' if i==0 else ('EC' if i==N-1 else 'EE')
        seg.add_wafer(lift=lift, rotation=0.0, cylinder_diameter=dia, outside_height=2.8,
                      wafer_type=wafer_type, start_pos=start, end_pos=end, curve_tangent=None)
    return seg, P, thetas

def run_freecad_wrapper(radius=6.0, pitch=4.0, turns=1.05, N=15, dia=2.25, csv_path=None):
    doc = FreeCAD.ActiveDocument
    if doc is None:
        doc = FreeCAD.newDocument()
    seg, P, thetas = build_synthetic_helix_segment(doc, 'HelixUnitTest', radius, pitch, turns, N, dia)

    rows = []
    print('idx, rot_deg(get_rotation_angle), ref_dtheta_deg')
    for i in range(N-1):
        w_i = seg.wafer_list[i]
        w_next = seg.wafer_list[i+1]

        ra = w_i.get_rotation_angle()  # radians
        ra_deg = math.degrees(ra)

        a1 = math.atan2(P[i+1][1], P[i+1][0])
        a0 = math.atan2(P[i][1],   P[i][0])
        dtheta = a1 - a0
        while dtheta <= -math.pi:
            dtheta += 2*math.pi
        while dtheta > math.pi:
            dtheta -= 2*math.pi
        dtheta_deg = math.degrees(dtheta)

        rows.append((i, ra_deg, dtheta_deg))
        print(f'{i:3d}, {ra_deg:9.3f}, {dtheta_deg:9.3f}')

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'rot_deg', 'ref_dtheta_deg'])
            w.writerows(rows)
        print(f'Wrote: {csv_path}')

if __name__ == '__main__':
    run_freecad_wrapper()
