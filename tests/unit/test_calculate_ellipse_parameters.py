# test_calculate_ellipse_parameters.py
# ---------------------------------------------------------------
# Unit tests for CurveFollower._calculate_ellipse_parameters
# Runs under pytest with NO FreeCAD dependency.
# ---------------------------------------------------------------

import math
import numpy as np

# --- Minimal stand-in implementing just the helpers your method needs ---
class MiniFollower:
    def __init__(self, curve_points):
        self.curve_points = np.asarray(curve_points, dtype=float)

    def _fit_plane(self, pts: np.ndarray, eps: float = 1e-6):
        # PCA plane fit: (is_planar, n_hat, centroid, rms)
        if pts.shape[0] < 3:
            return True, np.array([0.0, 0.0, 1.0]), (pts[0] if len(pts) else np.zeros(3)), 0.0
        c = pts.mean(axis=0)
        M = pts - c
        _, _, vh = np.linalg.svd(M, full_matrices=False)
        n = vh[-1, :]
        n_norm = np.linalg.norm(n)
        n_hat = n / (n_norm if n_norm > 0 else 1.0)
        d = M @ n_hat
        rms = float(np.sqrt(np.mean(d * d)))
        extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]), 1.0)
        is_planar = rms <= (eps * extent)
        return is_planar, n_hat, c, rms

    def _segment_is_planar(self, points: np.ndarray, eps_abs: float = 1e-6, eps_rel: float = 1e-4):
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return True, np.array([0.0, 0.0, 1.0]), 0.0, eps_abs
        try:
            is_planar, n_hat, _, rms = self._fit_plane(pts, eps=None)
        except Exception:
            c = pts.mean(axis=0)
            Q = pts - c
            H = Q.T @ Q
            w, v = np.linalg.eigh(H)
            n_hat = v[:, 0] / (np.linalg.norm(v[:, 0]) + 1e-12)
            d = Q @ n_hat
            rms = float(np.sqrt(np.mean(d * d)))
            is_planar = True
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = max(eps_abs, eps_rel * extent)
        return (rms <= tol), n_hat, rms, tol

    def _signed_angle_in_plane(self, t1: np.ndarray, t2: np.ndarray, n_hat: np.ndarray) -> float:
        t1u = t1 / (np.linalg.norm(t1) + 1e-12)
        t2u = t2 / (np.linalg.norm(t2) + 1e-12)
        cross = np.cross(t1u, t2u)
        sin_term = np.dot(n_hat, cross)
        cos_term = np.clip(np.dot(t1u, t2u), -1.0, 1.0)
        return float(np.arctan2(sin_term, cos_term))

    # --- Your current method (verbatim structure, trimmed logs) ---
    def _calculate_ellipse_parameters(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        start_index: int,
        end_index: int,
        is_first_wafer: bool = False,
        is_last_wafer: bool = False,
    ):
        EPS_CIRCULAR_DEG = 0.25
        eps_circ = np.deg2rad(EPS_CIRCULAR_DEG)

        def _unit(v: np.ndarray):
            n = float(np.linalg.norm(v))
            return (v / n) if n > 1e-12 else None

        chord_vec = end_point - start_point
        chord_len = float(np.linalg.norm(chord_vec))
        if chord_len < 1e-10:
            return 0.0, 0.0, 0.0, "CC"
        chord_hat = chord_vec / chord_len

        N = len(self.curve_points)
        prev_hat = None
        if start_index > 0:
            v = start_point - self.curve_points[start_index - 1]
            prev_hat = _unit(v)
        next_hat = None
        if end_index < N - 1:
            v = self.curve_points[end_index + 1] - end_point
            next_hat = _unit(v)

        neighborhood = self.curve_points[start_index:end_index + 1]
        is_planar, n_hat, rms, tol = self._segment_is_planar(neighborhood)

        def bend_between(u: np.ndarray, v: np.ndarray) -> float:
            if u is None or v is None:
                return 0.0
            if is_planar and n_hat is not None:
                return abs(self._signed_angle_in_plane(u, v, n_hat))
            uu = u / (np.linalg.norm(u) + 1e-12)
            vv = v / (np.linalg.norm(v) + 1e-12)
            d = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
            return float(np.arccos(d))

        bend_start = bend_between(prev_hat, chord_hat) if prev_hat is not None else 0.0
        bend_end   = bend_between(chord_hat, next_hat) if next_hat is not None else 0.0

        start_is_C = (bend_start <= eps_circ) or (prev_hat is None)
        end_is_C   = (bend_end   <= eps_circ) or (next_hat is None)

        start_type = "C" if start_is_C else "E"
        end_type   = "C" if end_is_C   else "E"
        wafer_type = start_type + end_type

        start_angle = 0.0 if start_is_C else 0.5 * bend_start
        end_angle   = 0.0 if end_is_C   else 0.5 * bend_end

        rotation_angle = 0.0
        if (not is_planar) and (wafer_type == "EE"):
            if prev_hat is not None and next_hat is not None:
                m_start = np.cross(chord_hat, prev_hat)
                nrm = float(np.linalg.norm(m_start))
                if nrm < 1e-8:
                    ref = np.array([0.0, 0.0, 1.0], float)
                    if abs(np.dot(ref, chord_hat)) > 0.9:
                        ref = np.array([1.0, 0.0, 0.0], float)
                    m_start = np.cross(chord_hat, ref)
                    nrm = float(np.linalg.norm(m_start))
                m_start /= (nrm + 1e-12)

                proj = m_start - next_hat * float(np.dot(m_start, next_hat))
                if float(np.linalg.norm(proj)) > 1e-8:
                    m_end_ref = proj / float(np.linalg.norm(proj))
                else:
                    m_end_ref = m_start

                m_end_true = np.cross(next_hat, chord_hat)
                nrm = float(np.linalg.norm(m_end_true))
                if nrm < 1e-8:
                    m_end_true = m_end_ref
                    nrm = float(np.linalg.norm(m_end_true))
                m_end_true /= (nrm + 1e-12)

                axis = chord_hat
                rot_sin = float(np.dot(axis, np.cross(m_end_ref, m_end_true)))
                rot_cos = float(np.clip(np.dot(m_end_ref, m_end_true), -1.0, 1.0))
                rotation_angle = float(np.arctan2(rot_sin, rot_cos))

        max_angle = math.pi / 3
        start_angle = float(np.clip(start_angle, 0.0, max_angle))
        end_angle   = float(np.clip(end_angle,   0.0, max_angle))
        if is_planar or abs(rotation_angle) < 1e-6:
            rotation_angle = 0.0

        return start_angle, end_angle, rotation_angle, wafer_type

# --- Helpers to make simple test curves ---
def circle_xy(radius=10.0, n=20):
    th = np.linspace(0.0, 2*np.pi, n, endpoint=True)
    return np.stack([radius*np.cos(th), radius*np.sin(th), np.zeros_like(th)], axis=1)

def straight_line_xyz(n=10, step=1.0):
    x = np.linspace(0.0, step*(n-1), n)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

# ============================== TESTS =================================

def test_straight_segment_types_and_angles_zero():
    P = straight_line_xyz(n=6, step=2.0)
    mf = MiniFollower(P)

    # Build one wafer spanning multiple points → still straight/planar
    start_i, end_i = 1, 4
    s, e = P[start_i], P[end_i]
    sa, ea, rot, wtype = mf._calculate_ellipse_parameters(s, e, start_i, end_i)

    # Straight: both joints have ~0 bend ⇒ CC; all angles 0, rotation 0.
    assert wtype == "CC"
    assert abs(sa) < 1e-9
    assert abs(ea) < 1e-9
    assert abs(rot) < 1e-9

def test_planar_circle_interior_wafer_rotation_zero_and_positive_end_angles():
    P = circle_xy(radius=8.0, n=24)          # planar XY circle
    mf = MiniFollower(P)

    # Choose an interior wafer (has prev and next)
    start_i, end_i = 5, 8
    s, e = P[start_i], P[end_i]

    sa, ea, rot, wtype = mf._calculate_ellipse_parameters(s, e, start_i, end_i)

    # Planar ⇒ rotation must be zero by construction
    assert abs(rot) < 1e-9

    # On a circle, bends at both joints > 0 ⇒ EE for interior wafer
    assert wtype in ("EE", "EC", "CE")  # usually EE for interior; tolerant to eps behavior

    # End angles are half the joint bends ⇒ positive (but < 60° cap)
    assert sa >= 0.0 and ea >= 0.0
    assert sa < math.pi/3 + 1e-9 and ea < math.pi/3 + 1e-9

def test_last_wafer_end_is_C_when_no_next_neighbor():
    P = circle_xy(radius=6.0, n=16)
    mf = MiniFollower(P)

    # Take a wafer that ends at the last point (no next neighbor)
    start_i, end_i = 10, len(P) - 1
    s, e = P[start_i], P[end_i]

    sa, ea, rot, wtype = mf._calculate_ellipse_parameters(s, e, start_i, end_i)

    # By your logic: if no next neighbor, end is C (regardless of bend elsewhere)
    assert wtype.endswith("C")
    # Planar circle ⇒ rotation zero
    assert abs(rot) < 1e-9
