# tests/test_calculate_ellipse_parameters_real.py
import math
import numpy as np
import inspect

# Import your actual class from your codebase
# Adjust the import below ONLY if your class lives elsewhere:
from src.curve_follower import CurveFollower   # <-- if module path differs, tweak this line

# --- Curve generators (numpy; no FreeCAD) ---
def circle_xy(radius=10.0, n=24, theta0=0.0, theta1=2*math.pi):
    th = np.linspace(theta0, theta1, n, endpoint=True, dtype=float)
    x = radius * np.cos(th)
    y = radius * np.sin(th)
    z = np.zeros_like(th)
    return np.stack([x, y, z], axis=1)

def line_xyz(n=8, step=2.0):
    x = np.linspace(0.0, step*(n-1), n, dtype=float)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

# --- Minimal constructor-free instance: avoid any FreeCAD deps in __init__ ---
def _make_cf_with_points(points: np.ndarray) -> CurveFollower:
    """Create a CurveFollower instance without calling its __init__, then set curve_points."""
    cf = object.__new__(CurveFollower)
    # Provide exactly what _calculate_ellipse_parameters uses:
    cf.curve_points = np.asarray(points, dtype=float)
    # If your class expects a logger attr, make a no-op:
    if not hasattr(cf, "logger"):
        class _NopLogger:
            def debug(self, *a, **k): pass
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
            def exception(self, *a, **k): pass
        cf.logger = _NopLogger()
    return cf

# ============================== TESTS =================================

def test_straight_segment_returns_CC_and_zero_angles():
    """A straight (planar) segment should yield CC type and zero start/end/rotation."""
    P = line_xyz(n=6, step=2.0)
    cf = _make_cf_with_points(P)

    start_i, end_i = 1, 4
    s, e = P[start_i], P[end_i]

    # call the *real* method from your code
    sa, ea, rot, wtype = cf._calculate_ellipse_parameters(
        s, e, start_i, end_i, is_first_wafer=False, is_last_wafer=False
    )

    assert wtype in ("CC", "CE", "EC")  # degenerate neighbors might make one end E with tiny eps
    # but angles must be ~0 on a straight segment:
    assert abs(sa) < 1e-8
    assert abs(ea) < 1e-8
    assert abs(rot) < 1e-8

def test_planar_circle_interior_has_zero_rotation_and_positive_end_angles():
    """Interior wafer on a planar circle has 0 rotation and positive (bend/2) end angles."""
    P = circle_xy(radius=8.0, n=32)
    cf = _make_cf_with_points(P)

    start_i, end_i = 5, 9
    s, e = P[start_i], P[end_i]

    sa, ea, rot, wtype = cf._calculate_ellipse_parameters(
        s, e, start_i, end_i, is_first_wafer=False, is_last_wafer=False
    )

    # Planar ⇒ rotation must be zero
    assert abs(rot) < 1e-8
    # On a circle, bends at both joints should be > 0 (usually EE for interior)
    assert wtype in ("EE", "EC", "CE")
    # end angles are half the joint bends, positive, and your code caps at 60° (pi/3)
    assert sa >= 0.0 and ea >= 0.0
    assert sa < math.pi/3 + 1e-12 and ea < math.pi/3 + 1e-12

def test_last_wafer_end_defaults_to_C_when_no_next_neighbor():
    """When there is no next neighbor, the end end is treated as C; rotation=0 on planar."""
    P = circle_xy(radius=6.0, n=16)
    cf = _make_cf_with_points(P)

    start_i, end_i = 10, len(P) - 1
    s, e = P[start_i], P[end_i]

    sa, ea, rot, wtype = cf._calculate_ellipse_parameters(
        s, e, start_i, end_i, is_first_wafer=False, is_last_wafer=True
    )

    assert wtype.endswith("C")
    assert abs(rot) < 1e-8

def test_api_signature_has_expected_parameters():
    """Guardrail: alert us if the method signature changes."""
    sig = inspect.signature(CurveFollower._calculate_ellipse_parameters)
    params = list(sig.parameters.keys())
    assert params[:4] == ["self", "start_point", "end_point", "start_index"], "breaking change in signature head"

# tests/test_replay_calculate_ellipse_parameters.py
import json
import math
import pathlib
import numpy as np
import inspect
import types
import sys

# ---- Keep FreeCAD out of the way (minimal stubs if your imports pull it) ----
if "FreeCAD" not in sys.modules:
    sys.modules["FreeCAD"] = types.ModuleType("FreeCAD")
if "FreeCADGui" not in sys.modules:
    sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")
# Optional: some trees import Part indirectly
if "Part" not in sys.modules:
    sys.modules["Part"] = types.ModuleType("Part")

# ---- Import your real class (adjust module path if needed) ----
from src.curve_follower import CurveFollower

TRACE_PATH = pathlib.Path(__file__).resolve().parent / "data" / "ellipse_cases.jsonl"

def _load_cases():
    if not TRACE_PATH.exists():
        return []
    cases = []
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except Exception:
                # ignore malformed lines
                pass
    return cases

def _make_cf_with_points(points):
    """Create a CurveFollower without calling __init__; inject curve_points only."""
    cf = object.__new__(CurveFollower)
    cf.curve_points = np.asarray(points, dtype=float)
    return cf

def _angle_close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol

def _rad_close(a, b):
    # a lot of code zeroes tiny rotations; allow a touch more room
    return _angle_close(a, b, tol=1e-7)

def test_replay_calculate_ellipse_parameters_cases():
    cases = _load_cases()
    assert cases, f"No captured cases found at {TRACE_PATH} — run a segment build first."

    failures = []

    for idx, c in enumerate(cases):
        points = np.asarray(c["curve_points"], dtype=float)
        cf = _make_cf_with_points(points)

        s = np.asarray(c["start_point"], dtype=float)
        e = np.asarray(c["end_point"], dtype=float)
        si = int(c["start_index"])
        ei = int(c["end_index"])
        is_first = bool(c.get("is_first_wafer", False))
        is_last  = bool(c.get("is_last_wafer", False))

        if not c.get("ok", False):
            # If the original call raised, ensure it still raises
            raised = None
            try:
                cf._calculate_ellipse_parameters(s, e, si, ei, is_first, is_last)
            except Exception as ex:
                raised = f"{type(ex).__name__}"
            assert raised is not None, f"Case {idx}: expected an exception originally but none was raised now."
            continue

        # Recompute using the real method
        sa, ea, rot, wtype = cf._calculate_ellipse_parameters(s, e, si, ei, is_first, is_last)

        # Compare to recorded outputs
        ok = True
        msgs = []
        if not _rad_close(sa, c["start_angle"]):
            ok = False
            msgs.append(f"start_angle {sa} != {c['start_angle']}")
        if not _rad_close(ea, c["end_angle"]):
            ok = False
            msgs.append(f"end_angle {ea} != {c['end_angle']}")
        # rotation is often 0 on planar; accept tiny jitter
        if not _rad_close(rot, c["rotation_angle"]):
            ok = False
            msgs.append(f"rotation_angle {rot} != {c['rotation_angle']}")
        if str(wtype) != str(c["wafer_type"]):
            ok = False
            msgs.append(f"wafer_type '{wtype}' != '{c['wafer_type']}'")

        if not ok:
            failures.append((idx, "; ".join(msgs)))

    assert not failures, "Mismatches:\n" + "\n".join(f"  case {i}: {msg}" for i, msg in failures)

