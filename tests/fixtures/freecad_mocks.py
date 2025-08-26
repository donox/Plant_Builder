# tests/freecad_mocks.py
# -------------------------------------------------------------------
# Minimal FreeCAD/Part stand-ins + helpers to build LCS objects for tests
# -------------------------------------------------------------------
import math
import types

# ---------- Vector ----------
class Vector:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = float(x), float(y), float(z)
    def __add__(self, o): return Vector(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vector(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, k): return Vector(self.x * float(k), self.y * float(k), self.z * float(k))
    __rmul__ = __mul__
    def __truediv__(self, k): k = float(k); return Vector(self.x / k, self.y / k, self.z / k)
    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z
    def cross(self, o): return Vector(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)

    def multiply(self, s: float):
        # FreeCAD's Vector.multiply(s) returns a NEW vector
        return self.__class__(self.x * s, self.y * s, self.z * s)

    def __mul__(self, s: float):
        return self.multiply(s)

    __rmul__ = __mul__

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return self.__class__(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


    @property
    def Length(self): return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def normalize(self):
        L = self.Length
        if L == 0.0: return Vector(0,0,0)
        self.x, self.y, self.z = self.x/L, self.y/L, self.z/L
        return self
    def normalized(self):
        L = self.Length
        return Vector(0,0,0) if L == 0.0 else Vector(self.x/L, self.y/L, self.z/L)
    def getAngle(self, o):
        a = self.normalized(); b = o.normalized()
        d = max(-1.0, min(1.0, a.dot(b)))
        return math.acos(d)
    def tuple(self): return (self.x, self.y, self.z)
    def __repr__(self): return f"Vector({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

def _matmul3(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

# ---------- Placement / Rotation (very thin stubs) ----------
class Rotation:
    """Minimal Rotation mock:
       - Rotation() -> identity
       - Rotation(axis: Vector, angle_deg: float)
       - Rotation(yaw_deg, pitch_deg, roll_deg)  # Z, Y, X order (Rz * Ry * Rx)
    """
    def __init__(self, *args):
        if len(args) == 0:
            self._m = [[1,0,0],[0,1,0],[0,0,1]]
        elif len(args) == 2 and isinstance(args[0], Vector):
            axis = args[0].normalized()
            ang = math.radians(args[1])
            c, s = math.cos(ang), math.sin(ang)
            ux, uy, uz = axis.x, axis.y, axis.z
            self._m = [
                [c+ux*ux*(1-c),   ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),   uy*uz*(1-c)-ux*s],
                [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)],
            ]
        elif len(args) == 3:
            yaw, pitch, roll = map(lambda d: math.radians(float(d)), args)
            cz, sz = math.cos(yaw),  math.sin(yaw)
            cy, sy = math.cos(pitch),math.sin(pitch)
            cx, sx = math.cos(roll), math.sin(roll)
            Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
            Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
            Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
            self._m = _matmul3(_matmul3(Rz, Ry), Rx)
        else:
            raise TypeError("Unsupported Rotation ctor signature")

    def multVec(self, v: Vector) -> Vector:
        m = self._m
        return Vector(
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z,
        )

class Placement:
    def __init__(self, Base=None, Rotation=None):
        self.Base = Base if Base is not None else Vector()
        self.Rotation = Rotation if Rotation is not None else globals()["Rotation"]()

    def multVec(self, v: Vector) -> Vector:
        # Apply rotation then translation
        return self.Rotation.multVec(v) + self.Base
# ---------- “Modules” to drop into sys.modules ----------
def install_freecad_stubs(sys_modules):
    freecad = types.SimpleNamespace(Vector=Vector, Placement=Placement, Rotation=Rotation)
    part = types.SimpleNamespace()  # nothing needed for these tests
    sys_modules['FreeCAD'] = freecad
    sys_modules['Part'] = part
    return freecad, part

# ---------- LCS helper ----------
class LCS:
    """
    Minimal Local Coordinate System object:
      .Placement.Base : origin (Vector)
      .x_axis, .y_axis, .z_axis : unit vectors
    """
    def __init__(self, origin: Vector, x_axis: Vector, y_axis: Vector, z_axis: Vector):
        self.Placement = types.SimpleNamespace(Base=origin, Rotation=Rotation())
        self.x_axis = x_axis.normalized()
        self.y_axis = y_axis.normalized()
        self.z_axis = z_axis.normalized()

def make_perp_frame_from_axis(tangent: Vector, up_hint: Vector = Vector(0,0,1)):
    z = tangent.normalized()
    x = z.cross(up_hint)
    if x.Length < 1e-12:
        # tangent || up_hint → choose an alternate hint
        up_hint = Vector(0,1,0) if abs(z.z) > 0.9 else Vector(0,0,1)
        x = z.cross(up_hint)
    x = x.normalized()
    y = z.cross(x).normalized()
    return x, y, z

def make_lcs(origin_xyz, tangent_xyz, up_hint_xyz=(0,0,1)):
    o = Vector(*origin_xyz)
    t = Vector(*tangent_xyz)
    u = Vector(*up_hint_xyz)
    x, y, z = make_perp_frame_from_axis(t, u)
    return LCS(o, x, y, z)
