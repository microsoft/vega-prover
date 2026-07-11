"""Compressed group-element (point) encoding for the canonical engine.

A group element serializes to **33 bytes**: a 1-byte flag followed by the
32-byte ``x`` coordinate in **big-endian**. The flag uses halo2curves'
``CompressedFlagConfig::Extra`` convention:

* ``0x40`` set  -> point at infinity (identity); the 32 x-bytes are all zero.
* otherwise      -> affine point, with ``0x80`` set iff ``y`` is odd
  (the "sign" bit = least-significant bit of ``y``'s little-endian ``to_repr``).

Decompression solves ``y^2 = x^3 - 3x + b`` for ``y`` and selects the root whose
parity matches the sign bit.

Points are parsed lazily as :class:`WirePoint` (raw bytes, no field work) so that
structural parsing stays cheap; call :meth:`WirePoint.point` to decompress to a
Sage curve point when the algebra is actually needed.

**Sage is imported lazily.** Structural parsing (digest, proof/VK layout) touches
no group algebra, so it never pays the multi-second Sage import; only the group
operations below (decompression, MSM, transcript encoding) trigger it.
"""

from .params import P, A, B, GX, GY, SCALAR_BYTES, FLAG_SIGN, FLAG_IDENTITY

# Sage-backed base field and curve, built on first use. This module is the *only*
# place Sage is used; the arithmetic core (field.py, polys.py) is Sage-free.
_SAGE = {}


def _ctx():
  """Return (Fp, E, G), constructing the Sage objects on first call."""
  if not _SAGE:
    from sage.all import GF, EllipticCurve

    Fp = GF(P)
    E = EllipticCurve(Fp, [Fp(A), Fp(B)])
    G = E(Fp(GX), Fp(GY))
    _SAGE["Fp"], _SAGE["E"], _SAGE["G"] = Fp, E, G
  return _SAGE["Fp"], _SAGE["E"], _SAGE["G"]


def base_field():
  """The Sage base field ``F_p`` (constructs Sage on first call)."""
  return _ctx()[0]


def curve():
  """The Sage curve ``E`` (constructs Sage on first call)."""
  return _ctx()[1]


def generator():
  """The Sage generator ``G`` (constructs Sage on first call)."""
  return _ctx()[2]


class WirePoint:
  """A group element in its 33-byte compressed wire form (lazily decoded)."""

  __slots__ = ("flag", "x_bytes", "_point")

  def __init__(self, flag: int, x_bytes: bytes):
    if len(x_bytes) != SCALAR_BYTES:
      raise ValueError(f"x needs {SCALAR_BYTES} bytes, got {len(x_bytes)}")
    self.flag = flag
    self.x_bytes = x_bytes
    self._point = None

  @classmethod
  def parse(cls, reader) -> "WirePoint":
    """Consume one compressed point (33 bytes) from a Reader."""
    flag = reader.u8()
    x_bytes = reader.take(SCALAR_BYTES)
    wp = cls(flag, x_bytes)
    wp._validate()
    return wp

  def _validate(self):
    if self.is_identity and self.x_bytes != b"\x00" * SCALAR_BYTES:
      raise ValueError("identity flag set but x is non-zero")
      # The only defined non-identity flags are 0x00 and 0x80.
    if not self.is_identity and (self.flag & ~FLAG_SIGN) != 0:
      raise ValueError(f"unexpected point flag byte 0x{self.flag:02x}")

  @property
  def is_identity(self) -> bool:
    return bool(self.flag & FLAG_IDENTITY)

  @property
  def sign(self) -> int:
    """The encoded sign bit (1 = y odd), meaningless for the identity."""
    return 1 if (self.flag & FLAG_SIGN) else 0

  def point(self):
    """Decompress to a Sage point on the T256 curve (cached)."""
    if self._point is None:
      self._point = self._decompress()
    return self._point

  def _decompress(self):
    Fp, E, _ = _ctx()
    if self.is_identity:
      return E(0)  # point at infinity
    x = Fp(int.from_bytes(self.x_bytes, "big"))
    rhs = x**3 + Fp(A) * x + Fp(B)
    if not rhs.is_square():
      raise ValueError("x is not on the curve (rhs is a non-residue)")
    y = rhs.sqrt()
    # Select the root whose parity matches the encoded sign bit.
    if (int(y) & 1) != self.sign:
      y = -y
    return E(x, y)

  def to_wire(self) -> bytes:
    return bytes([self.flag]) + self.x_bytes

  def __repr__(self):
    if self.is_identity:
      return "WirePoint(identity)"
    return f"WirePoint(sign={self.sign}, x={self.x_bytes.hex()})"


def read_point(reader) -> WirePoint:
  return WirePoint.parse(reader)


def point_to_wire(pt) -> bytes:
  """Compress a Sage curve point to its 33-byte wire form (prover side)."""
  if pt.is_zero():
    return bytes([FLAG_IDENTITY]) + b"\x00" * SCALAR_BYTES
  x = int(pt[0])
  y = int(pt[1])
  flag = FLAG_SIGN if (y & 1) else 0x00
  return bytes([flag]) + x.to_bytes(SCALAR_BYTES, "big")


def point_to_transcript(pt) -> bytes:
  """Transcript encoding of a group element: uncompressed ``x_LE || y_LE`` (64 B).

    Each coordinate is ``to_bytes`` (big-endian for the base field) reversed, i.e.
    little-endian. The identity has no coordinates; encoding it raises, matching
    the Rust ``coordinates().unwrap()`` panic (the identity is never absorbed).
    """
  if pt.is_zero():
    raise ValueError("identity has no transcript encoding (matches Rust panic)")
  x = int(pt[0])
  y = int(pt[1])
  return x.to_bytes(SCALAR_BYTES, "little") + y.to_bytes(SCALAR_BYTES, "little")
