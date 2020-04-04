### noble-secp256k1.py - MIT License (c) Paul Miller (paulmillr.com) ###

# initial:  0.177 / 36ms
# jacobian: 0.111 / 3.7ms
# + constant-time: 4ms
# + constant-time + precomputes: 2.6ms

class Curve:
    # Params: a, b
    a = 0
    b = 7
    # Field over which we'll do calculations
    P = 2 ** 256 - 2 ** 32 - 977
    # Subgroup order aka prime_order
    n = 2 ** 256 - 432420386565659656852420866394968145599
    # Cofactor
    h = 1
    # Base point (x, y) aka generator point
    Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
    Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424

    @classmethod
    def formula(cls, x: int) -> int:
        return (x ** 3 + cls.a * x + cls.b) % P

P = Curve.P
PRIME_ORDER = Curve.n

class ECCError(Exception):
    pass

# Python 3.8+ only. Use EGCD otherwise: brilliant.org/wiki/extended-euclidean-algorithm
def mod_inverse(num, mod = P):
    return pow(num, -1, mod=P)

POW_2_256_M1 = 2 ** 256 - 1

class JacobianPoint:
    precomputes = None
    @classmethod
    def from_affine(cls, point):
        return JacobianPoint(point.x, point.y, 1)

    def __init__(self, x: int, y: int, z: int) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return 'JacobianPoint({}, {}, {})'.format(self.x, self.y, self.z)

    def __eq__(self, other) -> bool:
        a, b = self, other
        az2, az3 = a.z ** 2 % P, a.z ** 3 % P
        bz2, bz3 = b.z ** 2 % P, b.z ** 3 % P
        return (a.x * bz2 % P) == (az2 * b.x % P) and (a.y * bz3 % P == az3 * b.y % P)

    def __neg__(self):
        return JacobianPoint(self.x, -self.y % P, self.z)

    def double(self):
        """
        Fast algo for doubling 2 Jacobian Points when curve's a=0.
        From: http://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
        Cost: 2M + 5S + 6add + 3*2 + 1*3 + 1*8.
        """
        X1, Y1, Z1 = self.x, self.y, self.z
        A = X1 ** 2 % P
        B = Y1 ** 2 % P
        C = B ** 2 % P
        D = (2 * ((X1 + B) ** 2 - A - C)) % P
        E = 3 * A % P
        F = E ** 2 % P
        X3 = (F - 2 * D) % P
        Y3 = (E * (D - X3) - 8 * C) % P
        Z3 = (2 * Y1 * Z1) % P
        return JacobianPoint(X3, Y3, Z3)

    def __add__(self, other):
        """
        Fast algo for adding 2 Jacobian Points when curve's a=0.
        http://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-1998-cmo-2
        Cost: 12M + 4S + 6add + 1*2.
        Note: 2007 Bernstein-Lange (11M + 5S + 9add + 4*2) is actually *slower*. No idea why.
        """
        X1, Y1, Z1 = self.x, self.y, self.z
        X2, Y2, Z2 = other.x, other.y, other.z
        if X2 == 0 or Y2 == 0:
            return self
        if X1 == 0 or Y1 == 0:
            return other
        Z1Z1 = Z1 ** 2 % P
        Z2Z2 = Z2 ** 2 % P
        U1 = X1 * Z2Z2 % P
        U2 = X2 * Z1Z1 % P
        S1 = Y1 * Z2 * Z2Z2 % P
        S2 = Y2 * Z1 * Z1Z1 % P
        H = (U2 - U1) % P
        r = (S2 - S1) % P
        # H = 0 meaning it's the same point.
        if H == 0:
            if r == 0:
                return self.double()
            else:
                return JacobianPoint.ZERO
        HH = (H ** 2) % P
        HHH = (H * HH) % P
        V = U1 * HH % P
        X3 = (r ** 2 - HHH - 2 * V) % P
        Y3 = (r * (V - X3) - S1 * HHH) % P
        Z3 = (Z1 * Z2 * H) % P
        return JacobianPoint(X3, Y3, Z3)

    def unsafe_mul(self, scalar: int):
        p = JacobianPoint.ZERO
        for i in range(0, 256):
            val = precomputes[i]
            if n & 1:
                p += val
            n >>= 1
        return p

    def to_affine(self, inv_z = None):
        if inv_z is None:
            inv_z = mod_inverse(self.z)
        inv_z2 = inv_z ** 2
        x = self.x * inv_z2 % P
        y = self.y * inv_z2 * inv_z % P
        return Point(x, y)

JacobianPoint.ZERO = JacobianPoint(0, 0, 1)

class Point:
    @classmethod
    def is_valid(cls, x: int, y: int) -> bool:
        if x == 0 or x == 0 or x >= P or y >= P:
            return False
        y_sqr = y * y % P
        y_equivalence = Curve.formula(x)
        left1 = y_sqr % P
        left2 = -y_sqr % P
        right1 = y_equivalence % P
        right2 = -y_equivalence % P
        return left1 == right1 or left1 == right2 or left2 == right1 or left2 == right2

    #@classmethod
    #def from_compressed_hex(arr: bytes) -> Point:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return 'Point({}, {})'.format(self.x, self.y)

    def __str__(self):
        return self.to_hex()

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        if not isinstance(other, Point):
            raise ECCError('Point#add: expected Point')
        return JacobianPoint.from_affine(self) + JacobianPoint.from_affine(other)

    def __mul__(self, scalar):
        return JacobianPoint.from_affine(self).unsafe_mul(scalar).to_affine()
    
    def simple_add(self, other):
        a, b = self, other
        X1, Y1, X2, Y2 = a.x, a.y, b.x, b.y
        if a == Point.ZERO:
            return b
        if b == Point.ZERO:
            return a
        if X1 == X2:
            if Y1 == Y2:
                return self.double()
            return Point.ZERO # Y1 == -Y2
        lam = ((Y2 - Y1) * mod_inverse(X2 - X1)) % P
        X3 = (lam * lam - X1 - X2) % P
        Y3 = (lam * (X1 - X3) - Y1) % P
        return Point(X3, Y3)
    
    def simple_double(self):
        X1, Y1 = self.x, self.y
        lam = 3 * X1 ** 2 * mod_inverse(2 * Y1, P)
        X3 = (lam * lam - 2 * X1) % P
        Y3 = (lam * (X1 - X3) - Y1) % P
        return Point(X3, Y3)
    
    def simple_mul(self, scalar: int):
        if isinstance(scalar, Point):
            raise TypeError("Point must be multiplied by scalar, not other Point")
        res = Point(0, 0)
        db = self
        while scalar > 0:
            if scalar & 1 == 1:
                res = res + db
            scalar >>= 1
            db = db.double()
        return res
    
    def __get_precomputes(self):
        dbl = self
        precomputes = []
        for i in range(0, 256):
            precomputes.append(dbl)
            dbl = dbl.double()
        return precomputes
    
    def dbl_mul(self, scalar: int):
        if not isinstance(scalar, int):
            raise ECCError('Point#multiply: expected number or bigint')
        n = scalar % PRIME_ORDER
        if n <= 0:
            raise ECCError('Point#multiply: invalid scalar, expected positive integer')
        p = Point.ZERO
        fake_p = Point.ZERO
        fake_n = POW_2_256_M1 ^ n

        precomputes = None
        if self == Point.BASE:
            if Point.precomputes is None:
                Point.precomputes = self.__get_precomputes()
            precomputes = Point.precomputes
        else:
            precomputes = self.__get_precomputes()

        for i in range(0, 256):
            val = precomputes[i]
            if n & 1:
                p += val
            else:
                fake_p += val
            n >>= 1
            fake_n >>= 1
        return p

    def to_hex(self, is_compressed=False):
        x = format(self.x, 'x').zfill(64)
        if is_compressed:
            head = "03" if self.y & 1 else "02"
            return head + x
        y = format(self.y, 'x').zfill(64)
        return "04{}{}".format(x, y)

Point.BASE = Point(Curve.Gx, Curve.Gy)
JacobianPoint.BASE = JacobianPoint.from_affine(Point.BASE)
Point.ZERO = Point(0, 0)

assert Point.BASE * 1 == Point(Curve.Gx, Curve.Gy)
assert Point.BASE * 2 == Point(89565891926547004231252920425935692360644145829622209833684329913297188986597, 12158399299693830322967808612713398636155367887041628176798871954788371653930)

def get_public_key(private_key: int):
    return Point.BASE * private_key


# Benchmarks.
def bench_get_public_key_1bit():
    get_public_key(2)
def bench_get_public_key():
    get_public_key(2 ** 255 - 1)
def format_bench(name, runs):
    import timeit
    time = timeit.timeit(
        "bench_{}()".format(name),
        setup="from __main__ import bench_{0}; bench_{0}()".format(name),
        number=runs
    )
    fmt = time / runs
    sym = 'sec'
    if fmt < 1:
        fmt = fmt * 1_000
        sym = 'ms'
    elif took < 0.001:
        fmt = fmt * 1_000_000
        sym = 'μs'
    print('{}: {:.3f} {}/op'.format(name, fmt, sym))
if __name__ == '__main__':
    format_bench('get_public_key_1bit', 100)
    format_bench('get_public_key', 100)
