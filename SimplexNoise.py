"""
 * SimplexNoise.py -- Perlin-style simplex noise for Python
 *    Should work with python 2 or 3
 *
 * Andrew Thall (based on Java/C++ versions by Stefan Gustavson and
 *               Eliot Eshelman respectively.)
 * Alma College
 * 5/6/16
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Python code A. Thall, algorithms and comments by E. Eshelman

2D, 3D and 4D Simplex Noise functions return 'random' values in (-1, 1).
This algorithm was originally designed by Ken Perlin, but my code has been
adapted from the implementation written by Stefan Gustavson (stegu@itn.liu.se)

Raw Simplex noise functions return the value generated by Ken's algorithm.

Scaled Raw Simplex noise functions adjust the range of values returned from the
traditional (-1, 1) to whichever bounds are passed to the function.

Multi-Octave Simplex noise functions combine multiple noise values to create a
more complex result. Each successive layer of noise is adjusted and scaled.

Scaled Multi-Octave Simplex noise functions scale the values returned from the
traditional (-1,1) range to whichever range is passed to the function.

In many cases, you may think you only need a 1D noise function, but in practice
2D  is almost always better.  For instance, if you're using the current frame
number  as the parameter for the noise, all objects will end up with the same
noise value  at each frame. By adding a second parameter on the second
dimension, you can ensure that each gets a unique noise value and they don't
all look identical.
"""

from math import sqrt

def octave_noise_2d(octaves, persistence, scale, x, y):
    """
    octave_noise_2d() -- 2D multi-octave Simplex noise.
    For each octave, a higher frequency/lower amplitude function will be added to the original.
    The higher the persistence [0-1], the more of each succeeding octave will be added.
    :param octaves:
    :param persistence:
    :param scale:
    :param x:
    :param y:
    :return:
    """
    total = 0.0
    frequency = scale
    amplitude = 1.0

    # We have to keep track of the largest possible amplitude,
    # because each octave adds more, and we need a value in [-1, 1].
    maxAmplitude = 0.0
    for i in range(octaves):
        total += raw_noise_2d(x*frequency, y*frequency)*amplitude
        frequency *= 2
        maxAmplitude += amplitude
        amplitude *= persistence

    return total/maxAmplitude

def octave_noise_3d(octaves, persistence, scale, x, y, z):
    """
    octave_noise_3d() -- 2D multi-octave Simplex noise.
    For each octave, a higher frequency/lower amplitude function will be added to the original.
    The higher the persistence [0-1], the more of each succeeding octave will be added.
    :param octaves:
    :param persistence:
    :param scale:
    :param x:
    :param y:
    :param z:
    :return:
    """
    total = 0.0
    frequency = scale
    amplitude = 1.0

    # We have to keep track of the largest possible amplitude,
    # because each octave adds more, and we need a value in [-1, 1].
    maxAmplitude = 0.0

    for i in range(octaves):
        total += raw_noise_3d(x*frequency, y*frequency, z*frequency )*amplitude
        frequency *= 2
        maxAmplitude += amplitude
        amplitude *= persistence

    return total/maxAmplitude

def octave_noise_4d(octaves, persistence, scale, x, y, z, w):
    """
    octave_noise_4d() -- 4D multi-octave Simplex noise.
    For each octave, a higher frequency/lower amplitude function will be added to the original.
    The higher the persistence [0-1], the more of each succeeding octave will be added.
    :param octaves:
    :param persistence:
    :param scale:
    :param x:
    :param y:
    :param z:
    :param w:
    :return:
    """
    total = 0.0
    frequency = scale
    amplitude = 1.0

    # We have to keep track of the largest possible amplitude,
    # because each octave adds more, and we need a value in [-1, 1].
    maxAmplitude = 0.0

    for i in range(octaves):
        total += SimplexNoise.raw_noise_4d(x*frequency, y*frequency, z*frequency, w*frequency)*amplitude
        frequency *= 2
        maxAmplitude += amplitude
        amplitude *= persistence

    return total/maxAmplitude

def scaled_octave_noise_2d(octaves, persistence, scale, loBound, hiBound, x, y):
    """
    scaled_octave_noise_2d() -- 2D Scaled Multi-octave Simplex noise.
    :param octaves:
    :param persistence:
    :param scale:
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :return: value between loBound and hiBound
    """
    return octave_noise_2d(octaves, persistence, scale, x, y)*(hiBound - loBound)/2 + (hiBound + loBound)/2

def scaled_octave_noise_3d(octaves, persistence, scale, loBound, hiBound, x, y, z):
    """
    scaled_octave_noise_3d() -- 3D Scaled Multi-octave Simplex noise.
    :param octaves:
    :param persistence:
    :param scale:
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :param z:
    :return: value between loBound and hiBound
    """
    return octave_noise_3d(octaves, persistence, scale, x, y, z)*(hiBound - loBound)/2 + (hiBound + loBound)/2

def scaled_octave_noise_4d(octaves, persistence, scale, loBound, hiBound, x, y, z, w):
    """
    scaled_octave_noise_4d() -- 4D Scaled Multi-octave Simplex noise.
    :param octaves:
    :param persistence:
    :param scale:
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :param z:
    :param w:
    :return: value will be between loBound and hiBound
    """
    return octave_noise_4d(octaves, persistence, scale, x, y, z, w)*(hiBound - loBound)/2 + (hiBound + loBound)/2

def scaled_raw_noise_2d(loBound, hiBound, x, y):
    """
    scaled_raw_noise_2d() -- 2D Scaled Simplex raw noise
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :return: value will be between loBound and hiBound
    """
    return raw_noise_2d(x, y)*(hiBound - loBound)/2 + (hiBound + loBound)/2

def scaled_raw_noise_3d(loBound, hiBound, x, y, z):
    """
    scaled_raw_noise_3d() -- 3D Scaled Simplex raw noise
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :param z:
    :return: value will be between loBound and hiBound
    """
    return raw_noise_3d(x, y, z)*(hiBound - loBound)/2 + (hiBound + loBound)/2

def scaled_raw_noise_4d(loBound, hiBound, x, y, z, w):
    """
    scaled_raw_noise_4d() -- 4D Scaled Simplex raw noise
    :param loBound:
    :param hiBound:
    :param x:
    :param y:
    :return: value will be between loBound and hiBound
    """
    return raw_noise_4d(x, y, z, w)*(hiBound - loBound)/2 + (hiBound + loBound)/2


def raw_noise_2d(x, y):
    """
    raw_noise_2d() -- 2D raw Simplex noise
    :param x:
    :param y:
    :return: value will be between [-1, 1]
    """
    # Noise contributions from the three corners
    n0, n1, n2 = 0.0, 0.0, 0.0

    # Skew the input space to determine which simplex cell we're in
    F2 = 0.5*(sqrt(3.0) - 1.0)
    # Hairy factor for 2D
    s = (x + y) * F2
    i = fastfloor( x + s )
    j = fastfloor( y + s )

    G2 = (3.0 - sqrt(3.0))/6.0
    t = (i + j)*G2
    # Unskew the cell origin back to (x,y) space
    X0 = i-t
    Y0 = j-t
    # The x,y distances from the cell origin
    x0 = x-X0
    y0 = y-Y0

    # For the 2D case, the simplex shape is an equilateral triangle.
    # Determine which simplex we are in.
    # Offsets for second (middle) corner of simplex in (i,j) coords
    if x0>y0:  # lower triangle, XY order: (0,0)->(1,0)->(1,1)
        i1=1
        j1=0
    else:  # upper triangle, YX order: (0,0)->(0,1)->(1,1)
        i1=0
        j1=1

    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    # c = (3-sqrt(3))/6
    x1 = x0 - i1 + G2     # Offsets for middle corner in (x,y) unskewed coords
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0*G2    # Offsets for last corner in (x,y) unskewed coords
    y2 = y0 - 1.0 + 2.0*G2

    # Work out the hashed gradient indices of the three simplex corners
    ii = i & 255
    jj = j & 255
    gi0 = perm[ii+perm[jj]] % 12
    gi1 = perm[ii+i1+perm[jj+j1]] % 12
    gi2 = perm[ii+1+perm[jj+1]] % 12

    # Calculate the contribution from the three corners
    t0 = 0.5 - x0*x0 - y0*y0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        # (x,y) of grad3 used for 2D gradient
        n0 = t0 * t0 * dot2(grad3[gi0][0], grad3[gi0][1], x0, y0)

    t1 = 0.5 - x1*x1 - y1*y1
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        n1 = t1 * t1 * dot2(grad3[gi1][0], grad3[gi1][1], x1, y1)

    t2 = 0.5 - x2*x2 - y2*y2
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        n2 = t2 * t2 * dot2(grad3[gi2][0], grad3[gi2][1], x2, y2)

    # Add contributions from each corner to get the final noise value.
    # The result is scaled to return values in the interval [-1,1].
    return 70.0*(n0 + n1 + n2)


def raw_noise_3d(x, y, z):
    """
    raw_noise_3d() -- 3D raw Simplex noise
    :param x: float
    :param y: float
    :param z: float
    :return: value will be between [-1, 1]
    """
    # Noise contributions from the four corners
    n0, n1, n2, n3 = 0.0, 0.0, 0.0, 0.0

    # Skew the input space to determine which simplex cell we're in
    F3 = 1.0/3.0
    s = (x+y+z)*F3   # Very nice and simple skew factor for 3D
    i = fastfloor(x+s)
    j = fastfloor(y+s)
    k = fastfloor(z+s)

    G3 = 1.0/6.0    # Very nice and simple unskew factor, too
    t = (i+j+k)*G3
    X0 = i-t         # Unskew the cell origin back to (x,y,z) space
    Y0 = j-t
    Z0 = k-t
    x0 = x-X0        # The x,y,z distances from the cell origin
    y0 = y-Y0
    z0 = z-Z0

    # For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    # Determine which simplex we are in.
    # i1, j1, k1  Offsets for second corner of simplex in (i,j,k) coords
    # i2, j2, k2  Offsets for third corner of simplex in (i,j,k) coords

    if x0 >= y0:
        if y0>=z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0   # X Y Z order
        elif x0>=z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1   # X Z Y order
        else:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1   # Z X Y order
    else:  # x0 < y0
        if y0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1   # Z Y X order
        elif x0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1   # Y Z X order
        else:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0   # Y X Z order

    # A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    # a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    # a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    # c = 1/6.
    x1 = x0 - i1 + G3 # Offsets for second corner in (x,y,z) coords
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0*G3 # Offsets for third corner in (x,y,z) coords
    y2 = y0 - j2 + 2.0*G3
    z2 = z0 - k2 + 2.0*G3
    x3 = x0 - 1.0 + 3.0*G3 # Offsets for last corner in (x,y,z) coords
    y3 = y0 - 1.0 + 3.0*G3
    z3 = z0 - 1.0 + 3.0*G3

    # Work out the hashed gradient indices of the four simplex corners
    ii = i & 255
    jj = j & 255
    kk = k & 255
    gi0 = perm[ii+perm[jj+perm[kk]]] % 12
    gi1 = perm[ii+i1+perm[jj+j1+perm[kk+k1]]] % 12
    gi2 = perm[ii+i2+perm[jj+j2+perm[kk+k2]]] % 12
    gi3 = perm[ii+1+perm[jj+1+perm[kk+1]]] % 12

    # Calculate the contribution from the four corners
    t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        n0 = t0 * t0 * dot3(grad3[gi0][0], grad3[gi0][1], grad3[gi0][2], x0, y0, z0)

    t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        n1 = t1 * t1 * dot3(grad3[gi1][0], grad3[gi1][1], grad3[gi1][2], x1, y1, z1)

    t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        n2 = t2 * t2 * dot3(grad3[gi2][0], grad3[gi2][1], grad3[gi2][2], x2, y2, z2)

    t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if t3 < 0:
        n3 = 0.0
    else:
        t3 *= t3
        n3 = t3 * t3 * dot3(grad3[gi3][0], grad3[gi3][1], grad3[gi3][2], x3, y3, z3)

    # Add contributions from each corner to get the final noise value.
    # The result is scaled to stay just inside [-1,1]
    return 32.0*(n0 + n1 + n2 + n3)

def raw_noise_4d(x, y, z, w ):
    """
    raw_noise_4d() -- 4D raw Simplex noise
    :param x:
    :param y:
    :param z:
    :param w:
    :return: value will be between [-1, 1]
    """
    # The skewing and unskewing factors are hairy again for the 4D case
    F4 = (sqrt(5.0) - 1.0)/4.0
    G4 = (5.0 - sqrt(5.0))/20.0
    n0, n1, n2, n3, n4 = 0.0, 0.0, 0.0, 0.0, 0.0 # Noise contributions from the five corners

    # Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    s = (x + y + z + w)*F4    # Factor for 4D skewing
    i = fastfloor(x + s)
    j = fastfloor(y + s)
    k = fastfloor(z + s)
    l = fastfloor(w + s)
    t = (i + j + k + l)*G4    # Factor for 4D unskewing
    X0 = i - t         # Unskew the cell origin back to (x,y,z,w) space
    Y0 = j - t
    Z0 = k - t
    W0 = l - t
    x0 = x - X0        # The x,y,z,w distances from the cell origin
    y0 = y - Y0
    z0 = z - Z0
    w0 = w - W0

    """
    For the 4D case, the simplex is a 4D shape I won t even try to describe.
    To find out which of the 24 possible simplices we're in, we need to
    determine the magnitude ordering of x0, y0, z0 and w0.
    The method below is a good way of finding the ordering of x,y,z,w and
    then find the correct traversal order for the simplex we're in.
    First, six pair-wise comparisons are performed between each possible pair
    of the four coordinates, and the results are used to add up binary bits
    for an integer index.
    """
    c1 = 23 if x0 > y0 else 0
    c2 = 16 if x0 > z0 else 0
    c3 =  8 if y0 > z0 else 0
    c4 =  4 if x0 > w0 else 0
    c5 =  2 if y0 > w0 else 0
    c6 =  1 if z0 > w0 else 0
    c = c1 + c2 + c3 + c4 + c5 + c6;

    # i1, j1, k1, l1   The integer offsets for the second simplex corner
    # i2, j2, k2, l2   The integer offsets for the third simplex corner
    # i3, j3, k3, l3   The integer offsets for the fourth simplex corner

    """
    simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
    Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
    impossible. Only the 24 indices which have non-zero entries make any sense.
    We use a thresholding to set the coordinates in turn from the largest magnitude.
    The number 3 in the "simplex" array is at the position of the largest coordinate.
    """
    i1 = 1 if simplex[c][0] >=3 else 0
    j1 = 1 if simplex[c][1] >=3 else 0
    k1 = 1 if simplex[c][2] >=3 else 0
    l1 = 1 if simplex[c][3] >=3 else 0
    # The number 2 in the "simplex" array is at the second largest coordinate.
    i2 = 1 if simplex[c][0] >=2 else 0
    j2 = 1 if simplex[c][1] >=2 else 0
    k2 = 1 if simplex[c][2] >=2 else 0
    l2 = 1 if simplex[c][3] >=2 else 0
    # The number 1 in the "simplex" array is at the second smallest coordinate.
    i3 = 1 if simplex[c][0] >=1 else 0
    j3 = 1 if simplex[c][1] >=1 else 0
    k3 = 1 if simplex[c][2] >=1 else 0
    l3 = 1 if simplex[c][3] >=1 else 0
    # The fifth corner has all coordinate offsets = 1, so no need to look that up.

    x1 = x0 - i1 + G4 # Offsets for second corner in (x,y,z,w) coords
    y1 = y0 - j1 + G4
    z1 = z0 - k1 + G4
    w1 = w0 - l1 + G4
    x2 = x0 - i2 + 2.0*G4 # Offsets for third corner in (x,y,z,w) coords
    y2 = y0 - j2 + 2.0*G4
    z2 = z0 - k2 + 2.0*G4
    w2 = w0 - l2 + 2.0*G4
    x3 = x0 - i3 + 3.0*G4 # Offsets for fourth corner in (x,y,z,w) coords
    y3 = y0 - j3 + 3.0*G4
    z3 = z0 - k3 + 3.0*G4
    w3 = w0 - l3 + 3.0*G4
    x4 = x0 - 1.0 + 4.0*G4 # Offsets for last corner in (x,y,z,w) coords
    y4 = y0 - 1.0 + 4.0*G4
    z4 = z0 - 1.0 + 4.0*G4
    w4 = w0 - 1.0 + 4.0*G4

    # Work out the hashed gradient indices of the five simplex corners
    ii = i & 255
    jj = j & 255
    kk = k & 255
    ll = l & 255
    gi0 = perm[ii+perm[jj+perm[kk+perm[ll]]]] % 32
    gi1 = perm[ii+i1+perm[jj+j1+perm[kk+k1+perm[ll+l1]]]] % 32
    gi2 = perm[ii+i2+perm[jj+j2+perm[kk+k2+perm[ll+l2]]]] % 32
    gi3 = perm[ii+i3+perm[jj+j3+perm[kk+k3+perm[ll+l3]]]] % 32
    gi4 = perm[ii+1+perm[jj+1+perm[kk+1+perm[ll+1]]]] % 32

    # Calculate the contribution from the five corners
    t0 = 0.6 - x0*x0 - y0*y0 - z0*z0 - w0*w0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        n0 = t0 * t0 * dot4(grad4[gi0][0], grad4[gi0][1], grad4[gi0][2], grad4[gi0][3], x0, y0, z0, w0)

    t1 = 0.6 - x1*x1 - y1*y1 - z1*z1 - w1*w1
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        n1 = t1 * t1 * dot4(grad4[gi1][0], grad4[gi1][1], grad4[gi1][2], grad4[gi1][3], x1, y1, z1, w1)

    t2 = 0.6 - x2*x2 - y2*y2 - z2*z2 - w2*w2
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        n2 = t2 * t2 * dot4(grad4[gi2][0], grad4[gi2][1], grad4[gi2][2], grad4[gi2][3], x2, y2, z2, w2)

    t3 = 0.6 - x3*x3 - y3*y3 - z3*z3 - w3*w3
    if t3 < 0:
        n3 = 0.0
    else:
        t3 *= t3
        n3 = t3 * t3 * dot4(grad4[gi3][0], grad4[gi3][1], grad4[gi3][2], grad4[gi3][3], x3, y3, z3, w3)

    t4 = 0.6 - x4*x4 - y4*y4 - z4*z4 - w4*w4
    if t4 < 0:
        n4 = 0.0
    else:
        t4 *= t4
        n4 = t4 * t4 * dot4(grad4[gi4][0], grad4[gi4][1], grad4[gi4][2], grad4[gi4][3], x4, y4, z4, w4)

    # Sum up and scale the result to cover the range [-1,1]
    return 27.0*(n0 + n1 + n2 + n3 + n4)

"""
SimplexTables.py
A. Thall
Alma College
5/6/16
Helper functions and gradient/permutation/simplex tables for
SimplexNoise.py
"""


def fastfloor(x):
    """
    return integer floor of x (rounded toward -INF)
    :param x: float
    :return: int
    """
    return int(x) if x > 0 else int(x) - 1


def dot2(gx, gy, x, y):
    """
    dot product
    :param gx: int
    :param gy: int
    :param x: float
    :param y: float
    :return: float
    """
    return gx*x + gy*y


def dot3(gx, gy, gz, x, y, z):
    """
    dot product
    :param gx: int
    :param gy: int
    :param gz: int
    :param x: float
    :param y: float
    :param z: float
    :return: float
    """
    return gx*x + gy*y + gz*z


def dot4(gx, gy, gz, gw, x, y, z, w):
    """
    dot product
    :param gx: int
    :param gy: int
    :param gz: int
    :param gw: int
    :param x: float
    :param y: float
    :param z: float
    :param w: float
    :return: float
    """
    return gx*x + gy*y + gz*z + gw*w

# The gradients are the midpoints of the vertices of a cube.
grad3 = [
    [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
    [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]
]

# The gradients are the midpoints of the vertices of a hypercube.
grad4 = [
    [0,1,1,1],  [0,1,1,-1],  [0,1,-1,1],  [0,1,-1,-1],
    [0,-1,1,1], [0,-1,1,-1], [0,-1,-1,1], [0,-1,-1,-1],
    [1,0,1,1],  [1,0,1,-1],  [1,0,-1,1],  [1,0,-1,-1],
    [-1,0,1,1], [-1,0,1,-1], [-1,0,-1,1], [-1,0,-1,-1],
    [1,1,0,1],  [1,1,0,-1],  [1,-1,0,1],  [1,-1,0,-1],
    [-1,1,0,1], [-1,1,0,-1], [-1,-1,0,1], [-1,-1,0,-1],
    [1,1,1,0],  [1,1,-1,0],  [1,-1,1,0],  [1,-1,-1,0],
    [-1,1,1,0], [-1,1,-1,0], [-1,-1,1,0], [-1,-1,-1,0]
]

# Permutation table.  The same list is repeated twice.
perm = [
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]

# A lookup table to traverse the simplex around a given point in 4D.
simplex = [
    [0,1,2,3],[0,1,3,2],[0,0,0,0],[0,2,3,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,3,0],
    [0,2,1,3],[0,0,0,0],[0,3,1,2],[0,3,2,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,3,2,0],
    [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
    [1,2,0,3],[0,0,0,0],[1,3,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,3,0,1],[2,3,1,0],
    [1,0,2,3],[1,0,3,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,3,1],[0,0,0,0],[2,1,3,0],
    [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
    [2,0,1,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,0,1,2],[3,0,2,1],[0,0,0,0],[3,1,2,0],
    [2,1,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,1,0,2],[0,0,0,0],[3,2,0,1],[3,2,1,0]
]

