#
# PointLinesLSQ.py -- find a point nearest a set of n lines in 3D space by least-squares
# A. Thall
# 7. October 2017
"""
Algorithm is math calculation in the public domain, did self, but forgot about
outerproduct and cribbed a bit from discussions on open internet forums.

The derivation of the formula just takes a little linalg and vector calc, but
the resulting formula for the 3x3 linear equation is not too ugly:

   [ SUM (n_i*trans(n_i) - I)] * p = SUM [(n_i*trans(n_i) - I) * a_i]

which is our S*p = C equation.

To compute:

(1)  Assuming an Array A of points and an Array N of unit-length vectors at each
point in A.  (We'll compute N below assuming lines are expressed as (A, B) given
two points defining the line.)

(2) From our list N of vectors, we create a list of 3x3 matrices by taking OUTER products
(see below) of each vector with itself and subtracting the 3x3 identity.

(3) The sum of all these 3x3 matrices gives us the left-hand matrix for our S*x = C
linear system.

(4) To create C, we take same list of 3x3 matrices, take matrix product of each with
corresponding point in list A to get list of vectors.

(5) The sum of these vectors gives us the right hand side C of our S*x = C system.
Then just call pn.linalg.solve(S, C) and life is good.

----
* outer product of two vectors is just v*trans(v), rather than dot product which
is trans(v)*v.  Where dot gives a scalar, outer gives a matrix:

      outer((a, b, c), (x, y, z)) = [[ ax, ay, az], [bx, by, bz], [cx, cy, cz]]
 with itself as above is just:
      outer((a, b, c), (a, b, c)) = [[ aa, ab, ac], [ba, bb, bc], [ca, cb, cc]]

In this case, it gives a 3x3 matrix from the product of two 1x3 vectors.
"""

import numpy as np

# We will assume lines specified by 2 points in 3D, and convert these to point-vector form
# for the computation.  So (a, b) becomes (pA = a, N = (b - a)/||b - a||)

A = np.array([[0, 0, 0], [2, 2, 0], [0, 2, 2]])
B = np.array([[0, 0, 2], [2, 0, 0], [2, 2, 2]])

# or can test with some random lines, from random points in unit cube
#A = np.random.rand(10, 3)
#B = np.random.rand(10, 3)

print(A)
print(B)

# compute vectors from Ai to Bi and normalize them
N = B - A
norms = np.apply_along_axis(np.linalg.norm, 0, N)
N = N/norms

print("N = \n", N)
myI = np.eye(3,3)
def self_outerprod_minus_I(v):
    return np.outer(v, v) - myI

# Now S = sum(outer(N_i, N_i) - I) over all i
Nouter = np.array([ self_outerprod_minus_I(n) for n in N])
S = np.sum(Nouter, 0)

print("S = \n", S)

# Now C = sum((outer(N_i, N_i) - I) dot a_i) over all i

C_array = np.array([np.dot(Nouter[i], A[i]) for i in range(len(A))])
C = np.sum(C_array, axis=0)
print("C = \n", C)

# Now we have the linear equation S*p = C .  Solve it!
p = np.linalg.solve(S, C)

print("Solving this system gives: ", p)



