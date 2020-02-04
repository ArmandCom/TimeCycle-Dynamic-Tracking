import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det, norm


def Hankel(s):
    # s = (l, 2)
    l = s.shape[0]
    if l % 2 == 0:  # l is even
        H = np.zeros([l, int(l/2) + 1])
    else:  # l is odd
        H = np.zeros([int(np.around(l / 2)) * 2, int(np.around(l / 2))])
    for i in range(int(H.shape[0]/2)):
            H[2 * i, :] = np.transpose(s[i:i + H.shape[1], 0])
            H[2 * i + 1, :] = np.transpose(s[i:i + H.shape[1], 1])
    return H


def stitchHankel(s, H0):
    hrows, hcols = H0.shape  # height of matrix H0
    l0 = hrows  # longitude of sequence s0
    if int(hrows/2) == hcols:  # if l0 is odd
        l0 = hrows - 1
    l = s.shape[0]  # longitude of sequence s
    dl = l - l0
    H = np.zeros([hrows, dl])
    for i in range(int(hrows/2)):
        H[2 * i, :] = np.transpose(s[hcols + i:hcols + i + dl, 0])
        H[2 * i + 1, :] = np.transpose(s[hcols + i:hcols + i + dl, 1])
    H = np.hstack([H0, H])
    return H


def JBLD(X, Y):
    d = np.log(det((X + Y)/2)) - 0.5*np.log(det(np.matmul(X, Y)))
    return d


def Gram(H, eps):
    N = np.power(eps, 2) * H.shape[0] * np.eye(H.shape[0])
    G = np.matmul(H, np.transpose(H)) + N
    Gnorm = G/norm(G, 'fro')
    return Gnorm


def createSinData(num_root, num_extra):
    # Sequences size (l, 2)
    x = np.linspace(1, 5, num=num_root + num_extra)
    y1 = np.log(x)
    s1 = np.transpose(np.vstack([x, y1]))
    s0 = s1[0:-num_extra, :]
    s2 = s1[-num_extra-1:, :].copy()
    s2[:, 1] = - s2[:, 1] + 2*s0[-1,1]
    return s0, s1, s2

# Create Data
s0, s1, s2 = createSinData(5, 3)

# Plot sequences
plt.plot(s1[:, 0], s1[:, 1], marker='o', ls='--', label='Sequence 1')
plt.plot(s2[:, 0], s2[:, 1], marker='o', ls='--', label='Sequence 2')
plt.plot(s0[:, 0], s0[:, 1], marker='o', c='k')
plt.legend()
plt.show()

# Create Hankel Matrices
H0 = Hankel(s0)
H1 = stitchHankel(s1, H0)
H2 = stitchHankel(s2, H0)

# Create Gram Matrices
eps = 0.1
G0 = Gram(H0, eps)
G1 = Gram(H1, eps)
G2 = Gram(H2, eps)

# Compute JBLD distances
d01 = JBLD(G1, G0)
d02 = JBLD(G2, G0)

print('d01:', np.around(d01, 2))
print('d02:', np.around(d02, 2))

w = np.argmin([d01, d02])
print('Sequence', w+1, 'has the most similar dynamics')

