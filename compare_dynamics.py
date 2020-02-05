import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det, norm


def Hankel(s):
    # s = (l, 2)
    l = s.shape[0]
    if l % 2 == 0:  # l is even
        H = np.zeros([l, int(l/2) + 1])
    else:  # l is odd
        if l == 3:
            H = np.zeros([4, 2])
        else:
            H = np.zeros([int(np.ceil(l / 2)) * 2, int(np.ceil(l / 2))])
    for i in range(int(H.shape[0]/2)):
            H[2 * i, :] = np.transpose(s[i:i + H.shape[1], 0])
            H[2 * i + 1, :] = np.transpose(s[i:i + H.shape[1], 1])
    return H


def stitchHankel(s, H0, reverse=False):
    hrows, hcols = H0.shape  # height of matrix H0
    l0 = hrows  # longitude of sequence s0
    if int(hrows/2) == hcols:  # if l0 is odd
        if hcols == 2:
            l0 = 3
        else:
            l0 = hrows - 1
    l = s.shape[0]  # longitude of sequence s
    dl = l - l0
    if reverse:  # stitch backwards
        H = np.zeros([hrows, l - hrows + 1])
        for i in range(int(np.ceil(hrows/2))):
                H[2 * i, :] = np.transpose(s[i:i + H.shape[1], 0])
                H[2 * i + 1, :] = np.transpose(s[i:i + H.shape[1], 1])
    else:
        H = np.zeros([hrows, dl])
        for i in range(int(np.ceil(hrows/2))):
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


def createSinData(num_root, num_extra, printValues=False):
    # num_root: Longitude of sequence root (s0)
    # num_extra: Longitude of s1 and s2
    # s0 size: (num_root, 2)
    # s1 and s2 size: (num_extra, 2)
    if num_extra <= 2 or num_root <= 2:
        raise ValueError('num_root and num_extra must be at least 3')
    if num_root < num_extra:
        raise ValueError('num_root must be higher than num_extra')
    x = np.linspace(1, 5, num=num_root + num_extra)
    s = np.transpose(np.vstack([x, np.log(x)]))
    s0 = s[0:-num_extra, :]
    s1 = s[num_root:, :]
    s2 = s1.copy()
    s2[:, 1] = - s2[:, 1] + 2*s0[-1,1]
    if printValues:
        print('s0: (x,y) \ns0 dim: ', s0.shape, '\n', np.around(s0), '\n')
        print('s1: (x,y) \ns1 dim: ', s1.shape, '\n', np.around(s1), '\n')
        print('s2: (x,y) \ns2 dim: ', s2.shape, '\n', np.around(s2), '\n')
    return s0, s1, s2


def compareSequences(s0, s1, s2, eps):
    H0root = Hankel(s0)
    H1root = Hankel(s1)
    H2root = Hankel(s2)

    H1withH0root = stitchHankel(np.vstack([s0, s1]), H0root)
    H2withH0root = stitchHankel(np.vstack([s0, s2]), H0root)
    H0withH1root = stitchHankel(np.vstack([s0, s1]), H1root, reverse=True)
    H0withH2root = stitchHankel(np.vstack([s0, s2]), H2root, reverse=True)

    d1_for = JBLD(Gram(H0root, eps), Gram(H1withH0root, eps))
    d1_back = JBLD(Gram(H1root, eps), Gram(H0withH1root, eps))
    d2_for = JBLD(Gram(H0root, eps), Gram(H2withH0root, eps))
    d2_back = JBLD(Gram(H2root, eps), Gram(H0withH2root, eps))

    d1 = (d1_for + d1_back)/2
    d2 = (d2_for + d2_back)/2

    print('d1_for:', np.around(d1_for, 2), 'd1_back:', np.around(d1_back, 2))
    print('d2_for:', np.around(d2_for, 2), 'd2_back:', np.around(d2_back, 2))

    w = np.argmin([d1, d2])
    print('Sequence', w + 1, 'has the most similar dynamics')

    return


# Create Data2
s0, s1, s2 = createSinData(6, 5)

# Test data
# s0 = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
# s1 = np.array([[10, 11], [12, 13], [14, 15], [16, 17]])
# s2 = np.array([[10, 11], [12, 13], [0, 0], [1, 1]])

# Plot sequences
plt.scatter(s0[:, 0], s0[:, 1], marker='o', c='k', label='Root Sequence')
plt.scatter(s1[:, 0], s1[:, 1], marker='o', label='Sequence 1')
plt.scatter(s2[:, 0], s2[:, 1], marker='o', label='Sequence 2')
plt.legend()
plt.show()

compareSequences(s0, s1, s2, 0.1)