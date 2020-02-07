import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det, norm


def Hankel(s0, stitch=False, s1=0):
    l0, dim = s0.shape
    l1 = 0
    if stitch:
        l1 = s1.shape[0]
        s0 = np.vstack([s0, s1])
    if l0 % 2 == 0:  # l is even
        num_rows = int(l0/2) * dim
        num_cols = int(l0/2) + 1 + l1
    else:
        num_rows = int(np.ceil(l0 / 2)) * dim
        num_cols = int(np.ceil(l0 / 2)) + l1
    H = np.zeros([num_rows, num_cols])
    for i in range(int(num_rows/dim)):
        for d in range(dim):
            H[dim * i + d, :] = np.transpose(s0[i:i + num_cols, d])
    if l1 == 1:  # Window
        H = H[:, 1:]
    return H


def JBLD(X, Y):
    d = np.log(det((X + Y)/2)) - 0.5*np.log(det(np.matmul(X, Y)))
    return d


def Gram(H, eps):
    N = np.power(eps, 2) * H.shape[0] * np.eye(H.shape[0])
    G = np.matmul(H, np.transpose(H)) + N
    Gnorm = G/norm(G, 'fro')
    return Gnorm


def createSinData(num_root, num_extra, printValues=False, plot=False):
    # num_root: Longitude of sequence root (s0)
    # num_extra: Longitude of s1 and s2
    # s0 size: (num_root, 2)
    # s1 and s2 size: (num_extra, 2)
    x = np.linspace(1, 5, num=num_root + num_extra)
    s = np.transpose(np.vstack([x, np.sin(x)]))
    s0 = s[0:-num_extra, :]
    s1 = s[num_root:, :]
    s2 = s1.copy()
    s2[:, 1] = - s2[:, 1] + 2*s0[-1,1]
    if printValues:
        print('s0: (x,y) \ns0 dim: ', s0.shape, '\n', np.around(s0), '\n')
        print('s1: (x,y) \ns1 dim: ', s1.shape, '\n', np.around(s1), '\n')
        print('s2: (x,y) \ns2 dim: ', s2.shape, '\n', np.around(s2), '\n')
    if plot:
        plt.scatter(s0[:, 0], s0[:, 1], marker='o', c='k', label='Root Sequence')
        plt.scatter(s1[:, 0], s1[:, 1], marker='o', label='Sequence 1')
        plt.scatter(s2[:, 0], s2[:, 1], marker='o', label='Sequence 2')
        plt.legend()
        plt.show()
    return s0, s1, s2


def compare_trajectories(s0, s1, s2, eps):
    H0 = Hankel(s0)
    H1stitchH0 = Hankel(s0, stitch=True, s1=s1)
    H2stitchH0 = Hankel(s0, stitch=True, s1=s2)
    d1 = JBLD(Gram(H0, eps), Gram(H1stitchH0, eps))
    d2 = JBLD(Gram(H0, eps), Gram(H2stitchH0, eps))
    return d1, d2


def compare_appearance(v0, v1, v2, eps):
    features = v0.shape[1]
    d_app = np.zeros([2, features])  # distance in appearance
    for f in range(features):
        v0_f = np.reshape(v0[:, f], (v0[:, f].shape[0], 1))  # (l0, 1)
        v1_f = np.reshape(v1[:, f], (v1[:, f].shape[0], 1))  # (l0, 1)
        v2_f = np.reshape(v2[:, f], (v2[:, f].shape[0], 1))  # (l0, 1)
        H0 = Hankel(v0_f)
        H0_1 = Hankel(v0_f, stitch=True, s1=v1_f)
        H0_2 = Hankel(v0_f, stitch=True, s1=v2_f)
        d_app[0, f] = JBLD(Gram(H0, eps), Gram(H0_1, eps))
        d_app[1, f] = JBLD(Gram(H0, eps), Gram(H0_2, eps))
    return np.mean(d_app[:, 0]), np.mean(d_app[:, 1])


l0 = 5
l1 = 3
l2 = 2
eps = 0.1

# Create temporal data
# s0, s1, s2 = createSinData(l0, l1, plot=True)
s0 = np.random.rand(l0, 4)
print(s0)
s1 = np.random.rand(l1, 4)
s2 = np.random.rand(l2, 4)

# Create appearance data
features = 63*63
v0 = np.random.rand(l0, features)
v1 = np.random.rand(l1, features)
v2 = np.random.rand(l2, features)

d1_traj, d2_traj = compare_trajectories(s0, s1, s2, eps)
d1_app, d2_app = compare_appearance(v0, v1, v2, eps)





