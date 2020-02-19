import torch
import numpy as np


def Hankel(s0, stitch=False, s1=0):
    l0 = s0.shape[0]
    l1 = 0
    if stitch:
        l1 = s1.shape[0]
        s0 = torch.cat([s0, s1])
    if l0 % 2 == 0:  # l is even
        num_rows = int(l0/2) * dim
        num_cols = int(l0/2) + 1 + l1
    else:
        num_rows = int(np.ceil(l0 / 2)) * dim
        num_cols = int(np.ceil(l0 / 2)) + l1
    H = torch.zeros([num_rows, num_cols])
    for i in range(int(num_rows/dim)):
        H[dim * i, :] = (s0[i:i + num_cols]).t()
    if l1 == 1:  # Window
        H = H[:, 1:]
    return H


def JBLD(X, Y):
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    return d


def Gram(H, eps):
    N = np.power(eps, 2) * H.shape[0] * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + N
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def compare_dynamics(data_root, data):
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps))
    dist = torch.mean(dist, 1)
    return dist


dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0')

BS = 100  # Batch Size
L0 = 5  # Longitude of the Root Sequence
L = 2  # Longitude of the Sequence being Tested
dim = 2  # Number of channels (x,y)
eps = 0.1  # Noise epsilon

# Create Random Data
data_root = torch.randn(BS, L0, dim,  device=device, dtype=dtype)  # size: (BS, L0, dim)
data = torch.randn(BS, L, dim, device=device, dtype=dtype)  # size: (BS, L, dim)

# Compare dynamics
d = compare_dynamics(data_root, data)  # size: (BS)
print(d)