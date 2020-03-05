import torch
import numpy as np
device = torch.device('cpu')


def Hankel(s0, stitch=False, s1=0):
    """ Creates the Hankel Matrix given a sequence
    Args:
        - s0: Root sequence
        - switch: Boolean to indicate if Hankel must be stitched or not
        - s1: Sequence to add if Hankel must be stitched
    Returns:
        - H: Hankel matrix
    """
    dim = 1  # if x and y want to be treated jointly, change to dim=2
    l0 = s0.shape[0]
    l1 = 0
    if stitch:
        l1 = s1.shape[0]
        s0 = torch.cat([s0, s1])
    if l0 % 2 == 0:  # l is even
        num_rows = int(l0/2) * dim
        num_cols = int(l0/2) + 1 + l1
    else:  # l is odd
        num_rows = int(np.ceil(l0 / 2)) * dim
        num_cols = int(np.ceil(l0 / 2)) + l1
    H = torch.zeros([num_rows, num_cols])
    for i in range(int(num_rows/dim)):
        H[dim * i, :] = (s0[i:i + num_cols]).view(1, num_cols)
    return H


def Gram(H, eps):
    """ Generates a normalized Gram matrix given the Hankel matrix and a noise factor
    Args:
        - H: Hankel matrix containing temporal information
        - eps: noise factor (1)
    Returns:
        - Gnorm: normalized Gram matrix
    """
    N = np.power(eps, 2) * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t())
    Gnorm = G/torch.norm(G, 'fro')
    Gnorm = Gnorm + N
    return Gnorm


def JBLD(X, Y, det):
    """ Computes the Jensen-Bregman LogDet (JBLD) distance between two Gram matrices
    Args:
        - X and Y: Normalized Gram matrices
        - det: Boolean indicating if the determinant is going to be computed or not
    Returns:
        - d: JBLD distance value between X and Y
    """
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    if not det:
        d = (torch.det((X + Y) / 2)) - 0.5 * (torch.det(torch.matmul(X, Y)))
        # print("torch.det((X+Y)) = ", torch.det(X+Y))
    return d



def compare_dynamics(data_root, data, eps, coord, BS=1):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence including data_root
        - eps: Noise value
        - coord: Coordinate value (0: x, 1: y, 2: mean)
        - BS: Batch Size
    Returns:
        - dist: (BS, 2) JBLD distance between sequences
    """
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps), True)
    if coord == 2:
        dist = torch.mean(dist, dim=1)
    else:
        dist = dist[0][coord].item()  # Print only x information
    return dist


def predict_Hankel(H):
    """ Predicts one trajectory value given the Hankel Matrix
    Args:
        - H: Hankel matrix of the sequence
    Returns:
        - first_term: Predicted value
    """
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:,-1]
    last_column_of_H = H[-1,:]
    last_column_of_H = last_column_of_H[1:]
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    return first_term