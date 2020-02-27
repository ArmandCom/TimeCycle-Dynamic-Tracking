import torch
import numpy as np
device = torch.device('cpu')


def Hankel(s0, stitch=False, s1=0):
    """ Generates a candidate sequence given an index
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
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    N = np.power(eps, 2) * H.shape[0] * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + N
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def JBLD(X, Y, det):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    if not det:
        d = (torch.det((X + Y) / 2)) - 0.5 * (torch.det(torch.matmul(X, Y)))
        # print("torch.det((X+Y)) = ", torch.det(X+Y))
    return d



def compare_dynamics(data_root, data, eps, BS=1):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps), False)
    dist = torch.mean(dist, 1)
    print(dist[0].item())
    return dist


def predict_Hankel(H):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:,-1]
    last_column_of_H = H[-1,:]
    last_column_of_H = last_column_of_H[1:]
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    return first_term