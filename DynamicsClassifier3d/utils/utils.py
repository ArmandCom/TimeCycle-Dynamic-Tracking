import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.misc as sm
# import factorization

# Note: Factorize and construct matrix K

def get_K(M):
    # assert len(M.size()) ==
    return torch.bmm(M,M.permute(0, 2, 1))

def factorize(K):
    F = None
    return F

def get_G(M):
    '''
    M: [*, T, f] --  T = n_frames_input
    G: [*, T/2 -1, T/2 -1] -- T/2 -1 = autoregressor_size
    '''

    size_G = M.size(1) // 2 + M.size(1) % 2
    G = torch.zeros(M.size(0), size_G, size_G).cuda()
    # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

    for i in range(M.size(1) - size_G + 1):
        a = M[:, i:i+size_G]

        G += torch.bmm(a, a.permute(0, 2, 1)) #- meanK # With unbias

    return G

def get_partial_G(M, L=0):
    '''
    M: [*, T, f] --  T = n_frames_input
    G: [*, T/2 -1, T/2 -1] -- T/2 -1 = autoregressor_size
    '''

    Gs = []
    # size_G = M.size(1) // 2 + M.size(1) % 2
    num_g = M.size(1) - 2*L + 1
    for idx_g in range(num_g):
        G = torch.zeros(M.size(0), L, L).cuda()
        for i in range(L):
            if idx_g == 26:
                print('a')
            a = M[:, idx_g+i:idx_g+i+L]
            G += torch.bmm(a, a.permute(0, 2, 1))
        Gs.append(G)
    Gs = torch.cat(Gs, dim=0)
    return Gs

def get_trace_K(M, flag='k'):

    '''
    :param M: [*, T, f] --  T = n_frames_input
    :return: scalar
    '''
    if flag == 'k':

        vecM = M.contiguous().view(M.size(0), 1, -1)

        # With unbias
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # tr = (vecM**2 - meanK).sum(dim=2, keepdim=True)

        # Without unbias
        tr = torch.bmm(vecM, vecM.permute(0, 2, 1))

    elif flag == 'g':
        tr = 0
        size_G = M.size(1) // 2 + M.size(1) % 2
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

        for i in range(M.size(1) - size_G + 1):
            a = M[:, i:i + size_G]
            veca = a.contiguous().view(a.size(0), 1, -1)

            # With unbias
            # tr += (veca ** 2 - meanK).sum(dim=2, keepdim=True)

            # Without unbias
            tr += torch.bmm(veca, veca.permute(0, 2, 1))

    else:
        print('Wrong flag')
        tr = None

    return tr

def get_dist(M, neigh):
    #
    # dist = []
    # # TODO: check in GPU
    # print(M.shape)
    # for i in range(M.size(1)):
    #     dist.append(torch.cdist(M[0:1, i], M[:, neigh[0:1, i]], p=2))
    #
    # dist1 = torch.cat(dist, dim=1).squeeze(2)

    K = get_K(M) #Note: erase cuda()

    i_index = torch.arange(0,M.size(1)).repeat(M.size(0),1,1).permute(0,2,1)
    # print(i_index.dtype)
    # print(neigh.dtype)
    neigh = neigh.long()
    neigh = torch.cat((i_index,neigh),2)
    i = neigh[:,:,0:1].cuda()
    j = neigh[:,:,1:].cuda()

    Kii = torch.gather(K,2,i)
    Kij = torch.gather(K,2,j)

    Kjj = []
    for b in range(M.size(0)):
        Kjj.append(K[b,j[b,:],j[b,:]].unsqueeze(0))

    Kjj = torch.cat(Kjj,0)


    dist = Kii + Kjj - 2*Kij

    # print(K, '\n', Kii, '\n', Kjj, '\n',Kij, '\n',)
    return dist

# def normalize():
#     """
#     """
#     torch.max()

def format_input(X, chan, shape, save_test = False):
    '''
    :param X: [N, channels(x,y,S), k points, Traj Length]
    :param chan: which channel use as coordinate
    :param shape: Height or width of the output tensor
    :return: Reshaped tensor: [N, shape, Traj Length]
    '''
    assert chan == 0 or chan == 1

    axis_values = X[:, chan]
    scores = X[:, -1]


    # Make sure trajectories fit in the image without losing their dynamics
    min_per_traj = torch.round(axis_values).min(2)[0].unsqueeze(2).repeat(1, 1, X.shape[-1])
    under0_idx = (min_per_traj < 0)
    if len(axis_values[under0_idx]) > 0:
        print('Warning: Negative coordinates found')
        axis_values[under0_idx] -= min_per_traj[under0_idx] - torch.rand(1)*shape/4
    max_per_traj = torch.round(axis_values).max(2)[0].unsqueeze(2).repeat(1, 1, X.shape[-1])
    overtop_idx = (max_per_traj >= shape)
    while len(axis_values[overtop_idx]) > 0:
        print('Warning: /2 size reduction for fitting')
        axis_values[overtop_idx] /= 2
        max_per_traj = torch.round(axis_values).max(2)[0].unsqueeze(2).repeat(1, 1, X.shape[-1])
        overtop_idx = (max_per_traj >= shape)

    # Extract indexes
    idx = torch.round(axis_values).long()

    # Create layout
    layout_scores = torch.zeros(X.shape[0], shape, X.shape[-1])
    layout_coords = torch.zeros(X.shape[0], shape, X.shape[-1])

    # Place scores in layout
    layout_scores.scatter_(1, idx, scores).unsqueeze(1)
    layout_coords.scatter_(1, idx, axis_values).unsqueeze(1)

    if save_test:
        for tr in range(axis_values.shape[1]):
            plt.plot(axis_values[0, tr])
            plt.savefig('test_plot_traj')
        sm.imsave('test_img_traj.png', layout_coords[0].numpy())

    return layout_scores, layout_coords, axis_values


def main():

    # n_frames_input = 6
    # n = n_frames_input // 2 + n_frames_input % 2
    # neigh = torch.randint(6,(1,6,2))
    #
    # '''Test get_G, get_K'''
    # M = torch.Tensor([np.linspace(11, 5, 7), np.linspace(4, 10, 7)]).unsqueeze(0).permute(0,2,1).float()
    # Gs = get_partial_G(M, 2).squeeze()
    # # K = get_K(M).squeeze()
    # # trK = torch.trace(K)
    # # trK2 = get_trace_K(M, 'k')
    # # trG = torch.trace(Gs)
    # # trG2 = get_trace_K(M, 'g')
    # # dist = get_dist(M, neigh)
    #
    # print(M.shape)
    # print(Gs.shape)

    traj_length = 9
    k_points = 2
    input = torch.randint(-1, 6, (2,2,k_points,traj_length)).float()
    scores = torch.ones((2,1,k_points,traj_length)).float()
    input = torch.cat([input, scores], dim=1)
    shape = 8 # 4
    chan = 0

    X = format_input(input, chan, shape, show=True)

if __name__ == "__main__":
    print("hello")
    main()