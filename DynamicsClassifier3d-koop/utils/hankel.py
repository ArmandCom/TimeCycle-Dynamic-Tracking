import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from matplotlib import *
from pylab import *
import time

def hankel_singular_values(H, compute_uv = True, gpu_id=0):
    sv = []
    for n in range(H.size(0)):
        sv.append(torch.svd(H[n,:])[1].unsqueeze(0))
    return torch.cat(sv, dim=0)

def represent_sv_heatmap(sv, name):
    '''
    :param sv: n_dim, n_sv
    :return:
    '''
    dir = '../figures'
    plt.figure()
    # svM = torch.max(sv, dim=1)[0].unsqueeze(1)
    # svM[svM == 0] = 1
    # sv = sv.div(svM)
    plt.pcolormesh(torch.t(sv).data.cpu().detach().numpy(), cmap='Reds')
    plt.colorbar()
    plt.savefig(os.path.join(dir, name + '.png'), dpi=200)  # , dpi=200
    # plt.show()
    plt.close()

    print(np.mean(sv.data.cpu().detach().numpy(), axis=0))
    return

def gram_nuclear_norm(y, gid):
    '''y shape:
    [m+p, n_features, T]'''
    wsize = int((y.shape[2] + 1) / 2)
    ksize = [1, wsize]
    poolL2  = nn.LPPool2d(2, ksize, stride=1).cuda(gid) # ceil_mode?
    avgpool = nn.AvgPool2d(ksize, stride=1).cuda(gid)
    gnn = avgpool(poolL2(y)**2) # * wsize

    return gnn

def unbias_seq(ts):
    mean = torch.mean(ts, dim=-1)
    ts = ts - mean.unsqueeze(-1)
    return ts

def hankel_matrix(y, unbiased = False, diff = False):

    '''y shape:
    [m+p, n_features, T]'''

    '''Create Hankel matrix'''

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    assert y.shape[2]%2 == 1
    sz = int((y.shape[2] + 1) / 2)


    chan = y.shape[0]
    nfeat = y.shape[1]
    nr = sz
    nc = sz
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)
    # reHidex = reHidex.numpy()

    T = y.shape[2]
    y = y.contiguous().view(-1,T)
    # H = y[:, :, reHidex].view(chan, T, nr, nc)
    H = y[:, reHidex].view(-1, nr, nc)

    return H

def gram_matrix(y, delta=0, unbiased = False, diff = False, normalize=True):

    '''y shape:
    [m+p, n_features, T]'''

    '''Create Hankel matrix'''

    if normalize:
        y_mean = y.mean(-1).unsqueeze(-1)
        y_std = y.std(-1).unsqueeze(-1)
        y = (y-y_mean)/(y_std + delta)

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    assert y.shape[2]%2 == 1
    sz = int((y.shape[2] + 1) / 2)


    chan = y.shape[0]
    nfeat = y.shape[1]
    nr = sz
    nc = sz
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)

    T = y.shape[2]
    y = y.contiguous().view(-1,T)
    # H = y[:, :, reHidex].view(chan, T, nr, nc)
    H = y[:, reHidex].view(-1, nr, nc)
    G = torch.matmul(H, H.permute(0,2,1))
    Gnorm = torch.norm(G.view(H.shape[0], 1, nr*nr), dim=2).unsqueeze(-1)
    G = G/Gnorm
    if delta!=0:
        G = G + delta*torch.eye(nr, nc).cuda()

    return G

def gram_matrix_sized(y, delta=0, size=None, unbiased = False, diff = False, normalize=True):

    '''y shape:
    [m+p, n_features, T]'''

    '''Create Hankel matrix'''

    # assert y.shape[-1] >= 2*size + 1

    if normalize:
        y_mean = y.mean(-1).unsqueeze(-1)
        y_std = y.std(-1).unsqueeze(-1)
        y = (y-y_mean)/(y_std + delta)

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    nr = int(size)
    nc = (y.shape[-1]+1)-nr
    # nc = int((y.shape[-1] + 1) / 2)

    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)

    T = y.shape[2]
    y = y.contiguous().view(-1,T)
    # H = y[:, :, reHidex].view(chan, T, nr, nc)
    H = y[:, reHidex].view(-1, nr, nc)
    G = torch.matmul(H, H.permute(0,2,1))
    Gnorm = torch.norm(G.view(H.shape[0], 1, nr*nr), dim=2).unsqueeze(-1)
    G = G/Gnorm

    if delta!=0:
        G = G + delta*torch.eye(nr, nr).cuda()

    return G

def JBLD(Gx,Gy):
    return torch.logdet((Gx + Gy)/2) - (torch.logdet(Gx) + torch.logdet(Gy))/2

def JBLDLoss(S, delta):
    # TODO: specify size of G (min of all)
    traj_length_1 = int(S.shape[-1] // 2)
    if S.shape[-1]%2 == 1:
        if traj_length_1%2 == 0:
            traj_length_1 += 1
            sz = int((traj_length_1 + 1) / 2) #size of gram matrix
            G1 = gram_matrix_sized(S[..., :traj_length_1],        delta=delta, size=sz)
            G2 = gram_matrix_sized(S[..., (traj_length_1-1):],    delta=delta, size=sz)
        else:
            sz = int((traj_length_1 + 1) / 2) #size of gram matrix
            G1 = gram_matrix_sized(S[..., :traj_length_1],        delta=delta, size=sz)
            G2 = gram_matrix_sized(S[..., traj_length_1:],        delta=delta, size=sz)
        G_tot = gram_matrix_sized(S, delta=2*delta, size=sz)
    else:
        traj_length_1 = int(S.shape[-1] // 2)
        if traj_length_1%2 == 0:
            traj_length_1 -= 1
        sz = int((traj_length_1+1)/2) #size of gram matrix
        G_tot = gram_matrix_sized(S[..., :-1], delta=2*delta, size=sz)
        G1 = gram_matrix_sized(S[..., :traj_length_1],        delta=delta, size=sz)
        G2 = gram_matrix_sized(S[..., traj_length_1:],        delta=delta, size=sz)

    return JBLD(G1, G_tot) + JBLD(G2, G_tot) #Note: Not working properly

def JBLDLoss_rolling(S, delta, sz=None):
    unbiased=False
    if sz is None and S.shape[-1]%2==0:
        sz = S.shape[-1] - 1
    elif sz is None and S.shape[-1]%2==1:
        sz = S.shape[-1] - 2
    else: assert sz%2==1

    jblds = []
    for t in range(S.shape[-1]-sz):
        G1 = gram_matrix(S[..., t  :t+sz  ],  delta=delta, unbiased=unbiased)
        G2 = gram_matrix(S[..., t+1:t+sz+1],  delta=delta, unbiased=unbiased)
        jblds.append(JBLD(G1, G2))
    jblds = torch.cat(jblds)


    return jblds

def reweighted_loss(G_old, G, gpu_id=0):

    Ginv_old = torch.inverse(G_old)

    '''Norm'''
    Ginv_old_vec = Ginv_old.contiguous().view(Ginv_old.shape[0], -1)
    Ginv_old_norm = Ginv_old_vec / torch.sum(torch.norm(Ginv_old_vec, p=2, dim=1))
    # Ginv_old_norm = Ginv_old  # back to normal

    Ginv_old_norm_vec = Ginv_old_norm.contiguous().view(-1, 1).squeeze()#.cuda(gid))
    G_vec = G.view(-1, 1).squeeze()
    rw_gnn = torch.dot(G_vec,Ginv_old_norm_vec)

    return rw_gnn

if __name__ == "__main__":

    gpu_id = 0
    # y = torch.arange(27).float().unsqueeze(0)
    y = Variable(torch.from_numpy(np.random.randint(5, size=(4, 3, 2))))\
        .type('torch.FloatTensor').cuda(gpu_id)
    # y = Variable(torch.from_numpy(np.array([[0, 1, 2],[3, 4, 6], [7, 8, 9]])))\
    #     .type('torch.FloatTensor').cuda(gpu_id)
    y_old = Variable(torch.from_numpy(np.random.randint(5, size=(2, 3, 5))))\
        .type('torch.FloatTensor').cuda(gpu_id)


    # y = y.view(2,3,-1).permute(1,2,0)
    # Y = y.expand(y.size(0), y.size(0), y.size(1), y.size(2))
    # Yt = Y.permute(1,0,2,3).contiguous().view(9,-1,2).permute(2,0,1)
    # Y = Y.contiguous().view(9,-1,2).permute(2,0,1)
    # # a = pd(Y.contiguous().view(9,-1), Yt.contiguous().view(9,-1))
    # a = Y - Yt

    # a = comb_pairwise_distance(y)
    #
    # # Hsize = torch.Size([14, 14])
    # # gnn = gram_nuclear_norm(y, gpu_id)
    # # hnn = hankel_nuclear_norm(y, Hsize, gpu_id)
    # G = comp_gram_matrix(y_old, delta = 0.001)

    # Re-weighted heuristic
    G_old = gram_matrix(y_old, delta = 0.001)
    G = gram_matrix(y)
    # G_old = torch.eye(3).unsqueeze(0).repeat(2, 1, 1).cuda(0)
    rwl = reweighted_loss(G_old, G)
    H = hankel_matrix(y)
    sv = hankel_singular_values(H)
    represent_sv_heatmap(sv, 'ranks')
    # G_old = torch.eye(4).unsqueeze(0).repeat(6,1,1)

    print(y, '\n', H, '\n', H.shape, '\n', sv.shape, '\n', sv)
    # print(G_old.shape, '\n', G_old, '\n', rwl,'\n', y)
