import torch
import torch.nn as nn
import torch.functional as F

class ARTransformerLayer(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''
    def __init__(self, k, traj_length):
        super(ARTransformerLayer, self).__init__()
        self.K = k
        self.T = traj_length
        self.unfold = nn.Unfold(kernel_size=[self.K, 3], padding=[0,1])
        self.idx = self.get_indices().cuda()

    def get_indices(self):

        ori_idx = torch.arange(self.K)
        # idx0 should point to the 'current' time-step
        idx0 = ori_idx.view(self.K, 1, 1).repeat(1, self.K, self.K).view(-1)
        idx1 = ori_idx.view(1, self.K, 1).repeat(self.K, 1, self.K).view(-1)
        idx2 = ori_idx.view(1, 1, self.K).repeat(self.K, self.K, 1).view(-1)

        idx_all = torch.stack([idx1, idx0, idx2], dim=1)\
            .unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        return idx_all

    def forward(self, x):
        batch_size, n_chans = x.shape[0], x.shape[1]
        x = self.unfold(x).view(batch_size, n_chans, self.K, 3, self.T)
        x = torch.gather(x, 2, self.idx.repeat(batch_size, n_chans, 1, 1, self.T))\
            .permute(0,1,2,4,3).reshape(batch_size, n_chans, self.K**3, -1)
        return x


class PermuteLastDims(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''
    def __init__(self):
        super(PermuteLastDims, self).__init__()

    def forward(self, x):
        return x.permute(0,1,3,2)

class PrintShape(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print('Shape of tensor: ', x.shape)
        return x

class ARMask(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''
    def __init__(self, shape, traj_length, n_act_t):
        super(ARMask, self).__init__()
        self.shape = shape
        self.T = traj_length
        self.mask = self.generate_mask(n_act_t)

    def generate_masks(self, n_act_t):
        mask_layout = torch.zeros(self.shape, self.T)
        mask_layout[:,:n_act_t] = torch.ones(self.shape, n_act_t).unsqueeze(0).unsqueeze(1)
        # mask_layout[:,:n_act_t] = torch.ones(self.shape, n_act_t).unsqueeze(0).unsqueeze(1)
        return mask_layout

    def forward(self, x):
        x = x * self.mask[..., :x.shape[-1]]
        return x

if __name__ == "__main__":

    B, ch, K, T = 1, 2, 4, 2
    x = torch.arange(0, K*ch*T*B).view(B, ch, K, T).float()
    transf = ARTransformerLayer(K, T)
    o = transf(x)
    # print(o[0, 0])
