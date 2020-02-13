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

if __name__ == "__main__":

    B, ch, K, T = 1, 2, 4, 2
    x = torch.arange(0, K*ch*T*B).view(B, ch, K, T).float()
    transf = ARTransformerLayer(K, T)
    o = transf(x)
    # print(o[0, 0])
