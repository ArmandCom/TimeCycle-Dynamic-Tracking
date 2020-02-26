from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

# from models.base import BaseModule
from models.networks.blocks_2d import DownsampleBlock_nonAR as DownsampleBlock
from models.networks.blocks_2d import UpsampleBlock, ResidualBlock


class TemporalEncoder(nn.Module):
    """
    MNIST model encoder.
    """
    def __init__(self, input_shape, ini_cond):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(TemporalEncoder, self).__init__()

        self.input_shape = input_shape
        self.initial_conditions_length = ini_cond

        c, h, t = input_shape
        self.T = t - self.initial_conditions_length
        self.h = h
        self.part_idx_reverse = torch.LongTensor([i for i in range(self.T-1, -1, -1)]).cuda()
        self.idx_reverse = torch.LongTensor([i for i in range(t-1, -1, -1)]).cuda()



        hc1, hc2, hc3 = 16, 32, 64

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        # self.conv = nn.Sequential(
        #     DownsampleBlock(channel_in=c,   channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
        #     DownsampleBlock(channel_in=hc1, channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
        #     DownsampleBlock(channel_in=hc1, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
        #     DownsampleBlock(channel_in=hc2, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
        #     DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        #     DownsampleBlock(channel_in=hc3, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        # )
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc1, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc2, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc3, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        )

        self.upsample = nn.Sequential(
            UpsampleBlock(channel_in=hc3, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc3, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc2, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc2, channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc1, channel_out=c,   traj_length=self.T, activation_fn=activation_fn),
        )

        self.conv_reverse = nn.Sequential(
            DownsampleBlock(channel_in=2*c,   channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc1, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc2, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc3, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        )

        self.deepest_shape = (hc3, h // 32, 2)

        self.res_ini = nn.Sequential(
            ResidualBlock(channel_in=1,  channel_out=hc2, ini_traj_length=self.initial_conditions_length, activation_fn=activation_fn),
            ResidualBlock(channel_in=hc2, channel_out=reduce(mul, self.deepest_shape), ini_traj_length=self.initial_conditions_length, activation_fn=activation_fn),
        )


        self.hidden_features = int(hc3//2) #int(hc3//2)
        # FC network
        self.gru = nn.GRU(reduce(mul, self.deepest_shape), reduce(mul, self.deepest_shape[0:2]),
                          num_layers=1, batch_first=True, bidirectional=False).cuda()
        self.gru_rev = nn.GRU(reduce(mul, self.deepest_shape), reduce(mul, self.deepest_shape[0:2]),
                          num_layers=1, batch_first=True, bidirectional=False).cuda()
        self.gru_bi = nn.GRU(reduce(mul, self.deepest_shape), reduce(mul, self.deepest_shape[0:2]),
                          num_layers=1, batch_first=True, bidirectional=True).cuda()
        self.gru_bi_fw = nn.GRU(reduce(mul, self.deepest_shape[0:2]), reduce(mul, self.deepest_shape[0:2]),
                             num_layers=1, batch_first=True, bidirectional=False).cuda()

        # .view(batch, seq_len, num_directions, hidden_size)
        self.fc1 = nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=self.hidden_features).cuda()
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_features).cuda()
        self.ac1 = activation_fn.cuda()
        self.fc2 = nn.Linear(in_features=self.hidden_features, out_features=1).cuda()
        self.ac2 = nn.ReLU().cuda() # nn.Sigmoid()


    def forward(self, x, x_coord):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        '''Conv network'''
        x_ini = x_coord[..., 0:1, :self.initial_conditions_length]
        x = x[..., self.initial_conditions_length:]
        h_ini = self.res_ini(x_ini).permute(0, 3, 1, 2).view(len(x), self.initial_conditions_length, -1)

        h = self.conv(x).permute(0, 3, 1, 2).view(len(x), self.T, -1)
        h_gru_fw_0 = torch.zeros((1, len(h),  h.shape[-1])).cuda()
        h = self.gru_bi_fw(h, h_gru_fw_0)[0].permute(0, 2, 1).reshape(h.shape[0], -1, 1, self.T)
        h_us = self.upsample(h)

        h_rev = torch.cat([x,h_us], dim=-3).index_select(-1, self.part_idx_reverse)
        h_rev = self.conv_reverse(h_rev)
        h_rev = h_rev.index_select(-1, self.part_idx_reverse)

        h       = h     .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h_rev   = h_rev .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h = torch.cat([h, h_rev], dim=-1)

        # Give known initial conditions
        h = torch.cat([h_ini, h], dim=-2)

        '''GRU - Recurrent Network'''
        # h_gru_0 = torch.zeros((1, len(h),  int(h.shape[-1]//2))).cuda()
        # out_gru, h_gru = self.gru(h, h_gru_0)
        # out_gru_rev, h_gru_rev = self.gru_rev(h.index_select(-2, self.idx_reverse),
        #                                       h_gru)
        # torch.cat([out_gru, out_gru_rev.index_select(-2, self.idx_reverse)], dim=-1)

        h_gru_0 = torch.zeros((2, len(h),  int(h.shape[-1]//2))).cuda()
        h, h_gru = self.gru_bi(h, h_gru_0)


        '''FC network'''
        # h = self.fc1(h).permute(0, 2, 1)
        # h = self.ac1(self.bn1(h)).permute(0,2,1) #TODO: no BN
        # o = self.fc2(h).permute(0,2,1).unsqueeze(1)

        '''FC network'''
        o = self.fc2(self.ac1(self.fc1(h))).permute(0,2,1).unsqueeze(1)

        return o + x_ini[..., 0:1]

    # def forward(self, x):
    #     # types: (torch.Tensor) -> torch.Tensor
    #     """
    #     Forward propagation.
    #
    #     :param x: the input batch of images.
    #     :return: the batch of latent vectors.
    #     """
    #     '''Conv network'''
    #     h = x
    #     h = self.conv(h)
    #     h_rev = x.index_select(-1, self.idx_reverse)
    #     h_rev = self.conv(h_rev)
    #     h_rev = h_rev.index_select(-1, self.idx_reverse)
    #
    #     h       = h     .permute(0, 3, 1, 2).view(len(h), self.T, -1)
    #     h_rev   = h_rev .permute(0, 3, 1, 2).view(len(h), self.T, -1)
    #     h = torch.cat([h, h_rev], dim=-1)
    #
    #     '''FC network'''
    #     h = self.fc1(h).permute(0, 2, 1)
    #     h = self.ac1(self.bn1(h)).permute(0,2,1) #+ self.h/2
    #     o = self.ac2(self.fc2(h)).permute(0,2,1).unsqueeze(1)
    #     return o