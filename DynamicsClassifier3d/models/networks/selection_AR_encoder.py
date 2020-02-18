from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

# from models.base import BaseModule
from models.networks.blocks_2d import DownsampleBlock
from models.networks.blocks_2d import UpsampleBlock


class TemporalEncoder(nn.Module):
    """
    MNIST model encoder.
    """
    def __init__(self, input_shape):
        # type: (Tuple[int, int, int]) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(TemporalEncoder, self).__init__()

        self.input_shape = input_shape

        c, h, t = input_shape
        self.T = t
        self.h = h
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
            DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        )

        self.upsample = nn.Sequential(
            UpsampleBlock(channel_in=hc3, channel_out=hc2, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc2, channel_out=hc1, activation_fn=activation_fn),
            UpsampleBlock(channel_in=hc1, channel_out=c, activation_fn=activation_fn),
        )

        self.conv_reverse = nn.Sequential(
            DownsampleBlock(channel_in=c,   channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc1, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
            DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
        )

        self.deepest_shape = (hc3, h // 8, 2)
        self.hidden_features = int(hc3//2)
        # FC network
        # Reduce applies multiplication through all elements of list reduce(mul, self.deepest_shape)
        self.fc1 = nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=self.hidden_features).cuda()
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_features).cuda()
        self.ac1 = activation_fn.cuda()
        self.fc2 = nn.Linear(in_features=self.hidden_features, out_features=1).cuda()
        self.ac2 = nn.ReLU().cuda() # nn.Sigmoid()


    def forward2(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        '''Conv network'''
        h = x
        h = self.conv(h)
        h_us = self.upsample(h)

        h_rev = (x+h_us).index_select(-1, self.idx_reverse)
        h_rev = self.conv_reverse(h_rev)
        h_rev = h_rev.index_select(-1, self.idx_reverse)

        h       = h     .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h_rev   = h_rev .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h = torch.cat([h, h_rev], dim=-1)

        '''FC network'''
        h = self.fc1(h).permute(0, 2, 1)
        h = self.ac1(self.bn1(h)).permute(0,2,1)
        o = self.ac2(self.fc2(h)).permute(0,2,1).unsqueeze(1)
        #TODO: check shape to output [B, 1, 1, T]
        return o

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        '''Conv network'''
        h = x
        h = self.conv(h)
        h_rev = x.index_select(-1, self.idx_reverse)
        h_rev = self.conv(h_rev)
        h_rev = h_rev.index_select(-1, self.idx_reverse)

        h       = h     .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h_rev   = h_rev .permute(0, 3, 1, 2).view(len(h), self.T, -1)
        h = torch.cat([h, h_rev], dim=-1)

        '''FC network'''
        h = self.fc1(h).permute(0, 2, 1)
        h = self.ac1(self.bn1(h)).permute(0,2,1) + self.h/2
        o = self.ac2(self.fc2(h)).permute(0,2,1).unsqueeze(1)
        return o