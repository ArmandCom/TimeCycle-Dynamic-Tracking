import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from typing import Tuple

# from models.networks.model_pieces.base import BaseModule
from models.networks.blocks_2d import DownsampleBlock



class ImageEncoder(nn.Module):
  """Mnist encoder"""

  def __init__(self, input_shape, feat_latent_size):
    super(ImageEncoder, self).__init__()

    self.input_shape = input_shape
    self.feat_latent_size = feat_latent_size

    c, h, t = input_shape
    self.T = t
    self.h = h
    # self.idx_reverse = torch.LongTensor([i for i in range(t - 1, -1, -1)]).cuda()

    hc1, hc2, hc3, hc4, hc5 = 16, 32, 64, 128, 256
    activation_fn = nn.LeakyReLU()

    # Convolutional network
    self.conv = nn.Sequential(
      DownsampleBlock(channel_in=c, channel_out=hc1, traj_length=self.T, activation_fn=activation_fn),
      DownsampleBlock(channel_in=hc1, channel_out=hc2, traj_length=self.T, activation_fn=activation_fn),
      DownsampleBlock(channel_in=hc2, channel_out=hc3, traj_length=self.T, activation_fn=activation_fn),
      DownsampleBlock(channel_in=hc3, channel_out=hc4, traj_length=self.T, activation_fn=activation_fn),
      DownsampleBlock(channel_in=hc4, channel_out=hc5, traj_length=self.T, activation_fn=activation_fn),
    )
    self.deepest_shape = (hc5, h // 32, 1)
    print(reduce(mul, self.deepest_shape))
    self.hidden_features = hc4

    # FC network
    self.fc = nn.Sequential(
      nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=self.hidden_features),
      nn.BatchNorm1d(num_features=self.hidden_features),
      activation_fn,
      nn.Linear(in_features=self.hidden_features, out_features=feat_latent_size),
      nn.ReLU()
    ).cuda()

  def forward(self, x):
    # types: (torch.Tensor) -> torch.Tensor
    """
    Forward propagation.
    :param x: the input batch of images.
    :return: the batch of latent vectors.
    """
    h = x
    h = self.conv(h).permute(0,3,2,1)
    h = h.reshape(x.shape[0] * self.T, -1)

    o = self.fc(h)#.view(x.shape[0], self.T)
    return o