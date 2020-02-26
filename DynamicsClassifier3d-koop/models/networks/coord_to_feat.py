import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from typing import Tuple

# from models.networks.model_pieces.base import BaseModule
from models.networks.blocks_2d import ResidualBlock



class CoordResidual(nn.Module):
  """Mnist encoder"""

  def __init__(self, input_shape, feat_latent_size):
    super(CoordResidual, self).__init__()

    self.input_shape = input_shape
    self.feat_latent_size = feat_latent_size

    c, _, t = input_shape
    self.ini_length = t
    # self.idx_reverse = torch.LongTensor([i for i in range(t - 1, -1, -1)]).cuda()

    hc = int(self.feat_latent_size//2)
    activation_fn = nn.LeakyReLU()

    # Convolutional network
    self.res = nn.Sequential(
      ResidualBlock(channel_in=1, channel_out=hc, ini_traj_length=self.ini_length,
                    activation_fn=activation_fn),
      ResidualBlock(channel_in=hc, channel_out=self.feat_latent_size,
                    ini_traj_length=self.ini_length, activation_fn=activation_fn),
    )

  def forward(self, x):
    # types: (torch.Tensor) -> torch.Tensor
    h = x[..., 0:1, :self.ini_length]
    o = self.res(h)
    return o.permute(0, 3, 1, 2).view(h.shape[0], self.ini_length, -1)