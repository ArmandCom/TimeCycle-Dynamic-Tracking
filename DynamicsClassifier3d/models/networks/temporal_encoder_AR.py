import torch
import torch.nn as nn
import torch.functional as F
#TODO: https://github.com/aimagelab/novelty-detection/tree/master/models

# class TemporalEncoder(nn.Module):
#     '''
#     Encodes images. Similar structure as DCGAN.
#     '''
#     def __init__(self, is_train, input_size, output_size, kernel_width=3, n_layers=3, use_groupnorm=False):
#         super(TemporalEncoder, self).__init__()
#
#
#
#     def forward(self, x):
#         x = self.main(x)
#         return x


from typing import List
# from models.base import BaseModule

'''-------------------------'''

class MaskedFullyConnection(nn.Linear):
    """
    Implements a Masked Fully Connection layer (MFC, Eq. 6).
    This is the autoregressive layer employed for the estimation of
    densities of image feature vectors.
    """
    def __init__(self, mask_type, in_channels, out_channels, *args, **kwargs):
        """
        Class constructor.
        :param mask_type: type of autoregressive layer, either `A` or `B`.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(MaskedFullyConnection, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B']
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.
        self.register_buffer('mask', self.weight.data.clone()) # Saves weights in buffer

        # Build mask
        self.mask.fill_(0)
        for f in range(0 if mask_type == 'B' else 1, self.out_features // self.out_channels):
            start_row = f*self.out_channels
            end_row = (f+1)*self.out_channels
            start_col = 0
            end_col = f*self.in_channels if mask_type == 'A' else (f+1)*self.in_channels
            if start_col != end_col:
                self.mask[start_row:end_row, start_col:end_col] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor.
        :return: the output of a MFC manipulation.
        """

        # Reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(len(x), -1)

        # Mask weights and call fully connection
        self.weight.data *= self.mask
        o = super(MaskedFullyConnection, self).forward(x)

        # Reshape again
        o = o.view(len(o), -1, self.out_channels)
        o = torch.transpose(o, 1, 2).contiguous()

        return o

class Estimator1D(nn.Module):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.
    Takes as input a latent vector and outputs cpds for each variable.
    """
    def __init__(self, code_length, fm_list, cpd_channels):
        # type: (int, List[int], int) -> None
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(Estimator1D, self).__init__()

        self.code_length = code_length
        self.fm_list = fm_list
        self.cpd_channels = cpd_channels

        activation_fn = nn.LeakyReLU()

        # Add autoregressive layers
        layers_list = []
        mask_type = 'A'
        fm_in = 1
        for l in range(0, len(fm_list)):

            fm_out = fm_list[l]
            layers_list.append(
                MaskedFullyConnection(mask_type=mask_type,
                                      in_features=fm_in * code_length,
                                      out_features=fm_out * code_length,
                                      in_channels=fm_in, out_channels=fm_out)
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedFullyConnection(mask_type=mask_type,
                                  in_features=fm_in * code_length,
                                  out_features=cpd_channels * code_length,
                                  in_channels=fm_in,
                                  out_channels=cpd_channels))

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of latent vectors.
        :return: the batch of CPD estimates.
        """
        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h

        return o


'''-------------------------'''

# from functools import reduce
# from operator import mul
#
# class BaseModule(nn.Module):
#     """
#     Implements the basic module.
#     All other modules inherit from this one
#     """
#     def load_w(self, checkpoint_path):
#         # type: (str) -> None
#         """
#         Loads a checkpoint into the state_dict.
#         :param checkpoint_path: the checkpoint file to be loaded.
#         """
#         self.load_state_dict(torch.load(checkpoint_path))
#
#     def __repr__(self):
#         # type: () -> str
#         """
#         String representation
#         """
#         good_old = super(BaseModule, self).__repr__()
#         addition = 'Total number of parameters: {:,}'.format(self.n_parameters)
#
#         return good_old + '\n' + addition
#
#     def __call__(self, *args, **kwargs):
#         return super(BaseModule, self).__call__(*args, **kwargs)
#
#     @property
#     def n_parameters(self):
#         # type: () -> int
#         """
#         Number of parameters of the model.
#         """
#         n_parameters = 0
#         for p in self.parameters():
#             if hasattr(p, 'mask'):
#                 n_parameters += torch.sum(p.mask).item()
#             else:
#                 n_parameters += reduce(mul, p.shape)
#         return int(n_parameters)