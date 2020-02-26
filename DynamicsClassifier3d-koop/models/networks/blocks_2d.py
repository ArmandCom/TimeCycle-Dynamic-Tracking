from typing import List
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module

# from models.base import BaseModule
def residual_op_upsample(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.

    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x

    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)
        # Residual connection
        out = ha + hb
    else:
        out = ha

    return activation_fn(out)


def residual_op_downsample_ar(x, functions, parallel_functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List, List[Module, Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.

    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    pf = parallel_functions
    f1, f2, f3 = functions
    bnt, bn1, bn2, bn3 = bns

    assert len(functions) + 1 == len(bns) == 4
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha_list = []
    for step, f in enumerate(pf):
        # test = f(ha) #NOTE: check output sizes
        ha_list.append(f(ha)[..., :1])
    ha = torch.cat(ha_list, dim=-1)
    if bnt is not None:
        ha = bnt(ha)
    ha = activation_fn(ha)
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)

def residual_op_downsample(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.

    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)

class BaseBlock(nn.Module):
    """ Base class for all blocks. """
    def __init__(self, channel_in, channel_out, traj_length, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'
        self._T = traj_length
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        # type: () -> Optional[Module]
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm2d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class UpsampleBlock(BaseBlock):
    """ Implements a Upsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, traj_length, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, traj_length, activation_fn, use_bn, use_bias)

        self._T = traj_length
        # Convolutions
        self.conv1a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=[5,1],
                                         padding=[2,0], stride=[2,1], output_padding=[1,0], bias=use_bias).cuda()
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=[3,1],
                                padding=[1,0], stride=1, bias=use_bias).cuda()
        self.conv2a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=1,
                                         padding=0, stride=[2,1], output_padding=[1,0], bias=use_bias).cuda()

        # Batch Normalization layers
        self.bn1a = self.get_bn().cuda()
        self.bn1b = self.get_bn().cuda()
        self.bn2a = self.get_bn().cuda()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op_upsample(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class DownsampleBlock_AR(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, traj_length, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock_AR, self).__init__(channel_in, channel_out, traj_length, activation_fn, use_bn, use_bias)
        self._T = traj_length

        # Convolutions
        self.conv1a = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=[3,1],
                                padding=[1,0], stride=[2,1], bias=use_bias).cuda()
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=[3,1],
                                padding=[1,0], stride=1, bias=use_bias).cuda()
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=[2,1], bias=use_bias).cuda()

        # Temporal Convolutions
        self.temp_conv = []
        for t in range(1, self._T+1):
            self.temp_conv.append(nn.Conv2d(in_channels=channel_in,
                                            out_channels=channel_out,
                                            kernel_size=[3, t],
                                            stride = 1,
                                            padding = [1,0]).cuda())

        # Batch Normalization layers
        self.bnt = self.get_bn().cuda()
        self.bn1a = self.get_bn().cuda()
        self.bn1b = self.get_bn().cuda()
        self.bn2a = self.get_bn().cuda()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op_downsample_ar(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            parallel_functions=self.temp_conv,
            bns=[self.bnt, self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )

class DownsampleBlock(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, traj_length, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, traj_length, activation_fn, use_bn, use_bias)
        self._T = traj_length

        # Convolutions
        self.conv1a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=[2,1], bias=use_bias).cuda()
            # Second convolution with dilation to compare 1-neighbors and 2-neighbors
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=[1,2], stride=1, dilation=[1,2], bias=use_bias).cuda()
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=[2,1], bias=use_bias).cuda()


        # Batch Normalization layers
        self.bn1a = self.get_bn().cuda()
        self.bn1b = self.get_bn().cuda()
        self.bn2a = self.get_bn().cuda()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op_downsample(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )

class ResidualBlock(BaseBlock):
    """ Implements a Residual block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, ini_traj_length, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        self._T_ini = ini_traj_length
        super(ResidualBlock, self).__init__(channel_in, channel_out, ini_traj_length, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                               padding=0, stride=1, bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=1,
                               padding=0, stride=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op_upsample(
            x,
            functions=[self.conv1, self.conv2, None],
            bns=[self.bn1, self.bn2, None],
            activation_fn=self._activation_fn
        )