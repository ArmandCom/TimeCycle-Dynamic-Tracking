import torch
import torch.nn as nn
import torch.functional as F
from models.networks.feature_transformer_layer import ARMask
from models.networks.feature_transformer_layer import PermuteLastDims
from models.networks.feature_transformer_layer import PrintShape
import math

class TemporalEncoder(nn.Module):

    def __init__(self, shape, traj_length, output_size, skip_connections = False):
        super(TemporalEncoder, self).__init__()

        self.skip_connections = skip_connections
        self.n_subnets = math.log2(shape)

        self.shape = shape # Shape of the layout
        self.T = traj_length # Length of the trajectories

        # Network that transforms and expands the input tensor to the desired shape
        # self.transformer = ARTransformerLayer(self.K, self.T)
        self.permute = PermuteLastDims()
        self.print_shape = PrintShape()

        # First hidden dimension
        hidden_size = 8

        # Check that channels keep increasing
        assert output_size >= hidden_size*(2**(self.n_subnets-1))

        # TODO: add residual blocks (as in code of anomaly detection)
        # TODO: merge channels to 1 value
        # TODO: add loss distance to 1NN
        '''Convolutional submodules construction'''
        self.ar_subnets = [0]*self.n_subnets
        self.spa_subnets = [0]*self.n_subnets

        self.ar_subnets[0] = self.build_subnet(
                                            input_size=1,
                                            output_size=hidden_size
                                            )
        self.spa_subnets[0] = nn.Sequential(*self.build_spa_subnet(
                                            input_size=hidden_size,
                                            output_size = hidden_size
                                            )).cuda()

        for sn in range(1,self.n_subnets):

            if sn < self.n_subnets - 1:
                self.ar_subnets[sn] = self.build_subnet(
                    input_size=hidden_size,
                    output_size=hidden_size * 2
                )
                self.spa_subnets[sn] = nn.Sequential(*self.build_spa_subnet(
                                                    input_size=hidden_size*2,
                                                    output_size=hidden_size*2)).cuda()

            else:
                self.ar_subnets[sn] = self.build_subnet(
                    input_size=hidden_size,
                    output_size=output_size
                )
                self.spa_subnets[sn] = nn.Sequential(*self.build_spa_subnet(
                                                    input_size=output_size,
                                                    output_size=output_size),
                                                    last=True).cuda()

            hidden_size *= 2


    def forward(self, x):

        # Input: [2N,1,K,T]
        batch_size, ini_chans, K, T = x.size()
        x = x.view(batch_size*ini_chans, 1, K, T)

        for sn in range(self.n_subnets):
            h = self.subnets[sn](x)
            if self.skip_connections and sn < self.n_subnets-2: # and sn%2==1
                #TODO: keep only first of every output
                x = x.repeat(1, int(h.shape[1]/x.shape[1]), 1, 1) + h
            else:
                x = h

        return x.squeeze()

    def build_subnet(self, input_size, output_size):

        parallel_modules = []
        for t in range(1, self.T):
            parallel_modules = nn.Sequential(*[
                      nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=[3, t],
                                stride = 1,
                                padding = [1, t]),
                      nn.BatchNorm2d(output_size),
                      nn.LeakyReLU(0.2, inplace=True)]).cuda()
            # ARMask(current_shape, self.T, n_act_t=t),

        return parallel_modules


    def build_spa_subnet(self, input_size, output_size, last=False):

        div = 4
        hidden_size = int(input_size//div)
        layers = [nn.Conv2d(in_channels=input_size,
                            out_channels=output_size,
                            kernel_size=[3,1],
                            stride = [2,1],
                            padding = [1,0])]
        if not last:
            layers += [nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.2, inplace=True)]

        return layers

    def build_selector_subnet(self, input_size, output_size, last=False):

        return None

if __name__ == "__main__":

    B, chan, K, T = 2, 1, 3, 5
    x = torch.arange(0, K*chan*T*B).view(B, chan, K, T).float()
    tenc = TemporalEncoder(K, T, 256)
    o = tenc(x)
    print(o[1,0,...])