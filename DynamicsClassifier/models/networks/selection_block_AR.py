import torch
import torch.nn as nn
import torch.functional as F
from models.networks.feature_transformer_layer import ARTransformerLayer
from models.networks.feature_transformer_layer import PermuteLastLayers
from models.networks.feature_transformer_layer import PrintShape

class TemporalEncoder(nn.Module):

    def __init__(self, k, traj_length, output_size, n_subnets=5, skip_connections = False):
        super(TemporalEncoder, self).__init__()

        self.skip_connections = skip_connections
        self.n_subnets = n_subnets

        self.K = k # Number of candidates
        self.T = traj_length # Length of the trajectories

        # Network that transforms and expands the input tensor to the desired shape
        self.transformer = ARTransformerLayer(self.K, self.T)
        self.permute = PermuteLastLayers()
        self.print_shape = PrintShape()
        # First hidden dimension
        hidden_size = 8

        # Check that channels keep increasing
        assert output_size >= hidden_size*(4**(self.n_subnets-3))

        '''Convolutional submodules construction'''
        self.subnets = [0]*n_subnets
        self.subnets[0] = nn.Sequential(*self.build_subnet(
                                                input_size=1,
                                                output_size=hidden_size
                                        )).cuda()


        for sn in range(1,self.n_subnets-2):
            self.subnets[sn] = nn.Sequential(*self.build_subnet(
                                                    input_size=hidden_size,
                                                    output_size=hidden_size * 4,
                                            )).cuda()
            hidden_size *= 4

        self.subnets[-2] = nn.Sequential(*self.build_subnet(
                                                input_size=hidden_size,
                                                output_size=output_size,
                                                last = True
                                        )).cuda()

        self.subnets[-1] = nn.Sequential(*self.build_selector_subnet(
                                            input_size=output_size,
                                            output_size = 1
                                        )).cuda()


    def forward(self, x):

        # Input: [2N,1,K,T]
        batch_size, ini_chans, K, T = x.size()
        x = x.view(batch_size*ini_chans, 1, K, T)

        for sn in range(self.n_subnets):
            h = self.subnets[sn](x)
            if self.skip_connections and sn < self.n_subnets-2: # and sn%2==1
                #Note: Do we want a skip connection after the last layers?
                #Note: Skip connection every 2+ subnets?
                x = x.repeat(1, int(h.shape[1]/x.shape[1]), 1, 1) + h
            else:
                x = h

        return x.squeeze()

    def build_subnet(self, input_size, output_size, last=False):

        layers = [self.transformer,
                  nn.Conv2d(in_channels=input_size,
                            out_channels=output_size,
                            kernel_size=[1, 3],
                            stride = [1, 3]),
                  nn.BatchNorm2d(output_size),
                  nn.LeakyReLU(0.2, inplace=True)]

        layers += [self.permute, nn.Linear(self.K**3, self.K), self.permute]
        # layers += [nn.Conv2d(in_channels=int(output_size//2),
        #                     out_channels=output_size,
        #                     kernel_size=[self.K**2, 1],
        #                     stride = [1, 1])]

        if not last:
            layers += [nn.BatchNorm2d(output_size),
                       nn.LeakyReLU(0.2, inplace=True)]
        else:
            layers += [nn.BatchNorm2d(output_size),
                       nn.LeakyReLU(0.2, inplace=True)]
        #  We keep the same output until we find an alternative

        return layers


    def build_selector_subnet(self, input_size, output_size):

        div = 4
        hidden_size = int(input_size//div)
        layers = [nn.Conv2d(in_channels=input_size,
                            out_channels=hidden_size,
                            kernel_size=1),
                  nn.BatchNorm2d(hidden_size),
                  nn.LeakyReLU(0.2, inplace=True)]

        # # Print shape for debug
        # layers += [self.print_shape]

        while hidden_size > 4*div:
            layers += [nn.Conv2d(in_channels=hidden_size,
                                out_channels=int(hidden_size//div),
                                kernel_size=1),
                      nn.BatchNorm2d(int(hidden_size//div)),
                      nn.LeakyReLU(0.2, inplace=True)]
            hidden_size = int(hidden_size//div)

        layers += [nn.Conv2d(in_channels=hidden_size,
                             out_channels=output_size,
                             kernel_size=1)]

        return layers

if __name__ == "__main__":

    B, chan, K, T = 2, 1, 3, 5
    x = torch.arange(0, K*chan*T*B).view(B, chan, K, T).float()
    tenc = TemporalEncoder(K, T, 256)
    o = tenc(x)
    print(o[1,0,...])