import torch
import torch.nn as nn
import torch.functional as F
from models.networks.feature_transformer_layer import ARTransformerLayer

class TemporalEncoder(nn.Module):

    def __init__(self, k, traj_length, output_size, n_subnets=4, skip_connections = False):
        super(TemporalEncoder, self).__init__()

        self.skip_connections = skip_connections
        self.n_subnets = n_subnets

        self.K = k # Number of candidates
        self.T = traj_length # Length of the trajectories

        # Network that transforms and expands the input tensor to the desired shape
        self.transformer = ARTransformerLayer(self.K, self.T)
        # First hidden dimension
        hidden_size = 8

        # Check that channels keep increasing
        assert output_size >= hidden_size*(4**(self.n_subnets-2))

        '''Convolutional submodules construction'''
        self.subnets = [0]*n_subnets
        self.subnets[0] = nn.Sequential(*self.build_subnet(
                                                input_size=1,
                                                output_size=hidden_size
                                        )).cuda()


        for sn in range(1,self.n_subnets-1):
            self.subnets[sn] = nn.Sequential(*self.build_subnet(
                                                    input_size=hidden_size,
                                                    output_size=hidden_size * 4,
                                            )).cuda()
            hidden_size *= 4

        self.subnets[-1] = nn.Sequential(*self.build_subnet(
                                                input_size=hidden_size,
                                                output_size=output_size,
                                                last = True
                                        )).cuda()


    def forward(self, x):

        # Input: [2N,1,K,T]
        batch_size, ini_chans, K, T = x.size()
        x = x.view(batch_size*ini_chans, 1, K, T)

        for sn in range(self.n_subnets):
            h = self.subnets[sn](x)
            if self.skip_connections and sn < self.n_subnets-1:
                #Note: Do we want a skip connection after the last layers?
                #Note: Skip connection every 2+ subnets?
                x = x+h
            else:
                x = h

        return x

    def build_subnet(self, input_size, output_size, last=False):

        layers = [self.transformer,
                  nn.Conv2d(in_channels=input_size,
                            out_channels=int(output_size//2),
                            kernel_size=[1, 3],
                            stride = [1, 3]),
                  nn.BatchNorm2d(int(output_size//2)),
                  nn.LeakyReLU(0.2, inplace=True)]

        layers += [nn.Conv2d(in_channels=int(output_size//2),
                            out_channels=output_size,
                            kernel_size=[self.K**2, 1],
                            stride = [self.K**2, 1])]

        if not last:
            layers += [nn.BatchNorm2d(output_size),
                       nn.LeakyReLU()]
        # else:
        #     layers += [nn.BatchNorm2d(output_size),
        #                nn.LeakyReLU()]
        #  We keep the same output until we find an alternative

        return layers

if __name__ == "__main__":

    B, chan, K, T = 2, 1, 3, 5
    x = torch.arange(0, K*chan*T*B).view(B, chan, K, T).float()
    tenc = TemporalEncoder(K, T, 256)
    o = tenc(x)
    print(o[1,0,...])