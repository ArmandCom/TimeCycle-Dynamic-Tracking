import torch
import torch.nn as nn
import torch.functional as F
from models.networks.feature_transformer_layer import ARTransformerLayer

class TemporalEncoder(nn.Module):

    def __init__(self, is_train, input_size, output_size=2048, kernel_width=3, n_layers=3):
        super(TemporalEncoder, self).__init__()

        """
        Each block has 2 convs.
        """
        self.n_layers = n_layers

        layers = []

        self.subnets = [] * n_layers
        for i in range(n_layers):

            if i == n_layers-1:
                hidden_size = output_size
            elif i == 0:
                hidden_size = input_size*2
            else:
                hidden_size *=2

            self.subnets.append(nn.Sequential(
                *self.build_subnet(
                    input_size,
                    hidden_size,
                    kernel_width
                )))

    def forward(self, x):

        # NCT -> NCT1 --> C=1?
        x = x.unsqueeze(-1)

        for i in range(self.n_layers):
            h = self.subnets[i](x)
            # NT1C -> NTC
            h = h.squeeze(-1)
            # skip connection
            x = x+h

        return x

    def build_subnet(self, input_size, output_size, kernel_width):

        if self.use_groupnorm:
            layers =  [nn.GroupNorm(num_groups=32, num_channels=1)] # num groups same as in TF default function
        else:
            layers =  [nn.BatchNorm2d(input_size)]

        layers += [nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=[kernel_width,1],
                             padding=int(kernel_width//2))]

        if self.use_groupnorm:
            layers += [nn.GroupNorm(num_groups=32, num_channels=1)]
        else:
            layers += [nn.BatchNorm2d(output_size)]

        layers += [nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=[kernel_width,1],
                             padding=int(kernel_width//2))] # initialize with Xavier

        return layers

if __name__ == "__main__":

    B, K, T = 2, 3, 5
    x = torch.arange(0, K*T*B).view(B, K, T).float()
    # transf = ARTransformerLayer(K, T)
    # o = transf(x)
    # print(o[1,...,1])