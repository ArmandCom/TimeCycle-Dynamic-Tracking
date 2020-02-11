import torch
import torch.nn as nn

class SelectionByAttention(nn.Module):
    """
    Implements several FC for selection of points.
    """
    def __init__(self, in_channels, indep_channels):
        super(SelectionByAttention, self).__init__()
        self.in_channels = in_channels
        self.num_layers = 2

        assert (self.in_channels / (2**self.num_layers)) > 4
        hidden_channels = int(self.in_channels//2)
        layers = [  nn.Linear(in_features=self.in_channels, out_features=hidden_channels),
                    nn.BatchNorm1d(num_features=indep_channels),
                    nn.ReLU()]

        for i in range(self.num_layers):

            layers += [ nn.Linear(in_features=hidden_channels, out_features=int(hidden_channels//2)),
                        nn.BatchNorm1d(num_features=indep_channels),
                        nn.ReLU()]
            hidden_channels = int(hidden_channels//2)

        layers += [nn.Linear(in_features=hidden_channels, out_features=1)] # nn.Sigmoid()

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        h = x.permute(0,2,3,1).view(x.shape[0], -1, self.in_channels)
        o = self.main(h).squeeze(-1).view(x.shape[0], x.shape[2], -1)

        return o