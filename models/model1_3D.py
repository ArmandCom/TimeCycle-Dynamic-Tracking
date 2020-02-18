import torch
import torch.nn as nn
import numpy as np

class Dyn3D(nn.Module):
    def __init__(self):
        super(Dyn3D, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(3,3,3), stride = 1),
            nn.LeakyReLU(),
            nn.MaxPool3d((1,2,2))
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(3, 6, kernel_size = (3,3,3)),
            nn.LeakyReLU(),
            nn.MaxPool3d((1,2,2))
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(6, 9, kernel_size = (3,3,3)),
            nn.LeakyReLU()
        )
        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose3d(9, 6, (3,6,6)),
            nn.LeakyReLU()
        )
        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose3d(6,3, (3,9,9)),
            nn.LeakyReLU()
        )
        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose3d(3,1, (3,11,11)),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        x = self.conv_layer1(x)
        print(x.size())
        x = self.conv_layer2(x)
        print(x.size())
        x = self.conv_layer3(x)
        print(x.size())
        x = self.trans_conv1(x)
        print(x.size())
        x = self.trans_conv2(x)
        print(x.size())
        x = self.trans_conv3(x)
        print(x.size())
        return x

if __name__ == '__main__':
    # (B, C, D, H, W)
    scores = torch.randn((2,1,9,25,25))
    xys = torch.empty((2,2,9,25,25))
    
    for i in range(25):
        xys[:, 0, :, i, :] = float(i)
        xys[:, 1, :, :, i] = float(i)
    print(xys[0,1,0,:,:])
    x = torch.cat([xys, scores], dim=1)
    model = Dyn3D()
    sm = nn.Softmax(dim = 3)
    y = model(x)
    y = y.view(2,1,9,-1)
    y = torch.exp(y)
    print(y[0,0,0,:])
    y = torch.exp(y)
    smy = sm(y).view(2,1,9,25,25)
    candidates_x = smy[0,0,0,:,:] * x[0,1,0,:,:]
    print(candidates_x)