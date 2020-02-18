import torch
import torch.nn as nn

x = torch.rand((1,1,5,5)) * 10.0
x = torch.clamp(x - 5.5, 0.001, 10.0)

x[0,0,3,3] = 9.98
print(x)
sm = nn.Softmax(2)
y = x.view(1,1,-1)
print(y.shape)
z = sm(y)
z = z.view(1,1,5,5)
print(torch.sum(z))
print(z)