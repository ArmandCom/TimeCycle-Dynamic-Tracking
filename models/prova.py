import torch
import torch.nn as nn
import os
import numpy as np
from model1_3D import Dyn3D

# Import data in a cutre way
datadir = '/data/Ponc/tracking/torch_data/resized'
llista = sorted(os.listdir(datadir))
inici = 12 - 2

sublist = llista[inici:inici+7] # 4 confident 3 not confident
x = torch.empty(1,3,7,55,98)
for i,ele in enumerate(sublist):
    # arr = np.load(os.path.join(datadir, sublist[i]))
    # print(arr.shape)
    # x[0,2,i,:,:] = torch.from_numpy(arr)
    

model = Dyn3D()