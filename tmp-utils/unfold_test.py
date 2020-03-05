import torch
import torch.nn as nn
import torch.nn.functional as F

unfold = nn.Unfold(kernel_size=(3,3), padding=(1,1))
ids = torch.arange(0, 16).view(1,1, 4,4)/1.
im = torch.arange(0, 256).view(16,16)
# np.add.at(predlbls, (h, w), lbl * corrfeat2[t, ww, hh, h, w][:, None])
# Note: arange(1,1601). All paddings assigned to minimum valued attention pixel.
patches = unfold(ids)[0].type(torch.LongTensor).permute(1,0).contiguous()

#Correcte!
patches_cyc = patches
for i in range(im.shape[0]):
    values = im[patches.view(-1), patches_cyc.view(-1)]

    print('ids_nor: ',patches[0,0:9])
    print('ids_cyc: ',patches_cyc[0,0:9])
    print('values: ',values[0:9])
    print('im: ',im[0:6,0:6])
    resh_values = values.view(16, -1)
    print('resh_values: ', resh_values[0,:])
    values = torch.sum(resh_values, dim=1)
    patches_cyc = torch.cat([patches_cyc[1:,:],patches_cyc[0:1, :]], dim=0)



# for i in range(im.shape[0]):
#     corrfeat2(hi, wi, hj, wj) = corrfeat2