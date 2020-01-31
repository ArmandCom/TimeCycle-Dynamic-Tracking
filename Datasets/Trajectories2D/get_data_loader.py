import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import scipy.misc

import data.video_transforms as vtransforms
from .moving_mnist import MovingMNIST
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


#TODO: change arguments for validation
#TODO: add args for frame_size, hidden_frame_size, velocity, object_size.

parser.add_argument('--is_train', type=int, default=True, help='training?')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
parser.add_argument('--gpus', type=str, default='0', help='visible GPU ids, separated by comma')

# data
parser.add_argument('--dset_dir', type=str, default=os.path.join('/data/Armand/', 'datasets/'))
parser.add_argument('--dset_name', type=str, default='moving-mnist')

# Moving MNIST
parser.add_argument('--num_objects', type=int, nargs='+', default=[1],
                         help='Max number of digits in Moving MNIST videos.') # default 2
parser.add_argument('--n_frames_input', type=int, default=9)
parser.add_argument('--n_frames_output', type=int, default=0)

opt = parser.parse_args()

opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)

def get_data_loader(opt):
  transform = transforms.Compose([vtransforms.ToTensor()])
  dset = MovingMNIST(opt.dset_path, opt.is_train, opt.n_frames_input,
                     opt.n_frames_output, opt.num_objects, transform)

  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader

if __name__ == '__main__':
    dloader = get_data_loader(opt)
    for step, data in enumerate(dloader):
        # data = data.reshape()
        scipy.misc.imsave('moving-mnist-example.jpg', data)
