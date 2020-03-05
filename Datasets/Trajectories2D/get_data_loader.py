import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import scipy.misc

from traj_tree import TrajectoryTree
from traj_multiple import TrajectoryMultiple
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


#TODO: change arguments for validation
#TODO: add args for frame_size, hidden_frame_size, velocity, object_size.

parser.add_argument('--is_train', type=int, default=True, help='training?')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
parser.add_argument('--gpus', type=str, default='0', help='visible GPU ids, separated by comma')

# data
parser.add_argument('--dset_dir', type=str, default=os.path.join('/data/Armand/', 'TimeCycle/'))
parser.add_argument('--dset_name', type=str, default='traj_multi')

# Moving MNIST
parser.add_argument('--traj_length', type=int, default=9)

opt = parser.parse_args()

opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)

def get_data_loader(opt):
    generate = True
    if opt.dset_name == 'traj':
        traj_trees = TrajectoryTree(opt.traj_length, dset_path=opt.dset_path, generate=generate)
        traj_data = traj_trees.load_data(opt.is_train)

    elif opt.dset_name == 'traj_multi':
        traj_trees = TrajectoryMultiple(opt.traj_length, dset_path=opt.dset_path, generate=generate)
        traj_data = traj_trees.load_data(opt.is_train)
    else:
        raise NotImplementedError

    dset = data.TensorDataset(*traj_data)
    dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
    return dloader

if __name__ == '__main__':
    dloader = get_data_loader(opt)
    for step, data in enumerate(dloader):

        # Traj tree
        # if step < 1:
        #     for i in range(len(data[1][0,:])):
        #         print('Fork id: ',data[1][0,i])
        #         fig = plt.plot(data[0][0,i,0].numpy())
        #         plt.savefig('example_traj_loaded')

        # Traj multiple
        if step < 1:
            for i in range(data.shape[1]):
                fig = plt.scatter(data[0,i,:].numpy())
                plt.savefig('example_traj_loaded')
            plt.close()
