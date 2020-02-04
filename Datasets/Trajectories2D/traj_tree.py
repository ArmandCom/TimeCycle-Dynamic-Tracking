import gzip
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(root):
  # Load MNIST dataset for generating training data.
  path = os.path.join(root, 'train-images-idx3-ubyte.gz')
  with gzip.open(path, 'rb') as f:
    mnist = np.frombuffer(f.read(), np.uint8, offset=16)
    mnist = mnist.reshape(-1, 28, 28)
  return mnist

def load_fixed_set(root, is_train):
  # Load the fixed dataset
  filename = 'mnist_test_seq.npy'
  path = os.path.join(root, filename)
  dataset = np.load(path)
  dataset = dataset[..., np.newaxis]
  return dataset

class TrajectoryTree(data.Dataset):
  def __init__(self, is_train, traj_length, dset_path = '', generate=False):
    '''
    param num_objects: a list of number of possible objects.
    '''
    super(TrajectoryTree, self).__init__()
    self.save_test = False
    self.save = True

    self.dataset = None
    self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

    self.traj_length = traj_length
    self.dset_path = dset_path

    # For generating data
    self.n_forks = 2
    self.n_traj = 2             # Number of coexisting trajectories
    # self.step_length = 0.5      # Sampling rate
    self.max_order = 2        # Maximum order of the LTI systems generated IRs
    self.noise_std = 0.2        # Variance of the 0-mean gaussian noise in the measurements
    self.missing_data_perc = 0  # How many missing points in the trajectory
    self.pole_low, self.pole_high = 0.5, 1.5
    self.clean = True

    if generate:
      self.generate_dataset(root=self.dset_path, training_samples=1e4, testing_samples=5e3)

  def get_random_trajectory(self, traj_length=None, pole_limits = None, n_realizations = 3):
    ''' Generate a random sequence of an IR '''

    if traj_length is None:
      traj_length = self.traj_length
    if pole_limits is None:
      pole_low, pole_high = self.pole_low, self.pole_high
    else: pole_low, pole_high = pole_limits[0], pole_limits[1]

    assert traj_length > self.max_order
    assert self.n_traj > 1

    rhos = np.random.uniform(pole_low, pole_high, self.max_order * self.n_traj) \
      .reshape(self.max_order, self.n_traj)

    traj_clean = []
    traj_noisy = []
    for tr in range(self.n_traj):

      ydata_clean = np.zeros((self.n_traj, traj_length))
      y = np.zeros((n_realizations, traj_length, self.n_traj))

      y[:, 0:self.max_order, :] = np.random.uniform(0, 1, n_realizations * self.max_order * self.n_traj) \
        .reshape(n_realizations, self.max_order, self.n_traj)

      # polys = []
      # for rr in range(self.n_traj):
      #   polys.append(np.poly(rhos[:,tr]))
      # reg = -np.flip(np.stack(polys), axis=1)

      poly = np.poly(rhos[:, tr])
      reg = -np.flip(poly)

      for i in range(self.max_order, traj_length):
        y[:, i, tr] = np.sum((y[:, i-self.max_order:i, tr]*reg[:-1]),axis=1)

      ydata_clean = y[...,tr]
      traj_clean.append(ydata_clean)

      # Adding noise
      if not self.clean:
        noise = np.random.normal(0, self.noise_std, traj_length)
        ydata = ydata_clean + noise
        traj_noisy.append(ydata)

      if self.save_test:
        fig = plt.plot(ydata_clean[0])
        plt.savefig('example_traj')

    fork_info = [-1]*self.n_traj
    if self.clean:
      trajs = np.stack(traj_clean, axis=0)
    else:
      trajs = np.stack(traj_noisy, axis=0)

    if self.save_test:
      plt.close()


    # TODO: generate 2D sequences
    '''
    Mix at different points of the trajectories - High order trajectories
    '''
    num_forks = 2
    data_ids = range(1,traj_length-1)
    forks = random.sample(data_ids, num_forks)

    final_trajs = []

    # TODO: also with 2 points forks per trajectory
    for pt in forks:
      trajs_cyc = np.concatenate([trajs[1:], trajs[0:1]], axis=0)
      for i in range(self.n_traj-1):

        final_trajs.append(np.concatenate([trajs[..., 0:pt],trajs_cyc[...,pt:]], axis=2))
        fork_info.extend([pt]*self.n_traj)
        trajs_cyc = np.concatenate([trajs_cyc[1:], trajs_cyc[0:1]], axis=0)


    trajs_mix = np.concatenate(final_trajs, axis=0)

    if self.save_test:
      for i in range(trajs_mix.shape[0]):
        fig = plt.plot(trajs_mix[i,0])
        plt.savefig('example_traj_mixed')

      plt.close()
    return  np.concatenate([trajs,trajs_mix], axis=0), np.stack(fork_info)

  def generate_dataset(self, root='', training_samples = 1e3, testing_samples = 5e2):

    N = (self.n_forks * self.n_traj * (self.n_traj - 1) + self.n_traj) # Number of trajectories per each realization
    n_train_iters = int(training_samples // N)
    n_test_iters = int(testing_samples // N)


    train_data_list, train_info_list, test_data_list, test_info_list = [],[],[],[]
    for i in range(n_train_iters):
      train_data, train_info = self.get_random_trajectory(traj_length = 9, n_realizations=1)
      train_data_list.append(train_data)
      train_info_list.append(train_info)
    for i in range(n_test_iters):
      test_data, test_info = self.get_random_trajectory(traj_length = 9, n_realizations=1)
      test_data_list.append(test_data)
      test_info_list.append(test_info)

    if self.save:
      train_data = np.stack(train_data_list, axis=0)
      test_data = np.stack(test_data_list, axis=0)
      train_info = np.stack(train_info_list, axis=0)
      test_info = np.stack(test_info_list, axis=0)

      train_dat_name = os.path.join(root, 'train_dat.npy')
      train_info_name = os.path.join(root, 'train_info.npy')
      test_dat_name = os.path.join(root, 'test_dat.npy')
      test_info_name = os.path.join(root, 'test_info.npy')

      np.save(train_dat_name, train_data)
      np.save(train_info_name, train_info)

      np.save(test_dat_name, test_data)
      np.save(test_info_name, test_info)

  def load_data(self, is_train):

    if is_train:
      data_name = 'train_dat.npy'
      info_name = 'train_info.npy'

    else:
      data_name = 'test_dat.npy'
      info_name = 'test_info.npy'

    data_path = os.path.join(self.dset_path, data_name)
    info_path = os.path.join(self.dset_path, info_name)
    data = np.load(data_path)
    info = np.load(info_path)

    inp = torch.FloatTensor(data)
    info = torch.LongTensor(info)

    return inp, info


def main():

  traj_length = 9
  traj2D = TrajectoryTree('', True, traj_length=traj_length)
  # traj = traj2D.get_random_trajectory()
  traj2D.generate_dataset(root='/data/Armand/TimeCycle/traj', n_traj=2)

if __name__=='__main__':
  main()