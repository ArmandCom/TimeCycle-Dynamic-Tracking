import gzip
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryMultiple(data.Dataset):
  def __init__(self, traj_length, dset_path = '', generate=False):
    '''
    param num_objects: a list of number of possible objects.
    '''
    super(TrajectoryMultiple, self).__init__()
    self.save_test = True
    self.save = False

    self.dataset = None
    self.branching = True # Whether we create branches or parallel trajectories

    self.traj_length = traj_length
    self.dset_path = dset_path

    # For generating data
    self.n_traj = 2 # Number of coexisting trajectories
    self.max_order = 2        # Maximum order of the LTI systems generated IRs
    self.missing_data_perc = 0  # How many missing points in the trajectory
    self.pole_low, self.pole_high = 0.5, 1.5
    self.clean = False

    # Specific parameters for generating multiple alternatives:
    # Gaussian distributions over different centroids
    self.n_alt_centroids = 1
    self.n_points_centroid = 3
    self.k = (self.n_alt_centroids + 1)*self.n_points_centroid
    self.centroid_min_randint, self.centroid_max_randint = 3, 10
    self.std_max_rand = 2 # std of the gaussian by which we generate the std of the gaussian around a centroid
    self.added_noise_std = 0.5   # Variance of the 0-mean gaussian noise in the measurements

    # For equal number of points around each centroid
    assert(self.k%(self.n_alt_centroids+1) == 0)
    # For this specific case [self.n_traj] must be 1 and n_realizations 2.
    # Note: We can also reverse it
    # assert self.n_traj == 1

    if generate:
      self.generate_dataset(root=self.dset_path, training_samples=1e4, testing_samples=5e3)



  def get_random_trajectory(self, traj_length=None, pole_limits = None, n_realizations = 2):
    ''' Generate a random sequence of an IR '''

    if traj_length is None:
      traj_length = self.traj_length
    if pole_limits is None:
      pole_low, pole_high = self.pole_low, self.pole_high
    else: pole_low, pole_high = pole_limits[0], pole_limits[1]

    assert traj_length > self.max_order

    rhos = np.random.uniform(pole_low, pole_high, self.max_order * self.n_traj) \
      .reshape(self.max_order, self.n_traj)

    traj_clean = []
    for tr in range(self.n_traj):

      ydata_clean = np.zeros((self.n_traj, traj_length))
      y = np.zeros((n_realizations, traj_length, self.n_traj))

      #TODO: check coef amplitude to correspond it to the image size
      y[:, 0:self.max_order, :] = np.random.uniform(0, 3, n_realizations * self.max_order * self.n_traj) \
        .reshape(n_realizations, self.max_order, self.n_traj)

      poly = np.poly(rhos[:, tr])
      reg = -np.flip(poly)

      for i in range(self.max_order, traj_length):
        y[:, i, tr] = np.sum((y[:, i-self.max_order:i, tr]*reg[:-1]),axis=1)

      ydata_clean = y[...,tr]
      traj_clean.append(ydata_clean)

      if self.save_test:
        fig = plt.plot(ydata_clean[0])
        plt.savefig('example_traj')

    trajs = np.stack(traj_clean, axis=0)

    if self.save_test:
      plt.close()

    branching = True
    if branching:

      '''
      --------------------------------------------------------------------------------------------
      Generate branches - keep only the once origined on the first one
      '''
      num_forks = 1
      data_ids = range(1, traj_length - 3)
      forks = random.sample(data_ids, num_forks)

      forked_trajs = []

      for pt in forks:
        trajs_cyc = np.concatenate([trajs[1:], trajs[0:1]], axis=0)
        for i in range(self.n_traj - 1):
          forked_trajs.append(np.concatenate([trajs[..., 0:pt], trajs_cyc[..., pt:]], axis=2)[0:1])
          # fork_info.extend([pt] * self.n_traj) # Note: no need to save where the forks are
          trajs_cyc = np.concatenate([trajs_cyc[1:], trajs_cyc[0:1]], axis=0)

      trajs_mix = np.concatenate(forked_trajs, axis=0)
      ori_trajectories = np.concatenate([trajs[0:1], trajs_mix], axis=0)[:,:,np.newaxis,:]

      if self.save_test:
        for i in range(ori_trajectories.shape[0]):
          fig = plt.plot(ori_trajectories[i, 0, 0])
          plt.savefig('example_traj_mixed')

        plt.close()

      '''
      --------------------------------------------------------------------------------------------
      Generate alternative trajectories with branching
      '''
      # Note: we are taking 2 realizations for x and y
      # TODO: remember suffling position of the trajectories (good one cant always be in the end).

      # Generate new centroids and distribute new values around centroids following gaussian
      centroids = np.zeros((2, self.n_traj, self.n_points_centroid,
                            self.traj_length))
      centroids[...] = np.transpose(ori_trajectories,(1,0,2,3))

      # Add points distributed around the centroids
      centroids_false = np.around(np.random.normal(0, self.std_max_rand,
                                                   size=(2, self.n_traj, self.n_points_centroid,
                                                         self.traj_length))).astype(int)
      # Set real traj deviation to 0
      centroids_false[:, :, -1, :] = 0

      trajectories = centroids + centroids_false

      # Add current generated variations to the original trajectory
      self.k = self.n_traj * self.n_points_centroid
      trajectories = trajectories.reshape(2, self.k, self.traj_length)

      if self.clean is False:
        # Add noise to all trajectories
        trajectories += np.around(np.random.normal(0, self.added_noise_std,
                                                   size=(2, self.k, self.traj_length))).astype(int)

      if self.save_test:
        for k in range(self.k):
          fig = plt.plot(trajectories[0, k])
          plt.savefig('example_traj_x_in_time')
        plt.close()

      # Add score to trajectories
      scores = np.ones((1, self.k, self.traj_length))
      trajectories = np.concatenate([trajectories.astype(float), scores], axis=0)

    else:
      '''
      --------------------------------------------------------------------------------------------
      Generate alternative trajectories without branching
      '''
      # Note: we are taking 2 realizations for x and y
      # TODO: remember suffling position of the trajectories (good one cant always be in the end).
      # TODO: Other centroids should have other poles? Also branches?
      # TODO: generate from traj_tree sequences --> keeping only one of the 2 original ones (lowrank)

      # Generate new centroids and distribute new values around centroids following gaussian
      centroids = np.zeros((2, self.n_alt_centroids + 1, self.n_points_centroid, self.traj_length)) # TODO: randomize the number of centroids


      # Generate centroids
      centroids[:, :-1, ...] = np.repeat(np.random.randint(self.centroid_min_randint, self.centroid_max_randint,
                                            size = (2,self.n_alt_centroids,1,self.traj_length)),
                                         self.n_points_centroid, axis=2)

      # Add points distributed around the centroids
      centroids_false = np.around(np.random.normal(0, self.std_max_rand,
                                    size = (2, self.n_alt_centroids + 1, self.n_points_centroid,
                                            self.traj_length))).astype(int)
      # Set real traj deviation to 0
      centroids_false[:, -1, -1, :] = 0

      centroids += centroids_false

      # Add current generated variations to the original trajectory
      centroids = centroids.reshape(2, self.k, self.traj_length)

      trajectories = centroids + trajs[0, :, np.newaxis, :] # Taking realizations as x,y / not n_traj

      if self.clean is False:
        # Add noise to all trajectories
        trajectories += np.around(np.random.normal(0, self.added_noise_std,
                                                size=(2, self.k, self.traj_length))).astype(int)

      if self.save_test:
        for k in range(self.k):
          fig = plt.plot(trajectories[0,k])
          plt.savefig('example_traj_x_in_time')
        plt.close()

      # Add score to trajectories
      scores = np.ones((1,self.k, self.traj_length))
      trajectories = np.concatenate([trajectories.astype(float), scores], axis=0)

    return trajectories

  def generate_dataset(self, root='', training_samples = 1e3, testing_samples = 5e2):

    # N = (self.n_forks * self.n_traj * (self.n_traj - 1) + self.n_traj)
    n_train_iters = int(training_samples)
    n_test_iters = int(testing_samples)


    train_data_list, test_data_list = [],[]
    for i in range(n_train_iters):
      train_data = self.get_random_trajectory(traj_length = 9)
      train_data_list.append(train_data)
    for i in range(n_test_iters):
      test_data = self.get_random_trajectory(traj_length = 9)
      test_data_list.append(test_data)

    if self.save:
      train_data = np.stack(train_data_list, axis=0)
      test_data = np.stack(test_data_list, axis=0)

      train_dat_name = os.path.join(root, 'train_dat.npy')
      test_dat_name = os.path.join(root, 'test_dat.npy')

      np.save(train_dat_name, train_data)
      np.save(test_dat_name, test_data)

  def load_data(self, is_train):

    if is_train:
      data_name = 'train_dat.npy'
    else:
      data_name = 'test_dat.npy'

    data_path = os.path.join(self.dset_path, data_name)
    data = np.load(data_path)

    inp = torch.FloatTensor(data) # Not sure it should be float

    return inp


def main():

  traj_length = 9
  traj2D = TrajectoryMultiple(traj_length=traj_length)
  traj = traj2D.get_random_trajectory()
  # traj2D.generate_dataset(root='/data/Armand/TimeCycle/traj', n_traj=2)

if __name__=='__main__':
  main()