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
  def __init__(self, root, is_train, traj_length):
    '''
    param num_objects: a list of number of possible objects.
    '''
    super(TrajectoryTree, self).__init__()
    self.save_test = True

    self.dataset = None
    self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

    self.is_train = is_train
    self.traj_length = traj_length

    # For generating data
    self.n_traj = 4             # Number of coexisting trajectories
    # self.step_length = 0.5      # Sampling rate
    self.max_order = 2        # Maximum order of the LTI systems generated IRs
    self.noise_std = 0.2        # Variance of the 0-mean gaussian noise in the measurements
    self.missing_data_perc = 0  # How many missing points in the trajectory
    self.n_realizations = 1
    self.pole_low, self.pole_high = 0.5, 1.5

  def get_random_trajectory(self, traj_length=None, pole_limits = None):
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
    traj_noisy = []
    for tr in range(self.n_traj):

      ydata_clean = np.zeros((self.n_traj, traj_length))
      y = np.zeros((self.n_realizations, traj_length))

      y[:, 0:self.max_order] = np.random.uniform(0, 1, self.n_realizations * self.max_order) \
        .reshape(self.n_realizations, self.max_order)

      # polys = []
      # for rr in range(self.n_traj):
      #   polys.append(np.poly(rhos[:,tr]))
      # reg = -np.flip(np.stack(polys), axis=1)

      poly = np.poly(rhos[:, tr])
      reg = -np.flip(poly)

      for i in range(self.max_order, traj_length):
        y[:, i] = (y[:, i-self.max_order:i]*reg[:-1]).sum()

      ydata_clean = y

      # Adding noise
      noise = np.random.normal(0, self.noise_std, traj_length)

      ydata = ydata_clean + noise

      if self.save_test:
        fig = plt.plot(ydata_clean[0])
        plt.savefig('example_traj')
        print(ydata_clean)

      traj_clean.append(ydata_clean)
      traj_noisy.append(ydata)

    traj_clean = np.concatenate(traj_clean, axis=0)
    traj_noisy = np.concatenate(traj_noisy, axis=0)

    # TODO: generate 2D sequences
    '''
    Mix at different points of the trajectories - High order trajectories
    '''
    num_forks = 2
    data_ids = range(1,traj_length-1)
    forks = random.sample(data_ids, num_forks)

    traj = []
    traj_clean_cyc = np.concatenate([traj_clean[1:],traj_clean[0:1]], axis=0)
    for pt in forks:
      for i in range(self.n_traj):
        traj.append(np.concatenate([traj_clean[:,0:pt],traj_clean_cyc[:,pt:]], axis=1))
        traj_clean_cyc = np.concatenate([traj_clean_cyc[1:], traj_clean_cyc[0:1]], axis=0)

    traj_mix = np.concatenate(traj)

    if self.save_test:
      for i in range(5):
        fig = plt.plot(traj_mix[i])
        plt.savefig('example_traj_mixed')


    return  traj, traj_mix

  def generate_dataset(self, training_samples = 1e4, testing_samples = 5e3):



def main():

  traj_length = 7
  traj2D = TrajectoryTree('', True, traj_length=traj_length)
  traj = traj2D.get_random_trajectory(traj_length=traj_length)

if __name__=='__main__':
  main()

# class MovingMNIST(data.Dataset):
#   def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
#                transform=None):
#     '''
#     param num_objects: a list of number of possible objects.
#     '''
#     super(MovingMNIST, self).__init__()
#
#     self.dataset = None
#     if is_train:
#       self.mnist = load_mnist(root)
#     else:
#       if num_objects[0] != 2:
#         self.mnist = load_mnist(root)
#       else:
#         self.dataset = load_fixed_set(root, False)
#     self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]
#
#     self.is_train = is_train
#     self.num_objects = num_objects
#     self.n_frames_input = n_frames_input
#     self.n_frames_output = n_frames_output
#     self.n_frames_total = self.n_frames_input + self.n_frames_output
#     self.transform = transform
#     # For generating data
#     self.image_size_ = 64
#     self.digit_size_ = 28
#     self.step_length_ = 0.5 #was: 0.5
#
#   def get_random_trajectory(self, seq_length):
#     ''' Generate a random sequence of a MNIST digit '''
#     canvas_size = self.image_size_ - self.digit_size_
#     x = random.random()
#     y = random.random()
#     theta = random.random() * 2 * np.pi
#     v_y = np.sin(theta)
#     v_x = np.cos(theta)
#
#     start_y = np.zeros(seq_length)
#     start_x = np.zeros(seq_length)
#     for i in range(seq_length):
#       # Take a step along velocity.
#       y += v_y * self.step_length_
#       x += v_x * self.step_length_
#
#       # Bounce off edges.
#       if x <= 0:
#         x = 0
#         v_x = -v_x
#       if x >= 1.0:
#         x = 1.0
#         v_x = -v_x
#       if y <= 0:
#         y = 0
#         v_y = -v_y
#       if y >= 1.0:
#         y = 1.0
#         v_y = -v_y
#       start_y[i] = y
#       start_x[i] = x
#
#     # Scale to the size of the canvas.
#     start_y = (canvas_size * start_y).astype(np.int32)
#     start_x = (canvas_size * start_x).astype(np.int32)
#     return start_y, start_x
#
#   def generate_moving_mnist(self, num_digits=1):
#     '''
#     Get random trajectories for the digits and generate a video.
#     '''
#     data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
#     for n in range(num_digits):
#       # Trajectory
#       start_y, start_x = self.get_random_trajectory(self.n_frames_total)
#       ind = random.randint(0, self.mnist.shape[0] - 1)
#       digit_image = self.mnist[ind]
#       for i in range(self.n_frames_total):
#         top    = start_y[i]
#         left   = start_x[i]
#         bottom = top + self.digit_size_
#         right  = left + self.digit_size_
#         # Draw digit
#         data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)
#
#     data = data[..., np.newaxis]
#     return data
#
#   def __getitem__(self, idx):
#     length = self.n_frames_input + self.n_frames_output
#     if self.is_train or self.num_objects[0] != 2:
#       # Sample number of objects
#       num_digits = random.choice(self.num_objects)
#       # Generate data on the fly
#       images = self.generate_moving_mnist(num_digits)
#     else:
#       images = self.dataset[:, idx, ...]
#
#     if self.transform is not None:
#       images = self.transform(images)
#     input = images[:self.n_frames_input]
#     if self.n_frames_output > 0:
#       output = images[self.n_frames_input:length]
#     else:
#       output = []
#
#     return input, output
#
#   def __len__(self):
#     return self.length