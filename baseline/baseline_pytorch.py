import torch
import pickle as pkl
import numpy as np
from TrackerDynBoxes import TrackerDynBoxes
from utils_visualization import *
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 1e-1  # Gram Matrix noise
T0 = 5  # Past temporal window
T = 2  # Future temporal window
smooth = True
W = 5  # Smoothing window size
s = int(np.ceil(W/2)-1)
coordinate = 0  # coordinate 0: x, 1: y

# Tracker data
directory = '/data/Ponc/tracking/centroids_tree_nhl.obj'
# directory = '/Users/marinaalonsopoal/Desktop/centroids_tree_nfl.obj'

with open(directory, 'rb') as f:
    data = pkl.load(f)

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, s=s, noise=eps, coord=coordinate)
len_output = len(data) - T0 - T + 1
points_tracked_npy = np.zeros((len_output, 2))


def generate_seq_from_tree(seq_lengths, window, idx):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per time (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    W = len(window)
    sequence = np.zeros((W, 2))
    new_idx = np.unravel_index(idx, seq_lengths)
    for time in range(W):
        sequence[time, :] = window[time][new_idx[time]][0]
    sequence = torch.from_numpy(sequence)
    return sequence


def smooth_data(window):
    smoothed = []  # list that contains the smoothed values in the data format
    num_points = []  # list that contains the number of points for each time step
    [num_points.append(len(p)) for p in window]
    num_seqs = reduce(mul, num_points)  # number of combinations
    for i in range(num_seqs):
        seq = generate_seq_from_tree(num_points, window, i)
        x = torch.mean(seq[:, 0])
        y = torch.mean(seq[:, 1])
        m = np.asarray([x, y])
        smoothed.append([m])
    return smoothed


Ts = 0
list_smoothed = []

for t, points in enumerate(data):
    # data[t] = points = [[array([x1, y1])], [array([x2, y2])]] - list
    # points[0] = [array([x1, y1])] - list
    # points[0][0] = [x1, y1] - numpy.ndarray
    # points[0][0][1] = y1 - numpy.int64 (or numpy.float64)

    print('\nTime:', t)
    if smooth:
        window = []
        for w in range(-s, s+1):
            if w+t < 0:
                window.append(data[0])
            elif w+t >= len(data):
                window.append(data[-1])
            else:
                window.append(data[w+t])
        points_smoothed = smooth_data(window)
        list_smoothed.append(points_smoothed)
        Ts = s


    # points_tracked = tracker.decide(points)
    #
    # if t >= T0 + T + Ts - 1:
    #     points_tracked_npy[t-T-T0-Ts+1, :] = np.asarray(points_tracked)

# jblds = np.asarray(tracker.JBLDs_x)


# Visualization
plot_data_and_smoothed(data, list_smoothed, W)
# plot_candidates_and_trajectory(data, points_tracked_npy, T0, T)
# plot_candidates_and_jblds(coordinate, data, points_tracked_npy, jblds, T0, T)