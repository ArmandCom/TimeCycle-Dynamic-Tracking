import torch
from functools import reduce
from operator import mul
import numpy as np


def generate_seq_from_tree(seq_lengths, window, idx):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per time (T)
        - window: list containing all candidates (W)
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
    """ Smoothes (computes mean) of a window given different candidates
    Args:
        - window: list containing different number of candidates per each time in the window
    Returns:
        - smoothed: list containing different smoothed candidates per one time
    """
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