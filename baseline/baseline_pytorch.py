import torch
import pickle as pkl
import numpy as np
from TrackerDynBoxes import TrackerDynBoxes
from utils_visualization import *
from utils_general import *

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 1e-1  # Gram Matrix noise
T0 = 6  # Past temporal window
T = 2  # Future temporal window
W = 1  # Smoothing window size - If W=1, no smoothing is performed
s = int(np.ceil(W/2)-1)
coordinate = 0  # coordinate 0: x, 1: y

# Tracker data
directory = '/data/Ponc/tracking/centroids_tree_nhl.obj'
# directory = '/Users/marinaalonsopoal/Desktop/centroids_tree_nhl.obj'

with open(directory, 'rb') as f:
    data = pkl.load(f)

# Bounding Boxes

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, noise=eps, coord=coordinate)
len_output = len(data) - T0 - T + 1  # Smoothing will not affect the length because of the mirroring
points_tracked_npy = np.zeros((len_output, 2))

Ts = 0
list_smoothed = []


for tin in range(len(data) + s):

    ts = tin - s
    tout = tin - s - T - T0 + 1

    if tin < len(data):
        points = data[tin]
        # data[t] = points = [[array([x1, y1])], [array([x2, y2])]] - list
        # points[0] = [array([x1, y1])] - list
        # points[0][0] = [x1, y1] - numpy.ndarray
        # points[0][0][1] = y1 - numpy.int64 (or numpy.float64)

    if ts >= 0:  # It can start smoothing
        window = []
        for w in range(-2*s, 1):
            if w+tin < 0:
                window.append(data[0])  # Mirroring
            elif w+tin >= len(data):
                window.append(data[-1])  # Mirroring
            else:
                window.append(data[w+tin])
        points = smooth_data(window)
        list_smoothed.append(points)
        points_tracked = tracker.decide(points)

    if tout >= 0:
        points_tracked_npy[tout, :] = np.asarray(points_tracked)

jblds = np.asarray(tracker.JBLDs_x)

# Visualization
# plot_data_and_smoothed(data, list_smoothed, W)
# plot_candidates_and_trajectory(data, list_smoothed, points_tracked_npy, T0, T, W, coordinate)
plot_candidates_and_jblds(coordinate, data, points_tracked_npy, jblds, T0, T)
plot_position_and_bboxes(data, bboxes)