import torch
import pickle as pkl
from TrackerDynBoxes import TrackerDynBoxes
from utils_visualization import *

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 1e-1  # Gram Matrix noise
T0 = 10
T = 3
coordinate = 1  # coordinate 0: x, 1: y

# Tracker data
directory = '/data/Ponc/tracking/centroids_definitius/centroids_tree_warmup_edmonton_97.obj'
# directory = '/Users/marinaalonsopoal/PycharmProjects/TimeCycle-Dynamic-Tracking/centroids_tree_nfl.obj'
# directory = '/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/centroids_tree_nhl.obj'
with open(directory, 'rb') as f:
    data = pkl.load(f)

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, noise=eps, coord=coordinate)
len_output = len(data) - T0 - T + 1
points_tracked_npy = np.zeros((len_output, 2))

for t, points in enumerate(data):
    points_tracked = tracker.decide(points)
    if t >= T0 + T - 1:
        points_tracked_npy[t-T-T0+1, :] = np.asarray(points_tracked)

jblds = np.asarray(tracker.JBLDs_x)


# Visualization
# plot_candidates_and_trajectory(data, points_tracked_npy, T0, T)
plot_candidates_and_jblds(coordinate, data, points_tracked_npy, jblds, T0, T)



