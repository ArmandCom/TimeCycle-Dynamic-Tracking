import torch
import pickle as pkl
from baseline.TrackerDynBoxes import TrackerDynBoxes
from baseline.utils_visualization import *

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 0.0001  # Gram Matrix noise
T0 = 8
T = 3

# Tracker data
directory = '/data/Ponc/tracking/centroids_tree_nfl.obj'
directory = '/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/centroids_tree_nhl.obj'
with open(directory, 'rb') as f:
    data = pkl.load(f)

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, noise=eps)
len_output = len(data) - T0 - T + 1
points_tracked_npy = np.zeros((len_output, 2))

for t, points in enumerate(data):
    print('\ntime:', t)
    points_tracked = tracker.decide(points)
    print(points_tracked)
    if t >= T0 + T - 1:
        points_tracked_npy[t-T-T0+1, :] = np.asarray(points_tracked)

jblds = np.asarray(tracker.JBLDs_x)

print('len points', points_tracked_npy.shape)
print('len jblds', jblds.shape)
print('len data', len(data))
print('supposed length:', len(data)-T0-T+1)


# Visualization
# plot_candidates_and_trajectory(data, points_tracked_npy, T0, T)
plot_candidates_and_jblds(data, points_tracked_npy, jblds, T0, T)



