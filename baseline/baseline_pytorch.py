import torch
import pickle as pkl
from baseline.TrackerDynBoxes import TrackerDynBoxes
from baseline.utils_visualization import *

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 0.05  # Gram Matrix noise
T0 = 5
T = 2
coordinate = 0  # coordinate 0: x, 1: y, 2: both mean, 3: combined Hankels

# Tracker data
directory = '/data/Ponc/tracking/centroids_tree_nfl.obj'
directory = '/Users/marinaalonsopoal/PycharmProjects/TimeCycle-Dynamic-Tracking/centroids_tree_nfl.obj'
directory = '/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/centroids_tree_nhl.obj'
with open(directory, 'rb') as f:
    data = pkl.load(f)


data = np.load('/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/opcio1.npy')
data = data.tolist()

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, noise=eps, coord=coordinate)
len_output = len(data) - T0 - T + 1
points_tracked_npy = np.zeros((len_output, 2))

data2=[]
for t, points in enumerate(data):
    data2.append([[points]])


for t, points in enumerate(data2):
    print('point', points)
    print('\n')
    points_tracked = tracker.decide(points)
    if t >= T0 + T - 1:
        points_tracked_npy[t-T-T0+1, :] = np.asarray(points_tracked)
jbldsX = np.asarray(tracker.JBLDs_x)


# Visualization
# plot_candidates_and_trajectory(data, points_tracked_npy, T0, T)
# plot_candidates_and_jblds(coordinate, data, points_tracked_npy, jblds, T0, T)



coordinate = 1  # coordinate 0: x, 1: y, 2: both mean, 3: combined Hankels

# Tracker
tracker = TrackerDynBoxes(T0=T0, T=T, noise=eps, coord=coordinate)
len_output = len(data) - T0 - T + 1
points_tracked_npy = np.zeros((len_output, 2))

for t, points in enumerate(data2):
    points_tracked = tracker.decide(points)
    if t >= T0 + T - 1:
        points_tracked_npy[t-T-T0+1, :] = np.asarray(points_tracked)

jbldsY = np.asarray(tracker.JBLDs_x)


plt.plot(jbldsX, label='X JBLD')
plt.plot(jbldsY, label='Y JBLD')
plt.legend()
plt.show()