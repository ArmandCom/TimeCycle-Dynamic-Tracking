import torch
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Tracker data
directory = '/Users/marinaalonsopoal/PycharmProjects/TimeCycle-Dynamic-Tracking/centroids_tree_nfl.obj'
directory = '/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/centroids_tree_nhl.obj'
with open(directory, 'rb') as f:
    data = pkl.load(f)

data_npy = np.zeros((len(data), 2))
for t, points in enumerate(data):
    data_npy[t, :] = np.asarray(points[0]).squeeze(0)
    print(t)
    print(points)
    if t == 19 or t == 20:
        data_npy[t, :] = np.asarray(points[1]).squeeze(0)
    elif t == 21:
        data_npy[t, :] = np.asarray(points[0]).squeeze(0)

np.save('/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/opcio1.npy', data_npy)

# Filtering
Tf = 3
if Tf == 3:
    data_npy_ext = np.vstack([data_npy[0, :], data_npy, data_npy[-1, :]])
elif Tf == 5:
    data_npy_ext = np.vstack([data_npy[0, :], data_npy[0, :], data_npy, data_npy[-1, :], data_npy[-1, :]])
elif Tf == 7:
    data_npy_ext = np.vstack([data_npy[0, :], data_npy[0, :], data_npy[0, :], data_npy, data_npy[-1, :], data_npy[-1, :], data_npy[-1, :]])

filter = (1/Tf)*np.ones(Tf)

datax = np.zeros(len(data))
datay = np.zeros(len(data))
for t in range(len(data)):
    data_x_window = data_npy_ext[t:t+Tf, 0]
    data_y_window = data_npy_ext[t:t + Tf, 1]
    convx = np.convolve(filter, data_x_window)
    convy = np.convolve(filter, data_y_window)
    datax[t] = convx[int(np.ceil((2*Tf-1)/2)-1)]
    datay[t] = convy[int(np.ceil((2 * Tf - 1) / 2) - 1)]

data_smoothed = np.zeros((len(data), 2))
data_smoothed[:, 0] = datax
data_smoothed[:, 1] = datay
print(data_smoothed)


fig, (ax1, ax2) = plt.subplots(2)

ax1.set_title('X coordinate')
# ax1.plot(np.arange(len(data)), datax, label='smoothed')
# ax1.scatter(np.arange(len(data)), datax)

ax1.plot(np.arange(len(data)), data_npy[:, 0], label='original', alpha=0.5)
ax1.scatter(np.arange(len(data)), data_npy[:, 0], alpha=0.5)

ax2.set_title('Y coordinate')
# ax2.plot(np.arange(len(data)), datay, label='smoothed')
# ax2.scatter(np.arange(len(data)), datay)
ax2.plot(np.arange(len(data)), data_npy[:, 1], label='original', alpha=0.5)
ax2.scatter(np.arange(len(data)), data_npy[:, 1], alpha=0.5)
ax1.legend()
ax2.legend()
plt.show()

a = data_smoothed.tolist()
for t, points in enumerate(a):
    print(points)

np.save('/Users/marinaalonsopoal/marinaaa.npy', a)