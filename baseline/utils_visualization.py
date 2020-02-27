import matplotlib.pyplot as plt
import numpy as np


def plot_candidates_and_trajectory(data, points_tracked_npy, T0, T, count_t=0):
    for t, points in enumerate(data):
        if (t > T0 + T):  # and t<=len(data)-T):
            # d("t = ", t)
            if t == 0:
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75, label='candidates')
                # pass
            if len(points) == 1:
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75)
                plt.scatter(t, points[0][0][1], s=50, c='k', zorder=1, alpha=0.75)
                # pass
            else:
                # pass
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=50, c='k', zorder=1, alpha=0.75)
                    plt.scatter(t, points[c][0][1], s=50, c='k', zorder=1, alpha=0.75)

            if (count_t < points_tracked_npy.shape[0]):
                plt.scatter(t - (T), int(points_tracked_npy[count_t, 0]), s=25, c='tomato', zorder=2, label='decided x')
                plt.scatter(t - (T), int(points_tracked_npy[count_t, 1]), s=25, c='orange', zorder=1, label='decided y')
                count_t = count_t + 1
                # print(count_t)
        else:
            if t == 0:
                # pass
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75, label='candidates')
            if len(points) == 1:
                # pass
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75)
                plt.scatter(t, points[0][0][1], s=50, c='k', zorder=1, alpha=0.75)
            else:
                # pass
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=50, c='k', zorder=1, alpha=0.75)
                    plt.scatter(t, points[c][0][1], s=50, c='k', zorder=1, alpha=0.75)

    tit = 'Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    plt.title(tit)
    plt.xlabel('Time')
    plt.show()



def plot_candidates_and_jblds(data, points_tracked_npy, jblds, T0, T):
    fig, (ax1, ax2) = plt.subplots(2)
    tit = 'X Coordinate Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    ax1.set_title(tit)
    ax2.set_title('JBLD')

    points_x = points_tracked_npy[:, 0]
    time_out = np.arange(T0, len(points_x) + T0)

    for t, points in enumerate(data):

        if len(points) == 1:
            ax1.scatter(t, points[0][0][0], s=25, c='k', zorder=1, alpha=0.75)
        else:
            for c in range(len(points)):
                ax1.scatter(t, points[c][0][0], s=25, c='k', zorder=1, alpha=0.75)
        ax1.axvline(x=t, color='gray', linestyle=':')
        ax2.axvline(x=t, color='gray', linestyle=':')

    ax1.scatter(time_out, points_x, s=15, c='tomato')
    ax2.plot(time_out, jblds)

    ax1.set_xlim(0, len(data))
    ax2.set_xlim(0, len(data))
    plt.show()
