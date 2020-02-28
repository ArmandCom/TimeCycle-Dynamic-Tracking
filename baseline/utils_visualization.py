import matplotlib.pyplot as plt
import numpy as np


def plot_candidates_and_trajectory(data, points_tracked_npy, T0, T, count_t=0):
    size_small = 15
    size_big = 50

    for t, points in enumerate(data):
        if len(points) == 1:
            plt.scatter(t, points[0][0][0], s=size_big, c='k', zorder=2, alpha=0.75)
            plt.scatter(t, points[0][0][1], s=size_big, c='k', zorder=2, alpha=0.75)
        else:
            for c in range(len(points)):
                plt.scatter(t, points[c][0][0], s=size_big, c='k', zorder=2, alpha=0.75)
                plt.scatter(t, points[c][0][1], s=size_big, c='k', zorder=2, alpha=0.75)
        plt.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)

    points_x = points_tracked_npy[:, 0]
    points_y = points_tracked_npy[:, 1]
    time_out = np.arange(T0, len(points_x) + T0)
    plt.scatter(time_out, points_x, s=size_small, c='tomato', zorder=3)
    plt.scatter(time_out, points_y, s=size_small, c='orange', zorder=3)
    tit = 'Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    plt.title(tit)
    plt.xlabel('Time')
    plt.show()



def plot_candidates_and_jblds(data, points_tracked_npy, jblds, T0, T):
    size_small = 15
    size_big = 50
    fig, (ax1, ax2) = plt.subplots(2)
    tit = 'X Coordinate Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    ax1.set_title(tit)
    ax2.set_title('JBLD')

    points_x = points_tracked_npy[:, 0]
    time_out = np.arange(T0, len(points_x) + T0)

    for t, points in enumerate(data):
        if len(points) == 1:
            ax1.scatter(t, points[0][0][0], s=size_big, c='k', zorder=2, alpha=0.75)
        else:
            for c in range(len(points)):
                ax1.scatter(t, points[c][0][0], s=size_big, c='k', zorder=2, alpha=0.75)
        ax1.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)
        ax2.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)

    ax1.scatter(time_out, points_x, s=size_small, c='tomato', zorder=3)
    ax2.plot(time_out, jblds, color='k')

    ax1.set_xlim(0, len(data))
    ax2.set_xlim(0, len(data))
    plt.show()
