import matplotlib.pyplot as plt
import numpy as np


def plot_candidates_and_trajectory(data, points_tracked_npy, T0, T, count_t=0):
    for t, points in enumerate(data):
        if (t > T0 + T - 1):  # and t<=len(data)-T):
            # print("t = ", t)
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

    # time = np.arange(len(data)-T-T0+1)
    # plt.scatter(time, points_tracked_npy[:, 0], s=25, c='tomato', zorder=2, label='decided x')
    # plt.scatter(time, points_tracked_npy[:, 1], s=25, c='orange', zorder=1, label='decided y')
    # plt.legend(loc="center left")
    tit = 'Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    plt.title(tit)
    plt.xlabel('Time')
    plt.show()



def plot_candidates_and_jblds(data, jblds, T0, T):
    mean_JBLD_window = np.mean(jblds[:T0])
    a_w0 = np.max(jblds[:T0])
    a = np.max(jblds)
    th = 10 * a_w0
    for t, dis in enumerate(jblds):
        if dis > th:
            plt.scatter(t, dis, c='r')
        else:
            plt.scatter(t, dis, c='b')
    plt.ylim(-a / 5, a)
    plt.show()