import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cpu')

def plot_candidates_and_trajectory(data, data_smoothed, points_tracked_npy, T0, T, W, coord):
    """ Plots the original candidates, the smoothed candidates and the tracked decisions.
    """
    size_small = 50
    size_big = 70
    s = int(np.ceil(W/2)-1)
    a = 0.5
    if coord == 0:
        coordinate = 'x'
    else:
        coordinate = 'y'

    for t, points in enumerate(data):
        if t == 0:
            plt.scatter(t, points[0][0][0], s=size_big, c='k', zorder=1, alpha=a, label='Original Data')
            plt.scatter(t, points[0][0][1], s=size_big, c='k', zorder=1, alpha=a)
        else:
            if len(points) == 1:
                plt.scatter(t, points[0][0][0], s=size_big, c='k', zorder=1, alpha=a)
                plt.scatter(t, points[0][0][1], s=size_big, c='k', zorder=1, alpha=a)
            else:
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=size_big, c='k', zorder=1, alpha=a)
                    plt.scatter(t, points[c][0][1], s=size_big, c='k', zorder=1, alpha=a)
        plt.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)

    for t, points in enumerate(data_smoothed):
        if t == 0:
            plt.scatter(t, points[0][0][0], s=size_big, c='b', zorder=2, alpha=a, label='Smoothed Data')
            plt.scatter(t, points[0][0][1], s=size_big, c='b', zorder=2, alpha=a)
        else:
            if len(points) == 1:
                plt.scatter(t, points[0][0][0], s=size_big, c='b', zorder=2, alpha=a)
                plt.scatter(t, points[0][0][1], s=size_big, c='b', zorder=2, alpha=a)
            else:
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=size_big, c='b', zorder=2, alpha=a)
                    plt.scatter(t, points[c][0][1], s=size_big, c='b', zorder=2, alpha=a)
        plt.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)

    points_x = points_tracked_npy[:, 0]
    points_y = points_tracked_npy[:, 1]
    tstart = T0
    time_out = np.arange(tstart, tstart + len(points_x))

    plt.scatter(time_out, points_x, s=size_small, c='r', edgecolors='k', zorder=3, label='X Coordinate')
    plt.scatter(time_out, points_y, s=size_small, c='yellow', edgecolors='k', zorder=3, label='Y Coordinate')
    tit = 'Trajectory Evolution with T0=' + str(T0) + ', T=' + str(T) + ' and W=' + str(W) + ' choosing coordinate: ' + coordinate
    plt.legend()
    plt.title(tit)
    plt.xlabel('Time')
    plt.show()



def plot_candidates_and_jblds(coord, data, points_tracked_npy, jblds, T0, T):
    """  In one axis plots the candidate points and the decided ones and on the other axis, the JBLD evolution.
    """
    th = 0.00045
    size_small = 15
    size_big = 50
    fig, (ax1, ax2) = plt.subplots(2)
    if coord == 0:
        tit_coord = 'X '
        col = 'tomato'
    else:
        tit_coord = 'Y '
        col = 'orange'

    tit = tit_coord + 'Coordinate Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    ax1.set_title(tit)
    ax2.set_title('JBLD')
    points_coord = points_tracked_npy[:, coord]
    time_out = np.arange(T0, len(points_coord) + T0)

    for t, points in enumerate(data):
        if len(points) == 1:
            ax1.scatter(t, points[0][0][coord], s=size_big, c='k', zorder=2, alpha=0.75)
        # else:
        #     for c in range(len(points)):
        #         ax1.scatter(t, points[c][0][coord], s=size_big, c='k', zorder=2, alpha=0.75)
        if t % 3 == 0:
            ax1.axvline(x=t, color='g', linestyle=':', linewidth=1, zorder=1)
            ax2.axvline(x=t, color='g', linestyle=':', linewidth=1, zorder=1)
        elif (t+1) % 3 == 0:
            ax1.axvline(x=t, color='r', linestyle=':', linewidth=1, zorder=1)
            ax2.axvline(x=t, color='r', linestyle=':', linewidth=1, zorder=1)
        else:
            ax1.axvline(x=t, color='b', linestyle=':', linewidth=1, zorder=1)
            ax2.axvline(x=t, color='b', linestyle=':', linewidth=1, zorder=1)

    ax1.scatter(time_out, points_coord, s=size_small, c=col, zorder=3)
    ax2.plot(time_out, jblds, color='k')
    # ax2.axhline(y=th)
    ax1.set_xlim(0, len(data))
    ax2.set_xlim(0, len(data))
    plt.show()


def plot_2_jblds(coord, data,points_coord,jblds, T0, T):
    size_small = 15
    size_big = 50
    fig, ax1 = plt.subplots(1)
    ax1.set_title('JBLD')


    time_out = np.arange(T0, len(points_coord) + T0)

    for t, points in enumerate(data):
        if t % 3 == 0:
            ax1.axvline(x=t, color='g', linestyle=':', linewidth=1, zorder=1)
            # ax2.axvline(x=t, color='g', linestyle=':', linewidth=1, zorder=1)
        elif (t+1) % 3 == 0:
            ax1.axvline(x=t, color='r', linestyle=':', linewidth=1, zorder=1)
            # ax2.axvline(x=t, color='r', linestyle=':', linewidth=1, zorder=1)
        else:
            ax1.axvline(x=t, color='b', linestyle=':', linewidth=1, zorder=1)
            # ax2.axvline(x=t, color='b', linestyle=':', linewidth=1, zorder=1)

   
    ax1.plot(time_out, jblds[:,0], color='k')
    ax1.plot(time_out, jblds[:,1], color='g')

    ax1.set_xlim(0, len(data))
    plt.show()
def plot_data_and_smoothed(data, list_smoothed, W):
    """ Plots original data and the smoothed version of the data
    """
    size = 40
    c1 = 'blue'
    c2 = 'red'
    a2 = 0.5
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Original and Smoothed (W =' + str(W) + ') Coordinate X')
    ax2.set_title('Original and Smoothed (W =' + str(W) + ') Coordinate Y')

    for t, points in enumerate(data):
        if len(points) == 1:
            if t == 0:
                ax1.scatter(t, points[0][0][0], c=c1, s=size, zorder=2, alpha=1, label='Original')
                ax2.scatter(t, points[0][0][1], c=c1, s=size, zorder=2, alpha=1, label='Original')
            else:
                ax1.scatter(t, points[0][0][0], c=c1, s=size, zorder=2, alpha=1)
                ax2.scatter(t, points[0][0][1], c=c1, s=size, zorder=2, alpha=1)

        else:
            for c in range(len(points)):
                ax1.scatter(t, points[c][0][0], c=c1, s=size, zorder=2, alpha=1)
                ax2.scatter(t, points[c][0][1], c=c1, s=size, zorder=2, alpha=1)

    for t, points in enumerate(list_smoothed):
        if len(points) == 1:
            if t == 0:
                ax1.scatter(t, points[0][0][0], c=c2, s=size, zorder=2, alpha=a2, label='Smoothed')
                ax2.scatter(t, points[0][0][1], c=c2, s=size, zorder=2, alpha=a2, label='Smoothed')
            else:
                ax1.scatter(t, points[0][0][0], c=c2, s=size, zorder=2, alpha=a2)
                ax2.scatter(t, points[0][0][1], c=c2, s=size, zorder=2, alpha=a2)

        else:
            for c in range(len(points)):
                ax1.scatter(t, points[c][0][0], c=c2, s=size, zorder=2, alpha=0.75)
                ax2.scatter(t, points[c][0][1], c=c2, s=size, zorder=2, alpha=0.75)
        ax1.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)
        ax2.axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax1.legend()
    ax2.legend()
    plt.show()
