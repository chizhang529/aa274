#!/usr/bin/python

############################################################
# PlotFunctions.py
############################################################

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------
# PlotScene
#
# Plots the room in which the robot collects rage data


def PlotScene():
    rectangles = np.array([[.5, 7, 1, 1, 0],
                           [2, 8.5, 1, 1, 0],
                           [3.5, 7, 1, 1, 0],
                           [2, 5.5, 1, 1, 0],
                           [1, 0, 2, .5, 0],
                           [4, 0, 2, .5, 0],
                           [8.9, 1.25, 1, 4, 0],
                           [7.5, .1, 1, 1, 0],
                           [7.5, 5.4, 1, 1, 0]])

    circles = np.array([[2.5, 7.5, .75],
                        [6, 9.3, .5],
                        [9.3, .6, .4],
                        [9.3, 5.9, .4],
                        [3, 2, .1],
                        [3, 3, .1],
                        [5.5, 2, .1],
                        [5.5, 3, .1]])

    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    plt.xlim([-0.5, 10.5])
    plt.ylim([-0.5, 10.5])

    p = plt.Rectangle((-.5, -.5), 11, 11, color=[.5, .5, .5])
    ax.add_patch(p)
    p = plt.Rectangle((0, 0), 10, 10, color='w')
    ax.add_patch(p)
    p = plt.Rectangle((2.5, 1.75), 3.5, 1.5, color='r', alpha=0.2)
    ax.add_patch(p)
    p = plt.Polygon([[6.5792, 8.5150],
                     [9.2104, 7.5574],
                     [9.6208, 8.6850],
                     [6.9896, 9.6426]], color='r')
    ax.add_patch(p)

    for rec in rectangles:
        p = plt.Rectangle(rec[0:2], rec[2], rec[3], color='r')
        ax.add_patch(p)

    for cir in circles:
        p = plt.Circle(cir[0:2], cir[2], color='g')
        ax.add_patch(p)

    return ax


#----------------------------------
# PlotData
#
# Plots range point measurements

def PlotData(RangeData, ax):

    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]
    x = x_r + rho*np.cos(theta)
    y = y_r + rho*np.sin(theta)

    ax.plot(x, y, 'b.', markersize=5)
    return ax


#----------------------------------
# PlotRays
#
# Plots lidar beams

def PlotRays(RangeData, ax):

    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]
    x = x_r + rho*np.cos(theta)
    y = y_r + rho*np.sin(theta)

    for i in range(len(x)):
        ax.plot([x_r, x[i]], [y_r, y[i]], 'b', linewidth=0.3)
    return ax

#----------------------------------
# PlotLines
#
# Plots extracted lines


def PlotLines(segend, ax):

    for seg in segend:
        ax.plot([seg[0], seg[2]], [seg[1], seg[3]], '-ko', linewidth=2,
                markersize=4, markerfacecolor='w', markeredgewidth=1)

    return ax

############################################################
