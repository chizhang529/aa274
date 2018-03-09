import numpy as np
import matplotlib.pyplot as plt
from numpy import cross

### PLOTTING

def plot_line_segments(segments, **kwargs):
    plt.plot([x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
             [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup], **kwargs)

### MAGICAL BLACK BOX COLLISION DETECTION

def ccw(A, B, C):
    return np.cross(B - A, C - A) > 0

def line_line_intersection(l1, l2):
    A, B = np.array(l1)
    C, D = np.array(l2)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)