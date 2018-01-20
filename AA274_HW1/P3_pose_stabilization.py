import numpy as np
from utils import wrapToPi

def ctrl_pose(x,y,th,x_g,y_g,th_g):
    #(x,y,th): current state
    #(x_g,y_g,th_g): desired final state

    #Code pose controller
    #...TODO...#

    #Define control inputs (V,om) - without saturation constraints

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
