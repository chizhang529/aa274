import numpy as np
from numpy import linalg
from P3_pose_stabilization import ctrl_pose

def ctrl_traj(x,y,th,ctrl_prev,x_d,y_d,xd_d,yd_d,xdd_d,ydd_d,x_g,y_g,th_g):
    # (x,y,th): current state
    # ctrl_prev: previous control input (V,om)
    # (x_d, y_d): desired position
    # (xd_d, yd_d): desired velocity
    # (xdd_d, ydd_d): desired acceleration
    # (x_g,y_g,th_g): desired final state

    # Timestep
    dt = 0.005

    # Gains
    # kpx = #...TODO...#
    # kpy =
    # kdx =
    # kdy =

    # Define control inputs (V,om) - without saturation constraints
    # Switch to pose controller once "close" enough, i.e., when
    # the robot is within 0.5m of the goal xy position.
    #...TODO...#

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
