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
    kpx = 1.0
    kpy = 1.0
    kdx = 2.0
    kdy = 2.0
    
    # Distance form current pos to the target
    dist = np.sqrt((x_g-x)**2 + (y_g-y)**2)
    # Define control inputs (V,om) - without saturation constraints
    # Switch to pose controller once "close" enough, i.e., when
    # the robot is within 0.5m of the goal xy position.
    if dist <= 0.5:
        V, om = ctrl_pose(x, y, th, x_g, y_g, th_g)
    else:
        if abs(ctrl_prev[0]) <= 1e-5:
            V_prev = np.sign(ctrl_prev[0])*1e-5 
        else:
            V_prev = ctrl_prev[0]
            
        x_dot = V_prev * np.cos(th)
        y_dot = V_prev * np.sin(th)
        
        # virtual control inputs
        u = np.array([xdd_d + kpx*(x_d-x) + kdx*(xd_d-x_dot),
                      ydd_d + kpy*(y_d-y) + kdy*(yd_d-y_dot)])
                      
        J = np.array([[np.cos(th), -V_prev*np.sin(th)],
                      [np.sin(th),  V_prev*np.cos(th)]])
        a, om = np.linalg.solve(J, u)
        V = V_prev + a*dt
        
    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
