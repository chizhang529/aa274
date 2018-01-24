import numpy as np
from utils import wrapToPi

def ctrl_pose(x,y,th,x_g,y_g,th_g):
    #(x,y,th): current state
    #(x_g,y_g,th_g): desired final state
    
    # Relative pos in global frame
    rel_pos_global = np.array([x-x_g, y-y_g])
    # Rotation matrix from relative to global
    R = np.array([[np.cos(th_g), -np.sin(th_g)],
                  [np.sin(th_g), np.cos(th_g)]])
    # Relative pos in relative frame
    rel_pos = R.T.dot(rel_pos_global)
    
    # New state variables
    theta = th - th_g
    rho = np.linalg.norm(rel_pos)
    delta = np.arctan2(rel_pos[1], rel_pos[0]) + np.pi
    alpha = delta - theta
    # Wrap angles to [-pi, pi]
    delta, alpha = wrapToPi(np.array([delta, alpha]))
    
    # Gains
    k1 = 0.5
    k2 = 0.8
    k3 = 0.8
   
    # Define control inputs (V,om) - without saturation constraints
    thresh = 1e-5
    
    if (np.array([rho, delta, alpha]) <= np.array([thresh]*3)).all():
        V, om = 0.0, 0.0
    else:
        V = k1 * rho * np.cos(alpha)
        om = k2 * alpha + k1 * np.sinc(2*alpha/np.pi) * (alpha + k3*delta)
    
    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
