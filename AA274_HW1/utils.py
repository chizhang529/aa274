import numpy as np

def car_dyn(x, t, ctrl, noise):

    u_0 = ctrl[0] + noise[0]
    u_1 = ctrl[1] + noise[1]

    dxdt = [u_0*np.cos(x[2]), u_0*np.sin(x[2]), u_1]

    return dxdt

def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi