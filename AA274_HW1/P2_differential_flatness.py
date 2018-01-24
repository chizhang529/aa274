import numpy as np
import math
from numpy import linalg
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# Constants
t_f = 15
V_max = 0.5
om_max = 1

# Initial conditions
x_0 = 0
y_0 = 0
V_0 = V_max
th_0 = -np.pi/2
x_dot_0 = V_0*np.cos(th_0)
y_dot_0 = V_0*np.sin(th_0)

# Final conditions

x_f = 5
y_f = 5
V_f = V_max
th_f = -np.pi/2
x_dot_f = V_f*np.cos(th_f)
y_dot_f = V_f*np.sin(th_f)

# Solve Linear equations:
index_mat = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [1, t_f, t_f**2, t_f**3],
                      [0, 1, 2*t_f, 3*t_f**2]])
A = np.zeros((8, 8))
A[:4, :4] = index_mat
A[-4:, -4:] = index_mat
b = np.array([x_0, x_dot_0, x_f, x_dot_f, y_0, y_dot_0, y_f, y_dot_f])
params = np.linalg.solve(A, b)
# params = [x1, x2, x3, x4, y1, y2, y3, y4]
#          [ 0,  1,  2,  3,  4,  5,  6,  7]

# Compute traj
dt = 0.005
N = int(t_f/dt)
t = dt*np.array(range(N+1)) # t[0],....,t[N]
t = t.T
data = np.zeros((N+1,9))

# Compute trajectory, store in data, format: [x,y,th,V,om,xd,yd,xdd,ydd]
x = lambda t: params[0] + params[1]*t + params[2]*t**2 + params[3]*t**3
y = lambda t: params[4] + params[5]*t + params[6]*t**2 + params[7]*t**3
xd = lambda t: params[1] + 2*params[2]*t + 3*params[3]*t**2
yd = lambda t: params[5] + 2*params[6]*t + 3*params[7]*t**2
xdd = lambda t: 2*params[2] + 6*params[3]*t
ydd = lambda t: 2*params[6] + 6*params[7]*t
th = lambda t: np.arctan2(yd(t), xd(t))
V = lambda t: np.sqrt(xd(t)**2 + yd(t)**2)
om = lambda t: (xd(t)*ydd(t) - yd(t)*xdd(t)) / (V(t)**2)

for i, _t in enumerate(t):
    data[i] = [x(_t), y(_t), th(_t), V(_t), om(_t), xd(_t), yd(_t), xdd(_t), ydd(_t)]

# Re-scaling - Compute scaled trajectory quantities at the N points along the geometric path above
# Compute arc-length s as a function of t (HINT: use the function cumtrapz)
V_t = data[:, 3]               # V(t)
om_t = data[:, 4]              # om(t)  
s = cumtrapz(V_t, t, initial=0)  # s(t)

# Compute V_tilde (HINT: at each timestep V_tilde should be computed as a minimum of
# the original value V, and values required to ensure both constraints are satisfied)
V_tilde = np.zeros(V_t.shape)
for i in range(V_tilde.shape[0]):
    V_tilde[i] = np.sign(V_t[i]) * min(abs(V_t[i]), V_max, om_max*abs(V_t[i]/om_t[i])) 

# Compute tau (HINT: use the function cumtrapz)
tau = cumtrapz(1/V_tilde, s, initial=0)

# Compute om_tilde
om_tilde = om_t / V_t * V_tilde

# Get new final time
tf_new = tau[-1]

# Generate new uniform time grid
N_new = int(tf_new/dt)
t_new = dt*np.array(range(N_new+1))
t_new = t_new.T

# Interpolate for state trajectory
data_scaled = np.zeros((N_new+1,9))
data_scaled[:,0] = np.interp(t_new,tau,data[:,0]) # x
data_scaled[:,1] = np.interp(t_new,tau,data[:,1]) # y
data_scaled[:,2] = np.interp(t_new,tau,data[:,2]) # th
# Interpolate for scaled velocities
data_scaled[:,3] = np.interp(t_new, tau, V_tilde)   # V
data_scaled[:,4] = np.interp(t_new, tau, om_tilde)  # om
# Compute xy velocities
data_scaled[:,5] = data_scaled[:,3]*np.cos(data_scaled[:,2]) # xd
data_scaled[:,6] = data_scaled[:,3]*np.sin(data_scaled[:,2]) # yd
# Compute xy acclerations
data_scaled[:,7] = np.append(np.diff(data_scaled[:,5])/dt,-V_f*data_scaled[-1,4]*np.sin(th_f)) # xdd
data_scaled[:,8] = np.append(np.diff(data_scaled[:,6])/dt, V_f*data_scaled[-1,4]*np.cos(th_f)) # ydd

# Save trajectory data
np.save('traj_data_differential_flatness',data_scaled)

# Plots
plt.rc('font', weight='bold', size=16)

plt.figure()
plt.plot(data_scaled[:,0], data_scaled[:,1],'k-',linewidth=2)
plt.grid('on')
plt.plot(x_0,y_0,'go',markerfacecolor='green',markersize=15)
plt.plot(x_f,y_f,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X'); plt.ylabel('Y')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, data[:,3:5],linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'], bbox_to_anchor=(0.01, 0.01), loc=3)
plt.title('Original')

plt.subplot(2,1,2)
plt.plot(t_new,data_scaled[:,3:5],linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'], bbox_to_anchor=(0.01, 0.01), loc=3)
plt.title('Scaled')

plt.figure()
plt.plot(t,s,'b-',linewidth=2)
plt.grid('on')
plt.plot(tau,s,'r-',linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Arc-length [m]')
plt.legend(['Original', 'Scaled'], bbox_to_anchor=(0.01, 0.99), loc=2)

plt.show()
