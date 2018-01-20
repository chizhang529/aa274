from sys import argv
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import car_dyn
from P3_pose_stabilization import ctrl_pose

# unpack argv
script_name,x_0, y_0, th_0, t_end = argv
x_0 = float (x_0)
y_0 = float(y_0)
th_0 = float(th_0)
t_end = float(t_end)

print 'script_name: %s' % script_name
print '(x_0, y_0, th_0) = (%.2f, %.2f, %.2f)' %(x_0, y_0, th_0)
print 't_f = %.2f' %t_end

#x_0 = 5
#y_0 = 3
#th_0 = 0
#t_end = 20

x_g = 5
y_g = 5
th_g = -np.pi/2

#timestep
dt = 0.005
N = int (t_end/dt)

# Set up simulation

time = dt*np.array(range(N+1)) #t[0]....t[N]
state = np.zeros((N+1,3))
state[0,:] = np.array([[x_0, y_0, th_0]])
x = state[0,:]

ctrl = np.zeros((N,2))

for i in range(N): #t[0]...t[N-1]
    ctrl_fbck = ctrl_pose(x[0],x[1],x[2],x_g,y_g,th_g)
    ctrl[i,0] = ctrl_fbck[0]
    ctrl[i,1] = ctrl_fbck[1]

    d_state = odeint(car_dyn,x,np.array([time[i], time[i+1]]), args = (ctrl[i,:],[0,0]))
    x = d_state[1,:]
    state[i+1,:] = x

# Plots

plt.figure()
plt.subplot(2,1,1)
plt.plot(state[:,0],state[:,1],linewidth=1)
plt.quiver(state[0:-1:200,0],state[0:-1:200,1],np.cos(state[0:-1:200,2]), np.sin(state[0:-1:200,2]))
plt.grid('on')
plt.plot(x_0,y_0,'go',markerfacecolor='green',markersize=15)
plt.plot(x_g,y_g,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')

plt.subplot(2,1,2)
plt.plot(time,state[:,2],linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['$\\theta$ [rad]'],loc='center left', bbox_to_anchor=(1,0.5))

plt.figure()
plt.plot(time[0:-1], ctrl,linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'],loc='center left', bbox_to_anchor=(1,0.5))
plt.show()
