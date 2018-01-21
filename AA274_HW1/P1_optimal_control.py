import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt


def q1_ode_fun(tau, z):
    # Code in the BVP ODEs
	# z = [z0, z1, z2, z3, z4, z5, z6]
	#   = [x, y, theta, p1, p2, p3, r]
	
	# control inputs
	V = -0.5 * (z[3]*np.cos(z[2]) + z[4]*np.sin(z[2]))
	omega = -0.5 * z[5]
	
	dz = np.zeros(7)
	dz[0] = z[6]*V*np.cos(z[2])
	dz[1] = z[6]*V*np.sin(z[2])
	dz[2] = z[6]*omega
	dz[3] = 0
	dz[4] = 0
	dz[5] = z[6]*(z[3]*V*np.sin(z[2]) - z[4]*V*np.cos(z[2]))
	dz[6] = 0
	
	return dz


def q1_bc_fun(za, zb):

    # lambda
    lambda_test = 0.25

    # goal pose
    x_g = 5
    y_g = 5
    th_g = -np.pi/2.0
    xf = [x_g, y_g, th_g]

    # initial pose
    x0 = [0, 0, -np.pi/2.0]
	
    # Code boundary condition residuals
    H_f = lambda_test - 0.5*(zb[3]*zb[4]*np.sin(zb[2])*np.cos(zb[2])) - 0.25*(zb[3]*np.cos(zb[2]))**2 - 0.25*(zb[4]*np.sin(zb[2]))**2 - 0.25*(zb[5]**2)
   	
   	# boundary conditions
    left_bc = np.array(za[:3]) - np.array(x0)
    right_bc = np.append(np.array(zb[:3]) - np.array(xf), H_f)
	
	# return a tuple of (left BC, right BC)
    return (left_bc, right_bc)

#Define solver state: z = [x, y, theta, p1, p2, p3, r]
problem = scikits.bvp_solver.ProblemDefinition(num_ODE = 7,
											num_parameters = 0,
                                            num_left_boundary_conditions = 3,
                                            boundary_points = (0,1),
                                            function = q1_ode_fun,
                                            boundary_conditions = q1_bc_fun)
                                            
initial_guess = (1.0, 1.0, -np.pi/2.0, -1.0, -1.0, 5.0, 10.0)

soln = scikits.bvp_solver.solve(problem, solution_guess = initial_guess)

dt = 0.005

# Test if time is reversed in bvp_solver solution
z_0 = soln(0)
flip = 0
if z_0[-1] < 0:
    t_f = -z_0[-1]
    flip = 1
else:
    t_f = z_0[-1]

t = np.arange(0,t_f,dt)
z = soln(t/t_f)
if flip:
    z[3:7,:] = -z[3:7,:]
z = z.T # solution arranged column-wise

# Recover optimal control histories
V = -0.5*(z[:,3]*np.cos(z[:,2]) + z[:,4]*np.sin(z[:,2]))
om = -0.5*z[:,5]

V = np.array([V]).T # Convert to 1D column matrices
om = np.array([om]).T

# Save trajectory data (state and controls)
data = np.hstack((z[:,:3],V,om))
np.save('traj_data_optimal_control',data)

# Plots
plt.rc('font', weight='bold', size=16)

plt.figure()
plt.plot(z[:,0], z[:,1],'k-',linewidth=2)
plt.quiver(z[1:-1:200,0],z[1:-1:200,1],np.cos(z[1:-1:200,2]),np.sin(z[1:-1:200,2]))
plt.grid('on')
plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X'); plt.ylabel('Y')

plt.figure()
plt.plot(t, V,linewidth=2)
plt.plot(t, om,linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'], bbox_to_anchor=(0.01, 0.99), loc=2)

plt.show()
