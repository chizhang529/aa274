import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)
        # update self.x, self.P
        self.x = g
        self.P = np.matmul(np.matmul(Gx, self.P), Gx.T) + dt * np.matmul(np.matmul(Gu, self.Q), Gu.T)

    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        # update self.x, self.P
        sigma = np.matmul(np.matmul(H, self.P), H.T) + R
        K = np.matmul(np.matmul(self.P, H.T), np.linalg.inv(sigma))
        self.x = self.x + K.dot(z).flatten()
        self.P = self.P - np.matmul(np.matmul(K, sigma), K.T)

    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x  # [x_(t-1), y_(t-1), theta_(t-1)]
        
        # compute g, Gx, Gu
        if abs(om) > 1e-5:  
            xt = x + v/om*(np.sin(th+om*dt) - np.sin(th))
            yt = y - v/om*(np.cos(th+om*dt) - np.cos(th))
            th_t = th + om*dt
            g = np.array([xt, yt, th_t])
            
            Gx = np.array([[1., 0., v/om*(np.cos(th+om*dt) - np.cos(th))],
                           [0., 1., v/om*(np.sin(th+om*dt) - np.sin(th))],
                           [0., 0., 1.]])
        
            dx_om = v/(om**2)*(np.sin(th) - np.sin(th+om*dt)) + v*dt/om*np.cos(th+om*dt)
            dy_om = v/(om**2)*(np.cos(th+om*dt) - np.cos(th)) + v*dt/om*np.sin(th+om*dt)
            Gu = np.array([[ 1./om*(np.sin(th+om*dt) - np.sin(th)), dx_om],
                           [-1./om*(np.cos(th+om*dt) - np.cos(th)), dy_om],
                           [0., dt]])
        else:  # L'Hospital's rule
            xt = x + v*np.cos(th)*dt
            yt = y + v*np.sin(th)*dt
            th_t = th
            g = np.array([xt, yt, th_t])
            
            Gx = np.array([[1., 0., -v*dt*np.sin(th)],
                           [0., 1., v*dt*np.cos(th)],
                           [0., 0., 1.]])
                           
            Gu = np.array([[np.cos(th)*dt, -v/2*(dt**2)*np.sin(th)],
                           [np.sin(th)*dt, v/2*(dt**2)*np.cos(th)],
                           [0., dt]])
        
        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m

        # compute h, Hx
        x, y, theta = self.x 
        x_cam, y_cam, theta_cam = self.tf_base_to_camera  # camera location in robot frame
        # compute camera location in world frame
        T_bot_to_w = np.array([[np.cos(theta), -np.sin(theta), x],
                               [np.sin(theta),  np.cos(theta), y],
                               [0., 0., 1.]])
        x_cam_w, y_cam_w, _ = T_bot_to_w.dot(np.array([x_cam, y_cam, 1.]))
        
        h = np.array([alpha - theta - theta_cam,
                      r - x_cam_w*np.cos(alpha) - y_cam_w*np.sin(alpha)])
        
        Hx = np.array([[0., 0., -1.],
                       [-np.cos(alpha), -np.sin(alpha), (y_cam*np.cos(alpha)-x_cam*np.sin(alpha))*np.cos(theta) + (x_cam*np.cos(alpha)+y_cam*np.sin(alpha))*np.sin(theta)]])
        
        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):
        # compute v_list, R_list, H_list
        v_list, R_list, H_list = [], [], []
        num_meas = rawZ.shape[1]
        num_map_lines = self.map_lines.shape[1]
        
        for i in range(num_meas):
            zi = rawZ[:, i]
            Ri = rawR[i]
            # record data
            d_min = self.g**2
            v, R, H = None, None, None
            for j in range(num_map_lines):
                hj, Hj = self.map_line_to_predicted_measurement(self.map_lines[:, j])

                vij = zi - hj
                Sij = np.matmul(np.matmul(Hj, self.P), Hj.T) + Ri
                dij = np.matmul(np.matmul(vij.T, np.linalg.inv(Sij)), vij)

                if dij < d_min:
                    d_min = dij
                    v, R, H = vij, Ri, Hj

            if d_min < self.g**2:
                v_list.append(v)
                R_list.append(R)
                H_list.append(H)
                
        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None
        
        if len(v_list) == 0:
            return None, None, None
        # compute z, R, H
        z = np.array(v_list).reshape(-1, 1)
        R = scipy.linalg.block_diag(*R_list)
        H = np.array(H_list).reshape(-1, H_list[0].shape[1])
        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        # compute g, Gx, Gu
        if abs(om) > 1e-5:  
            xt = x + v/om*(np.sin(th+om*dt) - np.sin(th))
            yt = y - v/om*(np.cos(th+om*dt) - np.cos(th))
            th_t = th + om*dt
            g = np.copy(self.x)
            g[:3] = [xt, yt, th_t]
            
            Gx = np.eye(self.x.size)
            Gx[0, 2] = v/om*(np.cos(th+om*dt) - np.cos(th))
            Gx[1, 2] = v/om*(np.sin(th+om*dt) - np.sin(th))
            
            Gu = np.zeros((self.x.size, 2))
            dx_om = v/(om**2)*(np.sin(th) - np.sin(th+om*dt)) + v*dt/om*np.cos(th+om*dt)
            dy_om = v/(om**2)*(np.cos(th+om*dt) - np.cos(th)) + v*dt/om*np.sin(th+om*dt)
            Gu[0, 0] = 1./om*(np.sin(th+om*dt) - np.sin(th))
            Gu[0, 1] = dx_om
            Gu[1, 0] = -1./om*(np.cos(th+om*dt) - np.cos(th))
            Gu[1, 1] = dy_om
            Gu[2, 1] = dt
            
        else:  # L'Hospital's rule
            xt = x + v*np.cos(th)*dt
            yt = y + v*np.sin(th)*dt
            th_t = th
            g = np.copy(self.x)
            g[:3] = [xt, yt, th_t]
            
            Gx = np.eye(self.x.size)
            Gx[0, 2] = -v*dt*np.sin(th)
            Gx[1, 2] = v*dt*np.cos(th)
            
            Gu = np.zeros((self.x.size, 2))
            Gu[0, 0] = np.cos(th)*dt
            Gu[0, 1] = -v/2*(dt**2)*np.sin(th)
            Gu[1, 0] = np.sin(th)*dt
            Gu[1, 1] = v/2*(dt**2)*np.cos(th)
            Gu[2, 1] = dt

        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        if len(v_list) == 0:
            return None, None, None
            
        z = np.array(v_list).reshape(-1, 1)
        R = scipy.linalg.block_diag(*R_list)
        H = np.array(H_list).reshape(-1, H_list[0].shape[1])
        return z, R, H

    # Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
    #
    # Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
    # so that we know which components of the Jacobian to fill in.
    def map_line_to_predicted_measurement(self, j):
        alpha, r = self.x[(3+2*j):(3+2*j+2)]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

        # compute h, Hx (you may find the skeleton for computing Hx below useful)
        x, y, theta = self.x[:3] 
        x_cam, y_cam, theta_cam = self.tf_base_to_camera  # camera location in robot frame
        
        # compute camera location in world frame
        T_bot_to_w = np.array([[np.cos(theta), -np.sin(theta), x],
                               [np.sin(theta),  np.cos(theta), y],
                               [0., 0., 1.]])
        x_cam_w, y_cam_w, _ = T_bot_to_w.dot(np.array([x_cam, y_cam, 1.]))
        # compute h, Hx
        h = np.array([alpha - theta - theta_cam,
                      r - x_cam_w*np.cos(alpha) - y_cam_w*np.sin(alpha)])
        
        Hx = np.zeros((2,self.x.size))
        Hx[:,:3] = np.array([[0., 0., -1.],
                            [-np.cos(alpha), -np.sin(alpha), (y_cam*np.cos(alpha)-x_cam*np.sin(alpha))*np.cos(theta) + (x_cam*np.cos(alpha)+y_cam*np.sin(alpha))*np.sin(theta)]])
        # First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
        if j > 1:
            Hx[0, 3+2*j] = 1.
            Hx[1, 3+2*j] = x_cam_w*np.sin(alpha) - y_cam_w*np.cos(alpha)
            Hx[0, 3+2*j+1] = 0.
            Hx[1, 3+2*j+1] = 1.

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):
        # compute v_list, R_list, H_list
        v_list, R_list, H_list = [], [], []
        num_meas = rawZ.shape[1]
        num_map_lines = (len(self.x) - 3) / 2 # self.x = [x, y, th, alpha_k, r_k...] (k = 1,2,3,...,j)
        
        for i in range(num_meas):
            zi = rawZ[:, i]
            Ri = rawR[i]
            # record data
            d_min = self.g**2
            v, R, H = None, None, None
            for j in range(num_map_lines):
                hj, Hj = self.map_line_to_predicted_measurement(j)

                vij = zi - hj
                Sij = np.matmul(np.matmul(Hj, self.P), Hj.T) + Ri
                dij = np.matmul(np.matmul(vij.T, np.linalg.inv(Sij)), vij)

                if dij < d_min:
                    d_min = dij
                    v, R, H = vij, Ri, Hj

            if d_min < self.g**2:
                v_list.append(v)
                R_list.append(R)
                H_list.append(H)

        return v_list, R_list, H_list
