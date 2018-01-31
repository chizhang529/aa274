#!/usr/bin/python

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np
import glob

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length))) # n_rows, n_cols, dims
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions (480 x 640)
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]


    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            if k[0] != 0:
                u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
                ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            # Flip Y-axis to traditional direction (aka NOT CV coordinate system)
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])
        
        return u_meas, v_meas   # Lists of arrays (one per chessboard)

    def genCornerCoordinates(self, u_meas, v_meas):
        # part (i)
        '''
            u_meas and v_meas are not necessarily needed in this function since the world frame is
            NOT relative to pixel/camera frame. Here, we define the top left corner as the origin.
        '''
        X = []
        Y = []
        
        row_coords = np.linspace(0., self.d_square*(self.n_corners_x-1), num=self.n_corners_x)
        col_coords = np.linspace(0., self.d_square*(self.n_corners_y-1), num=self.n_corners_y).reshape(self.n_corners_y, 1)
        col_coords = col_coords[-1, :] - col_coords
        corner_world_x = np.broadcast_to(row_coords, (self.n_corners_y, self.n_corners_x)).reshape(-1)
        corner_world_y = np.broadcast_to(col_coords, (self.n_corners_y, self.n_corners_x)).reshape(-1)
        
        X = [corner_world_x] * self.n_chessboards
        Y = [corner_world_y] * self.n_chessboards
        return X, Y             # Lists of arrays (length: 23)

    def estimateHomography(self, u_meas, v_meas, X, Y):
        # part (ii)
        # u_meas/v_meas: (63,)
        # X/Y: (63,)
        assert u_meas.shape == X.shape
        num_points = u_meas.shape[0]  # 63 points
        
        # each point: 2 constraints (9 items) --> P: 2nx9 matrix
        # minimize Ph = 0 subject to 2-norm(h) = 1
        P = np.zeros((2*num_points, 9))
        for i in range(num_points):
            P[2*i] = [X[i], Y[i], 1, 0, 0, 0, -u_meas[i]*X[i], -u_meas[i]*Y[i], -u_meas[i]]
            P[2*i+1] = [0, 0, 0, X[i], Y[i], 1, -v_meas[i]*X[i], -v_meas[i]*Y[i], -v_meas[i]]
        
        _, _, V_T = np.linalg.svd(P)
        h = V_T[-1, :]
        H = np.reshape(h, (3, 3))
        return H

    def getCameraIntrinsics(self, H):
        # part (iii): n images of the model plane, construct (2n, 6) V matrix
        num_images = len(H)
        # solve Vb = 0
        V = np.zeros((2*num_images, 6))
        for n in range(num_images):
            _H = H[n].T  # follow the notation of literature
            v_ij = lambda i, j: np.array([_H[i-1, 0]*_H[j-1, 0],
                                          _H[i-1, 0]*_H[j-1, 1]+_H[i-1, 1]*_H[j-1, 0],
                                          _H[i-1, 1]*_H[j-1, 1], 
                                          _H[i-1, 2]*_H[j-1, 0]+_H[i-1, 0]*_H[j-1, 2],
                                          _H[i-1, 2]*_H[j-1, 1]+_H[i-1, 1]*_H[j-1, 2], 
                                          _H[i-1, 2]*_H[j-1, 2]])
            V[2*n] = v_ij(1, 2)
            V[2*n+1] = v_ij(1, 1) - v_ij(2, 2)
            
        _, _, V_T = np.linalg.svd(V)
        b = V_T[-1, :]
        B11, B12, B22, B13, B23, B33 = b
        
        # extract intrinsic parameters
        v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
        _lambda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
        alpha = np.sqrt(_lambda / B11)
        beta = np.sqrt(_lambda*B11 / (B11*B22 - B12**2))
        gamma = -B12*(alpha**2)*beta / _lambda
        u0 = gamma*v0 / beta - B13*(alpha**2) / _lambda
        
        # sanity check
        # print("u0: {}".format(u0/self.w_pixels))
        # print("v0: {}".format(v0/self.h_pixels))
        # print("gamma << alpha: {}".format(abs(gamma)/alpha))
        
        # construct intrinsic matrix
        A = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
        
        return A

    def getExtrinsics(self, H, A):
        # part (iv)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lambda1 = 1. / np.linalg.norm(np.linalg.inv(A).dot(h1))
        lambda2 = 1. / np.linalg.norm(np.linalg.inv(A).dot(h2))
        # sanity check (lambda1 = lambda2 theoretically)
        _lambda = 0.5 * (lambda1 + lambda2) if abs(lambda1 - lambda2) > 1e-5 else lambda1
        
        r1 = _lambda * np.linalg.inv(A).dot(h1)
        r2 = _lambda * np.linalg.inv(A).dot(h2)
        r3 = np.cross(r1, r2)
        t = _lambda * np.linalg.inv(A).dot(h3)
        
        # initial R
        Q = np.hstack((r1.reshape(-1, 1), r2.reshape(-1, 1), r3.reshape(-1, 1)))
        U, _, V_T = np.linalg.svd(Q)
        R = np.matmul(U, V_T)
        
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):
        """
        Note: The transformation functions should only process one chessboard at a time!
        This means X, Y, Z, R, t should be individual arrays
        """
        # part (v)
        num_points = X.shape[0]
        
        P = np.array([X[0], Y[0], Z[0], 1.]).reshape(4, 1)
        for i in range(1, num_points):
            P = np.hstack((P, np.array([X[i], Y[i], Z[i], 1.]).reshape(4, 1)))
            
        assert P.shape == (4, 63)
        
        Rt = np.hstack((R, t.reshape(-1, 1))) # 3x4
        p = np.matmul(Rt, P) # 3x63
        p = np.divide(p, p[-1, :])
        
        x = p[0, :].tolist()
        y = p[1, :].tolist()
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):
        # part (v)
        num_points = X.shape[0]
        
        P = np.array([X[0], Y[0], Z[0], 1.]).reshape(4, 1)
        for i in range(1, num_points):
            P = np.hstack((P, np.array([X[i], Y[i], Z[i], 1.]).reshape(4, 1)))
            
        
        T = np.matmul(A, np.hstack((R, t.reshape(-1, 1)))) # 3x4
        p = np.matmul(T, P) # 3x63
        p = np.divide(p, p[-1, :])
        
        u = p[0, :].tolist()
        v = p[1, :].tolist()
        return u, v

    def transformWorld2NormImageDist(self, X, Y, R, t, k):
        # part (vi)
        x, y = self.transformWorld2NormImageUndist(X, Y, np.zeros_like(X), R, t)
        x, y = np.array(x), np.array(y)
        x_br = x + x * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        y_br = y + y * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        return x_br, y_br

    def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):
        # part (vi)
        u, v = self.transformWorld2PixImageUndist(X, Y, np.zeros_like(Z), R, t, A)
        u_0, v_0 = A[0, 2], A[1, 2]
        x, y = self.transformWorld2NormImageUndist(X, Y, np.zeros_like(Z), R, t)
        
        u, v = np.array(u), np.array(v)
        x, y = np.array(x), np.array(y)
        u_0, v_0 = np.array([u_0]*u.shape[0]), np.array([v_0]*v.shape[0])
        u_br = u + (u - u_0) * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        v_br = v + (v - v_0) * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        return u_br, v_br
