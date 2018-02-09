from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def tf_histogram_of_oriented_gradients(img_raw, x_kernel=[-1.0, 0.0, 1.0], y_kernel=[-1.0, 0.0, 1.0], pixels_in_cell=8, cells_in_block=1, n_angle_bins=9):
    ## COMPUTE GRADIENT MAGNITUDES/ORIENTATIONS
    img_raw = img_raw if len(img_raw.shape) == 4 else tf.expand_dims(img_raw, 0)                           # convert single image to batch of 1
    img = tf.to_float(img_raw)                                                                             # convert int pixel values to float
    x_kernel = tf.reshape(x_kernel, [3,1,1,1])                                                             # x kernel is a row matrix
    y_kernel = tf.reshape(y_kernel, [1,3,1,1])                                                             # y kernel is a column matrix
    x_grad_by_c = tf.nn.depthwise_conv2d(img, tf.tile(x_kernel, [1,1,3,1]), [1,1,1,1], "SAME")             # computing channel x/y gradients by convolution
    y_grad_by_c = tf.nn.depthwise_conv2d(img, tf.tile(y_kernel, [1,1,3,1]), [1,1,1,1], "SAME")             # (kernel tiled across 3 RGB channels)
    grad_mag_by_c = tf.sqrt(tf.square(x_grad_by_c) + tf.square(y_grad_by_c))                               # gradient magnitude
    grad_ang_by_c = tf.atan2(y_grad_by_c, x_grad_by_c)                                                     # gradient orientation
    grad_mag = tf.reduce_max(grad_mag_by_c, axis=-1)                                                       # select largest channel gradient
    grad_ang = tf.reduce_sum(grad_ang_by_c * tf.one_hot(tf.argmax(grad_mag_by_c, axis=-1), 3), axis=-1)    # select corresponding orientation

    ## GROUP VALUES INTO CELLS (8x8)
    p = pixels_in_cell
    grad_mag = tf.extract_image_patches(tf.expand_dims(grad_mag, -1), [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID")
    grad_ang = tf.extract_image_patches(tf.expand_dims(grad_ang, -1), [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID")

    ## COMPUTE CELL HISTOGRAMS
    bin_width = np.pi / n_angle_bins
    grad_ang = tf.mod(grad_ang, np.pi)    # unsigned gradients
    grad_ang_idx = grad_ang / bin_width
    lo_bin = tf.floor(grad_ang_idx)
    hi_bin = lo_bin + 1
    lo_weight = (hi_bin - grad_ang_idx)*grad_mag
    hi_weight = (grad_ang_idx - lo_bin)*grad_mag
    hi_bin = tf.mod(hi_bin, n_angle_bins)

    lo_bin = tf.to_int32(lo_bin)
    hi_bin = tf.to_int32(hi_bin)
    cell_hogs = tf.reduce_sum(tf.one_hot(lo_bin, n_angle_bins)*tf.expand_dims(lo_weight, -1) +
                              tf.one_hot(hi_bin, n_angle_bins)*tf.expand_dims(hi_weight, -1), -2)

    ## ASSEMBLE AND NORMALIZE BLOCK HISTOGRAM VECTORS
    unnormalized_hog = tf.extract_image_patches(cell_hogs, [1, cells_in_block, cells_in_block, 1], [1, 1, 1, 1], [1, 1, 1, 1], "VALID")
    hog_descriptor = tf.reshape(tf.nn.l2_normalize(unnormalized_hog, -1), [unnormalized_hog.shape[0],-1])

    return cell_hogs, hog_descriptor

def hog_descriptor(img_raw, x_kernel=[-1.0, 0.0, 1.0], y_kernel=[-1.0, 0.0, 1.0], pixels_in_cell=8, cells_in_block=1, n_angle_bins=9):
    return tf_histogram_of_oriented_gradients(img_raw, x_kernel, y_kernel, pixels_in_cell, cells_in_block, n_angle_bins)[1]

def plot_cell_hogs(cell_hogs, pixels_in_cell=8, n_angle_bins=9, color="yellow"):
    th = np.arange(9)*np.pi/n_angle_bins
    cth = np.cos(th)
    sth = np.sin(th)
    cell_hogs_normalized = np.squeeze(cell_hogs / np.linalg.norm(cell_hogs, axis=-1, keepdims=True))
    X = []
    Y = []
    for r in range(cell_hogs_normalized.shape[0]):
        for c in range(cell_hogs_normalized.shape[1]):
            xc = pixels_in_cell*(c + 0.5)
            yc = pixels_in_cell*(r + 0.5)
            for ct, st, l in zip(cth, sth, cell_hogs_normalized[r,c]):
                X.extend([xc - ct*l*pixels_in_cell/2.5, xc + ct*l*pixels_in_cell/2.5, None])
                Y.extend([yc + st*l*pixels_in_cell/2.5, yc - st*l*pixels_in_cell/2.5, None])    # something about images being plotted top to bottom? ... or HOG bug
    plt.plot(X, Y, color=color)
