from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil


def generate_data(P='svm', a=1, b=8, c=-4, d=5, sigma=2.5, N=1000):
    if P == 'svm':
        x = np.random.randn(N,2)
        w = np.array([-.00, 0.01, 0.1, -0.04, 0.09, 0.02])
        features = np.hstack([np.ones([N,1]), x, x**2, x[:,:1]*x[:,1:2]])
        f = np.dot(features, w)
        labels = 2*((f + np.random.randn(N)*0.02)>0) - 1
        y = np.expand_dims(labels, axis=-1)
        return x, y
    if P == 'svm_eg':
        p1 = np.array([[a,b]])
        p2 = np.array([[b,a]])
        c1 = np.random.randn(N, 2)*sigma + p1
        c2 = np.random.randn(N, 2)*sigma + p2
        labels = np.concatenate([np.ones([N]), -np.ones([N])], axis=0)
        perm = np.random.choice(range(2*N), 2*N, replace=False)
        return np.concatenate([c1, c2], axis=0)[perm,:], np.expand_dims(labels, axis=-1)[perm,:]
    elif P == 'LS_linear':
        x = np.random.rand(N,1)*d
        y = a*x - b + np.random.randn(N,1)*sigma
        return x, y
    elif P == 'LS_quad':
        x = np.random.rand(N,1)*d
        y = a*x**2 + b*x + c + np.random.randn(N,1)*sigma
        return x, y
    else:
        return
