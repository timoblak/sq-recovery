import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import cv2
from time import time
from matplotlib import pyplot as plt
from pylab import cm
import keras.backend as K
import tensorflow as tf
from tensorflow import math as M
from tensorflow_graphics.geometry.transformation import quaternion as quat
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as rmat

size = 64
BATCH = 4
rng = list(np.arange(-size // 2, size // 2, 1).astype(np.float32))
tf.compat.v1.enable_eager_execution()

p1 = [0.8267967, 0.8670515, 0.60345936, 0.541294, 0.574811, 0.65267265, 0.62242144, 0.35946697, -0.20178, -0.243733, 0.789973, -0.525188]
#p1 = [16.5, 17, 13.75, 0.541294, 0.574811, 9.6, 8, -9.6,  0.20178, -0.243733, 0.789973, -0.525188]
#p1 = [16.5, 17, 13.75, 0.541294, 0.574811, 9.6, 8, -9.6,  0,0,0,1]

p2 = [16.5, 17, 13.75, 0.541294, 0.574811, 9.6, 8, -9.6,  0.20178, -0.243733, 0.789973, -0.525188]
p3 = [0.4729304, 0.78276294, 0.58753353, 0.319357, 0.153823, 0.42050555,
      0.5497313, 0.5376073, 0.38441, 0.462276, - 0.430949, 0.672914]
p4 = [0.1976376, 0.67498684, 0.07473186, 0.760474, 0.81913, 0.5578802,
      0.6376708, 0.6082252, 0.604945, 0.406145, 0.39579, - 0.558962]
params = np.array([p1, p2, p3, p4], dtype=np.float32)

xyz_list = tf.meshgrid(rng, rng, rng, indexing="ij")
xyz_grid = tf.stack(xyz_list)
xyz_points = tf.stack(xyz_list, axis=-1)
two = tf.constant(2, dtype=tf.float32)


def preprocess_sq(p):
    a, e, t, q = tf.split(p, (3, 2, 3, 4), axis=-1)
    a = M.multiply(a, tf.constant(12.5)) + tf.constant(6.25)
    t = M.multiply(t, tf.constant(64.)) + tf.constant(-32.)
    return tf.concat([a, e, t, q], axis=-1)


def io1(batch, preprocess=True):
    @tf.function
    def ins_outs_1(p):
        if preprocess:
            p = preprocess_sq(p)
        results = []
        for i in range(batch):
            rot = rmat.from_quaternion(p[i, 8:])
            # Rotate translation
            t = quat.rotate(p[i, 5:8], p[i, 8:])
            # Rotate coordinate system
            m = tf.einsum('ij,jabc->iabc', rot, xyz_grid)
            # Calculate
            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = M.divide(m[0] - t[0], p[i, 0])
            y_translated = M.divide(m[1] - t[1], p[i, 1])
            z_translated = M.divide(m[2] - t[2], p[i, 2])
            # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
            A = M.pow(M.abs(x_translated), M.divide(two, p[i, 4]))
            B = M.pow(M.abs(y_translated), M.divide(two, p[i, 4]))
            C = M.pow(M.abs(z_translated), M.divide(two, p[i, 3]))
            D = A + B
            E = M.pow(M.abs(D), M.divide(p[i, 4], p[i, 3]))
            inout = E + C

            results.append(inout)
        return tf.stack(results)
    return ins_outs_1


def io2(batch, preprocess=True):
    @tf.function
    def ins_outs_2(p):
        # Rotate translation
        if preprocess:
            p = preprocess_sq(p)
        results = []
        for i in range(batch):
            _, t, q = tf.split(p[i], (5, 3, 4), axis=-1)
            t = quat.rotate(t, q)
            m = quat.rotate(xyz_points, q)
            m = tf.unstack(m, axis=-1)

            # Rotate coordinate system
            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = M.divide(m[0] - t[0], p[i, 0])
            y_translated = M.divide(m[1] - t[1], p[i, 1])
            z_translated = M.divide(m[2] - t[2], p[i, 2])
            # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
            A = M.pow(M.abs(x_translated), M.divide(two, p[i, 4]))
            B = M.pow(M.abs(y_translated), M.divide(two, p[i, 4]))
            C = M.pow(M.abs(z_translated), M.divide(two, p[i, 3]))
            D = A + B
            E = M.pow(M.abs(D), M.divide(p[i, 4], p[i, 3]))
            inout = E + C

            results.append(inout)
        return tf.stack(results)
    return ins_outs_2


loss = io2(4, True)

t = time()
b = loss(params)
print(time() - t)

t = time()
b = loss(params)
print(time() - t)

md = b[0].numpy()
MESH = xyz_grid.numpy()
print("-----------------")
# assert (a == m[0]).all()


# md = np.log(md)
# print(md.max())
# print(md.min())
# print(mlog)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")
print(md)
disp = (md < 1)
#disp = (md >= 0)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp], MESH[1][disp], MESH[2][disp], c=md[disp].ravel(), marker='o', alpha=0.3)
# yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
