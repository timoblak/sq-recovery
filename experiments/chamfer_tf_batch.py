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
BATCH = 2
rng = list(np.arange(-size // 2, size // 2, 1).astype(np.float64))
tf.compat.v1.enable_eager_execution()

#p1 = [0.987007678, 1.01116645, 0.397809595, 0.813163, 0.373285, 0.429682, 0.537141502, 0.640188515, -0.412896, 0.821542, 0.327043, 0.21824]
p1 = [0.2, 0.6, 0.2, 1, 1, 0, 0, 0, 0, 0, 0.8509035, 0.525322]
#p2 = [1, 0.0883607417, 0, 0.01, 0.855431676, 0, 0.646699309, 0.932791829, 0.280940086, 0, 1, 0.558223188]
p2 = [0.2, 0.6, 0.2, 1, 1, 1, 1, 1, 0, 0, 0, 1]
params = np.array([p1, p2], dtype=np.float64)

xyz_list = tf.meshgrid(rng, rng, rng, indexing="ij")
XYZ_GRID = tf.stack(xyz_list)
XYZ_POINTS = tf.stack(xyz_list, axis=-1)
TWO = tf.constant(2, dtype=tf.float64)


def preprocess_sq(p):
    a, e, t, q = tf.split(p, (3, 2, 3, 4), axis=-1)
    a = M.multiply(a, tf.constant(12.5, dtype=tf.float64)) + tf.constant(6.25, dtype=tf.float64)
    t = M.multiply(t, tf.constant(64., dtype=tf.float64)) + tf.constant(-32., dtype=tf.float64)
    return tf.concat([a, e, t, q], axis=-1)


def io1(batch, preprocess=True):
    @tf.function
    def ins_outs_1(p):
        if preprocess:
            p = preprocess_sq(p)
        results = []
        for i in range(batch):
            a, e, t, q = tf.split(p[i], (3, 2, 3, 4), axis=-1)
            # Create a rotation matrix from quaternion

            rot = rmat.from_quaternion(q)
            # Rotate translation vector using quaternion
            #t = quat.rotate(t, q)
            # Rotate coordinate system using rotation matrix

            m = tf.einsum('ij,jabc->iabc', rot, XYZ_GRID)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = M.divide(m[0] - t[0], a[0])
            y_translated = M.divide(m[1] - t[1], a[1])
            z_translated = M.divide(m[2] - t[2], a[2])
            # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
            A = M.pow(M.abs(x_translated), M.divide(TWO, e[1]))
            B = M.pow(M.abs(y_translated), M.divide(TWO, e[1]))
            C = M.pow(M.abs(z_translated), M.divide(TWO, e[0]))
            D = A + B
            E = M.pow(M.abs(D), M.divide(e[1], e[0]))
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
            a, e, t, q = tf.split(p[i], (3, 2, 3, 4), axis=-1)
            t = quat.rotate(t, q)
            m = quat.rotate(XYZ_POINTS, q)
            m = tf.unstack(m, axis=-1)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = M.divide(m[0] - t[0], a[0])
            y_translated = M.divide(m[1] - t[1], a[1])
            z_translated = M.divide(m[2] - t[2], a[2])
            # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
            A = M.pow(M.abs(x_translated), M.divide(TWO, e[1]))
            A = tf.Print(A, [tf.reduce_any(tf.math.is_inf(A))])
            B = M.pow(M.abs(y_translated), M.divide(TWO, e[1]))
            B = tf.Print(B, [tf.reduce_any(tf.math.is_inf(B))])
            C = M.pow(M.abs(z_translated), M.divide(TWO, e[0]))
            C = tf.Print(C, [tf.reduce_any(tf.math.is_inf(C))])
            D = A + B
            E = M.pow(M.abs(D), M.divide(e[1], e[0]))
            inout = E + C

            results.append(inout)
        return tf.stack(results)
    return ins_outs_2


def chamfer_loss(ts1, ts2):
    ts1 = tf.clip_by_value(tf.log(ts1), clip_value_min=0., clip_value_max=10000000)
    ts2 = tf.clip_by_value(tf.log(ts2), clip_value_min=0., clip_value_max=10000000)

    ab_diff = ts1 - ts2

    return tf.reduce_mean(tf.abs(ab_diff)).numpy(), ab_diff.numpy()

loss = io2(BATCH, True)

t = time()
b = loss(params)
print(time() - t)

t = time()
b = loss(params)
print(time() - t)
print(b.shape)
#md1 = np.log(b[0].numpy())
#md2 = np.log(b[1].numpy())

final_loss, final_diff = chamfer_loss(b[0], b[1])
print(final_diff)
print("Final loss is: " + str(final_loss))

md1 = b[0].numpy()
md2 = b[1].numpy()
md3 = final_diff
MESH = XYZ_GRID.numpy()
print("-----------------")
# assert (a == m[0]).all()


# md = np.log(md)
# print(md.max())
# print(md.min())
# print(mlog)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax.set_aspect("auto")
#print(md)
disp1 = (md1 <= 1)
disp2 = (md2 <= 1)
disp3 = (md3 < np.inf)

# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp1], MESH[1][disp1], MESH[2][disp1], c=md1[disp1].ravel(), marker='o', alpha=0.3)
# yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax = fig.add_subplot(132, projection='3d')
ax.set_aspect("auto")
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp2], MESH[1][disp2], MESH[2][disp2], c=md2[disp2].ravel(), marker='o', alpha=0.3)
# yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


ax = fig.add_subplot(133, projection='3d')
ax.set_aspect("auto")
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp3], MESH[1][disp3], MESH[2][disp3], c=md3[disp3].ravel(), marker='o', alpha=0.3)
# yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
