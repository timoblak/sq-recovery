import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import cv2
from time import time
from matplotlib import pyplot as plt
import keras.backend as K
import tensorflow as tf
from tensorflow import math as M
from tensorflow_graphics.geometry.transformation import quaternion as quat
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as rmat

size = 64
rng = list(np.arange(-size // 2, size // 2, 1).astype(np.float32))
tf.enable_eager_execution()

e = [0.1, 0.9]
a = [32.0, 16.0, 16.0]
q = [0, 0, 1,1]
c = [0, -32, 0]
params = np.concatenate([a, e, c, q]).astype(np.float32)
params = np.array([0.74,       0.5    ,    0.9 ,       0.804617,   0.116392 ,  0.60326314, 0.47432035, 0.3563426, 0, 0, 0, 1], dtype=np.float32)

xyz_list = tf.meshgrid(rng, rng, rng, indexing="ij")
xyz = tf.stack(xyz_list)
xyz2 = tf.stack(xyz_list, axis=-1)
print(xyz2.shape)
# x = tf.cast(x, dtype=tf.float32)
# y = tf.cast(y, dtype=tf.float32)
# z = tf.cast(z, dtype=tf.float32)

two = tf.constant(2, dtype=tf.float32)

def preprocess_sq(p):
    a, e, t, q = tf.split(p, (3, 2, 3, 4), axis=-1)
    a = M.multiply(a, tf.constant(12.5)) + tf.constant(6.25)
    t = M.multiply(t, tf.constant(64.)) + tf.constant(-32.)
    q = quat.normalize(q)
    return tf.concat([a, e, t, q], axis=-1)

@tf.function
def ins_outs_1(p):

    p = preprocess_sq(p)

    # Create a rotation matrix from the quaternion
    rot = rmat.from_quaternion(p[8:])
    # Rotate translation using quaternion
    t = quat.rotate(p[5:8], p[8:])
    # Rotate coordinate system using rotation matrix
    print(xyz.shape, rot.shape)
    m = tf.einsum('ij,jabc->iabc', rot, xyz)

    # #### Calculate inside-outside equation ####
    # First the inner parts of equation (translate + divide with axis sizes)
    x_translated = M.divide(m[0] - t[0], p[0])
    y_translated = M.divide(m[1] - t[1], p[1])
    z_translated = M.divide(m[2] - t[2], p[2])
    # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
    A = M.pow(M.abs(x_translated), M.divide(two, p[4]))
    B = M.pow(M.abs(y_translated), M.divide(two, p[4]))
    C = M.pow(M.abs(z_translated), M.divide(two, p[3]))
    D = A + B
    E = M.pow(M.abs(D), M.divide(p[4], p[3]))
    inout = E + C
    return inout


@tf.function
def ins_outs_2(p):
    p = preprocess_sq(p)

    # Rotate translation
    _, t, q = tf.split(p, (5, 3, 4), axis=-1)
    t = quat.rotate(t, q)
    m = quat.rotate(xyz2, q)
    m = tf.unstack(m, axis=-1)
    # Rotate coordinate system
    # Calculate
    # #### Calculate inside-outside equation ####
    # First the inner parts of equation (translate + divide with axis sizes)
    x_translated = M.divide(m[0] - t[0], p[0])
    y_translated = M.divide(m[1] - t[1], p[1])
    z_translated = M.divide(m[2] - t[2], p[2])
    # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
    A = M.pow(M.abs(x_translated), M.divide(two, p[4]))
    B = M.pow(M.abs(y_translated), M.divide(two, p[4]))
    C = M.pow(M.abs(z_translated), M.divide(two, p[3]))
    D = A + B
    E = M.pow(M.abs(D), M.divide(p[4], p[3]))
    inout = E + C
    return inout


a = ins_outs_1(params)

t = time()
b = ins_outs_1(params)
print(time() - t)

md = a.numpy()
print(md)
MESH = xyz.numpy()

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

disp = (md < 1)
# disp = (md >= 0)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
# yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp], MESH[1][disp], MESH[2][disp], c=md[disp].ravel(), marker='o', alpha=0.3)
# yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
