import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow import math as M
from tensorflow_graphics.geometry.transformation import quaternion as quat
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as rmat


size = 64
rng = list(np.arange(-size//2, size//2, 1).astype(np.float32))
xyz_list = tf.meshgrid(rng, rng, rng, indexing="ij")
# 3*64*64*64 grid, representing the coordinate system for each axis inside 3D space
XYZ_GRID = tf.stack(xyz_list)
# 64*64*64*3 points, representing the coordinates of individual voxels inside the 3D space
XYZ_POINTS = tf.stack(xyz_list, axis=-1)
TWO = tf.constant(2, dtype=tf.float32)


def preprocess_sq(p):
    # Preprocess superquadrics to fit scale of the 3D space.
    a, e, t, q = tf.split(p, (3, 2, 3, 4), axis=-1)
    a = M.multiply(a, tf.constant(12.5)) + tf.constant(6.25)
    t = M.multiply(t, tf.constant(64.)) + tf.constant(-32.)
    return tf.concat([a, e, t, q], axis=-1)

@tf.function
def _ins_outs_1(p, batch):
    p = preprocess_sq(p)
    results = []
    for i in range(batch):
        a, e, t, q = tf.split(p[i], (3, 2, 3, 4), axis=-1)
        # Create a rotation matrix from quaternion
        rot = rmat.from_quaternion(q)
        # Rotate translation vector using quaternion
        t = quat.rotate(t, q)
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

@tf.function
def _ins_outs_2(p, batch):
    # Rotate translation
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
        B = M.pow(M.abs(y_translated), M.divide(TWO, e[1]))
        C = M.pow(M.abs(z_translated), M.divide(TWO, e[0]))
        D = A + B
        E = M.pow(M.abs(D), M.divide(e[1], e[0]))
        inout = E + C

        results.append(inout)
    return tf.stack(results)


def quaternion_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))


def quaternion_loss_np(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1))


def chamfer_loss(y_true, y_pred):
    batch = 1
    #y_pred = tf.Print(y_pred, [y_pred], " B: ", )
    #y_true = tf.Print(y_true, [y_true], " A: ", )

    a = _ins_outs_1(y_true, batch)
    b = _ins_outs_1(y_pred, batch)
    ab_diff = a - b

    ab_diff_sq = tf.square(ab_diff)

    return tf.reduce_mean(ab_diff_sq)