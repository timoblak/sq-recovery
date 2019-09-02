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
rng = list(np.arange(-size//2, size//2, 1).astype(np.float32))
tf.enable_eager_execution()

p1 = [32.0, 16.0, 16.0,    0.1, 1.0,    5.0, 0.0, 0.0,    0.0, 0.0, 0.7071068, 0.7071068]
p2 = [32.0, 16.0, 16.0,    1, 1.0,    5.0, 0.0, 0.0,    0.0, 0.0, 0.7071068, 0.7071068]
p3 = [11.0, 20, 16.0,    1, 1.0,    5.0, 0.0, 0.0,    0.0, 0.0, 0.7071068, 0.7071068]
p4 = [32.0, 16.0, 16.0,    1, 1.0,    5.0, 0.0, 0.0,    0.0, 0.0, 0, 1]
params = np.array([p1, p2, p3, p4], dtype=np.float32)

xyz_list = tf.meshgrid(rng, rng, rng, indexing="ij")
xyz = tf.stack(xyz_list)
xy = tf.stack([tf.stack(xyz_list, axis=-1)]*4)


xyz2 = tf.stack(xyz_list, axis=-1)
print(xy.shape)
#x = tf.cast(x, dtype=tf.float32)
#y = tf.cast(y, dtype=tf.float32)
#z = tf.cast(z, dtype=tf.float32)

two = tf.constant(2, dtype=tf.float32)
@tf.function
def ins_outs_1(p):
    rot = rmat.from_quaternion(p[8:])
    # Rotate translation
    t = quat.rotate(p[5:8], p[8:])
    # Rotate coordinate system
    m = tf.einsum('ij,jabc->iabc', rot, xyz)
    # Calculate
    return M.pow(M.pow(M.divide(m[0] - t[0], p[0]), M.divide(two, p[4])) +
                 M.pow(M.divide(m[1] - t[1], p[1]), M.divide(two, p[4])), M.divide(p[4], p[3])) + \
           M.pow(M.divide(m[2] - t[2], p[2]), M.divide(two, p[3]))

@tf.function
def ins_outs_2(p):
    # Rotate translation
    results = []
    for i in range(BATCH):
        t = quat.rotate(p[i, 5:8], p[i, 8:])
        m = quat.rotate(xyz2, p[i, 8:])
        print(m.shape)
        m = tf.unstack(m, axis=-1)

        # Rotate coordinate system
        # Calculate
        results.append(M.pow(M.pow(M.divide(m[0] - t[0], p[0]), M.divide(two, p[4])) +
                 M.pow(M.divide(m[1] - t[1], p[1]), M.divide(two, p[4])), M.divide(p[4], p[3])) + \
           M.pow(M.divide(m[2] - t[2], p[2]), M.divide(two, p[3])))
    return tf.stack(results)


a = ins_outs_2(params)

t = time()
b = ins_outs_2(params)
print(time()-t)

md = a[0].numpy()
MESH = xyz.numpy()
exit()
print("-----------------")
#assert (a == m[0]).all()


#md = np.log(md)
#print(md.max())
#print(md.min())
#print(mlog)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")

disp = (md < 1)
#disp = (md >= 0)
#yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=m_rot[0][:32].ravel(), marker='o', alpha=0.5)
#yg = ax.scatter(MESH[0][:32], MESH[1][:32], MESH[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)
yg = ax.scatter(MESH[0][disp], MESH[1][disp], MESH[2][disp], c=md[disp].ravel(), marker='o', alpha=0.3)
#yg = ax.scatter(MESH[0], MESH[1], MESH[2], c=md.ravel(), marker='o', alpha=0.5)
ax.set(xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()