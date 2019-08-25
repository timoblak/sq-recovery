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

def ins_outs(meshgrid, p):
    rot = quat2mat(p[8:])
    # Rotate translation
    t = quat_product(quat_product(q, p[5:8]+[0]), quat_conjungate(q))
    # Rotate coordinate system
    m = np.einsum('ij,jabc->iabc', rot, meshgrid)
    # Calculate
    return np.power(np.power((m[0] - t[0]) / p[0], 2 / p[4]) +
                    np.power((m[1] - t[1]) / p[1], 2 / p[4]), p[4] / p[3]) + \
           np.power((m[2] - t[2]) / p[2], 2 / p[3])


size = 64
rng = list(np.arange(-size//2, size//2, 1))


e = [1, 1]
a = [32, 16, 16]
q = [0, 0, 0.7071068, 0.7071068]
c = [20, 0, 0]
params = a + e + c + q

x, y, z = tf.meshgrid(rng, rng, rng, indexing="ij")
x = tf.cast(x, dtype=tf.float32)
y = tf.cast(y, dtype=tf.float32)
z = tf.cast(z, dtype=tf.float32)
tf.pow
def ins_outs(p):
    return M.pow(M.pow(M.divide(x, p[0]), M.divide(2.0, p[4])) +
                 M.pow(M.divide(y, p[1]), M.divide(2.0, p[4])), M.divide(p[4], p[3])) + \
           M.pow(M.divide(z, p[2]), M.divide(2.0, p[3]))


sess = tf.Session()
params_tf = tf.Variable(params, tf.float32)

with sess.as_default():   # or `with sess:` to close on exit
    t = time()
    a = ins_outs(params_tf).eval()
    print(time()-t)


    print(a.max())
    print(a.min())
    print(a.shape)
    print("-----------------")
    #assert (a == m[0]).all()


exit()


md = np.log(md)
#print(mlog)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")
yg = ax.scatter(m[0][:32], m[1][:32], m[2][:32], c=md[:32].ravel(), marker='o', alpha=0.5)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()