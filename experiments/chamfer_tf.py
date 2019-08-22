import numpy as np
import cv2
from time import time
from matplotlib import pyplot as plt
from pylab import cm
import keras.backend as K
import tensorflow as tf
from tensorflow import math as M


def ins_outs_np(m):
    return np.power(np.power(m[0] / a1, 2 / e2) + np.power(m[1] / a2, 2 / e2), e2 / e1) + np.power(m[2] / a3, 2 / e1)


size = 64
rng = list(range(-size//2, size//2))
x, y, z = tf.meshgrid(rng, rng, rng, indexing="ij")

e1, e2 = 1, 1
a1, a2, a3 = 12, 20, 5


def ins_outs():
    return M.pow(M.pow(M.divide(x, a1), 2 / e2) +
                 M.pow(M.divide(y, a2), 2 / e2), e2 / e1) + \
           M.pow(M.divide(z, a3), 2 / e1)


sess = tf.Session()
with sess.as_default():   # or `with sess:` to close on exit
    t = time()
    a = ins_outs().eval()
    print(time()-t)

    t = time()
    a = ins_outs().eval()
    print(time() - t)

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