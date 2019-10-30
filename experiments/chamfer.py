import numpy as np
import cv2
from time import time
from matplotlib import pyplot as plt
from pylab import cm
import keras.backend as K
import tensorflow as tf

def preprocess(p):
    # Preprocess superquadrics to fit scale of the 3D space.
    a, e, t, q = p[:3], p[3:5], p[5:8], p[8:]
    a = a * 12.5 + 6.25
    t = t * 64. + -32.
    return np.concatenate([a, e, t, q])

def ins_outs(meshgrid, p):
    p = preprocess(p)
    print(p)
    rot = quat2mat(p[8:])
    # Rotate translation
    t = quat_product(quat_product(p[8:], np.concatenate([p[5:8], [0]])), quat_conjungate(p[8:]))
    # Rotate coordinate system
    m = np.einsum('ij,jabc->iabc', rot, meshgrid)
    # Calculate
    t0 = np.abs(m[0] - t[0]) / p[0]
    t1 = np.abs(m[1] - t[1]) / p[1]
    t2 = np.abs(m[2] - t[2]) / p[2]

    p1 = np.power(t0, 2 / p[4]) * np.sign(t0)
    p2 = np.power(t1, 2 / p[4])

    p3 = np.power(t2, 2 / p[3]) * np.sign(t2)
    p4 = np.power(np.abs(p1 + p2), p[4] / p[3]) * np.sign(p1 + p2)
    return p4 + p3


def quat2mat(q):
    u_q = q / np.sqrt(np.square(q).sum())
    x, y, z, w = u_q
    M = [[1 - 2 * (y**2 + z**2), 2*x*y - 2*w*z, 2*x*z + 2*w*y],
         [2*x*y + 2*w*z, 1 - 2*(x**2 + z**2), 2*y*z - 2*w*x],
         [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*(x**2 + y**2)]]
    return np.array(M)

def quat_conjungate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_product(q1, q2):
    print(q1, q2)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return [x, y, z, w]

size = 16

MESH = np.mgrid[-size/2:size/2, -size/2:size/2, -size/2:size/2].astype(np.float32)
print(MESH)
print(MESH.shape)
pm = np.stack(MESH, axis=-1)
#exit()
#q = [0, 0, 0, 1]

#params = [32.0, 16.0, 16.0, 0.541294, 0.574811, 0, 0, 0.0, 0, 0, 0.7071068, 0.7071068 ]
params = [0.66,       0.86,       0.1,        0.098257,   0.360296 ,  0.4093721,
 0.5807533,  0.43790472 , 0, 0, 0, 1]

t = time()


t = time()
md = ins_outs(MESH, np.array(params))
print(time()-t)
print(md.shape)
print(MESH.shape)
#print(md)


#md = np.log(md)
#print(md.max())
#print(md.min())
#print(mlog)

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