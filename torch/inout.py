import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt


device = torch.device("cuda:0")
render_size = 32.
render_type = np.float64
eps = 1e-8

#step = 1 / render_size
#range = torch.tensor(np.arange(-self.render_size // 2, self.render_size // 2+1, 1).astype(self.render_type))
#t_range = torch.tensor(np.arange(0, 1+step, step).astype(render_type))
#xyz_list = torch.meshgrid([t_range, t_range, t_range])
#xyz = torch.stack(xyz_list).to(device)


def preprocess_sq(p):
    a, e, t = torch.split(p, (3, 2, 3))
    a = (a * 0.196) + 0.098
    # e = torch.clamp(e, 0.1, 1)
    # t = (t * 64.) + -32.
    return torch.cat([a, e, t], dim=-1)

def inout(p):
    p = p.double()
    results = []
    for i in range(p.shape[0]):
        perameters = preprocess_sq(p[i])
        a, e, t = torch.split(perameters, (3, 2, 3))

        # Create a rotation matrix from quaternion

        # rot = rmat.from_quaternion(q)

        # Rotate translation vector using quaternion
        # t = quat.rotate(t, q)

        # Rotate coordinate system using rotation matrix

        # m = tf.einsum('ij,jabc->iabc', rot, XYZ_GRID)

        # TWO = tf.cast(TWO, tf.float64)

        # #### Calculate inside-outside equation ####
        # First the inner parts of equation (translate + divide with axis sizes)
        grid = np.arange(-render_size, render_size + 1)
        x_translated = (torch.tensor(grid) - t[0]) / a[0]
        #y_translated = (xyz[1] - t[1]) / a[1]
        #z_translated = (self.xyz[2] - t[2]) / a[2]
        # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
        print(x_translated)
        A = torch.pow(torch.abs(x_translated), (2 / e[0]))
        B = (2 / e[0]) * torch.log(torch.abs(x_translated))
        #B = torch.pow(torch.abs(y_translated), (2 / e[1]))  # * torch.sign(y_translated)
        #C = torch.pow(torch.abs(z_translated), (2 / e[0]))  # * torch.sign(z_translated)
        #D = A + B
        #E = torch.pow(torch.abs(D), (e[1] / e[0]))
        #inout = E + C

        results.append(A)
        results.append(B)
    return results


data = [
    [10., 5., 5., 1., 1., 0., 0., 0.]
]
y = inout(torch.tensor(data))
print(y)
x = np.arange(-render_size, render_size+1)
plt.plot(x, y[0])
plt.plot(x, y[1]+100)
plt.plot(x, len(x) * [1])

plt.axvline(x=1)
plt.axvline(x=-1)
plt.xlim([-5, 5])
plt.ylim([-0, 3])
plt.show()


