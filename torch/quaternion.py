import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

def rotate(point, quaternion):
    point = F.pad(point, [0, 1])
    point = multiply(quaternion, point)

    point = multiply(point, conjugate(quaternion))
    xyz, _ = torch.split(point, (3, 1), dim=-1)
    return xyz


def conjugate(quaternion):
    xyz, w = torch.split(quaternion, (3, 1), dim=-1)
    return torch.cat((-xyz, w), dim=-1)


def multiply(quaternion1, quaternion2):
    x1, y1, z1, w1 = torch.split(quaternion1, (1, 1, 1, 1), dim=-1)
    x2, y2, z2, w2 = torch.split(quaternion2, (1, 1, 1, 1), dim=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return torch.cat((x, y, z, w), dim=-1)


def mat_from_quaternion(quaternion):
    # supports 1 quaternion of shape (4)
    x, y, z, w = torch.split(quaternion, (1, 1, 1, 1), dim=-1)

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = torch.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                          txy + twz, 1.0 - (txx + tzz), tyz - twx,
                          txz - twy, tyz + twx, 1.0 - (txx + tyy)), dim=-1)

    output_shape = (1, 3, 3)
    return torch.reshape(matrix, shape=output_shape)


def test_quat_loss(ytrue, ypred, reduce=True):
    theta = torch.tensor(1) - 2 * torch.abs(torch.tensor(0.5) - torch.pow(torch.einsum('ji,ji->j', ytrue, ypred), 2))
    if reduce:
        return torch.mean(theta)
    return theta


if __name__ == "__main__":
    pt = np.array([1, 1, 1])
    pt = torch.tensor(pt, dtype=torch.float64, device="cuda:0")
    q = np.array([[-0.3438,  0.6873,  0.6210,  0.1540],
        [-0.1391, -0.9361, -0.2694, -0.1783],
        [ 0.7519,  0.1729, -0.6150,  0.1626],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [ 0.1096, -0.9567, -0.2165, -0.1609]])
    q2 = np.array([[-0.5187,  0.7938, -0.1713, -0.2675],
        [ 0.4670, -0.4831, -0.7378, -0.0651],
        [-0.4694,  0.3033,  0.1275, -0.8194],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0.7071068, 0.7071068],
        [ 0.7743,  0.2355, -0.4961, -0.3146]])
    q = torch.tensor(q, dtype=torch.float64, device="cuda:0")
    q2 = torch.tensor(q2, dtype=torch.float64, device="cuda:0")
    print(test_quat_loss(q2, q, reduce=False))
