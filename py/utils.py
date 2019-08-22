import numpy as np
import cv2

def quat2mat(q):
    u_q = q / np.sqrt(np.square(q).sum())
    x, y, z, w = u_q
    M = [[1 - 2 * (y**2 + z**2), 2*x*y - 2*w*z, 2*x*z + 2*w*y],
         [2*x*y + 2*w*z, 1 - 2*(x**2 + z**2), 2*y*z - 2*w*x],
         [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*(x**2 + y**2)]]
    return np.array(M)


def quat_conjungate(q):
    q[:3] = -q[:3]
    return q

def quat_product(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return [x, y, z, w]


def randquat():
    u = np.random.uniform(0, 1, (3,))
    q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                  np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                  np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                  np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
    return q


def mat2quat(M):
    qr = np.sqrt(1+M[0][0]+M[1][1]+M[2][2])/2
    qi = (M[2][1] - M[1][2])/(4*qr)
    qj = (M[0][2] - M[2][0])/(4*qr)
    qk = (M[1][0] - M[0][1])/(4*qr)
    return qr, qi, qj, qk


def get_command(scanner_loc, fn, params):
    command = scanner_loc + "scanner " + fn + " "
    dims = params[:3]
    command += ("%f " * 3) % tuple(dims)
    shape = params[3:5]
    command += ("%f " * 2) % tuple(shape)
    pos_new = params[5:8]
    command += "%f %f %f " % tuple(pos_new.ravel())
    M = params[8:]
    command += ("%f " * 9) % tuple(M.ravel())
    command += "\n"
    return command


def to_pc(img, name):
    fimg = cv2.flip(img, 0)

    nz = np.nonzero(fimg)
    ls = np.transpose((nz[1], nz[0], fimg[np.nonzero(fimg)]))

    text = ""
    for p in ls:
        text += str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"

    with open("ales_sq/point_clouds/" + name + "_points.txt", "w") as handle:
        handle.write(text)