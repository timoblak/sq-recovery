import os
import numpy as np
import torch
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from time import sleep

def norm_img(img):
    img -= img.min()
    return img / img.max()


def quat2mat(q):
    # Transforms a quaternion into a rotation matrix
    u_q = q / np.sqrt(np.square(q).sum())
    x, y, z, w = u_q
    M = [[1 - 2 * (y**2 + z**2), 2*x*y - 2*w*z, 2*x*z + 2*w*y],
         [2*x*y + 2*w*z, 1 - 2*(x**2 + z**2), 2*y*z - 2*w*x],
         [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*(x**2 + y**2)]]
    return np.array(M)


def get_command(scanner_loc, fn, params):
    # Creates os command for the renderer
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


def save_model(path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_model(path, model, optimizer):
    print("Loading model: " + path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def save_compare_images(params_true, params_pred):
    for i, (true, pred) in enumerate(zip(params_true, params_pred)):
        M = quat2mat(true[-4:])
        params = np.concatenate((true[:3] * 255., true[3:5], true[5:8] * 255, M.ravel()))
        command = get_command("../", "examples/" + str(i) + "_true.bmp", params)
        os.system(command)
        M = quat2mat(pred[-4:])

        params = np.concatenate((np.clip(pred[:3], 0.05, 1) * 255., np.clip(pred[3:5], 0.1, 1), np.clip(pred[5:8], 0, 1)* 255, M.ravel()))
        command = get_command("../", "examples/" + str(i) + "_pred.bmp", params)
        os.system(command)


def change_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


def plot_render(meshgrid, np_array, mode="all", figure=1, lims=(0, 1), eps=0.1):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figure)
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    if mode == "all":
        disp = (np_array >= 0)
        opacity = 0.1
    elif mode == "in":
        disp = (np_array < 1)
        opacity = 0.1
    elif mode == "in_inv":
        disp = (np_array > 0.01)
        opacity = 0.1
    elif mode == "bit":
        disp = (np_array == 1)
        opacity = 0
    elif mode == "shell":
        disp = (np_array < 1+eps) & (np_array > 1-eps)
        opacity = 0
    np_array = np_array.ravel()
    clr = np.zeros(shape=(np_array.shape[0], 4))
    dsp = disp.ravel()
    np_max, np_min = np_array.max(), np_array.min()
    np_array = -1 + ((np_array - np_min)/(np_max - np_min)) * 2

    for i in range(np_array.shape[0]):
        r, b, g, a = gray_to_jet(np_array[i])
        if not dsp[i]:
            a = opacity
        clr[i] = np.array([r, b, g, a])

    ax.scatter(
        meshgrid[0],
        meshgrid[1],
        meshgrid[2],
        color=clr, marker='o'
        #np_array[disp].ravel(), marker='o', alpha=0.3
    )
    lims2 = (1, 0)
    ax.set(xlim=lims, ylim=lims, zlim=lims)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')


def plot_points(xs, ys, zs, figure=2, lims=(-1, 1), subplot=111):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figure)
    ax = fig.add_subplot(subplot, projection='3d')
    ax.set_aspect("auto")
    ax.scatter(xs, ys, zs, marker='o')
    ax.set(xlim=lims, ylim=lims, zlim=lims)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')


def parse_csv(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")
    labels = []
    print("Parsing csv " + csvfile)
    for line in lines:
        if line == "":
            continue
        split_line = line.split(",")
        # Get Image ID

        #label_id = split_line[0].split("/")[1].split(".")[0]
        #list_ids.append(label_id)

        # Cast, normalize the values
        values = []
        for i in range(1, 9):
            val = float(split_line[i])
            if i in [1, 2, 3]:
                #val = (val - 25) / 50
                val /= 255.0
            elif i in [6, 7, 8]:
                val /= 255.0
            values.append(val)  # delete \t
        for i in range(-4, 0):
            values.append(float(split_line[i]))
        labels.append(np.array(values, dtype=np.float32))
    print("Size of data: " + str(len(labels)))
    print("----------------------------------------------------------------")
    return labels


def graph2img(net, filename="graph.png"):
    x = torch.randn(1, 1, 256, 256).requires_grad_(True)
    y = net(x)
    dot = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    dot.format = filename.split(".")[1]
    dot.render(filename.split(".")[0])
    print("Graph visualization saved to " + filename)


def gray_to_jet(gray):
    def interpolate(val, y0, x0, y1, x1 ):
        return (val - x0) * (y1 - y0) / (x1 - x0) + y0
    def base(val):
        if val <= -0.75:
            return 0
        elif val <= -0.25:
            return interpolate(val, 0.0, -0.75, 1.0, -0.25 )
        elif val <= 0.25:
            return 1.0
        elif val <= 0.75:
            return interpolate(val, 1.0, 0.25, 0.0, 0.75 )
        else:
            return 0.0
    red = base(gray - 0.5)
    green = base(gray)
    blue = base(gray + 0.5)
    alpha = 1
    return red, green, blue, alpha


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                sleep(1)
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

def randquat():
    u = np.random.uniform(0, 1, (3,))
    q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                  np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                  np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                  np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
    return q


def slerp(v0, v1, t_array):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :]), v1
