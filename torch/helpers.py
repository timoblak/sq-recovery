import numpy as np
import torch
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from time import sleep


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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def change_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr

def plot_render(meshgrid, np_array, mode="all", figure=1):
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
    elif mode == "bit":
        disp = (np_array == 1)
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

    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
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
                val = (val - 25) / 50
            elif i in [6, 7, 8]:
                val /= 255.0
            values.append(val)
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