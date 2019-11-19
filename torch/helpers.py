import numpy as np
import torch
from torchviz import make_dot
from matplotlib import pyplot as plt


def plot_render(meshgrid, np_array, mode="all"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    if mode == "all":
        disp = (np_array >= 0)
    else:
        disp = (np_array < 1)

    np_array = np_array.ravel()
    clr = np.zeros(shape=(np_array.shape[0], 4))
    dsp = disp.ravel()
    np_max, np_min = np_array.max(), np_array.min()
    np_array = -1 + ((np_array - np_min)/(np_max - np_min)) * 2

    for i in range(np_array.shape[0]):
        r, b, g, a = gray_to_jet(np_array[i])
        if not dsp[i]:
            a = 0.1
        clr[i] = np.array([r, b, g, a])
    print(disp.shape)
    print(dsp.shape)
    print(clr.shape)

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
    plt.show()

def parse_csv(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")
    labels = {}
    list_ids = []
    print("Parsing csv " + csvfile)
    for line in lines:
        if line == "":
            continue
        split_line = line.split(",")
        # Get Image ID
        label_id = split_line[0].split("/")[1].split(".")[0]
        list_ids.append(label_id)
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
        labels[label_id] = np.array(values, dtype=np.float)
    print("Size of data: " + str(len(list_ids)))
    print("----------------------------------------------------------------")
    return list_ids, labels


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