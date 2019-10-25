import numpy as np
import torch
from torchviz import make_dot
from matplotlib import pyplot as plt


def plot_render(meshgrid, np_array):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    disp = (np_array < 1)
    ax.scatter(
        meshgrid[0][disp],
        meshgrid[1][disp],
        meshgrid[2][disp],
        c=np_array[disp].ravel(), marker='o', alpha=0.3
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
    print("Parsing csv...")
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
    print("----------------------------------------------------------------")
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
