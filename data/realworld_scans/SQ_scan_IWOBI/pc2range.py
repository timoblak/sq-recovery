from matplotlib import pyplot as plt
import pptk

point_list_file = "test8.asc"

with open(point_list_file, "r") as handle:
    point_list = handle.read()

xs, ys, zs = [], [], []
pl = []
for point in point_list.split("\n"):
    try:
        x, y, z = [float(x) for x in point.split(" ")]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        pl.append([x,y,z])
    except ValueError:
        print(point)


x_bins = 256
y_bins = 256



v = pptk.viewer(pl)


