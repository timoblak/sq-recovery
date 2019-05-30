import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns  # for nicer graphics
import scipy.stats as stats


def ff(suma):
    def format_func(value, tick_number):
        return np.round(value*suma, 2)
    return format_func

with open("errors.pkl", "rb") as handle:
    data = pickle.load(handle)

param = np.array(data)
order = [0, 1, 2, 3, 5, 6, 7, 4]

print(param.shape)

size_xlim = 5
shape_xlim = 0.1
pos_xlim = 8
size_ylim = 0.35
shape_ylim = 25
pos_ylim = 0.5

size_c = "salmon"
shape_c = "lightskyblue"
pos_c = "lightgreen"

size_bins = 64
shape_bins = 50
pos_bins = 64

xlims = [size_xlim, size_xlim, size_xlim, shape_xlim, shape_xlim, pos_xlim, pos_xlim, pos_xlim]
ylims = [size_ylim, size_ylim, size_ylim, shape_ylim, shape_ylim, pos_ylim, pos_ylim, pos_ylim]

bins = [size_bins, size_bins, size_bins, shape_bins, shape_bins, pos_bins, pos_bins, pos_bins]
color = [size_c, size_c, size_c, shape_c, shape_c, pos_c, pos_c, pos_c]
tit = [r"$a_1$", r"$a_2$", r"$a_3$", r"$\epsilon_1$", r"$\epsilon_2$", r"$x_0$", r"$y_0$", r"$z_0$"]

fig = plt.figure(1)
for cell, i in enumerate(order):
    print(i)
    x = -param[:, i]
    #len(x)
    ax = plt.subplot(2, 4, cell+1)

    xz = np.linspace(x.mean() - 4 * x.std(), x.mean() + 4 * x.std(), 100)


    n, b, p = ax.hist(x, density=True, bins=bins[i], color=color[i])
    ax.plot(xz, stats.norm.pdf(xz, xz.mean(), xz.std()))
    print(tit[i] + " --- Mean: " + str(x.mean()) + ", Std: " + str(x.std()))
    #second_ax = ax.twinx()
    #sns.distplot(x, ax=second_ax, kde=True, hist=False)
    #print(np.round(n*np.diff(b), 3))
    # Removing Y ticks from the second axis
    #second_ax.set_yticks([])
    sns.kdeplot(x, color="b")
    #ax.plot(n*np.diff(b))
    #print(np.array(1 / sum(n) * n))

    ax.set_xlim([-xlims[i], xlims[i]])
    ax.set_ylim([0, ylims[i]])
    ax.set_title(tit[i])
    #ax.set_yticks([])

    ax.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(ff(1/sum(n))))
    #ax.set_xlabel('Error')
    #ax.set_ylabel('Probability density')

fig.text(0.5, 0.04, 'Prediction Error', ha='center')
fig.text(0.04, 0.5, 'Probability Density', va='center', rotation='vertical')
plt.show()