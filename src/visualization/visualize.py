import os
import matplotlib.pyplot as plt


def gen_fig(filename, x, ys, xlabel, ylabels, ycolors, dpi=300):
    fig, ax = plt.subplots()
    for y, ylabel in zip(ys, ylabels):
        ax.plot(x, y, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.save_fig(os.path.join('reports', filename), dpi=dpi)
    