import matplotlib.pyplot as plt
import os

def savefig(fname, verbose=True):
    plt.tight_layout()
    path = os.path.join('.', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))