import matplotlib.pyplot as plt
import os

def savefig(fname, verbose=True):
    plt.tight_layout()
    path = os.path.join('.', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))


def clusterToMarker(cluster):
    c2m = ["b+", "ro", "cp", "kx", "gs"]
    if (cluster >= len(c2m)):
        #out of range, return default
        return "y2"

    return c2m[cluster]