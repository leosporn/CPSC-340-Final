import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def savefig(fname, verbose=True):
    plt.tight_layout()
    path = os.path.join('.', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))


def clusterToMarker(cluster):
    c2m = ["b+", "ro", "cp", "kx", "gs", "bs", "r+", "g+"]
    if (cluster >= len(c2m)):
        #out of range, return default
        return "y2"

    return c2m[cluster]

def filterLabels(label):
    #whitelist = ["US", "Canada", "Iran", "Italy", "Mainland China", "Spain", "Vietnam", "Finland", "Bahrain", "Iceland", "Andorra", "Luxembourg"]
    whitelist = ["Wyoming", "Michigan", "Alabama", "New York", "Florida", "Washington", "California", "Ontario", "British Columbia", "Quebec", "Arizona", "Hawaii", "Alberta"]
    if (label in whitelist):
        return label
    return ""


def elbow(X):
    inertia_by_i = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        inertia_by_i.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), inertia_by_i)
    savefig("elbow.png")
    
