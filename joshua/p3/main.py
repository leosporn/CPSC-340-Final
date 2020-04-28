import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.manifold import Isomap



def getPca(X):
    pcaS = PCA(n_components = 2).fit(X)
    return pcaS.transform(X)

def getIso(X, neighbs):
    isoS = Isomap(n_components= 2, n_neighbors=neighbs).fit(X)
    return isoS.transform(X)

def divideByRow(X, y):
    N, D = X.shape
    for i in range(N):
        X[i, :] = np.divide(X[i, :], y[i])
    return X

clusters = 4
neighbours = 2


data = pd.read_csv("./data/covid_19_Countries.csv")
labels = data.loc[:, "country"]
data = data.drop(["country"], axis=1)
pops = pd.read_csv("./data/country_pops_covid.csv")
pops = pops.drop(["Country"], axis=1)


X = data.to_numpy()
pop = pops.to_numpy(dtype=np.float32).flatten()
#X[:,:-1] = pop.T
X = divideByRow(X, pop)

N, D = X.shape
X = normalize(X, axis=0)
X = normalize(X, axis=1)
'''
#PCA implementation
X = getPca(X)
modelNm = "PCA"
'''

#ISOMAP implementation
X = getIso(X, neighbours)
modelNm = "ISOMAP with " + str(neighbours) + " neighbours"

#uncomment this to get an elbow diagram
#utils.elbow(X)

kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
print(kmeans.inertia_)

plt.figure()
for i in range(N):
    plt.plot(X[i, 0], X[i, 1], utils.clusterToMarker(kmeans.labels_[i]))
plt.xlabel("c1")
plt.ylabel("c2")
plt.title(modelNm + " w 2-components with k = " + str(clusters))


for i in range(N):
    plt.annotate(utils.filterLabels(labels[i]), (X[i, 0], X[i, 1]))

utils.savefig('pca1.png')