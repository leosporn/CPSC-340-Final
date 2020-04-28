import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utils
import numpy as np




data = pd.read_csv("./data/covid_19_Countries.csv")
labels = data.loc[:, "country"]
data = data.drop(["country"], axis=1)
pops = pd.read_csv("./data/country_pops_covid.csv")
pops = pops.drop(["Country"], axis=1)



X = data.to_numpy()
pop = np.matrix(pops.to_numpy(dtype=np.float32).flatten())

X[:,:-1] = pop.T

N, D = X.shape

'''for i in range(N):
    print(X[i, :])
    np.divide(X[i, :], pop[i])
    print(X[i, :])

print(X)
'''
#for i in range(N):
    #data[i, :] = data[i, :].div(pops[]) 

pcaS = PCA(n_components = 2).fit(X)
pca = pcaS.transform(X)

plt.figure()
plt.scatter(pca[:, 0], pca[:, 1])
plt.xlabel("c1")
plt.ylabel("c2")


for i in range(N):
    plt.annotate(labels[i], (pca[i, 0], pca[i, 1]))

utils.savefig('pca1.png')