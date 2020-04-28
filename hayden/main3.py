import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

import pandas as pd
import utils

from manifold import MDS, ISOMAP

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        data = pd.read_csv('../data/covid_19_Countries.csv')

        X = data.values
        names = X[:,0]
        # get rid of country names
        X = X[:,1:]
        X = np.array(X).astype(np.float32)
        X[X<=0] = 1.
        X[np.isnan(X)] = 1.
        X[np.isinf(X)] = 1.
        X = np.log(X)
        #X = X / X.max()
        n,d = X.shape
        print("n =", n)
        print("d =", d)

        for n_neighbours in [2,3]:
            model = ISOMAP(n_components=2, n_neighbours=n_neighbours)
            Z = model.compress(X)
            fig, ax = plt.subplots()
            ax.scatter(Z[:,0], Z[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(names[i], (Z[i,0], Z[i,1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)
        #pca = PCA(n_components = 2)
        #pca.fit(X)
        #pcaTransform = pca.transform(X)
        #f1 = pcaTransform[:, 0]
        #f2 = pcaTransform[:, 1] #np.random.choice(d, size=2, replace=False)
        #print(pca.singular_values_)
        #print(pca.explained_variance_ratio_.sum())
        # tsne = TSNE(n_components=2)
        # results = tsne.fit_transform(X)
        # f1 = results[:,0]
        # f2 = results[:,1]

        #print(pca.explained_variance_ratio_.sum())

        # plt.figure()
        # plt.scatter(f1,f2)
        # #plt.scatter(X[:,f1], X[:,f2])
        # plt.xlabel("component 1")
        # plt.ylabel("component 2")
        # # plt.xlabel("$x_{%d}$" % f1)
        # # plt.ylabel("$x_{%d}$" % f2)
        # for i in range(n):
        #     plt.annotate(names[i], (f1[i], f2[i]))
        #     #plt.annotate(animals[i], (X[i,f1], X[i,f2]))
        
        # #D = utils.euclidean_dist_squared(X,X)
        # #D = np.sqrt(D)
        # #MDS._fun_obj_z(f1,D)
        
        # utils.savefig('PCAunscaledLOG.png')


    elif question == '1.3':

        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('MDS_animals.png')

    elif question == '1.4':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        for n_neighbours in [2,3]:
            model = ISOMAP(n_components=2, n_neighbours=n_neighbours)
            Z = model.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z[:,0], Z[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(animals[i], (Z[i,0], Z[i,1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)

    elif question == '1.5':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        tsne = TSNE(n_components=2)
        results = tsne.fit_transform(X)
        f1 = results[:,0]
        f2 = results[:,1]

        plt.figure()
        plt.scatter(f1,f2)
        #plt.scatter(X[:,f1], X[:,f2])
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        # plt.xlabel("$x_{%d}$" % f1)
        # plt.ylabel("$x_{%d}$" % f2)
        for i in range(n):
            plt.annotate(animals[i], (f1[i], f2[i]))
            #plt.annotate(animals[i], (X[i,f1], X[i,f2]))
        
        #D = utils.euclidean_dist_squared(X,X)
        #D = np.sqrt(D)
        #MDS._fun_obj_z(f1,D)
        
        utils.savefig('tsne.png')


    elif question == "2":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50]
        model = NeuralNet(hidden_layer_sizes)

        t = time.time()
        model.fit(X,Y)
        print("Fitting took %d seconds" % (time.time()-t))

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == "2.4":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        print("n =", X.shape[0])
        print("d =", X.shape[1])        

        #model = MLPClassifier()
        model = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', batch_size = 1, max_iter = 200, early_stopping = True)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)    