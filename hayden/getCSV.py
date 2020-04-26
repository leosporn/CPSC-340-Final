import pandas as pd
import numpy as np

#requires you to put the covid_19_data.csv file into the common data folder
data = pd.read_csv('../data/covid_19_data.csv')

print(data.shape)
X = data.values
X = np.array(X)

#Important Columns and their Column numbers
date = 1
ProvState = 2
Ctry = 3
confirmed = 6
deaths = 7
recovered = 8

#filter to US/Canada

XCan = X[X[:,Ctry] == "Canada"]
XUS = X[X[:,Ctry] == "US"]
X = np.concatenate((XCan,XUS), axis = 0)

print(X.shape)
#now we have the right countries, but each row is a measure for a given county and day
dfX = pd.DataFrame(X)
dfX.to_csv("../data/covid_19_USCAN.csv")
