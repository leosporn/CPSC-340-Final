import pandas as pd
import numpy as np

# Requires that you put the covid_19_data.csv file into the common data folder
data = pd.read_csv('../../data/covid_19_data.csv')

# Important Columns and their Column numbers
Date = 0
ProvState = 1
Ctry = 2
Confirmed = 4
Deaths = 5
Recovered = 6

X = data.values

# Get Unique provinces/states, done by checking last day
places = X[X[:,Date] == "04/25/2020"]
uniquePlaces = np.unique(places[:,Ctry])
numPlaces = uniquePlaces.size

# Get unique dates
uniqueDates = np.unique(X[:,Date])

# Set dataframe indices to be the provinces/states
df2 = pd.DataFrame(index = uniquePlaces)

# Track deaths/confirmed for previous day to find new cases on a date
prevTotalConfirmed = np.zeros(numPlaces)
prevTotalDeaths = np.zeros(numPlaces)
prevTotalRecovered = np.zeros(numPlaces)

for date in uniqueDates:

	confirmed = np.zeros(numPlaces)
	deaths = np.zeros(numPlaces)
	totalconfirmed = np.zeros(numPlaces)
	totaldeaths = np.zeros(numPlaces)
	recovered = np.zeros(numPlaces)
	totalRecovered = np.zeros(numPlaces)

	rowsWithThisDate = X[X[:,Date] == date]

	# Iterate through rows and change 0s to correct values for index
	# idx is a province/states spot in uniquePlaces
	for row in rowsWithThisDate:

		idx = np.where(uniquePlaces == row[Ctry])[0]

		totalconfirmed[idx] += row[Confirmed]
		totaldeaths[idx] += row[Deaths]
		totalRecovered[idx] += row[Recovered]

	if date != "01/22/2020":
		confirmed = totalconfirmed - prevTotalConfirmed
		deaths = totaldeaths - prevTotalDeaths
		recovered = totalRecovered - prevTotalRecovered

	prevTotalDeaths = totaldeaths
	prevTotalConfirmed = totalconfirmed
	prevTotalRecovered = totalRecovered

	# Add new columns
	df2[date + " new confirmed"] = confirmed
	df2[date + " total confirmed"] = totalconfirmed
	df2[date + " new deaths"] = deaths
	df2[date + " total deaths"] = totaldeaths
	df2[date + " recovered"] = recovered
	df2[date + " total recovered"] = totalRecovered

# Save to CSV
# print(df2)
df2.to_csv("../../data/covid_19_Countries.csv")
