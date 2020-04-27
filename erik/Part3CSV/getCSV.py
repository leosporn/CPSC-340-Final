import pandas as pd
import numpy as np

# Requires that you put the covid_19_data_CA_US_no_cities.csv file into the common data folder
# Justin sent this file over discord
data = pd.read_csv('../../data/covid_19_data_CA_US_no_cities.csv')

# Important Columns and their Column numbers
Date = 1
ProvState = 2
Ctry = 3
Confirmed = 5
Deaths = 6
Recovered = 7

X = data.values

# Get Unique provinces/states, done by checking last day
places = X[X[:,Date] == "04/25/2020"]
uniquePlaces = np.unique(places[:,2])
numPlaces = uniquePlaces.size

# Get unique dates
uniqueDates = np.unique(X[:,1])

# Set dataframe indices to be the provinces/states
df2 = pd.DataFrame(index = uniquePlaces)

# Track deaths/confirmed for previous day to find new cases on a date
prevTotalConfirmed = np.zeros(numPlaces)
prevTotalDeaths = np.zeros(numPlaces)

for date in uniqueDates:

	confirmed = np.zeros(numPlaces)
	deaths = np.zeros(numPlaces)
	totalconfirmed = np.zeros(numPlaces)
	totaldeaths = np.zeros(numPlaces)
	# recovered = np.zeros(numPlaces)

	rowsWithThisDate = X[X[:,Date] == date]

	# Iterate through rows and change 0s to correct values for index
	# idx is a province/states spot in uniquePlaces
	for row in rowsWithThisDate:

		idx = np.where(uniquePlaces == row[ProvState])[0]

		totalconfirmed[idx] = row[Confirmed]
		totaldeaths[idx] = row[Deaths]
		# recovered[idx] = row[Recovered]

		if date != "01/22/2020":
			confirmed[idx] = row[Confirmed] - prevTotalConfirmed[idx]
			deaths[idx] = row[Deaths] - prevTotalDeaths[idx]

	prevTotalDeaths = totaldeaths
	prevTotalConfirmed = totalconfirmed

	# Add new columns
	df2[date + " new confirmed"] = confirmed
	df2[date + " total confirmed"] = totalconfirmed
	df2[date + " new deaths"] = deaths
	df2[date + " total deaths"] = totaldeaths
	# df2[date + " recovered"] = recovered

# Save to CSV
df2.to_csv("../data/covid_19_USCANtouchedup.csv")
