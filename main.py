# This sample code is taken from a problem set in my statistics class that was supposed to be done in Excell.
# However, given that I haven't much practiced graphing on python,
# and needed to brush up on handling data set, I did it in python. I particularily enjoyed working
# on this project because it combined my two majors, Economics and Computer Science.

# This program was made to compute a data set of U.S GDP
# which can be downloaded from https://fred.stlouisfed.org/series/GDPC1
# we will be importing pandas,matplotlib,numpy,and scripy

import pandas as pd
import matplotlib.pyplot as plt
from math import log
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

# the initial reading of the data set will need path adjustement based on where the csv file is located
# in my case,the csv file is in the same location as the code so there in no need to specifcy the path.

df = pd.read_csv('GDP.csv')
df.plot()

# Next, we plot the logarithm of the data set
# takes the logarithm and stores it in new data set "logdf"

logdf = df
logdf['GDP'] = np.log(df['GDP'])

# plots new data set

logdf.plot()
plt.show()

# creates a new collum in the data set and set it take floats as input

logdf["Delta log"] = .0
logdf.to_csv("sample.csv", index=False)

# creates the list delta_log that will store the values of the delta value
# of the logarithms of the data set

delta_log = []
time = 0
value = 0
for i in range(0, 301):
    value = logdf['GDP'][time + 1] - logdf['GDP'][time]
    delta_log.append(value)
    time = time + 1

# appends delta_log to the data set logdf in the newly created column

for i in range(0, 301):
    logdf["Delta log"][i] = delta_log[i]

# plots the new data set chronologically with the delta_log values as the y-values

logdf[['DATE', 'Delta log']].plot()
plt.show()

plt.hist(logdf["Delta log"], bins=100)
plt.show()

# Data analysis here are 2 ways of computing data. The first two funtions takes lists of floats as input annd returns
# the variance and standard deviation of the data. We then use scripy.stats as a shortcut to compute skewness
# and covariance.

from math import sqrt


def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    print('the Variance of the delta p is:', variance)


def standard_error(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    standard_error = sqrt(sum(deviations) / n)
    print("the Standard Error of delta p is:", standard_error)


variance(logdf['Delta log'])
standard_error(logdf["Delta log"])

# Alternatively, scripy.stats can compute

print('the coefficient of skewness is', skew(logdf['Delta log']))
print('the coefficient of kurtosis is', kurtosis(logdf['Delta log']))