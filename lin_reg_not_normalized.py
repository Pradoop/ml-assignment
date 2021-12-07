import sys
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas  # To read data
from sklearn import linear_model

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
print(dataframe)

x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

regr = linear_model.LinearRegression()
regr.fit(x, y)

