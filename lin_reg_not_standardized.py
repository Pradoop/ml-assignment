import sys

import numpy
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# separating features dataset
x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

# splitting dataset to training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
regr = linear_model.LinearRegression()

# fit is essentially training your model
regr.fit(x_train, y_train.values)

# this gives us the values for the intercept and the coefficient for each feature
print("Intercept: ", regr.intercept_)
print("Coefficients: ", list(zip(x, regr.coef_)))

# for test set
y_predicted_result = regr.predict(x_test)
print("Predicted price values: ", list(map('{:.2f}'.format, y_predicted_result)))

# comparison
comparison = pandas.DataFrame({'Actual test value': y_test, 'Predicted value': y_predicted_result})
print(comparison.head())

# Evaluation
meanAbErr = metrics.mean_absolute_error(y_test, y_predicted_result)
meanSqErr = metrics.mean_squared_error(y_test, y_predicted_result)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predicted_result))

print('R squared: {:.2f}'.format(regr.score(x_test, y_test) * 100))
print('Mean Absolute Error: {:.2f}'.format(meanAbErr))
print('Mean Square Error: {:.2f}'.format(meanSqErr))
print('Root Mean Square Error: {:.2f}'.format(rootMeanSqErr))
