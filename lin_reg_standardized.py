import sys
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

ford = pd.read_csv(sys.path[0] + "/files/ford.csv", sep=';')

x = ford[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = ford['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

x_train_new = x_train[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
x_train[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']] = (x_train_new - x_train_new.mean()) / x_train_new.std()

mean = x_train_new.mean()
dev = x_train_new.std()

x_test_new = x_test[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
x_test[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']] = (x_test_new-x_test_new.mean())/x_test_new.std()
#x_test[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']] = (x_test_new - mean) / dev

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print("Intercept: ", regr.intercept_)
print("Coefficients: ", list(zip(x, regr.coef_)))

y_predicted_result = regr.predict(x_test)
print("Predicted price values: ", list(map('{:.2f}'.format, y_predicted_result)))

comparison = pd.DataFrame({'Training value': y_test, 'Predicted value': y_predicted_result})
print(comparison.head())

meanAbErr = metrics.mean_absolute_error(y_test, y_predicted_result)
meanSqErr = metrics.mean_squared_error(y_test, y_predicted_result)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predicted_result))

print('R squared: {:.2f}'.format(regr.score(x, y) * 100))
print('Mean Absolute Error: {:.2f}'.format(meanAbErr))
print('Mean Square Error: {:.2f}'.format(meanSqErr))
print('Root Mean Square Error: {:.2f}'.format(rootMeanSqErr))
