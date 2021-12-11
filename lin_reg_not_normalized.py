import sys
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')

x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print("Intercept: ", regr.intercept_)
print("Coefficients: ")
print(list(zip(x, regr.coef_)))

y_predicted_result = regr.predict(x_test)

print("Prediction for test set: {}".format(y_predicted_result))

mlr_diff = pandas.DataFrame({'Actual value': y_test, 'Predicted value': y_predicted_result})
mlr_diff.head()

meanAbErr = metrics.mean_absolute_error(y_test, y_predicted_result)
meanSqErr = metrics.mean_squared_error(y_test, y_predicted_result)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_predicted_result, y_predicted_result))
print('R squared: {:.2f}'.format(regr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)