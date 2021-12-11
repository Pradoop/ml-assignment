import sys
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

x_train_new = (x_train - x_train.mean()) / x_train.std()
x_test_new = (x_test - x_test.mean()) / x_test.std()

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# crossValidation scores
scores = cross_val_score(regr, x_train, y_train, cv=5)
print("Cross-validated scores:", scores)

predictions = cross_val_predict(regr, x_train, y_train, cv=5)
plt.scatter(y_train, predictions)

print("Intercept: ", regr.intercept_)
print("Coefficients: ", list(zip(x, regr.coef_)))

y_predicted_result = regr.predict(x_test_new)
print("Predicted price values: ", list(map('{:.2f}'.format, y_predicted_result)))

comparison = pandas.DataFrame({'Training value': y_test, 'Predicted value': y_predicted_result})
print(comparison.head())

meanAbErr = metrics.mean_absolute_error(y_test, y_predicted_result)
meanSqErr = metrics.mean_squared_error(y_test, y_predicted_result)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predicted_result))

print('R squared: {:.2f}'.format(regr.score(x, y) * 100))
print('Mean Absolute Error: {:.2f}'.format(meanAbErr))
print('Mean Square Error: {:.2f}'.format(meanSqErr))
print('Root Mean Square Error: {:.2f}'.format(rootMeanSqErr))
