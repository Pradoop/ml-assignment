import sys
import numpy
import numpy as np
from matplotlib import pyplot as plt
import pandas  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# separating features dataset
x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

# splitting dataset to training, validation, test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
regr = linear_model.LinearRegression()

# fit is essentially training your model
regr.fit(x_train, y_train.values)

# testSet intercept and coefficients
print("Intercept: ", regr.intercept_)
print("Coefficients: ", list(zip(x, regr.coef_)))

# crossValidation scores and plot
scores = cross_val_score(regr, x_train, y_train, cv=5)
print("Cross-validated scores:", scores)
predictions = cross_val_predict(regr, x_train, y_train, cv=5)

# for test set
y_predicted_result = regr.predict(x_test)
print("Predicted price values: ", list(map('{:.2f}'.format, y_predicted_result)))

# comparison
comparison = pandas.DataFrame({'Actual test value': y_test, 'Predicted value': y_predicted_result})
print(comparison.head())

# plot
#plt.scatter(x_test['year'], y_test, color='red')
#plt.plot(x_test['year'], y_predicted_result, color='k', label='Regression model')
#plt.title('Year VS price in GBP', fontsize=14)
#plt.xlabel('Year', fontsize=14)
#plt.ylabel('Price in GBP', fontsize=14)
#plt.grid(True)
#plt.show()

# Evaluation metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_predicted_result)
meanSqErr = metrics.mean_squared_error(y_test, y_predicted_result)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predicted_result))

print('R squared: {:.2f}'.format(regr.score(x_test, y_test) * 100))
print('Mean Absolute Error: {:.2f}'.format(meanAbErr))
print('Mean Square Error: {:.2f}'.format(meanSqErr))
print('Root Mean Square Error: {:.2f}'.format(rootMeanSqErr))
