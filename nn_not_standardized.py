import math
import sys

import numpy as np
import pandas  # To read data
import pandas as pd
import statsmodels.api as sm
import tensorflow
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
# dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# separating features dataset and mentioning categorical values so that neural network doesnt rank them
x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']

x_model = x.model.astype("category").cat.codes
x_trans = x.transmission.astype("category").cat.codes
x_fuel = x.fuelType.astype("category").cat.codes

# IMPORTANT REMARK: THESE SERIES VALUES DECREASE BY ONE. THIS MEANS THAT THE MODEL WITH VALUE 1 BECOMES 0
# COMPARE THE DATASET WITH x_data
x_model = pd.Series(x_model)
x_trans = pd.Series(x_trans)
x_fuel = pd.Series(x_fuel)

x_data = np.column_stack((x_model, x['year'], x['engineSize'], x_trans, x['mileage'], x_fuel, x['tax'], x['mpg']))
y_data = np.row_stack(y)
# following line adds a column with value unity for the bias
x_data = sm.add_constant(x_data, prepend=True)
print(x_data)
print(y_data)

# splitting dataset to training, validation, test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=100)

# input layer has 9 neurons = number of features in dataset + 1
# 1 hidden layer for the moment, can always be expanded
# neurons = #samples in training/(alpha*(#input neurons + #output neurons) = 540

model = Sequential()
model.add(Dense(8, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(2670, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# Epochs 30 and batch size 150
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(x_train, y_train, epochs=30, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(x_test)
print("Predictions for Epochs 30 and batch size 150", predictions)
print("MAE for Epochs 30 and batch size 150", mean_absolute_error(y_test, predictions))
print("MSE for Epochs 30 and batch size 150", mean_squared_error(y_test, predictions))
print("RMSE for Epochs 30 and batch size 150", math.sqrt(mean_squared_error(y_test, predictions)))
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for 30 epochs and batch size of 150')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Epochs and batch size 30
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(x_train, y_train, epochs=30, batch_size=30, verbose=1, validation_split=0.2)
predictions = model.predict(x_test)
print("Predictions for Epochs 30 and batch size 30", predictions)
print("MAE for Epochs 30 and batch size 30", mean_absolute_error(y_test, predictions))
print("MSE for Epochs 30 and batch size 30", mean_squared_error(y_test, predictions))
print("RMSE for Epochs 30 and batch size 30", math.sqrt(mean_squared_error(y_test, predictions)))
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for 30 epochs and batch_size of 30')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Epochs 100 and batch size 30
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(x_train, y_train, epochs=100, batch_size=30, verbose=1, validation_split=0.2)
predictions = model.predict(x_test)
print("Predictions for Epochs 100 and batch size 30", predictions)
print("MAE for Epochs 100 and batch size 30", mean_absolute_error(y_test, predictions))
print("MSE for Epochs 100 and batch size 30", mean_squared_error(y_test, predictions))
print("RMSE for Epochs 100 and batch size 30", math.sqrt(mean_squared_error(y_test, predictions)))
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for 100 epochs and batch size of 30')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Epochs 100 and batch size 150
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(x_train, y_train, epochs=100, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(x_test)
print("Predictions for Epochs 100 and batch size 150", predictions)
print("MAE for Epochs 100 and batch size 150", mean_absolute_error(y_test, predictions))
print("MSE for Epochs 100 and batch size 150", mean_squared_error(y_test, predictions))
print("RMSE for Epochs 100 and batch size 150", math.sqrt(mean_squared_error(y_test, predictions)))
print(history.history.keys())

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for 100 epochs and batch size of 150')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
