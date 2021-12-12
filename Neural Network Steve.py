import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys

dataframe = pd.read_csv(sys.path[0] + "/files/ford.csv", sep=';')

y_data = np.array(dataframe['price'])
x_model = dataframe.model.astype("category").cat.codes
x_trans = dataframe.transmission.astype("category").cat.codes
x_fuel = dataframe.fuelType.astype("category").cat.codes

x_model = pd.Series(x_model)
x_trans = pd.Series(x_trans)
x_fuel = pd.Series(x_fuel)

x_data = np.column_stack((x_model, dataframe['year'], dataframe['engineSize'], x_trans, dataframe['mileage'], x_fuel, dataframe['tax'], dataframe['mpg']))
x_data = sm.add_constant(x_data, prepend=True)
X_train, X_val, y_train, y_val = train_test_split(x_data, y_data)

y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))
#scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()
scaler_x = StandardScaler()
scaler_y = StandardScaler()

print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
print(scaler_x.fit(X_val))
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)
print(scaler_y.fit(y_val))
yval_scale=scaler_y.transform(y_val)

model=Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(540, activation='relu'))
model.add(Dense(540, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=100, batch_size=30, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scale)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = scaler_y.inverse_transform(predictions)
mae = mean_absolute_error(y_val, predictions)
print(mae)
mse = mean_squared_error(y_val, predictions)
math.sqrt(mse)
print(mse)
np.mean(y_val)
np.mean(predictions)