import sys
import numpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import pandas  # To read data
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# file opening
dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
# dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# separating features dataset and mentioning categorical values
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
# print(x_data)
# following line adds a column with value unity for the bias
x_data = sm.add_constant(x_data, prepend=True)
print(x_data)

# splitting dataset to training, validation, test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
