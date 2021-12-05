import sys
import pandas
from sklearn import linear_model
import random

# file opening
data = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
print(data)


