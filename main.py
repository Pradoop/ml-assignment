import pandas
from sklearn import linear_model

df = pandas.read_csv("ford.csv")

X = df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, y)
