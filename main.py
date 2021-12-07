# test comment
# test comment v.2

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ford = pd.read_csv("ford.csv", header = 0)
    #usecols = ['model','year','price','transmission','mileage','fuelType','tax','mpg','engineSize']
    #X = ford[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
    #y = ford['price']

    print(ford)

    df = pd.DataFrame(data = ford, columns = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize'])

    #columns = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize')

    print(df)

    #X = ford[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
    #y = ford['price']

    #print(df['mileage'])
    #print(df['price'])
    #plt.scatter(df['mileage'], df['price'], color = 'red')
    #plt.title('mileage Vs price', fontsize = 14)
    #plt.xlabel('mileage', fontsize=14)
    #plt.ylabel('price', fontsize = 14)
    #plt.grid(True)
    #plt.show()

# regr = linear_model.LinearRegression()
# regr.fit(X, y)
