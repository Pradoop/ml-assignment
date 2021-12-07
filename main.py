# test comment
# test comment v.2

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ford = pd.read_csv("ford.csv", delimiter = ';')
    df = pd.DataFrame(data = ford)

    X = df
    y = df
    X = X.reindex([['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']])
    y = y.reindex(['price'])

    plt.scatter(df['mileage'], df['price'], color = 'red')
    plt.title('mileage Vs price', fontsize = 14)
    plt.xlabel('mileage', fontsize=14)
    plt.ylabel('price', fontsize = 14)
    plt.grid(True)
    plt.show()
    #Werkt zoals bedoeld

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    predictedPrice = regr.predict([[2,2018,1,12449,4,145,57,1]])
    #ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    print(predictedPrice)
