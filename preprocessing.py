# file opening
import sys

import pandas
from matplotlib import pyplot as plt

dataframe = pandas.read_csv(sys.path[0] + "/files/ford.csv", sep=';')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# separating features dataset
x = dataframe[['model', 'year', 'engineSize', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg']]
y = dataframe['price']


# plot for linearity in dataset between year and price
def yearPlot():
    plt.scatter(dataframe['year'], dataframe['price'], color='red')
    plt.title('year VS Price in GBP', fontsize=14)
    plt.xlabel('year', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between engineSize and price
def engineSizePlot():
    plt.scatter(dataframe['engineSize'], dataframe['price'], color='red')
    plt.title('engineSize VS Price in GBP', fontsize=14)
    plt.xlabel('engineSize', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between mileage and price
def mileagePlot():
    plt.scatter(dataframe['mileage'], dataframe['price'], color='red')
    plt.title('mileage in Miles VS Price in GBP', fontsize=14)
    plt.xlabel('mileage', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between tax and price
def taxPlot():
    plt.scatter(dataframe['tax'], dataframe['price'], color='red')
    plt.title('tax in GBP VS Price in GBP', fontsize=14)
    plt.xlabel('tax', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between mpg and price
def mpgPlot():
    plt.scatter(dataframe['mpg'], dataframe['price'], color='red')
    plt.title('mpg VS Price in GBP', fontsize=14)
    plt.xlabel('mileage', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    yearPlot()
    mpgPlot()
    engineSizePlot()
    taxPlot()
    mileagePlot()
