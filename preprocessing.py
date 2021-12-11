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
    plt.title('Year VS price in GBP', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Price in GBP', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between engineSize and price
def engineSizePlot():
    plt.scatter(dataframe['engineSize'], dataframe['price'], color='red')
    plt.title('Engine size VS price in GBP', fontsize=14)
    plt.xlabel('Engine size', fontsize=14)
    plt.ylabel('Price in GBP', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between mileage and price
def mileagePlot():
    plt.scatter(dataframe['mileage'], dataframe['price'], color='red')
    plt.title('Mileage in miles VS price in GBP', fontsize=14)
    plt.xlabel('Mileage', fontsize=14)
    plt.ylabel('Price in GBP', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between tax and price
def taxPlot():
    plt.scatter(dataframe['tax'], dataframe['price'], color='red')
    plt.title('Tax in GBP VS price in GBP', fontsize=14)
    plt.xlabel('Tax', fontsize=14)
    plt.ylabel('Price in GBP', fontsize=14)
    plt.grid(True)
    plt.show()


# plot for linearity in dataset between mpg and price
def mpgPlot():
    plt.scatter(dataframe['mpg'], dataframe['price'], color='red')
    plt.title('MPG VS price in GBP', fontsize=14)
    plt.xlabel('MPG', fontsize=14)
    plt.ylabel('Price in GBP', fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    yearPlot()
    mpgPlot()
    engineSizePlot()
    taxPlot()
    mileagePlot()
