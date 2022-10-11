import csv
from re import I, X
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    
    with open(filepath) as csvfile:
        output = csv.DictReader(csvfile)
        
        # pop #,Name,Type 1,Type 2,Total
        popData = ['#','Name','Type 1','Type 2','Total']
        for data in output:
            for key in popData:
                data.pop(key)

    return output

# input: dict representing one Pokemon
# output: numpy array of shape (6,) and  dtype int64. The first element is x1 and so
# on with the sixth element being x6.


def calc_features(row):
    return np.array(list(row.items()))


def hac(features):

    pass


def imshow_hac(Z):
    return plt.show()
