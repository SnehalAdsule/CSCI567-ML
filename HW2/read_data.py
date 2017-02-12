import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd
# Load the diabetes dataset
def main():
    boston_housing = datasets.load_boston()
    test_array=np.array([])
    print boston_housing.keys()
    print boston_housing.feature_names

    a= boston_housing.data
    y= boston_housing.target
    x_test_array=[]
    x_train_array=[]
    y_test_array=[]
    y_train_array=[]
    for x in range(len(a)):
        if((x)%7==0):
            x_test_array.append(a[x])
            y_test_array.append(y[x])

        else:
            x_train_array.append(a[x])
            y_train_array.append(y[x])
    print len(x_test_array),len(x_train_array)
    np.savetxt('test.csv', x_test_array,delimiter=',')
    np.savetxt('train.csv', x_train_array,delimiter=',')
    np.savetxt('test_target.csv', y_test_array,delimiter=',')
    np.savetxt('train_target.csv', y_train_array,delimiter=',')

