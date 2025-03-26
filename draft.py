

import os
os.getcwd()
os.chdir("../MLExtrem")

import numpy as np

x = np.array([3,6,2,8])
np.argsort(x)[::-1]


import numpy as np
x_raw = np.array([[10, 20], [30, 40], [50, 60]])
rank_transformation(x_raw)


x_train = np.array([[1, 2], [3, 4], [5, 6]])
x_test = np.array([[2, 3], [4, 5]])
rank_transform_test(x_train, x_test)

    # array([[1.5, 1.5],
    #        [3. , 3. ]])

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

ytrue = np.linspace(0,5,num=5)
ypred= ytrue + 0.6
mean_squared_error(ytrue,ypred)

myfun = mean_squared_error

def toto(y1,y2,FUN):
    result = FUN(y1,y2)
    return result

toto(ytrue,ypred, mean_squared_error)
