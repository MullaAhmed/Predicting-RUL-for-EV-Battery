#  Importing libraries
#general stuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

#apna time ayega
from preprocessing import loadMat
from util import *

#  Importing dataset (temp 24)
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')

#Creating dataframe
dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
dfB0007 = getDataframe(B0007)
dfB0018 = getDataframe(B0018)

x_train, x_test, y_train, y_test = train_test_split(dfB0005['cycle'], dfB0005['capacity'], test_size=0.2,random_state=0)
lst_x, lst_y = rollingAverage(x_train, y_train)
x_train=np.array(lst_x)
y_train=np.array(lst_y)

#training model
from sklearn.svm import SVR

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

best_svr.fit(x_train,y_train)

y_pred = best_svr.predict(x_test.values.reshape(-1, 1))

plt.plot(dfB0005['cycle'], dfB0005['capacity'],color='black')
plt.plot(dfB0005['cycle'],best_svr.predict(dfB0005["cycle"].values.reshape(-1, 1)))
plt.xlabel='Cycles'
plt.ylabel='Battery Capacity'
plt.title='Model performance for Battery 05'
plt.show()


# Evaluating the Model Performance
from sklearn.metrics import r2_score
print(r2_score(list(y_test),list(y_pred)))

