###Directly predicts capacity at given cycle

#  Importing libraries
#general stuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#apna time ayega
from preprocessing import *
from util import *

#sk learn
from sklearn.svm import SVR
from sklearn.model_selection import  train_test_split

#  Importing dataset (temp 24)
B0005 = loadMat('B0005.mat')

#Creating dataframe
dfB0005 = getDataframe(B0005)

#####################
#                   #
#Model for battery 5#
#                   #
#####################

x_train, x_test, y_train, y_test = train_test_split(dfB0005['cycle'], dfB0005['capacity'], test_size=0.2,random_state=2)
x_train=np.array(x_train)
y_train=np.array(y_train)

x_train = x_train.reshape(-1, 1) #changes from 1 d array to 2 d array
y_train = y_train.reshape(-1, 1)

#Fitting model
regressor = SVR(epsilon=0.0001,kernel='rbf')
regressor.fit(x_train,y_train)

#Predicting data
y_pred = regressor.predict(x_test.values.reshape(-1, 1))

#Plotting curve
plt.plot(dfB0005['cycle'], dfB0005['capacity'],color='black')
plt.plot(dfB0005['cycle'],regressor.predict(dfB0005["cycle"].values.reshape(-1, 1)))
plt.xlabel='Cycles'
plt.ylabel='Battery Capacity'
plt.title='Model performance for Battery 05'
plt.show()


# Evaluating the Model Performance
from sklearn.metrics import r2_score
print(r2_score(list(y_test),list(y_pred)))

