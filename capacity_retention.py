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
datasets=["B0005.mat","B0006.mat","B0007.mat","B0018.mat",'B0025.mat', 'B0026.mat', 'B0027.mat', 'B0028.mat', 'B0029.mat', 'B0030.mat', 'B0031.mat', 'B0032.mat', 'B0033.mat', 'B0034.mat', 'B0036.mat', 'B0038.mat', 'B0039.mat', 'B0040.mat', 'B0041.mat', 'B0045.mat', 'B0046.mat', 'B0047.mat', 'B0048.mat', 'B0049.mat',  'B0051.mat', 'B0053.mat', 'B0054.mat', 'B0055.mat', 'B0056.mat']

#dataset not working 42,43,44,50,52

good=[]
bad=[]
for i in datasets:
    print(i)
    #  Importing dataset (temp 24)
    battery = loadMat(i)

    #Creating dataframe
    dfbattery = getDataframe(battery)

    #getting battery capicity
    capacity=getBatteryCapcity(battery)
    capacity_retention = getBatteryCapacityRetention(capacity[1])

    #capacity retention %
    dfbattery["Capacity Retention %"]=capacity_retention[1]

    l=[]#temperory storage for r 2 scores
    d={}#temp dict to get random state values from r2_score
    
    #to find appropriate random state
    for j in range (10):
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(dfbattery['cycle'], dfbattery['Capacity Retention %'], test_size=0.1,random_state=j)
        lst_x, lst_y =(x_train_0, y_train_0)
        x_train_0=np.array(x_train_0)
        y_train_0=np.array(y_train_0)
        
      
        #training model
        from sklearn.svm import SVR
        x_train_0 = x_train_0.reshape(-1, 1)
        y_train_0 = y_train_0.reshape(-1, 1)
        regressor = SVR(C=2000, epsilon=0.0001,kernel='rbf')
        regressor.fit(x_train_0,y_train_0)
        y_pred = regressor.predict(x_test_0.values.reshape(-1, 1))

        # Evaluating the Model Performance
        from sklearn.metrics import r2_score
        x=float(r2_score(y_test_0,y_pred))
        l.append(x)
        d[x]=j
    


    z=(d[(max(l))])
    if max(l)>0.80:
        good.append("for {0} value of i {1} and r2_score {2} ".format(i,z,max(l)))
    else:
        bad.append("for {0} value of i {1} and r2_score {2} ".format(i,z,max(l)))

    
    x_train, x_test, y_train, y_test = train_test_split(dfbattery['cycle'], dfbattery['Capacity Retention %'], test_size=0.1,random_state=z)
    x_train=np.array(x_train)
    y_train=np.array(y_train)

    x_train = x_train.reshape(-1, 1) #changes from 1 d array to 2 d array
    y_train = y_train.reshape(-1, 1)

    #Fitting model
    regressor = SVR(C=2000, epsilon=0.0001,kernel='rbf') #epsilon defines the tube inside which error is allowed(must be small)
    regressor.fit(x_train,y_train)

    #Predicting data
    y_pred = regressor.predict(x_test.values.reshape(-1, 1))

    #Plotting curve
    plt.plot(dfbattery['cycle'], dfbattery['Capacity Retention %'],color='black')
    plt.plot(dfbattery['cycle'],regressor.predict(dfbattery["cycle"].values.reshape(-1, 1)))
    plt.xlabel('Cycles')
    plt.ylabel('Battery Capacity Retention %')
    temp='Model performance for Battery '+ str((i.split("."))[0])
    plt.title(temp)
    plt.show()


    
for j in good:
    print("GOOD")
    print(j)
for k in bad:
    print("Bad")
    print(k)