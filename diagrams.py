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

# battery capacity
B0005_capacity = getBatteryCapcity(B0005)
B0006_capacity = getBatteryCapcity(B0006)
B0007_capacity = getBatteryCapcity(B0007)
B0018_capacity = getBatteryCapcity(B0018)

# plotting disintegrating of battery capacity
plt.plot(B0005_capacity[0], B0005_capacity[1], color='blue', label='Battery-05')
plt.plot(B0006_capacity[0], B0006_capacity[1], color='green', label='Battery-06')
plt.plot(B0007_capacity[0], B0007_capacity[1], color='red', label='Battery-07')
plt.plot(B0018_capacity[0], B0018_capacity[1], color='purple', label='Battery-18')
plt.xlabel('Discharge cycles')
plt.ylabel('Capacity/Ah')
plt.title('Capacity degradation at ambient temperature of 24°C')
plt.legend()
plt.show()  



# capacity retention percent
B0005_capacity_retention = getBatteryCapacityRetention(B0005_capacity[1])

#Plotting(battery retention % remains same for 1 ambient temperature) 
plt.plot(B0005_capacity_retention[0], B0005_capacity_retention[1], color='blue', label='Battery-05')
plt.xlabel('Discharge cycles')
plt.ylabel('Capacity retention percentage')
plt.title('Capacity degradation at ambient temperature of 24°C')
plt.legend()
plt.show()  

pred_range = []
for i in range(1, 270):
    pred_range.append(i)

#brute force linear fitting
poly = np.polyfit(B0005_capacity[0], B0005_capacity[1], 1)
B0005_capacity_pred = np.polyval(poly, pred_range)

plt.plot(B0005_capacity[0], B0005_capacity[1], color='orange', label='24°C (Battery-05)')
plt.plot(pred_range, B0005_capacity_pred, color='orange')
plt.xlabel('Discharge cycles')
plt.ylabel('Capacity/Ah')
plt.title('Capacity degradation ')
plt.legend()
plt.show()
