import pandas as pd
import numpy as np
from pandas import concat
from sklearn.model_selection import train_test_split

#gets value of capacity vale from every discharging cycle
def getBatteryCapcity(Battery):
    cycle = []
    capacity = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            capacity.append(Bat['data']['Capacity'][0])
            i += 1
    return [cycle, capacity]

#calculates battery retention percentage
def getBatteryCapacityRetention(capacity):
    retention=[]
    cycle=[]

    for i in range (len(capacity)):
        temp=(i/capacity[0])*100
        retention.append(temp)
        cycle.append(i+1)        
    return[cycle,capacity]
    print(cycle[-1])

#getting all values from charging cycle (2d array)
def getChargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return [index, Battery['Voltage_measured'], Battery['Current_measured'], Battery['Temperature_measured'], Battery['Voltage_charge'], Battery['Time']]

#getting all values from discharging cycle (2d array)
def getDischargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return (pd.DataFrame({"Voltage measured": Battery['Voltage_measured'],"Current measured" : Battery['Current_measured'], "Temperature measured": Battery['Temperature_measured'],"Voltage load": Battery['Voltage_load'],"Time" : Battery['Time']}))

#gets maximum discharging temperature from each discharging
def getMaxDischargeTemp(Battery):
    cycle = []
    temp = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle, temp]

def getMaxChargeTemp(Battery):
    cycle = []
    temp = []
    i = 1
    x=len(getMaxDischargeTemp(Battery)[0])+1
    for Bat in Battery :
        if Bat['cycle'] == 'charge' and i<x :
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle, temp]#change was made here

#makes a dataframe
def getDataframe(Battery):
    l = getBatteryCapcity(Battery)
    l1 = getMaxDischargeTemp(Battery)
    l2 = getMaxChargeTemp(Battery)#no. of charge and discharge temps are different
    data = {'cycle':l[0],'capacity':l[1], 'max_discharge_temp':l1[1], 'max_charge_temp':l2[1]}
    return pd.DataFrame(data)


#same as the one used for stocks(try with exp / weighted moving avgs)
def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    #makes a [1,1,1...] (len=window_size)
    #divides those 1 by window_size and returns that array

    return np.convolve(data, window, 'same')
    #convolve function smae mode(there are 2 other modes too)(returns same lenght as input)
    #check notes

#select a time frame say 5 days values are 2,3,4,5,6 on respective days and 11 on 6th day
#moving avg= (2+3+4+5+6)/5 on 5th day
#moving avg=(3+4+5+6+8)/5 on the 6ht day

def rollingAverage(x_stuff, y_stuff):
    window_size = 10

    avg = moving_average(y_stuff, window_size) #this will give moving averages of capacity with window of 10
    avg_list=list(avg)
    residual = y_stuff - avg
    
    
    testing_std = residual.rolling(window_size).std()

    
    testing_std_as_df = pd.DataFrame(testing_std)
  
    
    rolling_std = testing_std_as_df.replace(np.nan,testing_std_as_df.iloc[window_size - 1]).round(3).iloc[:,0].tolist()

    std = np.std(residual)
    lst=[]
    lst_index = 0
    lst_count = 0
    for i in y_stuff.index:
        if (y_stuff[i] > avg_list[lst_index] + (1.5 * rolling_std[lst_index])) | (y_stuff[i] < avg_list[lst_index] - (1.5 * rolling_std[lst_index])):
            lt=[i,x_stuff[i], y_stuff[i],avg_list[lst_index],rolling_std[lst_index]]
            lst.append(lt)
            lst_count+=1
        lst_index+=1

    lst_x = []
    lst_y = []

    for i in range (0,len(lst)):
        lst_x.append(lst[i][1])
        lst_y.append(lst[i][2])

    return lst_x, lst_y


