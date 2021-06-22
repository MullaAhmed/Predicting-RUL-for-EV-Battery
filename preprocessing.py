import scipy.io
import numpy as np
from datetime import datetime
import pandas as pd


def convert_to_time(hmm):
	return datetime(year=int(hmm[0]),month=int(hmm[1]),day=int(hmm[2]), hour=int(hmm[3]),minute=int(hmm[4]),second=int(hmm[5]))

def loadMat(matfile):
	data = scipy.io.loadmat(matfile)
	filename = matfile.split(".")[0] #B0005.mat --> B0005(str)
	col = data[filename]#data(var) is a dict(len 4- but the 1st 3 are not useful ) and pure data(value) is extracted here 
	col = col[0][0][0][0]
	size = col.shape[0] #shape returns lenght of multidimentional array as an array

	da = []

	for i in range(size):
		k=list(col[i][3][0].dtype.fields.keys())#k will return list properities(headings) from each cycle
		d1 = {}
		d2 = {}
		if str(col[i][0][0]) != 'impedance':#this will look for charging discharging impedence part
			for j in range(len(k)):
				t=col[i][3][0][0][j][0] # j here gives fields and t returns value of each field
				l=list(t)
				d2[k[j]]=l #after running through the j loop d2 will have all the info about 1 particular cycle
		
		d1['cycle']=str(col[i][0][0])
		d1['temp']=int(col[i][1][0])
		d1['time']=str(convert_to_time(col[i][2][0]))
		d1['data']=d2
		da.append(d1)

	return da






