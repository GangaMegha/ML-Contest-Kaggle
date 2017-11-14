from scipy.stats import mode

import csv
import numpy as np
import pandas as pd
import pickle

def csv_writer(data,filename,mode):
	n = train_data.shape[1]
	with open(filename,mode) as fname:
		for i in range(n):
			if i!=n-1 :
				fname.write("f_{},".format(i+1))
			else :
				fname.write("f_{}\n".format(i+1))
		np.savetxt(fname,data,delimiter=',',newline='\n')

train_data = np.genfromtxt('../Dataset/train_cv_p1.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/valid_cv_p1.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_cv_p1.csv", delimiter=',')

data = np.zeros((train_data.shape[0]+validation_data.shape[0]+test_data.shape[0], train_data.shape[1]))

data[:train_data.shape[0]] = train_data
data[train_data.shape[0]:train_data.shape[0]+validation_data.shape[0]] = validation_data
data[train_data.shape[0]+validation_data.shape[0]:] = test_data

csv_writer(data, "data.csv","wb")
