from scipy.stats import mode

import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA
import sklearn.metrics as met


class_labels=[]
for i in range(1,30):
	class_labels.append('class'+str(i))

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')

the_good_ones=['models/one_last_try_Y_out_1200_300_300_cross_val2_p150.csv','models/one_last_try_Y_out_1200_300_300_cross_val2_p100.csv','models/one_last_try_Y_out_1200_300_300_cross_val2_p80.csv','models/one_last_try_Y_out_1200_300_300_cross_val1_p60.csv',
'models/one_last_try_Y_out_3_cross_val3_p80.csv','models/one_last_try_Y_out_800_300_300_cross_val2_p100.csv','models/one_last_try_Y_out_800_300_300_cross_val2_p60.csv','models/one_last_try_Y_out_3_cross_val3_p80.csv','models/one_last_try_Y_out_700_1400_400_cross_val1_p40.csv']
#print("BEFORE")
X=np.loadtxt("models/one_last_try_Y_out_p200.csv",delimiter=',', skiprows=1)[:,1]
X_out=np.zeros([X.shape[0],len(the_good_ones)])
Y_out=np.zeros([X.shape[0],2])
Y_out[:,0]=np.arange(X_out.shape[0])
							

select=[200,300,400]
i=0
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1
for output in the_good_ones:

	

	X_out[:,i]=np.loadtxt(output,delimiter=',', skiprows=1)[:,1]
	#print(met.classification_report(Y_validate,Y_val, target_names=class_labels))


	i+=1


Y_out[:,1]=(mode(X_out,axis=1)[0]).reshape(X_out.shape[0])	
np.savetxt("models/fourth_last_try_combine.csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
