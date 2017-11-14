import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics as met

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')
#print("BEFORE")
train_data=preprocessing.scale(np.loadtxt('../../datasets/csv_file/imputed_train_median.csv', delimiter=','))
test_data=preprocessing.scale(np.loadtxt('../../datasets/csv_file/imputed_test_median.csv', delimiter=','))

train_labels=np.loadtxt('../../datasets/csv_file/train_labels.csv', delimiter=',')
X_train, X_validate, Y_train, Y_validate = train_test_split(train_data, train_labels,stratify=train_labels, test_size=0.25)
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1

lda =LinearDiscriminantAnalysis(n_components=80)

X_train=lda.fit_transform(X_train,Y_train)
X_validate=lda.transform(X_validate)
X_test=lda.transform(test_data)
print(X_train.shape)
csv_writer(X_train,"../../datasets/csv_file/X_train_80.csv","wb")
csv_writer(X_test,"../../datasets/csv_file/X_test_80.csv","wb")
csv_writer(X_validate,"../../datasets/csv_file/X_validate_80.csv","wb")
csv_writer(Y_validate,"../../datasets/csv_file/Y_validate.csv","wb")
csv_writer(Y_train,"../../datasets/csv_file/Y_train.csv","wb")


