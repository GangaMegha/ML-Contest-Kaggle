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
#print("BEFORE")
train_data=(np.loadtxt('clean_train.csv', delimiter=','))
test_data=(np.loadtxt('clean_test.csv', delimiter=','))
#select=[200,400,500,800,1000,1200]
train_labels=np.loadtxt('train_labels.csv', delimiter=',')
X_train, X_validate, Y_train, Y_validate = train_test_split(train_data, train_labels,stratify=train_labels, test_size=0.25)
Y_out=np.zeros([test_data.shape[0],2])
Y_out[:,0]=np.arange(test_data.shape[0])
select=[200,300,400]
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1
for ncomp in select:

	pca =PCA(n_components=ncomp)

	X_train_red=pca.fit_transform(X_train)
	X_validate_red=pca.transform(X_validate)
	X_test=pca.transform(test_data)

	mlp=MLPClassifier(hidden_layer_sizes=(900,500,300), activation='relu', solver='adam', 
	alpha=0.0005, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.0005, 
	power_t=0.5, max_iter=15000, shuffle=True, random_state=5, tol=0.0001, verbose=False, 
	warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
	validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	mlp.fit(X_train_red,Y_train)
	Y_val=mlp.predict(X_validate_red)
	print(mlp.score(X_validate_red,Y_validate))

	Y_out[:,1]=mlp.predict(X_test)
	print(met.classification_report(Y_validate,Y_val, target_names=class_labels))

	np.savetxt("../dataset/models/one_last_try_Y_out_p"+str(ncomp)+".csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')



