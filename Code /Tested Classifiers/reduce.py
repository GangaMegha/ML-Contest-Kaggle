import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA,KernelPCA
import sklearn.metrics as met

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


train_data=preprocessing.scale(np.loadtxt('../dataset/clean_train.csv', delimiter=','))
test_data=preprocessing.scale(np.loadtxt('../dataset/clean_test.csv', delimiter=','))

train_labels=np.loadtxt('../dataset/train_labels.csv', delimiter=',')
X_train, X_validate, Y_train, Y_validate = train_test_split(train_data, train_labels,stratify=train_labels, test_size=0.15)
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1

pca =KernelPCA(n_components=800, kernel='rbf', gamma=10**-2, kernel_params=None, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1)
#pca=PCA(n_components=500)
X_train=pca.fit_transform(X_train)
X_validate=pca.transform(X_validate)
X_test=pca.transform(test_data)
print(X_train.shape)
csv_writer(X_train,"../dataset/X_train_kp.csv","wb")
csv_writer(X_test,"../dataset/X_test_kp.csv","wb")
csv_writer(X_validate,"../dataset/X_validate_kp.csv","wb")
#csv_writer(Y_validate,"../dataset/Y_validate.csv","wb")
csv_writer(Y_train,"../dataset/Y_train.csv","wb")


