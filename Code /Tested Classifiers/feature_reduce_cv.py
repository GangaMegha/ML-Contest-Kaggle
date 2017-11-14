import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')

# #Impute test
# test = np.genfromtxt("../../Dataset/test.csv", delimiter=',', missing_values='NaN', skip_header=1)	
# impute=Imputer(missing_values='NaN', strategy='median', axis=0)
# test=impute.fit_transform(test)
# test = preprocessing.scale(test)
# print(test[:15,:15])

# csv_writer(test,"../Dataset/test.csv","wb")

# Read data from files
for i in range(5):
	train = np.genfromtxt('../Dataset/X_train_{}.csv'.format(i+1), delimiter=',')
	valid = np.genfromtxt("../Dataset/X_validate_{}.csv".format(i+1), delimiter=',')
	train_labels = np.genfromtxt('../Dataset/train_labels_{}.csv'.format(i+1), delimiter=',')
	# test = np.genfromtxt("../Dataset/test.csv")
	test = np.genfromtxt("../Dataset/imputed_test_median.csv", delimiter=',')
	# print("\n\n\n")
	# print(test[:15,:15])
	# print(test.shape)
	train = preprocessing.scale(train[:,1:])
	valid = preprocessing.scale(valid[:,1:])
	test = preprocessing.scale(test)

	# LDA
	lda = LinearDiscriminantAnalysis(n_components=100)
	train_data = lda.fit_transform(train, train_labels)
	valid_data = lda.transform(valid)
	test_data = lda.transform(test)

	csv_writer(train_data,"../Dataset/train_cv_l{}.csv".format(i+1), "wb")
	csv_writer(valid_data,"../Dataset/valid_cv_l{}.csv".format(i+1), "wb")
	csv_writer(test_data,"../Dataset/test_cv_l{}.csv".format(i+1), "wb")

	#PCA
	pca = PCA(n_components=500)
	train_data = pca.fit_transform(train, train_labels)
	valid_data = pca.transform(valid)
	test_data = pca.transform(test)

	csv_writer(train_data,"../Dataset/train_cv_p{}.csv".format(i+1), "wb")
	csv_writer(valid_data,"../Dataset/valid_cv_p{}.csv".format(i+1), "wb")
	csv_writer(test_data,"../Dataset/test_cv_p{}.csv".format(i+1), "wb")