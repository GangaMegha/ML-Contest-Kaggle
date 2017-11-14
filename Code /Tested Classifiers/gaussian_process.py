import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

# Read data from files
#LDA
train_data = np.genfromtxt('../Dataset/train_mean_28_lda.csv', delimiter=',')
train_labels = np.genfromtxt('../Dataset/train_labels_mean_28_lda.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/validation_mean_28_lda.csv", delimiter=',')
validation_labels = np.genfromtxt("../Dataset/validation_labels_mean_28_lda.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_mean_28_lda.csv", delimiter=',')
X = np.genfromtxt("../Dataset/X_mean_28_lda.csv", delimiter=',')
labels = np.genfromtxt("../Dataset/train_labels.csv", delimiter=',')



#PCA
# train_data = np.genfromtxt('../Dataset/train_mean_500_pca.csv', delimiter=',')
# train_labels = np.genfromtxt('../Dataset/train_labels_mean_500_pca.csv', delimiter=',')
# validation_data = np.genfromtxt("../Dataset/validation_mean_500_pca.csv", delimiter=',')
# validation_labels = np.genfromtxt("../Dataset/validation_labels_mean_500_pca.csv", delimiter=',')
# test_data = np.genfromtxt("../Dataset/test_mean_500_pca.csv", delimiter=',')
# X = np.genfromtxt("../Dataset/X_mean_500_pca.csv", delimiter=',')
# labels = np.genfromtxt("../Dataset/train_labels.csv", delimiter=',')


# print(data.shape)
# print(labels.shape)


# # k-fold stratified cross validation
# skf = StratifiedKFold(n_splits=5)
# rf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=4, min_samples_leaf=2)

# print("\nRunning 5-fold cross validation...................................................\n")
# for train_indx, test_indx in skf.split(data, labels) :
# 	train_data, train_labels = data[train_indx], labels[train_indx]
# 	test_data, test_labels = data[test_indx], labels[test_indx]

# 	rf.fit(train_data,train_labels)
# 	y_prediction = rf.predict(test_data)
# 	print("\n\nResults....\n\n")
# 	print(classification_report(test_labels, y_prediction))



# Validation
print("\n\n\nRunning Validation..................................\n")

kernel = 1.0 * RBF(np.ones(train_data.shape[1]))
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel)
gpc_rbf_anisotropic.fit(train_data,train_labels)

y_prediction = rf.predict(validation_data)

print(classification_report(validation_labels, y_prediction))



# Test
print("\n\nRunning test........................")
kernel = 1.0 * RBF(np.ones(train_data.shape[1]))
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel)
gpc_rbf_anisotropic.fit(X, labels)

y_prediction = rf.predict(test_data)
item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_gaussian.csv","wb")

print("\n\nDone :)")
