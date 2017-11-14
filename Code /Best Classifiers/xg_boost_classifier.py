# XGBOOST

from scipy.stats import mode

import csv
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
# from xgboost import XGBClassifier


def csv_writer(data,filename,mode):
	np.savetxt(filename, data, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')

		# with open(filename,mode) as fname:
		# 	fname.write("id,label\n")
		# 	np.savetxt(fname,data,delimiter=',',newline='\n')



with open("../Results/xgbc_results2.txt", "w") as text_file:
	for i in range(5):

		# Read data from files
		#LDA
		# train_data = np.genfromtxt('../Dataset/train_cv_l{}.csv'.format(i+1), delimiter=',')
		# train_labels = np.genfromtxt('../Dataset/train_labels_{}.csv'.format(i+1), delimiter=',')
		# validation_data = np.genfromtxt("../Dataset/valid_cv_l{}.csv".format(i+1), delimiter=',')
		# validation_labels = np.genfromtxt("../Dataset/valid_labels_{}.csv".format(i+1), delimiter=',')
		# test_data = np.genfromtxt("../Dataset/test_cv_l{}.csv".format(i+1), delimiter=',')

		# #PCA
		train_data = np.genfromtxt('../Dataset/train_cv_p{}.csv'.format(i+1), delimiter=',')
		train_labels = np.genfromtxt('../Dataset/train_labels_{}.csv'.format(i+1), delimiter=',')
		validation_data = np.genfromtxt("../Dataset/valid_cv_p{}.csv".format(i+1), delimiter=',')
		validation_labels = np.genfromtxt("../Dataset/valid_labels_{}.csv".format(i+1), delimiter=',')
		test_data = np.genfromtxt("../Dataset/test_cv_p{}.csv".format(i+1), delimiter=',')


		# Validation
		print("\n\n\nRunning Validation..................................\n")

		xgb = XGBClassifier(n_estimators=1500)
		xgb.fit(train_data,train_labels)

		# Save model
		pickle.dump(xgb, open("../Models/xgbc{}_2.model".format(i+1), 'wb'))

		y_prediction = xgb.predict(validation_data)

		# if(i!=0):
		# 	item1 = np.column_stack((item1, y_prediction))
		# else:
		# 	item1 = y_prediction

		text_file.write("\nCross-Validation {}".format(i+1))
		text_file.write(classification_report(validation_labels, y_prediction))
		text_file.write("\n\n\n")

		# #Train
		# y_prediction = xgb.predict(train_data)

		# if(i!=0):
		# 	item2 = np.column_stack((item2, y_prediction))
		# else:
		# 	item2 = y_prediction

		# Test
		print("\n\nRunning test........................")

		y_prediction = xgb.predict(test_data)

		if(i!=0):
			item3 = np.column_stack((item3, y_prediction))
		else:
			item3 = y_prediction
		print(item3.shape)

# t1 = mode(np.array(item1), axis=1)[0].reshape(item1.shape[0])
	
# t2 = mode(np.array(item2), axis=1)[0].reshape(item2.shape[0])

t3 = mode(np.array(item3), axis=1)[0].reshape(item3.shape[0])

# t1 = np.column_stack((np.arange(len(t1)), t1))
# t2 = np.column_stack((np.arange(len(t2)), t2))
t3 = np.column_stack((np.arange(len(t3)), t3))
		
# csv_writer(t1, "../Results/xgb_valid.csv","wb")
# csv_writer(t2, "../Results/xgb_train.csv","wb")
csv_writer(t3, "../Results/xgbc_test2.csv","wb")

print("\n\nDone :)")